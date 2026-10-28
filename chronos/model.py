"""One diffusion transformer: physical time supplies temporal position."""

from dataclasses import dataclass
import math
import re

import numpy as np
import torch
from torch import nn
from timm.layers import Mlp, RmsNorm

from .attention import CrossAttention, SelfAttention, time_angles, time_frequencies


def projector(input_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(approximate="tanh"),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(approximate="tanh"),
        nn.Linear(hidden_dim, hidden_dim),
    )


def source_group(name):
    """Keep camera identity separate from arm identity (including bimanual cameras)."""
    name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if "agentview" in name or re.search(r"(?:^|_)global(?:view|camera)?(?:_|$)", name):
        return "global"
    if "right_hand" in name:
        return "arm0"
    if "left_hand" in name:
        return "arm1"
    match = re.search(r"(?:^|_)shoulder_?camera_?(\d+)(?:_|$)", name)
    match = match or re.search(r"(?:^|_)robot_?(\d+)(?:_|$)", name)
    return f"arm{int(match[1])}" if match else f"source:{name}"


def identity_embedding(count, dim):
    """Sin/cos initialization of learned identity embeddings, not temporal slots."""
    phase = np.outer(np.arange(count), 1 / 10000 ** (np.arange(dim // 2) / (dim / 2)))
    return torch.from_numpy(np.concatenate((np.sin(phase), np.cos(phase)), -1)).float()[None]


def grid_embedding(shape, dim):
    """The original baseline's frame/row/column positional initialization."""
    result = torch.zeros(*shape, dim)
    axes = [axis for axis, size in enumerate(shape) if size > 1]
    width = (dim // max(1, len(axes))) // 2 * 2
    for index, axis in enumerate(axes):
        view = [1] * len(shape) + [width]
        view[axis] = shape[axis]
        result[..., index * width : (index + 1) * width] = identity_embedding(
            shape[axis], width
        ).reshape(view)
    return result.reshape(1, -1, dim)


class DiffusionStep(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, step):
        half = self.dim // 2
        frequency = torch.exp(
            torch.arange(half, device=step.device) * -(math.log(10000) / (half - 1))
        )
        phase = step[..., None] * frequency
        return torch.cat((phase.sin(), phase.cos()), -1)


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = (RmsNorm(dim, eps=1e-6) for _ in range(3))
        self.attn = SelfAttention(dim, heads)
        self.cross_attn = CrossAttention(dim, heads)
        self.ff = Mlp(dim, hidden_features=dim, act_layer=nn.GELU)

    def forward(self, x, kv, angles, valid, image_valid):
        x = x + self.attn(self.norm1(x), angles, valid)
        x = x + self.cross_attn(self.norm2(x), kv, angles, image_valid)
        return x + self.ff(self.norm3(x))


@dataclass
class Context:
    proprio: torch.Tensor
    images: torch.Tensor
    query_angles: tuple | None
    image_angles: tuple | None
    proprio_valid: torch.Tensor
    action_valid: torch.Tensor
    image_valid: torch.Tensor | None
    input_position: torch.Tensor
    kv: tuple | None = None


class ChronosTransformer(nn.Module):
    def __init__(
        self,
        action_dim,
        cameras,
        proprio_keys,
        *,
        feature_dim=768,
        dim=512,
        layers=16,
        heads=8,
        patch_grid=16,
        rotary_pairs=16,
        positional_encoding="time",
        observation_horizons=None,
        horizon=16,
        diffusion_conditioning="additive",
        mask_self_attention=True,
    ):
        super().__init__()
        if min(feature_dim, dim, layers, heads, patch_grid, rotary_pairs) < 1:
            raise ValueError("Model dimensions and layer counts must be positive")
        if positional_encoding not in ("time", "index"):
            raise ValueError("positional_encoding must be 'time' or 'index'")
        if diffusion_conditioning not in ("additive", "token"):
            raise ValueError("diffusion_conditioning must be 'additive' or 'token'")
        if positional_encoding == "time" and diffusion_conditioning != "additive":
            raise ValueError("Chronos adds diffusion time to actions without an untimed token")
        if (
            dim % heads
            or dim % 4
            or (positional_encoding == "time" and 2 * rotary_pairs > dim // heads)
        ):
            raise ValueError(
                "Dimensions must fit the attention heads, RoPE pairs, and 2D patch grid"
            )
        self.cameras, self.proprio_keys = tuple(cameras), tuple(proprio_keys)
        self.positional_encoding = positional_encoding
        self.diffusion_conditioning = diffusion_conditioning
        self.mask_self_attention = mask_self_attention
        self.observation_horizons = dict(
            observation_horizons or {name: 2 for name in (*cameras, *proprio_keys)}
        )
        groups = {source_group(name) for name in (*cameras, *proprio_keys)}
        self.groups = sorted(
            groups,
            key=lambda g: (
                (0, 0)
                if g == "global"
                else ((1, int(g[3:])) if re.fullmatch(r"arm\d+", g) else (2, g))
            ),
        )
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        if positional_encoding == "time":
            self.role = nn.Parameter(identity_embedding(5, dim))
            self.arm = nn.Parameter(identity_embedding(len(self.groups), dim))
            self.proprio_source = nn.Parameter(identity_embedding(len(proprio_keys), dim))
            self.camera_source = nn.Parameter(identity_embedding(len(cameras), dim))
            axis = identity_embedding(patch_grid, dim // 2)[0]
            spatial = torch.cat(
                (
                    axis[:, None, :].expand(-1, patch_grid, -1),
                    axis[None, :, :].expand(patch_grid, -1, -1),
                ),
                -1,
            ).reshape(1, patch_grid**2, dim)
            self.spatial = nn.Parameter(spatial if patch_grid > 1 else torch.zeros_like(spatial))
        else:
            if any(
                self.observation_horizons.get(name, 0) < 1 for name in (*cameras, *proprio_keys)
            ):
                raise ValueError("Index encoding needs a positive history length for each stream")
            length = horizon + sum(self.observation_horizons[name] for name in proprio_keys)
            length += int(diffusion_conditioning == "token")
            frames = sum(self.observation_horizons[name] for name in cameras)
            self.input_position = nn.Parameter(identity_embedding(length, dim))
            self.image_position = nn.Parameter(
                grid_embedding((frames, patch_grid, patch_grid), dim)
            )
        self.diffusion_step = nn.Sequential(
            DiffusionStep(dim), nn.Linear(dim, dim * 4), nn.Mish(), nn.Linear(dim * 4, dim)
        )
        self.action_projector = projector(action_dim, dim)
        self.image_projector = projector(feature_dim, dim)
        self.proprio_projector = projector(feature_dim, dim)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(layers))
        self.final_norm = RmsNorm(dim, eps=1e-6)
        self.final = Mlp(
            dim,
            hidden_features=dim,
            out_features=action_dim,
            act_layer=lambda: nn.GELU(approximate="tanh"),
        )
        self.register_buffer("frequencies", time_frequencies(rotary_pairs), persistent=False)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def prepare(self, features, times, action_times, valid, action_valid, *, cache=False):
        """Center clocks before float32 conversion; all masks use True = usable."""
        if cache and self.training:
            raise ValueError("Condition caching requires model.eval()")
        temporal = self.positional_encoding == "time"
        origin = action_times[:, :1] if temporal else None
        if not temporal:
            for name, value in features.items():
                if value.shape[1] != self.observation_horizons[name]:
                    raise ValueError(
                        f"Index encoding expects {self.observation_horizons[name]} frames for {name}"
                    )
        proprio, positions, query_times, proprio_masks = [], [], [], []
        images, image_positions, image_times, image_masks = [], [], [], []
        for i, name in enumerate(self.proprio_keys):
            value = features[name]
            proprio.append(value)
            if temporal:
                arm = self.arm[:, self.groups.index(source_group(name)) :][:, :1]
                position = self.proprio_source[:, i : i + 1] + arm + self.role[:, 1:2]
                positions.append(position.expand(-1, value.shape[1], -1))
                query_times.append((times[name] - origin).float())
            proprio_masks.append(valid[name])
        for i, name in enumerate(self.cameras):
            value = features[name]
            images.append(value.flatten(1, 2))
            if temporal:
                arm = self.arm[:, self.groups.index(source_group(name)) :][:, :1]
                position = (
                    self.camera_source[:, i : i + 1, None]
                    + arm[:, :, None]
                    + self.spatial[:, None]
                    + self.role[:, 3:4, None]
                )
                image_positions.append(position.expand(-1, value.shape[1], -1, -1).flatten(1, 2))
                image_times.append(
                    (times[name] - origin).float().repeat_interleave(value.shape[2], 1)
                )
            image_masks.append(valid[name].repeat_interleave(value.shape[2], 1))
        if temporal:
            positions.append(self.role[:, 2:3].expand(-1, action_times.shape[1], -1))
            query_times.append((action_times - origin).float())
        context = Context(
            torch.cat(proprio, 1),
            self.image_projector(torch.cat(images, 1))
            + (torch.cat(image_positions, 1) if temporal else self.image_position),
            time_angles(torch.cat(query_times, 1), self.frequencies) if temporal else None,
            time_angles(torch.cat(image_times, 1), self.frequencies) if temporal else None,
            torch.cat(proprio_masks, 1),
            action_valid,
            torch.cat(image_masks, 1),
            torch.cat(positions, 1) if temporal else self.input_position,
        )
        if cache:
            context.kv = tuple(
                block.cross_attn.project_kv(context.images, context.image_angles)
                for block in self.blocks
            )
        return context

    def forward(self, noisy_actions, step, context):
        action = self.action_projector(noisy_actions)
        action = torch.where(
            context.action_valid[..., None], action, self.mask_token.type_as(action)
        )
        proprio = self.proprio_projector(context.proprio)
        proprio = torch.where(
            context.proprio_valid[..., None], proprio, self.mask_token.type_as(proprio)
        )
        diffusion = self.diffusion_step(step)[:, None]
        if self.diffusion_conditioning == "token":
            x = torch.cat((diffusion, proprio, action), 1)
        else:
            x = torch.cat((proprio, action + diffusion), 1)
        x = x + context.input_position
        valid = torch.cat((context.proprio_valid, context.action_valid), 1)
        if self.diffusion_conditioning == "token":
            valid = torch.cat((torch.ones_like(valid[:, :1]), valid), 1)
        if not self.mask_self_attention:
            valid = None
        for i, block in enumerate(self.blocks):
            kv = (
                context.kv[i]
                if context.kv is not None
                else block.cross_attn.project_kv(context.images, context.image_angles)
            )
            x = block(x, kv, context.query_angles, valid, context.image_valid)
        return self.final(self.final_norm(x[:, -noisy_actions.shape[1] :]))
