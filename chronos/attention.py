"""The Chronos attention change: rotate Q/K by physical time in seconds."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import RmsNorm


def time_frequencies(pairs=16, min_period=0.1, max_period=2.0):
    """Angular frequencies spanning the physical period band in the paper."""
    if pairs < 1 or not 0 < min_period <= max_period:
        raise ValueError("Expected positive rotary pairs and 0 < min_period <= max_period")
    periods = torch.linspace(
        math.log(min_period), math.log(max_period), pairs, dtype=torch.float64
    ).exp()
    return (2 * math.pi / periods).float()


def time_angles(seconds, frequencies):
    """Compute once per request; timestamps have already been centered in float64."""
    angles = seconds.float()[:, None, :, None] * frequencies[None, None, None, :]
    return angles.cos(), angles.sin()


def rotate(x, angles):
    """x: [batch, heads, tokens, head_dim]; leave the remaining channels unchanged."""
    if angles is None:  # The index-position baseline uses ordinary attention.
        return x
    cos, sin = (factor.to(x.dtype) for factor in angles)
    width = cos.shape[-1] * 2
    even, odd = x[..., :width:2], x[..., 1:width:2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return torch.cat((rotated.flatten(-2), x[..., width:]), dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = RmsNorm(self.head_dim)
        self.k_norm = RmsNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, angles, valid=None):
        batch, length, dim = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(batch, length, 3, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        q = rotate(self.q_norm(q), angles)
        k = rotate(self.k_norm(k), angles)
        mask = None if valid is None else valid[:, None, None, :]
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(x.transpose(1, 2).reshape(batch, length, dim))


class CrossAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.q_norm = RmsNorm(self.head_dim)
        self.k_norm = RmsNorm(self.head_dim)
        self.linear = nn.Linear(dim, dim)

    def project_kv(self, images, angles):
        batch, length, _ = images.shape
        k, v = (
            self.kv(images)
            .reshape(batch, length, 2, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        return rotate(self.k_norm(k), angles), v.contiguous()

    def forward(self, x, kv, angles, valid=None):
        batch, length, dim = x.shape
        q = self.q(x).reshape(batch, length, self.heads, self.head_dim).permute(0, 2, 1, 3)
        q = rotate(self.q_norm(q), angles)
        # Match the evaluated cross-attention mask layout, including all-invalid rows.
        mask = (
            None
            if valid is None
            else valid[:, None, None, :].expand(batch, self.heads, length, valid.shape[1])
        )
        x = F.scaled_dot_product_attention(q, *kv, attn_mask=mask)
        x = self.linear(x.transpose(1, 2).reshape(batch, length, dim))
        return x if valid is None else x * valid.any(1)[:, None, None].to(x.dtype)
