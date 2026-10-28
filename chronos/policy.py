"""A compact diffusion policy around the Chronos transformer."""

import torch
from torch import nn
from diffusers import DDIMScheduler

from .model import ChronosTransformer


class ChronosPolicy(nn.Module):
    """Inputs are normalized proprioception and RGB [0,1] or encoded image patches.

    Each observation has its own [batch, history] capture-time array in seconds.
    Action times are [batch, horizon] physical landing times on that same clock.
    """

    def __init__(
        self,
        action_dim,
        proprio_dims,
        cameras,
        *,
        vision=None,
        feature_dim=768,
        horizon=16,
        dim=512,
        layers=16,
        heads=8,
        patch_grid=16,
        rotary_pairs=16,
        train_steps=24,
        inference_steps=8,
        prefix_length=0,
        prefix_weight=1.0,
        seed=0,
        positional_encoding="time",
        observation_horizons=None,
        diffusion_conditioning="additive",
        use_validity_masks=True,
        mask_self_attention=True,
        loss_per_sample=True,
    ):
        super().__init__()
        if not cameras or not proprio_dims or action_dim < 1 or horizon < 1:
            raise ValueError("Provide cameras, proprioception streams, and a positive action shape")
        if train_steps < inference_steps or train_steps % inference_steps:
            raise ValueError("Training diffusion steps must be divisible by inference steps")
        if not 0 <= prefix_length <= horizon or prefix_weight <= 0:
            raise ValueError("Invalid executed-prefix loss weights")
        self.action_dim, self.horizon, self.seed = action_dim, horizon, seed
        self.positional_encoding = positional_encoding
        self.use_validity_masks = use_validity_masks
        self.loss_per_sample = loss_per_sample
        self.cameras, self.proprio_keys = tuple(cameras), tuple(proprio_dims)
        self.vision = vision
        self.proprio = nn.ModuleDict(
            {name: nn.Linear(width, feature_dim) for name, width in proprio_dims.items()}
        )
        self.model = ChronosTransformer(
            action_dim,
            cameras,
            self.proprio_keys,
            feature_dim=feature_dim,
            dim=dim,
            layers=layers,
            heads=heads,
            patch_grid=patch_grid,
            rotary_pairs=rotary_pairs,
            positional_encoding=positional_encoding,
            observation_horizons=observation_horizons,
            horizon=horizon,
            diffusion_conditioning=diffusion_conditioning,
            mask_self_attention=mask_self_attention,
        )
        self.scheduler = DDIMScheduler(
            num_train_timesteps=train_steps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="sample",
        )
        self.inference_steps = inference_steps
        self.prefix_length, self.prefix_weight = prefix_length, prefix_weight
        self._generators = {}

    def reset(self):
        """Reset per-episode diffusion-noise streams between independent rollouts."""
        self._generators.clear()

    def _context(
        self, observations, times, action_times, valid=None, action_valid=None, *, cache=False
    ):
        first = observations[self.cameras[0]]
        temporal = self.positional_encoding == "time"
        if not temporal:
            # No clock values, including supplied ones, reach the index baseline.
            action_times = torch.zeros(first.shape[0], self.horizon, device=first.device)
        if action_times is None or action_times.ndim != 2 or action_times.shape[1] != self.horizon:
            raise ValueError("action_times must have shape [batch, horizon]")
        if temporal and (
            not torch.isfinite(action_times).all() or not (action_times.diff(dim=1) > 0).all()
        ):
            raise ValueError("Action times must be finite and strictly increasing")
        valid = dict(valid or {})
        if not self.use_validity_masks:
            valid, action_valid = {}, None
        features = {}
        for name in (*self.cameras, *self.proprio_keys):
            value = observations[name]
            if value.shape[0] != action_times.shape[0]:
                raise ValueError("Observation batch sizes differ")
            if temporal and (times is None or times[name].shape != value.shape[:2]):
                raise ValueError(f"Observation/timestamp shape mismatch for {name}")
            if value.device != action_times.device or (
                temporal and times[name].device != value.device
            ):
                raise ValueError("Values and clocks must share a device")
            if temporal and not torch.isfinite(times[name]).all():
                raise ValueError("Capture times must be finite")
            if name not in valid:
                valid[name] = torch.ones(value.shape[:2], dtype=torch.bool, device=value.device)
            if valid[name].shape != value.shape[:2] or valid[name].dtype != torch.bool:
                raise ValueError("Observation validity must be boolean [batch, history]")
            if name in self.proprio:
                features[name] = self.proprio[name](value)
            elif value.ndim == 4:
                features[name] = value  # Already encoded [batch, frames, patches, feature_dim].
            elif value.ndim == 5 and self.vision is not None:
                rgb = value.flatten(0, 1)
                encoded = self.vision(pixel_values=(rgb - 0.5) / 0.5).last_hidden_state
                features[name] = encoded.reshape(*value.shape[:2], *encoded.shape[1:])
            else:
                raise ValueError("Supply image patch features, or RGB with a vision encoder")
        if action_valid is None:
            action_valid = torch.ones_like(action_times, dtype=torch.bool)
        if action_valid.shape != action_times.shape or action_valid.dtype != torch.bool:
            raise ValueError("Action validity must be boolean [batch, horizon]")
        context = self.model.prepare(
            features, times, action_times, valid, action_valid, cache=cache
        )
        if not self.use_validity_masks:
            context.image_valid = None
        return context

    def loss(
        self,
        observations,
        times=None,
        action_times=None,
        actions=None,
        *,
        valid=None,
        action_valid=None,
        noise=None,
        diffusion_step=None,
    ):
        """Predict clean, future-aligned actions; normalize masked L1 within each sample."""
        expected = (observations[self.cameras[0]].shape[0], self.horizon, self.action_dim)
        if actions is None or actions.shape != expected:
            raise ValueError("actions must have shape [batch, horizon, action_dim]")
        context = self._context(observations, times, action_times, valid, action_valid)
        if noise is None:
            noise = torch.randn_like(actions)
        if diffusion_step is None:
            diffusion_step = torch.randint(
                self.scheduler.config.num_train_timesteps,
                (actions.shape[0],),
                device=actions.device,
            )
        noisy = self.scheduler.add_noise(actions, noise, diffusion_step)
        predicted = self.model(noisy, diffusion_step, context)
        weights = torch.ones(self.horizon, device=actions.device, dtype=actions.dtype)
        weights[: self.prefix_length] = self.prefix_weight
        weights = (context.action_valid.to(actions.dtype) * weights)[..., None].expand_as(actions)
        error = (predicted - actions).abs() * weights
        if self.loss_per_sample:
            return (error.sum((1, 2)) / weights.sum((1, 2)).clamp_min(1)).mean()
        return error.sum() / weights.sum().clamp_min(1)

    @torch.no_grad()
    def predict(
        self,
        observations,
        times=None,
        action_times=None,
        *,
        valid=None,
        action_valid=None,
        episode_ids=None,
        noise=None,
        cache=True,
    ):
        """Run DDIM. Pass explicit noise for comparisons, or episode IDs for seeded rollouts."""
        if self.training:
            raise ValueError("Call policy.eval() before prediction")
        context = self._context(observations, times, action_times, valid, action_valid, cache=cache)
        batch_size, horizon = context.action_valid.shape
        device = context.images.device
        if noise is None:
            if episode_ids is None or len(episode_ids) != batch_size:
                raise ValueError("Provide one episode ID per batch item, or explicit noise")
            samples = []
            for episode in episode_ids:
                key = (str(device), int(episode))
                if key not in self._generators:
                    seed = (
                        self.seed * 6_364_136_223_846_793_005
                        + int(episode) * 1_442_695_040_888_963_407
                        + 1_013_904_223
                    ) % (2**63 - 1)
                    self._generators[key] = torch.Generator(device=device).manual_seed(seed)
                samples.append(
                    torch.randn(
                        (self.horizon, self.action_dim),
                        device=device,
                        generator=self._generators[key],
                    )
                )
            noise = torch.stack(samples)
        if noise.shape != (batch_size, horizon, self.action_dim):
            raise ValueError("noise must have shape [batch, horizon, action_dim]")
        self.scheduler.set_timesteps(self.inference_steps)
        action = noise
        for step in self.scheduler.timesteps.tolist():
            timestep = torch.full((action.shape[0],), float(step), device=action.device)
            predicted = self.model(action, timestep, context)
            action = self.scheduler.step(predicted, step, action).prev_sample
        return action
