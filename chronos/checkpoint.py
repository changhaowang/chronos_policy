"""Load published Chronos or original DT weights without the research framework."""

from dataclasses import dataclass

import torch
import yaml

from .policy import ChronosPolicy


@dataclass
class PaperCheckpoint:
    policy: ChronosPolicy
    metadata: dict
    normalizers: dict
    action_keys: tuple

    def normalize(self, observations):
        result = dict(observations)
        for name, value in observations.items():
            if self.metadata[name]["normalizer"] == "identity":
                continue
            scale, offset = self.normalizers[name]
            result[name] = (value - offset.to(value)) / scale.to(value)
        return result

    def decode(self, actions):
        """Split the action vector into named controller streams, then undo normalization."""
        result, start = {}, 0
        for name in self.action_keys:
            width = int(self.metadata[name]["shape"][0])
            value = actions[..., start : start + width]
            scale, offset = self.normalizers[name]
            result[name] = value * scale.to(value) + offset.to(value)
            start += width
        return result


def load_paper_checkpoint(path, device="cpu"):
    """Load a trusted Chronos or timestamp-blind DT checkpoint with its original settings."""
    from transformers import SiglipVisionConfig, SiglipVisionModel

    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    raw_config = checkpoint["cfg_str_unresolved"]
    if "${" in raw_config:
        raise ValueError(
            "Expected an original resolved paper checkpoint; use the pinned runner for other formats"
        )
    config = yaml.safe_load(raw_config)
    model = config["workspace"]["model"]
    dataset = config["workspace"]["train_dataset"]
    network = model["denoising_network_partial"]
    rope = network["rope"]
    temporal = bool(rope["enabled"])
    required = {
        "enabled": True,
        "input_unit": "seconds",
        "position_mode": "factorized",
        "frequency_mode": "physical_periods",
        "time_scale": 1.0,
        "min_period_seconds": 0.1,
        "max_period_seconds": 2.0,
    }
    if temporal and any(rope.get(key) != value for key, value in required.items()):
        raise ValueError("This loader supports the published physical-time Chronos model")
    if (
        network["diffusion_conditioning"] != ("additive" if temporal else "token")
        or network["projector_type"] != "mlp3x_gelu"
        or network.get("relative_value", {}).get("enabled", False)
        or network.get("action_latency_conditioning", {}).get("enabled", False)
    ):
        raise ValueError("Checkpoint uses an architecture outside the minimal Chronos model")
    image_config = model["global_cond_encoder"]["image_encoder_partial"]
    if image_config["feature_aggregation"] != "patches" or not image_config["apply_image_norm"]:
        raise ValueError("Expected normalized SigLIP patch features")
    metadata = dataset["output_data_meta"]
    if any(entry["normalizer"] not in ("identity", "range") for entry in metadata.values()):
        raise ValueError("Expected the paper identity/range normalizers")
    image_name = image_config["model_name"]
    vision_config = SiglipVisionConfig.from_pretrained(image_name)
    vision = SiglipVisionModel(vision_config)
    action_keys = tuple(dataset["action_keys"])
    use_masks = model.get("use_temporal_validity_masks")
    if use_masks is None:
        use_masks = temporal
    policy = ChronosPolicy(
        action_dim=sum(int(metadata[name]["shape"][0]) for name in action_keys),
        proprio_dims={name: int(metadata[name]["shape"][0]) for name in dataset["proprio_keys"]},
        cameras=dataset["image_keys"],
        vision=vision,
        feature_dim=vision_config.hidden_size,
        horizon=int(model["action_length"]),
        dim=int(network["hidden_dim"]),
        layers=int(network["layer_num"]),
        heads=int(network["head_num"]),
        patch_grid=vision_config.image_size // vision_config.patch_size,
        rotary_pairs=int(rope["rotary_pair_num"] or 16),
        seed=int(model["seed"]),
        train_steps=int(model["noise_scheduler"]["train_step_num"]),
        inference_steps=int(model["noise_scheduler"]["inference_step_num"]),
        prefix_length=int(model.get("action_loss_executed_prefix_length") or 0),
        prefix_weight=float(model["action_loss_executed_prefix_weight"]),
        positional_encoding="time" if temporal else "index",
        observation_horizons={
            name: int(metadata[name]["length"])
            for name in (*dataset["image_keys"], *dataset["proprio_keys"])
        },
        diffusion_conditioning=network["diffusion_conditioning"],
        use_validity_masks=bool(use_masks),
        mask_self_attention=temporal,
        loss_per_sample=bool(model.get("action_loss_per_sample_normalized", False)),
    )
    state = checkpoint.get("ema_model_state_dict", checkpoint["model_state_dict"])
    converted = {}
    replacements = {
        "factorized_role_embedding": "role",
        "factorized_arm_embedding": "arm",
        "factorized_proprio_source_embedding": "proprio_source",
        "factorized_image_source_embedding": "camera_source",
        "factorized_image_spatial_embedding": "spatial",
        "diffusion_step_encoder.": "diffusion_step.",
        "action_projector.model.": "action_projector.",
        "global_cond_projector.model.": "image_projector.",
        "local_cond_projector.model.": "proprio_projector.",
        "action_final_layer.norm.": "final_norm.",
        "action_final_layer.linear.": "final.",
        "input_pos_embedding": "input_position",
        "global_cond_pos_embedding": "image_position",
    }
    vision_prefix = f"shared_model_manager.vit_models.{image_name}."
    for name, tensor in state.items():
        if name.startswith(vision_prefix):
            name = "vision." + name.removeprefix(vision_prefix)
        elif name.startswith("global_cond_encoder."):
            if not name.endswith((".img_mean", ".img_std")) or not torch.equal(
                tensor, torch.full_like(tensor, 0.5)
            ):
                raise ValueError(f"Unexpected image-encoder parameter: {name}")
            continue
        elif name.startswith("local_cond_encoder.encoder_dict."):
            name = "proprio." + name.removeprefix("local_cond_encoder.encoder_dict.").replace(
                ".model.model.", "."
            )
        elif name.startswith("denoising_network."):
            name = name.removeprefix("denoising_network.")
            for old, new in replacements.items():
                if name.startswith(old):
                    name = new + name[len(old) :]
                    break
            name = "model." + name
        else:
            raise ValueError(f"Unrecognized checkpoint parameter: {name}")
        if name in converted:
            raise ValueError(f"Duplicate checkpoint parameter: {name}")
        converted[name] = tensor
    policy.load_state_dict(converted, strict=True)
    policy.to(device).eval()
    norm = checkpoint["normalizer_state_dict"]
    normalizers = {
        name: (norm[f"normalizers.{name}.scale"], norm[f"normalizers.{name}.offset"])
        for name in metadata
    }
    return PaperCheckpoint(policy, metadata, normalizers, action_keys)
