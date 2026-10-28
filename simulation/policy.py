"""Run a Chronos or DT checkpoint on delivered simulator observations."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from chronos.checkpoint import load_paper_checkpoint
from chronos.policy import ChronosPolicy

# Frozen image encoder for policies trained with scripts/train.py on exported windows.
ENCODER = "google/siglip2-base-patch16-256"
ENCODER_IMAGE_SIZE = 256


def preprocess_images(images, size=ENCODER_IMAGE_SIZE):
    """uint8 [frames, H, W, 3] -> float [frames, 3, size, size] in [0, 1], as in training."""
    image = torch.from_numpy(np.asarray(images).transpose(0, 3, 1, 2).astype(np.float32) / 255)
    image = F.interpolate(image, size=(size, size), mode="bilinear", align_corners=True)
    # Contiguous NCHW: a channels-last layout selects different conv kernels.
    return image.contiguous()


def load_encoder(device):
    from transformers import SiglipVisionModel

    return SiglipVisionModel.from_pretrained(ENCODER).to(device).eval()


@torch.no_grad()
def encode_images(encoder, images, device, batch_size=64):
    """Patch features [frames, patches, feature_dim] from uint8 images."""
    features = []
    for start in range(0, len(images), batch_size):
        pixels = preprocess_images(images[start : start + batch_size]).to(device)
        features.append(encoder(pixel_values=(pixels - 0.5) / 0.5).last_hidden_state)
    return torch.cat(features)


class Policy:
    """Wraps a published checkpoint (``.ckpt``) or one from ``scripts/train.py`` (``.pt``)."""

    def __init__(self, path, device="cuda:0"):
        path = Path(path)
        self.device = torch.device(device)
        if path.suffix == ".ckpt":
            paper = load_paper_checkpoint(path, self.device)
            self.model = paper.policy
            self._normalize, self._decode = paper.normalize, paper.decode
            self.action_keys = paper.action_keys
            self.encoder = None
            metadata = paper.metadata
            names = (*self.model.cameras, *self.model.proprio_keys)
            self.horizons = {name: int(metadata[name]["length"]) for name in names}
            self.image_size = int(metadata[self.model.cameras[0]]["shape"][-1])
        else:
            saved = torch.load(path, map_location="cpu", weights_only=True)
            self.model = ChronosPolicy(**saved["config"]).to(self.device).eval()
            self.model.load_state_dict(saved["state_dict"])
            self.normalizers = {
                name: {k: v.to(self.device) for k, v in stats.items()}
                for name, stats in saved["normalizers"].items()
            }
            self._normalize, self._decode = self._normalize_trained, self._decode_trained
            self.action_keys = ("actions",)
            self.encoder = load_encoder(self.device)
            self.horizons = dict(saved["config"]["observation_horizons"])
            self.image_size = ENCODER_IMAGE_SIZE
        self.cameras = self.model.cameras
        self.temporal = self.model.positional_encoding == "time"
        self.horizon = self.model.horizon

    def _normalize_trained(self, values):
        result = dict(values)
        for name in self.model.proprio_keys:
            stats = self.normalizers[name]
            result[name] = (values[name].float() - stats["offset"]) / stats["scale"]
        return result

    def _decode_trained(self, actions):
        stats = self.normalizers["actions"]
        return {"actions": actions * stats["scale"] + stats["offset"]}

    def reset(self):
        self.model.reset()

    def _tensor(self, name, value):
        if name in self.cameras:
            if self.encoder is not None:
                return encode_images(self.encoder, value, self.device)[None]
            return preprocess_images(value, self.image_size)[None].to(self.device)
        return torch.from_numpy(np.asarray(value))[None].to(self.device)

    def __call__(self, observations, episode, action_times=None):
        """Predict ``horizon`` actions from ``{name: (values, capture_seconds, valid)}``.

        Capture and action times are seconds relative to the request; DT
        ignores them.
        """
        values = {name: self._tensor(name, value) for name, (value, _, _) in observations.items()}
        valid = {
            name: torch.from_numpy(v)[None].to(self.device)
            for name, (_, _, v) in observations.items()
        }
        times = query = None
        if self.temporal:
            times = {
                name: torch.from_numpy(t)[None].to(self.device)
                for name, (_, t, _) in observations.items()
            }
            query = torch.from_numpy(np.asarray(action_times))[None].to(self.device)
        actions = self.model.predict(
            self._normalize(values), times, query, valid=valid, episode_ids=[episode]
        )
        decoded = self._decode(actions)
        return torch.cat([decoded[name] for name in self.action_keys], -1)[0].cpu().numpy()
