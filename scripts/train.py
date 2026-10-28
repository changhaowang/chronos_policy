"""Train Chronos or DT on episode files containing demonstration windows."""

import argparse
from bisect import bisect_right
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from chronos.policy import ChronosPolicy


def load_episode(path):
    episode = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    actions = episode["actions"]
    if actions.ndim != 3 or len(actions) == 0:
        raise ValueError(f"{path}: actions must have shape [windows, horizon, action_dim]")
    for group in ("observations", "capture_times", "valid"):
        for value in episode.get(group, {}).values():
            if len(value) != len(actions):
                raise ValueError(f"{path}: inconsistent window counts in {group}")
    for key in ("action_times", "action_valid"):
        if key in episode and len(episode[key]) != len(actions):
            raise ValueError(f"{path}: inconsistent window counts in {key}")
    return episode


class Episodes(Dataset):
    """Keep only one episode in memory per loader worker."""

    def __init__(self, directory):
        self.paths = sorted(Path(directory).glob("*.pt"))
        if not self.paths:
            raise ValueError(f"No episode .pt files in {directory}")
        self.ends = []
        count = 0
        for path in self.paths:
            count += len(load_episode(path)["actions"])
            self.ends.append(count)
        self.cached_index, self.cached_episode = None, None

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        episode_index = bisect_right(self.ends, index)
        if episode_index != self.cached_index:
            self.cached_episode = load_episode(self.paths[episode_index])
            self.cached_index = episode_index
        offset = index - (self.ends[episode_index - 1] if episode_index else 0)
        return tree_map(lambda tensor: tensor[offset], self.cached_episode)


def tree_map(function, value):
    if isinstance(value, dict):
        return {key: tree_map(function, item) for key, item in value.items()}
    return function(value)


def validate_ignored_indices(config, ignored):
    dimensions = {**config["proprio_dims"], "actions": config["action_dim"]}
    if not isinstance(ignored, dict):
        raise ValueError("normalizer_ignore_indices must map stream names to coordinate lists")
    for name, indices in ignored.items():
        if name not in dimensions:
            raise ValueError(f"Unknown normalization stream: {name}")
        if not isinstance(indices, list) or any(
            type(index) is not int or not 0 <= index < dimensions[name] for index in indices
        ):
            raise ValueError(f"Invalid normalization indices for {name}: {indices}")


def fit_normalizers(dataset, config, ignored=None):
    """Fit per-coordinate min/max using valid training observations and actions."""
    ignored = {} if ignored is None else ignored
    validate_ignored_indices(config, ignored)
    bounds = {}
    for path in dataset.paths:
        episode = load_episode(path)
        values = {name: episode["observations"][name] for name in config["proprio_dims"]}
        values["actions"] = episode["actions"]
        for name, value in values.items():
            mask = (
                episode.get("action_valid")
                if name == "actions"
                else episode.get("valid", {}).get(name)
            )
            value = value.reshape(-1, value.shape[-1]).float()
            if mask is not None:
                value = value[mask.reshape(-1)]
            if not len(value):
                continue
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite training values in {path}: {name}")
            low, high = value.amin(0), value.amax(0)
            if name in bounds:
                low = torch.minimum(low, bounds[name][0])
                high = torch.maximum(high, bounds[name][1])
            bounds[name] = (low, high)
    expected = {*config["proprio_dims"], "actions"}
    if set(bounds) != expected:
        raise ValueError("Training data has no valid values for one or more streams")
    normalizers = {
        name: {
            "offset": (low + high) / 2,
            "scale": torch.where(high > low, (high - low) / 2, torch.ones_like(low)),
        }
        for name, (low, high) in bounds.items()
    }
    for name, indices in ignored.items():
        normalizers[name]["offset"][indices] = 0.0
        normalizers[name]["scale"][indices] = 1.0
    return normalizers


def batch_loss(policy, batch, normalizers, device):
    batch = tree_map(lambda tensor: tensor.to(device, non_blocking=True), batch)
    observations = batch["observations"]
    for name in policy.cameras:
        observations[name] = observations[name].float()
    for name in policy.proprio_keys:
        stats = normalizers[name]
        observations[name] = (observations[name].float() - stats["offset"]) / stats["scale"]
    stats = normalizers["actions"]
    actions = (batch["actions"].float() - stats["offset"]) / stats["scale"]
    return policy.loss(
        observations,
        batch.get("capture_times"),
        batch.get("action_times"),
        actions,
        valid=batch.get("valid"),
        action_valid=batch.get("action_valid"),
    )


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("chronos", "dt"), default="chronos")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.workers < 0 or args.lr <= 0:
        parser.error(
            "Epochs, batch size, and learning rate must be positive; workers must be nonnegative"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA is unavailable. Install a CUDA-enabled PyTorch build or use --device cpu"
        )
    torch.manual_seed(args.seed)
    config = json.loads(args.config.read_text())
    config["positional_encoding"] = "time" if args.method == "chronos" else "index"
    config["seed"] = args.seed
    ignored = config.pop("normalizer_ignore_indices", {})
    validate_ignored_indices(config, ignored)
    train, val = Episodes(args.data / "train"), Episodes(args.data / "val")
    if {p.name for p in train.paths} & {p.name for p in val.paths}:
        raise ValueError("Use distinct episode IDs in train/ and val/; split before making windows")
    saved = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    settings = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "data": str(args.data.resolve()),
    }
    if saved and (saved["config"] != config or saved["settings"] != settings):
        raise ValueError(
            "Resume requires the same model, data path, batch size, learning rate, and seed"
        )
    if saved and saved.get("normalizer_ignore_indices", {}) != ignored:
        raise ValueError("Resume requires the same normalizer_ignore_indices")
    args.output.mkdir(parents=True, exist_ok=True)
    if not saved and any((args.output / name).exists() for name in ("last.pt", "best.pt")):
        raise ValueError("Output already contains checkpoints; use --resume or a new directory")
    if saved and args.output.resolve() != args.resume.resolve().parent:
        raise ValueError("Resume into the checkpoint's original output directory")
    normalizers = saved["normalizers"] if saved else fit_normalizers(train, config, ignored)
    gpu_normalizers = tree_map(lambda tensor: tensor.to(device), normalizers)
    policy = ChronosPolicy(**config).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    start, best = 0, float("inf")
    history = []
    if saved:
        policy.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        start, best, history = saved["epoch"], saved["best_val_loss"], saved["history"]
        generator.set_state(saved["loader_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
    options = dict(
        batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda"
    )
    train_loader = DataLoader(train, shuffle=True, generator=generator, **options)
    val_loader = DataLoader(val, shuffle=False, **options)
    for epoch in range(start, args.epochs):
        policy.train()
        train_sum = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(policy, batch, gpu_normalizers, device)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss; check demonstration values and masks")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            train_sum += loss.item() * len(batch["actions"])
        policy.eval()
        val_sum = 0.0
        # Reuse validation noise each epoch without changing the training RNG.
        devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.random.default_generator.manual_seed(args.seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(args.seed)
            for batch in val_loader:
                loss = batch_loss(policy, batch, gpu_normalizers, device)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite validation loss")
                val_sum += loss.item() * len(batch["actions"])
        val_loss = val_sum / len(val)
        improved = val_loss < best
        best = min(best, val_loss)
        record = {"epoch": epoch + 1, "train_loss": train_sum / len(train), "val_loss": val_loss}
        history.append(record)
        checkpoint = dict(
            config=config,
            settings=settings,
            normalizers=normalizers,
            normalizer_ignore_indices=ignored,
            state_dict=policy.state_dict(),
            optimizer=optimizer.state_dict(),
            epoch=epoch + 1,
            best_val_loss=best,
            history=history,
            torch_rng=torch.get_rng_state(),
            loader_rng=generator.get_state(),
            cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        )
        save_checkpoint(args.output / "last.pt", checkpoint)
        if improved:
            save_checkpoint(args.output / "best.pt", checkpoint)
        (args.output / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
