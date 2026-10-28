# Train and use a policy

## Train on a new task

[train.py](../scripts/train.py) trains on GPU, validates after each epoch, and
saves `best.pt`, `last.pt`, and `metrics.json`. The best checkpoint is selected
by validation loss; evaluate policy rollouts separately for task success.
New policies use uniform action weights. Paper reproduction restores the
weighting recorded for each experiment.

Save one `.pt` file per demonstration episode under `data/my_task/train/` or
`data/my_task/val/`, using distinct episode names. For the simulation tasks,
[`simulation/export.py`](../simulation/export.py) writes this format. Split episodes before making
overlapping windows. Each file is a dictionary of tensors:

| Key | Shape |
| --- | --- |
| `observations["camera"]` | `[N, frames, patches, feature_dim]` |
| `observations["robot_state"]` | `[N, frames, state_dim]` |
| `actions` | `[N, horizon, action_dim]` |
| `capture_times[name]` (Chronos Policy) | `[N, frames]` per observation stream |
| `action_times` (Chronos Policy) | `[N, horizon]` |

`N` is the number of windows in that episode. Use float32 robot state and
actions, and float16 or float32 camera features. Precompute image patch features with your vision
encoder, using the same image preprocessing for training and inference.
The trainer fits robot-state and action normalization on the training split
and saves it with the checkpoint; supply these values in their original units.

For Chronos Policy, timestamps are seconds on the same clock, and targets must match
the requested action times. DT needs no timestamps. Optional boolean `valid`
masks have shape `[N, frames]` per stream; `action_valid` is `[N, horizon]`.
Fill padded entries with finite values and mark them invalid.

Save a model configuration as `data/my_task/config.json`, adjusting names and
dimensions to your data:

```json
{
  "action_dim": 10,
  "proprio_dims": {"robot_state": 10},
  "cameras": ["camera"],
  "observation_horizons": {"camera": 2, "robot_state": 2},
  "feature_dim": 768,
  "patch_grid": 16,
  "horizon": 16,
  "dim": 512,
  "layers": 16,
  "heads": 8,
  "normalizer_ignore_indices": {
    "robot_state": [3, 4, 5, 6, 7, 8],
    "actions": [3, 4, 5, 6, 7, 8]
  }
}
```

This example leaves the six rotation coordinates unchanged in both robot state
and actions. Indices start at zero; adjust them to your representation. Omit
`normalizer_ignore_indices` to normalize every coordinate. These settings are
saved with the checkpoint and preserved when resuming.

`patch_grid=16` means 256 patches per image. Train with:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --data data/my_task --config data/my_task/config.json \
  --method chronos --epochs 100 --batch-size 32 --output outputs/my_task/chronos
```

Use `--method dt --output outputs/my_task/dt` for DT. Reduce `--batch-size` if
GPU memory is limited. Resume with the same command plus
`--resume outputs/my_task/chronos/last.pt`; `--epochs` is the total epoch count.

## Run your trained policy

```python
import torch
from chronos.policy import ChronosPolicy

saved = torch.load("outputs/my_task/chronos/best.pt", map_location="cuda", weights_only=True)
policy = ChronosPolicy(**saved["config"]).to("cuda")
policy.load_state_dict(saved["state_dict"])
policy.eval()

# observations contains camera features and robot state, already on CUDA.
for name in policy.proprio_keys:
    stats = saved["normalizers"][name]
    observations[name] = (observations[name] - stats["offset"]) / stats["scale"]
actions = policy.predict(observations, capture_times, action_times, episode_ids=[0])
stats = saved["normalizers"]["actions"]
actions = actions * stats["scale"] + stats["offset"]
```

For DT, omit `capture_times` and `action_times`. This example uses one rollout;
pass one episode ID per batch item. Use a new ID for each rollout, or call
`policy.reset()` between independent rollouts. Actions are returned in the
original controller units.

## Load a paper checkpoint

Install the vision extra (`pip install -e ".[vision]"`); the loader fetches the
SigLIP configuration from Hugging Face. Load a trained Chronos Policy or DT policy:

```python
from chronos.checkpoint import load_paper_checkpoint

checkpoint = load_paper_checkpoint("checkpoints/conveyor/chronos/seed0.ckpt", "cuda:0")
observations = checkpoint.normalize(observations)
actions = checkpoint.policy.predict(
    observations, capture_times, action_times, episode_ids=[201]
)
actions = checkpoint.decode(actions)
```

Use `checkpoint.metadata` for observation names and history lengths. Supply
images and robot state with the same preprocessing used during training.
Timestamps are in seconds.

For DT, load a checkpoint from `checkpoints/conveyor/dt/` and omit timestamps:

```python
actions = checkpoint.policy.predict(observations, episode_ids=[201])
actions = checkpoint.decode(actions)
```
