"""Export a simulation dataset as training windows for scripts/train.py.

    python -m simulation.export --task pivoting --data data/pivoting --output data/pivoting-windows

One window every 50 ms, each observation stream with its recorded history and
16 action targets at 20 Hz. ``--timing randomized`` resamples every training
window's sensor rates, sensor delays and inference delay, so Chronos Policy
learns from the irregular timing it is evaluated under; ``nominal`` keeps the
fixed grid. Camera frames are stored as frozen SigLIP patch features (float16).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import zarr

from .policy import encode_images, load_encoder

PERIOD = 5  # frames per 20 Hz step in the 100 Hz recordings
HORIZON = 16
POSE_ROTATION = [3, 4, 5, 6, 7, 8]  # rotation coordinates of a 10-D pose, left unnormalized


def dexmimicgen(cameras, proprio, actions):
    # Each 20 Hz command is held for five frames, starting at the observation frame.
    return dict(
        cameras=cameras,
        proprio=proprio,
        actions=actions,
        horizons={name: 2 for name in (*cameras, *proprio)},
        periods={name: PERIOD for name in (*cameras, *proprio)},
        query_phase=0,
        action_offset=0,
        ignore={},
    )


TWO_ARM_10D = dict(
    proprio={"robot0_10d": 10, "robot1_10d": 10}, actions=("action0_10d", "action1_10d")
)
HAND_CAMERAS = ("agentview_image", "robot0_eye_in_hand_image", "robot1_eye_in_hand_image")
TWO_HAND_CAMERAS = (
    "agentview_image",
    "robot0_eye_in_right_hand_image",
    "robot0_eye_in_left_hand_image",
)
TASKS = {
    "threading": dexmimicgen(HAND_CAMERAS, **TWO_ARM_10D),
    "assembly": dexmimicgen(HAND_CAMERAS, **TWO_ARM_10D),
    "transport": dexmimicgen(
        (*HAND_CAMERAS, "shouldercamera0_image", "shouldercamera1_image"), **TWO_ARM_10D
    ),
    "coffee": dexmimicgen(
        TWO_HAND_CAMERAS, {"robot0_20d": 20, "robot1_20d": 20}, ("action0_15d", "action1_15d")
    ),
    "drawer_cleanup": dexmimicgen(
        HAND_CAMERAS, {"robot0_21d": 21, "robot1_21d": 21}, ("action0_15d", "action1_15d")
    ),
    "pouring": dexmimicgen(
        TWO_HAND_CAMERAS, {"robot0_20d": 20, "robot1_20d": 20}, ("action0_15d", "action1_15d")
    ),
    "conveyor": dict(
        cameras=("third_person_camera",),
        proprio={"robot0_10d": 10},
        actions=("action0_10d",),
        horizons={"third_person_camera": 2, "robot0_10d": 2},
        periods={"third_person_camera": PERIOD, "robot0_10d": PERIOD},
        # Frame i is recorded at 10 ms * (i + 1), so policy calls fall on frames 5k - 1;
        # the first action is the target for the next 20 Hz control step.
        query_phase=4,
        action_offset=5,
        ignore={"robot0_10d": POSE_ROTATION, "actions": POSE_ROTATION},
    ),
    "pivoting": dict(
        cameras=("third_person_camera",),
        proprio={"robot0_pose9": 9, "robot0_wrench6": 6},
        actions=("action0_19d",),
        horizons={"third_person_camera": 2, "robot0_pose9": 3, "robot0_wrench6": 32},
        periods={"third_person_camera": PERIOD, "robot0_pose9": PERIOD, "robot0_wrench6": 1},
        # Policy calls start at t = 0; the first action targets the request time.
        query_phase=0,
        action_offset=0,
        ignore={
            "robot0_pose9": POSE_ROTATION,
            "actions": [3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17],
        },
    ),
}
# Randomized timing: per-sample interval scale, sensor delay and inference delay.
INTERVAL_SCALE = (0.6, 1.65)  # 12.5-33 Hz around a 20 Hz stream
SENSOR_DELAY_FRAMES = 5  # up to 50 ms
INFERENCE_DELAYS = (0, 1, 2, 3, 4)  # in 20 Hz steps, up to 200 ms


def rot6d(rotation):
    """First two rows of the rotation matrix, the checkpoints' 6-D convention."""
    return rotation.as_matrix()[:, :2].reshape(-1, 6).astype(np.float32)


def stream(episode, name):
    """Read a stream, deriving 10-D poses from end-effector poses and absolute actions."""
    if name in episode:
        return episode[name][:]
    if name.startswith("robot"):
        robot = int(name[5])
        position = episode[f"robot{robot}_eef_pos"][:]
        rotation = Rotation.from_quat(episode[f"robot{robot}_eef_quat"][:][:, [1, 2, 3, 0]])
        gripper = episode[f"robot{robot}_gripper_qpos"][:, :1]
        return np.concatenate([position, rot6d(rotation), gripper], 1).astype(np.float32)
    robot = int(name[6])
    action = episode["abs_actions"][:, 7 * robot : 7 * robot + 7]
    rotation = Rotation.from_rotvec(action[:, 3:6])
    return np.concatenate([action[:, :3], rot6d(rotation), action[:, 6:]], 1).astype(np.float32)


def stamps(episode, name):
    key = f"{name}_timestamp"
    return episode[key][:] if key in episode else episode["timestamp"][:]


def windows(episode, spec, encoder, device, rng=None):
    frames = len(episode["timestamp"])
    reserve = max(INFERENCE_DELAYS) if rng is not None else 0
    last = frames - 1 - spec["action_offset"] - PERIOD * (HORIZON - 1 + reserve)
    queries = np.arange(spec["query_phase"], last + 1, PERIOD)
    if len(queries) == 0:
        return None
    count = len(queries)
    delays = rng.choice(INFERENCE_DELAYS, count) if rng is not None else np.zeros(count, int)
    result = {"observations": {}, "capture_times": {}, "valid": {}}
    for name, horizon in spec["horizons"].items():
        period = spec["periods"][name]
        if rng is None:
            ages = period * np.arange(horizon - 1, -1, -1)[None].repeat(count, 0)
        else:
            # Newest sample after a random delay, older ones at jittered intervals.
            steps = rng.uniform(*INTERVAL_SCALE, (count, horizon - 1)) * period
            gaps = np.round(np.cumsum(steps[:, ::-1], 1)[:, ::-1]).astype(int)
            delay = rng.integers(0, SENSOR_DELAY_FRAMES + 1, (count, 1))
            ages = np.concatenate([gaps, np.zeros((count, 1), int)], 1) + delay
        index = queries[:, None] - ages
        valid = index >= 0
        index = index.clip(0)
        times = stamps(episode, name)
        result["capture_times"][name] = torch.from_numpy(times[index] - times[queries][:, None])
        result["valid"][name] = torch.from_numpy(valid)
        if name in spec["cameras"]:
            used = np.unique(index)
            features = encode_images(
                encoder, episode[name].get_orthogonal_selection((used,)), device
            )
            lookup = torch.as_tensor(np.searchsorted(used, index), device=features.device)
            result["observations"][name] = features[lookup].half().cpu()
        else:
            result["observations"][name] = torch.from_numpy(stream(episode, name)[index])
    targets = (
        queries[:, None] + spec["action_offset"] + PERIOD * (delays[:, None] + np.arange(HORIZON))
    )
    actions = np.concatenate([stream(episode, name) for name in spec["actions"]], 1)
    result["actions"] = torch.from_numpy(actions[targets])
    result["action_times"] = torch.from_numpy(0.05 * (delays[:, None] + np.arange(HORIZON)))
    return result


def split(store):
    """Use the dataset's split when it has one, otherwise hold out every tenth episode."""
    keys = sorted(store.keys(), key=lambda k: int(k.split("_")[1]))
    splits = store.attrs.get("split_episode_indices")
    if splits:
        train = {f"episode_{i}" for i in splits["train"]}
        val = {f"episode_{i}" for i in splits["validation"]}
        return [k for k in keys if k in train], [k for k in keys if k in val]
    return [k for i, k in enumerate(keys) if i % 10], [k for i, k in enumerate(keys) if not i % 10]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument(
        "--data", type=Path, required=True, help="Directory with episode_data.zarr."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing", choices=("randomized", "nominal"), default="randomized")
    parser.add_argument("--episodes", type=int, help="Export only the first N episodes per split.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    spec = TASKS[args.task]
    store = zarr.open(str(args.data / "episode_data.zarr"), "r")
    encoder = load_encoder(args.device)
    splits = split(store)
    for name, keys in zip(("train", "val"), splits):
        directory = args.output / name
        directory.mkdir(parents=True, exist_ok=True)
        for index, key in enumerate(keys[: args.episodes]):
            # Validation windows keep the nominal grid so losses stay comparable.
            randomized = args.timing == "randomized" and name == "train"
            rng = np.random.default_rng([args.seed, index]) if randomized else None
            episode = windows(store[key], spec, encoder, args.device, rng)
            if episode is not None:
                torch.save(episode, directory / f"{key}.pt")
        print(f"{name}: {len(keys[: args.episodes])} episodes", flush=True)
    first = store[splits[0][0]]
    config = dict(
        action_dim=sum(stream(first, name).shape[1] for name in spec["actions"]),
        proprio_dims=spec["proprio"],
        cameras=list(spec["cameras"]),
        observation_horizons=spec["horizons"],
        feature_dim=int(encoder.config.hidden_size),
        patch_grid=encoder.config.image_size // encoder.config.patch_size,
        horizon=HORIZON,
        normalizer_ignore_indices=spec["ignore"],
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
