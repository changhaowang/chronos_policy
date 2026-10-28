"""Evaluate a checkpoint on a DexMimicGen task under one sensor and inference timing condition.

    python -m simulation.dexmimicgen.evaluate --task coffee --data data/coffee \
        --checkpoint checkpoints/coffee/chronos/seed0.ckpt --condition nominal \
        --output results/coffee-nominal.json

Defaults reproduce the paper's evaluation batch of the task (50 scenes from
seed 201). Episodes run sequentially; each starts with a seeded hard reset.
"""

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np

from .env import TASKS, DexMimicGenEnv, action_keys, scheduled_action, to_env_actions
from .timing import (
    NOMINAL,
    STEP_SECONDS,
    DeliveredSensors,
    PlanSchedule,
    SensorTiming,
    inference_delay_s,
)

MAX_STEPS = dict(
    threading=400, assembly=300, transport=1200, coffee=400, drawer_cleanup=550, pouring=400
)
SOURCE_HISTORY = 400  # retained 100 Hz frames; 1 Hz sensors need about 2.1 s

SENSOR_ID = SensorTiming((12.5, 33.333333333333336), (0.0, 0.05))
SENSOR_OOD = SensorTiming((1.0, 50.0), (0.0, 0.1))
CONDITIONS = {  # (sensor timing, maximum inference delay in s)
    "nominal": (NOMINAL, 0.0),
    "sensor_id": (SENSOR_ID, 0.0),
    "sensor_ood": (SENSOR_OOD, 0.0),
    "inference_0_100ms": (NOMINAL, 0.1),
    "inference_0_200ms": (NOMINAL, 0.2),
    "combined_sensor_ood_inference_0_200ms": (SENSOR_OOD, 0.2),
}


class Source:
    """The retained 100 Hz observation record of one episode."""

    def __init__(self, frame):
        self.frames = deque([frame], maxlen=SOURCE_HISTORY)

    def extend(self, frames):
        self.frames.extend(frames)

    def times(self):
        return np.array([frame["time"] for frame in self.frames], dtype=np.float64)

    def values(self, indices, key):
        return np.stack([self.frames[int(i)][key] for i in indices])


def proprio_stream(source, indices, position, quaternion, gripper, width):
    """xyz, the two first rotation-matrix rows and the hand state, in float32."""
    import torch

    pos = torch.from_numpy(source.values(indices, position).astype(np.float32))
    quat = torch.from_numpy(source.values(indices, quaternion).astype(np.float32))
    hand = torch.from_numpy(source.values(indices, gripper).astype(np.float32))[..., :width]
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = quat.unbind(dim=-1)
    rows = torch.stack(
        (
            1 - 2 * y * y - 2 * z * z,
            2 * x * y - 2 * w * z,
            2 * x * z + 2 * w * y,
            2 * x * y + 2 * w * z,
            1 - 2 * x * x - 2 * z * z,
            2 * y * z - 2 * w * x,
        ),
        dim=-1,
    )
    return torch.cat((pos, rows, hand), dim=-1).numpy()


def observe(source, sensors, cameras, proprio, horizons):
    """``{name: (values, capture_seconds, valid)}`` as delivered at the newest frame."""
    clock = source.times()
    observations = {}

    def timing(selection):
        # Capture clocks travel to the policy as float32, as in the paper runner.
        return selection.capture_s.astype(np.float32).astype(np.float64), ~(
            selection.padding | selection.duplicate
        )

    for camera in cameras:
        selection = sensors.select(camera, True, clock, horizons[camera])
        observations[camera] = (source.values(selection.indices, camera), *timing(selection))
    for name, *keys, width in proprio:
        selection = sensors.select(name, False, clock, horizons[name])
        observations[name] = (
            proprio_stream(source, selection.indices, *keys, width),
            *timing(selection),
        )
    return observations


def run_episode(env, policy, condition, scene_seed, args):
    timing, delay_max_s = CONDITIONS[condition]
    policy.reset()
    source = Source(env.reset(scene_seed))
    sensors = DeliveredSensors(timing, args.sensor_seed, scene_seed)
    plans = PlanSchedule(args.execute_steps)
    previous = None
    call = 0
    while not env.done:
        delay = inference_delay_s(args.inference_seed, scene_seed, call, delay_max_s)
        observations = observe(source, sensors, env.cameras, env.proprio, policy.horizons)
        # Actions are queried on their landing clock: D + 50 ms * j after the request.
        query = (np.arange(policy.horizon, dtype=np.float32) * STEP_SECONDS).astype(
            np.float64
        ) + delay
        # Diffusion noise is drawn per episode, keyed by the scene seed.
        actions = to_env_actions(env.task, policy(observations, scene_seed, query))
        schedule = plans.update(actions, delay, previous)
        for k in range(args.execute_steps):
            if env.done:
                break
            source.extend(env.step(schedule, k * STEP_SECONDS))
        previous = scheduled_action(schedule, args.execute_steps * STEP_SECONDS - 1e-12)[0]
        call += 1
    return dict(
        scene_seed=int(scene_seed), success=env.success, policy_steps=env.steps, policy_calls=call
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Directory with episode_data.zarr (and evaluation.hdf5).",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, default="nominal")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--first-episode", type=int, default=201, help="Scene seed of the first episode."
    )
    parser.add_argument("--sensor-seed", type=int, default=0)
    parser.add_argument("--inference-seed", type=int, default=0)
    parser.add_argument(
        "--execute-steps", type=int, default=12, help="Actions executed per policy call."
    )
    parser.add_argument(
        "--max-steps", type=int, help="Episode length in 20 Hz steps (default: the paper's)."
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.max_steps = args.max_steps or MAX_STEPS[args.task]
    if round(CONDITIONS[args.condition][1] / STEP_SECONDS) >= args.execute_steps:
        parser.error("The inference delay must be shorter than the executed action chunk")
    import torch  # imported late so the rollout code itself needs no torch at import

    from ..policy import Policy

    torch.set_num_threads(2)
    policy = Policy(args.checkpoint, args.device)
    expected_keys = action_keys(args.task)
    expected_dim = sum(int(name.rsplit("_", 1)[1][:-1]) for name in expected_keys)
    if policy.model.action_dim != expected_dim or tuple(policy.action_keys) not in (
        expected_keys, ("actions",)
    ):
        parser.error(
            f"The checkpoint predicts {policy.action_keys}, not the {args.task} actions {action_keys(args.task)}"
        )
    env = DexMimicGenEnv(args.task, args.data, policy.cameras, args.max_steps)
    episodes = []
    try:
        for seed in range(args.first_episode, args.first_episode + args.episodes):
            episodes.append(run_episode(env, policy, args.condition, seed, args))
            print(f"episode {seed}: success={episodes[-1]['success']}", flush=True)
    finally:
        env.close()
    successes = sum(e["success"] for e in episodes)
    result = dict(
        task=args.task,
        condition=args.condition,
        checkpoint=str(args.checkpoint),
        success_rate=successes / len(episodes),
        successes=successes,
        settings={
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
            if k not in ("checkpoint", "output", "device")
        },
        episodes=episodes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{args.task} {args.condition}: {successes}/{len(episodes)} successes")


if __name__ == "__main__":
    main()
