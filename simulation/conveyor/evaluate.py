"""Evaluate a checkpoint on Conveyor under one sensor and inference timing condition.

    python -m simulation.conveyor.evaluate --checkpoint checkpoints/conveyor/chronos/seed0.ckpt \
        --condition nominal --output results/conveyor-nominal.json

Defaults reproduce the paper's first evaluation batch (50 scenes from seed
600851). Episodes run sequentially in one simulator, as in the paper, because
each episode inherits a little physics state from the previous one.
"""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np

from .env import ConveyorEnv, pose10_to_action, sample_episode
from .timing import (
    ACTION_PERIOD_S,
    DeliveredSensors,
    PlanSchedule,
    SensorTiming,
    inference_delay_steps,
)

CAMERA = "third_person_camera"
PROPRIO = "robot0_10d"
SOURCE_HISTORY = 400  # retained 100 Hz frames; 1 Hz sensors need about 3 s

SENSOR_ID = SensorTiming((12.5, 33.333333333333336), (0.0, 0.05))
SENSOR_OOD = SensorTiming((1.0, 50.0), (0.0, 0.1))
CONDITIONS = {  # (sensor timing or None for the nominal 20 Hz grid, maximum inference delay in s)
    "nominal": (None, 0.0),
    "sensor_id": (SENSOR_ID, 0.0),
    "sensor_ood": (SENSOR_OOD, 0.0),
    "inference_0_100ms": (None, 0.1),
    "inference_0_200ms": (None, 0.2),
    "combined_sensor_ood_inference_0_200ms": (SENSOR_OOD, 0.2),
}


class Source:
    """The dense 100 Hz observation record of one episode."""

    def __init__(self, frame):
        self.times = [frame["time"]]
        self.frames = deque([frame], maxlen=SOURCE_HISTORY)

    def extend(self, frames):
        self.times.extend(frame["time"] for frame in frames)
        self.frames.extend(frames)

    def values(self, indices, key):
        offset = len(self.times) - len(self.frames)
        if int(indices.min()) < offset:
            raise RuntimeError("A selected frame is older than the retained history")
        return np.stack([self.frames[int(i) - offset][key] for i in indices])


def observe(source, sensors, horizons):
    """``{name: (values, capture_seconds, valid)}`` as delivered at the newest frame."""
    clock = np.asarray(source.times)
    query = float(clock[-1])
    observations = {}
    for name, key in ((CAMERA, "image"), (PROPRIO, "proprio")):
        selection = sensors.select(
            "image:" + CAMERA if name == CAMERA else "proprio", clock, query, horizons[name]
        )
        values = source.values(selection.indices, key)
        observations[name] = (
            values.astype(np.float32) if key == "proprio" else values,
            selection.capture_s - query,
            selection.valid,
        )
    return observations


def run_episode(env, policy, condition, scene_seed, episode_index, args):
    timing, delay_max_s = CONDITIONS[condition]
    episode = sample_episode(scene_seed)
    policy.reset()
    source = Source(env.reset(episode))
    sensors = DeliveredSensors(timing, args.sensor_seed, scene_seed)
    plans = PlanSchedule(policy.horizon, args.execute_steps)
    steps = call = 0
    while not env.done:
        delay = inference_delay_steps(args.inference_seed, scene_seed, call, delay_max_s)
        observations = observe(source, sensors, policy.horizons)
        # Actions are queried on their landing clock: D + 50 ms * j after the request.
        query = np.arange(policy.horizon, dtype=np.float64) / 20.0 + delay * ACTION_PERIOD_S
        fresh = [pose10_to_action(a) for a in policy(observations, episode_index, query)]
        actions = plans.update(fresh, delay, env.executed_action)
        for tcp, gripper in actions[: args.execute_steps]:
            source.extend(env.step(tcp, gripper))
            steps += 1
            if env.done:
                break
        call += 1
    return dict(
        episode=episode_index,
        scene_seed=int(scene_seed),
        success=env.success,
        reward=env.judge.reward,
        control_steps=steps,
        policy_calls=call,
        conveyor_speed=episode["speed"],
        **env.judge.states(),
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, default="nominal")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--first-episode", type=int, default=600851, help="Scene seed of the first episode."
    )
    parser.add_argument("--sensor-seed", type=int, default=1801)
    parser.add_argument("--inference-seed", type=int, default=1811)
    parser.add_argument(
        "--execute-steps", type=int, default=8, help="Actions executed per policy call."
    )
    parser.add_argument(
        "--max-steps", type=int, default=600, help="Episode limit in 20 Hz action steps."
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.max_steps != 600:
        parser.error("The paper's 30 s episode limit (600 steps) is built into the environment")
    if round(CONDITIONS[args.condition][1] / ACTION_PERIOD_S) > 16 - args.execute_steps:
        parser.error("The inference delay must leave enough fresh actions for the executed chunk")
    import torch  # imported late so the rollout code itself needs no torch

    from ..policy import Policy

    torch.set_num_threads(2)
    policy = Policy(args.checkpoint, args.device)
    env = ConveyorEnv()
    episodes = []
    try:
        for index in range(args.episodes):
            # Diffusion noise follows the episode index; timing follows the scene seed.
            episodes.append(
                run_episode(env, policy, args.condition, args.first_episode + index, index, args)
            )
            print(
                f"episode {args.first_episode + index}: success={episodes[-1]['success']}",
                flush=True,
            )
    finally:
        env.close()
    successes = sum(e["success"] for e in episodes)
    result = dict(
        task="conveyor",
        condition=args.condition,
        checkpoint=str(args.checkpoint),
        success_rate=successes / len(episodes),
        successes=successes,
        settings={
            k: v for k, v in vars(args).items() if k not in ("checkpoint", "output", "device")
        },
        episodes=episodes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{args.condition}: {successes}/{len(episodes)} successes")


if __name__ == "__main__":
    main()
