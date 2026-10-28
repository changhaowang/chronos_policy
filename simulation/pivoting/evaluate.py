"""Evaluate a checkpoint on Pivoting under one sensor and inference timing condition.

    python -m simulation.pivoting.evaluate --checkpoint checkpoints/pivoting/chronos/seed0.ckpt \
        --condition nominal --output results/pivoting-nominal.json

Defaults reproduce the paper's first evaluation batch (50 scenes from seed 600251).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch

from ..policy import Policy
from .timing import DeliveredSensors, SensorTiming, inference_delay_ms, schedule_update
from .env import PivotingEnv, pose9_to_pose7

ACTION_PERIOD_MS = 50  # policy actions are 20 Hz knots
ACTION_PERIOD_S = 0.05
PLAN_SECONDS = 0.75  # 16 knots span 15 periods
CAMERA_PERIOD_MS = 10  # the camera renders at 100 Hz
PROPRIO_PERIOD_MS = 10  # pose and wrench are sampled at 100 Hz
STREAMS = ("third_person_camera", "robot0_pose9", "robot0_wrench6")


def rates(camera, pose, wrench):
    return dict(zip(STREAMS, (camera, pose, wrench)))


NOMINAL = SensorTiming(rates((20.0, 20.0), (20.0, 20.0), (100.0, 100.0)))
SENSOR_ID = SensorTiming(
    rates((15.0, 20.0), (15.0, 20.0), (75.0, 100.0)),
    first_capture_ms=dict(zip(STREAMS, (50, 50, 10))),
)
SENSOR_OOD = SensorTiming(rates((1.0, 30.0), (1.0, 50.0), (10.0, 100.0)), latency_ms=100)
CONDITIONS = {
    "nominal": (NOMINAL, 0),
    "sensor_id": (SENSOR_ID, 0),
    "sensor_ood": (SENSOR_OOD, 0),
    "inference_0_100ms": (NOMINAL, 100),
    "inference_0_200ms": (NOMINAL, 200),
    "combined_sensor_ood_inference_0_200ms": (SENSOR_OOD, 200),
}


class ActionPlan:
    """Interpolates 16 action knots: nominal pose, virtual pose and stiffness."""

    def __init__(self, actions, origin_tick):
        self.origin_tick = int(origin_tick)
        self.actions = np.asarray(actions)
        assert self.actions.shape == (16, 19) and np.isfinite(self.actions).all()
        self.knots = np.arange(16) * ACTION_PERIOD_S
        self.poses = [
            np.stack([pose9_to_pose7(a[start : start + 9]) for a in self.actions])
            for start in (0, 9)
        ]
        self.rotations = [
            Slerp(self.knots, Rotation.from_quat(p[:, [4, 5, 6, 3]])) for p in self.poses
        ]

    def sample(self, ticks):
        relative = (np.asarray(ticks, dtype=np.int64) - self.origin_tick) * 0.001
        assert relative.min() >= -1e-9 and relative.max() <= PLAN_SECONDS + 1e-9
        relative = np.clip(relative, 0, PLAN_SECONDS)
        outputs = []
        for poses, rotation in zip(self.poses, self.rotations):
            xyz = np.column_stack([np.interp(relative, self.knots, poses[:, i]) for i in range(3)])
            q = rotation(relative).as_quat()
            outputs.append(np.column_stack([xyz, q[:, [3, 0, 1, 2]]]))
        return *outputs, np.interp(relative, self.knots, self.actions[:, 18])


def run_episode(env, policy, timing, delay_max_ms, episode, args):
    execute_ms = args.execute_steps * ACTION_PERIOD_MS
    info = env.reset(episode)
    policy.reset()
    pose, wrench = env.observation()
    latest = dict(zip(STREAMS, ((0, env.render()), (0, pose), (0, wrench))))
    horizons = {name: policy.horizons[name] for name in STREAMS}
    sensors = DeliveredSensors(
        timing, horizons, episode, {k: v[1] for k, v in latest.items()}, args.sensor_seed
    )
    tick = call = 0
    previous = None
    success = False
    while tick < args.max_steps * ACTION_PERIOD_MS and not success:
        delay = inference_delay_ms(episode, call, delay_max_ms, args.inference_seed)
        observations = {name: sensors.observation(name, tick) for name in STREAMS}
        query = np.arange(policy.horizon) * ACTION_PERIOD_S + delay * 0.001
        fresh = ActionPlan(policy(observations, episode, query), tick + delay)
        for nominal, virtual, stiffness in zip(
            *schedule_update(fresh, previous, tick, delay, execute_ms)
        ):
            env.step(nominal, virtual, stiffness)
            tick += 1
            if tick % PROPRIO_PERIOD_MS == 0:
                pose, wrench = env.observation()
                latest["robot0_pose9"] = (tick, pose)
                latest["robot0_wrench6"] = (tick, wrench)
            if tick % CAMERA_PERIOD_MS == 0:
                latest["third_person_camera"] = (tick, env.render())
            sensors.update(tick, latest)
            if env.success():
                success = True
                break
        previous = fresh
        call += 1
    return dict(
        **info,
        success=success,
        physics_steps=tick,
        policy_calls=call,
        final_angle_deg=env.book_angle_deg(),
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
        "--first-episode", type=int, default=600251, help="Scene seed of the first episode."
    )
    parser.add_argument("--sensor-seed", type=int, default=601)
    parser.add_argument("--inference-seed", type=int, default=607)
    parser.add_argument(
        "--execute-steps", type=int, default=8, help="Actions executed per policy call."
    )
    parser.add_argument(
        "--max-steps", type=int, default=600, help="Episode limit in 20 Hz action steps."
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    timing, delay_max_ms = CONDITIONS[args.condition]
    if delay_max_ms >= args.execute_steps * ACTION_PERIOD_MS:
        parser.error("The inference delay must be shorter than the executed action chunk")
    torch.set_num_threads(2)
    policy = Policy(args.checkpoint, args.device)
    env = PivotingEnv()
    episodes = []
    try:
        for episode in range(args.first_episode, args.first_episode + args.episodes):
            episodes.append(run_episode(env, policy, timing, delay_max_ms, episode, args))
            print(f"episode {episode}: success={episodes[-1]['success']}", flush=True)
    finally:
        env.close()
    successes = sum(e["success"] for e in episodes)
    result = dict(
        task="pivoting",
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
