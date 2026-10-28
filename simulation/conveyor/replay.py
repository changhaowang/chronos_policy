"""Replay a recorded Conveyor demonstration in the simulator.

    python -m simulation.conveyor.replay --data data/conveyor/episode_data.zarr \
        --episode 0 --video replay.mp4

The dataset stores 100 Hz frames without the scene seed. Frame ``i`` is the
state ``0.01 (i + 1)`` s after reset, and ``action0_10d`` interpolates from
the observed pose to each 20 Hz command, so command ``k`` is
``action0_10d[5k + 4]``. The scene seed is recovered by matching the first
recorded image against candidate scenes of the episode's speed bin; the
demonstration is then re-simulated open loop from its 20 Hz commands.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import zarr

from .env import ConveyorEnv, SPEED_RANGE, pose10_to_action, sample_episode

EPISODES_PER_BIN = 300  # collected per speed bin, before failed demonstrations were dropped


def speed_range(store, index):
    """The collection's sampling range for the episode's 0.05 m/s speed bin."""
    attrs = store[f"episode_{index}"].attrs
    low = max(attrs["conveyor_speed_bin_min_m_per_s"], SPEED_RANGE[0])
    high = attrs["conveyor_speed_bin_max_m_per_s"]
    if high < SPEED_RANGE[1] - 1e-12:
        high -= 1e-6  # bins do not overlap
    return float(f"{low:.6f}"), float(f"{high:.6f}")


def seed_candidates(store, index):
    """Seeds are indices within the bin; dropped failures shift later keys down."""
    key = int(store[f"episode_{index}"].attrs["source_episode_key"].split("_")[1])
    bins = [
        b
        for b in store.attrs["speed_bins"]
        if b["first_output_index"] <= index <= b["last_output_index"]
    ]
    dropped = EPISODES_PER_BIN - bins[0]["episode_num"] if bins else 0
    return range(key, key + dropped + 1)


def first_frame(env, seed, speeds, command):
    env.reset(sample_episode(seed, speeds))
    return env.step(*pose10_to_action(command))[0]["image"]


def find_seed(store, index, speeds):
    episode = store[f"episode_{index}"]
    target = episode["third_person_camera"][0].astype(np.float32)
    command = episode["action0_10d"][4]
    env = ConveyorEnv()
    try:
        errors = {
            seed: float(np.abs(first_frame(env, seed, speeds, command) - target).mean())
            for seed in seed_candidates(store, index)
        }
    finally:
        env.close()
    seed = min(errors, key=errors.get)
    return seed, errors[seed]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to episode_data.zarr.")
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument(
        "--seed", type=int, help="Scene seed; recovered from the first image when omitted."
    )
    parser.add_argument(
        "--video", type=Path, help="Write simulated and recorded frames side by side as an MP4."
    )
    args = parser.parse_args()
    store = zarr.open(str(args.data), "r")
    episode = store[f"episode_{args.episode}"]
    speeds = speed_range(store, args.episode)
    match_error = None
    if args.seed is None:
        args.seed, match_error = find_seed(store, args.episode, speeds)
    commands = episode["action0_10d"][:]
    recorded = episode["robot0_10d"][:]
    images = episode["third_person_camera"]
    env = ConveyorEnv(render=args.video is not None)  # fresh, so the replay is reproducible
    frames, deviation, index = [], 0.0, 0
    try:
        env.reset(sample_episode(args.seed, speeds))
        for k in range(len(commands) // 5):
            for frame in env.step(*pose10_to_action(commands[5 * k + 4])):
                deviation = max(
                    deviation, float(np.abs(frame["proprio"][:3] - recorded[index, :3]).max())
                )
                if args.video:
                    frames.append(np.concatenate([frame["image"], images[index]], axis=1))
                index += 1
            if env.done:
                break
        result = dict(
            episode=args.episode,
            scene_seed=args.seed,
            speed_range=speeds,
            image_match_error=match_error,
            frames=int(len(commands)),
            replayed_frames=index,
            success=env.success,
            **env.judge.states(),
            max_tcp_deviation_m=round(deviation, 6),
        )
    finally:
        env.close()
    if args.video:
        writer = cv2.VideoWriter(
            str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), 100.0, frames[0].shape[1::-1]
        )
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
