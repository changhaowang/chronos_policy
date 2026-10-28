"""Replay a recorded Pivoting demonstration in the simulator.

    python -m simulation.pivoting.replay --data data/pivoting/episode_data.zarr \
        --episode 0 --video replay.mp4

By default every recorded MuJoCo state is restored and rendered, which
reproduces the dataset frames. ``--actions`` instead re-simulates the
episode from its recorded 100 Hz commands (open loop, so it drifts slightly
from the demonstration). Demonstrations end at the last frame before the book
passes 15 degrees.

Only floating-gripper demonstrations can be replayed; the dataset's older
episodes were recorded with a UR5e arm controller.
"""

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
import zarr

from .env import Book, PivotingEnv, pose9_to_pose7

PHYSICS_STEPS_PER_FRAME = 10  # 1 kHz physics, 100 Hz recording


def load_episode(store, index):
    episode = store[f"episode_{index}"]
    if episode.attrs["domain"] != "new_floating":
        raise ValueError(
            f"episode_{index} was recorded with the arm controller and cannot be replayed"
        )
    meta = episode.attrs["source_metadata"]
    return (
        episode,
        Book(**meta["physical_properties"]),
        np.array(meta["book_color_rgba"]),
        np.array(meta["initial_tool_position_world_m"]),
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to episode_data.zarr.")
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument(
        "--actions", action="store_true", help="Re-simulate from recorded commands."
    )
    parser.add_argument("--video", type=Path, help="Write the rendered replay as an MP4.")
    args = parser.parse_args()
    try:
        episode, book, color, start = load_episode(zarr.open(str(args.data), "r"), args.episode)
    except ValueError as error:
        parser.error(str(error))
    states = episode["mujoco_state"][:]
    env = PivotingEnv()
    env.configure(book, color, start)
    frames, deviation = [], 0.0
    try:
        if args.actions:
            for index, action in enumerate(episode["action0_19d"][:]):
                if args.video:
                    frames.append(env.render())
                nominal, virtual = pose9_to_pose7(action[:9]), pose9_to_pose7(action[9:18])
                for _ in range(PHYSICS_STEPS_PER_FRAME):
                    env.step(nominal, virtual, action[18])
                if index + 1 < len(states):
                    recorded = states[index + 1][1 : 1 + env.model.nq]  # state[0] is time
                    deviation = max(deviation, float(np.abs(env.data.qpos - recorded).max()))
        else:
            for state in states:
                mujoco.mj_setState(
                    env.model.ptr, env.data.ptr, state, mujoco.mjtState.mjSTATE_INTEGRATION
                )
                mujoco.mj_forward(env.model.ptr, env.data.ptr)
                if args.video:
                    frames.append(env.render())
        angle = env.book_angle_deg()
    finally:
        env.close()
    if args.video:
        writer = cv2.VideoWriter(
            str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), 100.0, frames[0].shape[1::-1]
        )
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
    summary = dict(
        episode=args.episode,
        mode="actions" if args.actions else "states",
        frames=len(states),
        final_angle_deg=round(angle, 2),
        recorded_final_angle_deg=round(float(episode["book_angle_deg"][-1]), 2),
    )
    if args.actions:
        summary["max_qpos_deviation"] = round(deviation, 4)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
