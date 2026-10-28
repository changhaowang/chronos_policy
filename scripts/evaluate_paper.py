"""Run the paper's evaluation jobs with their recorded scenes and seeds.

    python scripts/evaluate_paper.py --task conveyor --method all

Each job evaluates one checkpoint under one timing condition on one batch of
50 scenes and writes results/<task>/<method>/seed<seed>/<batch>/<condition>.json.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys

if __package__:
    from .download import JOBS, TASKS, sha256
else:
    from download import JOBS, TASKS, sha256

CONDITIONS = (
    "nominal",
    "sensor_id",
    "sensor_ood",
    "inference_0_100ms",
    "inference_0_200ms",
    "combined_sensor_ood_inference_0_200ms",
)
DEXMIMICGEN = ("threading", "assembly", "transport", "coffee", "drawer_cleanup", "pouring")


def job_id(job):
    return f"{job['task']}/{job['method']}/seed{job['training_seed']}/{job['batch']}/{job['condition']}"


def select(task="all", method="all", seed=None, condition="all"):
    return [
        job
        for job in JOBS["jobs"]
        if task in ("all", job["task"])
        and method in ("all", job["method"])
        and seed in (None, job["training_seed"])
        and condition in ("all", job["condition"])
    ]


def command(job, checkpoint, output, args):
    task = job["task"]
    module = "dexmimicgen" if task in DEXMIMICGEN else task
    result = [sys.executable, "-m", f"simulation.{module}.evaluate"]
    if task in DEXMIMICGEN:
        result += [
            "--task",
            task,
            "--data",
            str(args.data / task),
        ]
    return result + [
        "--checkpoint",
        str(checkpoint),
        "--condition",
        job["condition"],
        "--output",
        str(output),
        "--episodes",
        str(job["n_test"]),
        "--first-episode",
        str(job["scene_seed"]),
        "--sensor-seed",
        str(job["sensor_seed"]),
        "--inference-seed",
        str(job["inference_seed"]),
        "--execute-steps",
        str(job["execute_steps"]),
        "--max-steps",
        str(job["max_steps"]),
        "--device",
        args.device,
    ]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
    parser.add_argument("--method", choices=("chronos", "dt", "all"), default="chronos")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--condition", choices=(*CONDITIONS, "all"), default="all")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Evaluate your own checkpoint; needs one task, method and seed.",
    )
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands only.")
    args = parser.parse_args()
    if args.checkpoint and ("all" in (args.task, args.method) or args.seed is None):
        parser.error("--checkpoint requires one --task, one --method and --seed")
    for name in ("checkpoint", "checkpoints", "data", "output"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"), PYTHONPATH=str(root))
    verified = set()
    for job in select(args.task, args.method, args.seed, args.condition):
        record = JOBS["checkpoints"][f"{job['task']}/{job['method']}/seed{job['training_seed']}"]
        checkpoint = args.checkpoint or args.checkpoints / record["file"]
        output = args.output / f"{job_id(job)}.json"
        cmd = command(job, checkpoint, output, args)
        if args.dry_run:
            print(" ".join(cmd))
            continue
        if output.exists():
            print("Skipping completed", output)
            continue
        if args.checkpoint is None and checkpoint not in verified:
            if (
                checkpoint.stat().st_size != record["bytes"]
                or sha256(checkpoint) != record["sha256"]
            ):
                raise ValueError(f"{checkpoint} differs from the published checkpoint")
            verified.add(checkpoint)
        output.parent.mkdir(parents=True, exist_ok=True)
        print(job_id(job), flush=True)
        subprocess.run(cmd, env=env, cwd=root, check=True)


if __name__ == "__main__":
    main()
