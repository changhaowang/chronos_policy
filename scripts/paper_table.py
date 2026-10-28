"""Print the published paper table, or the same table from your evaluation results.

    python scripts/paper_table.py --reference
    python scripts/paper_table.py --results results --output paper-table.md
"""

import argparse
import json
from pathlib import Path
import statistics

if __package__:
    from .evaluate_paper import CONDITIONS, job_id, select
else:
    from evaluate_paper import CONDITIONS, job_id, select

TASKS = {
    "threading": "Threading",
    "assembly": "Assembly",
    "transport": "Transport",
    "coffee": "Coffee",
    "drawer_cleanup": "Drawer Cleanup",
    "pouring": "Pouring",
    "conveyor": "Conveyor",
    "pivoting": "Pivoting",
}
CONDITION_NAMES = dict(
    zip(
        CONDITIONS,
        (
            "Nominal",
            "Sensor ID",
            "Sensor OOD",
            "Inference 0–100 ms",
            "Inference 0–200 ms",
            "Combined",
        ),
    )
)
METHODS = {"dt": "DT", "chronos": "Chronos"}


def successes(job, results):
    if results is None:
        return job["expected_successes"]
    path = results / f"{job_id(job)}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation: {path}")
    result = json.loads(path.read_text())
    if len(result["episodes"]) != job["n_test"]:
        raise ValueError(f"Incomplete evaluation: {path}")
    return result["successes"]


def summarize(results):
    """Average batches within a training seed, then mean and sample SD across seeds."""
    seeds = {}
    for job in select():
        key = job["task"], job["method"], job["condition"]
        rate = 100.0 * successes(job, results) / job["n_test"]
        seeds.setdefault(key, {}).setdefault(job["training_seed"], []).append(rate)
    cells = {}
    for key, batches in seeds.items():
        means = [statistics.mean(rates) for rates in batches.values()]
        cells[key] = f"{statistics.mean(means):.2f} ± {statistics.stdev(means):.2f}"
    return cells


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--reference", action="store_true", help="Use the published paper results.")
    source.add_argument(
        "--results", type=Path, help="Directory written by scripts/evaluate_paper.py."
    )
    parser.add_argument("--output", type=Path, help="Save Markdown instead of printing it.")
    args = parser.parse_args()
    try:
        cells = summarize(None if args.reference else args.results)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    title = "Published paper results" if args.reference else "Evaluation results"
    lines = [
        f"# {title}",
        "",
        "Success rate (%), mean ± sample standard deviation across three training seeds.",
        "",
        "| Task | Method | " + " | ".join(CONDITION_NAMES.values()) + " |",
        "| --- | --- | " + " | ".join(["---:"] * len(CONDITIONS)) + " |",
    ]
    for task, name in TASKS.items():
        for method, label in METHODS.items():
            values = [cells[task, method, condition] for condition in CONDITIONS]
            lines.append(f"| {name} | {label} | " + " | ".join(values) + " |")
    table = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(table, encoding="utf-8")
    else:
        print(table, end="")


if __name__ == "__main__":
    main()
