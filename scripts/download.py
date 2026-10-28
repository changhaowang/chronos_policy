"""Download the paper checkpoints or training data from Hugging Face and verify them.

    python scripts/download.py checkpoints --task conveyor --method all
    python scripts/download.py data --task conveyor
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

JOBS = json.loads((Path(__file__).with_name("paper_jobs.json")).read_text())
TASKS = sorted({job["task"] for job in JOBS["jobs"]})
MODEL_REPO, MODEL_REVISION = (
    "changhaowang/chronos-policy",
    "d18161c367978564b0a8414868e9f0d9d2b56efb",
)
DATA_REPO, DATA_REVISION = (
    "changhaowang/chronos-policy-data",
    "300f1db4cfb9f82befcb7c3c63397a0f0774981e",
)
HUB_NAMES = {"pivoting": "flipup"}  # task names used on the Hub


def hub_file(name, *, model, cache_dir):
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            MODEL_REPO if model else DATA_REPO,
            name,
            repo_type="model" if model else "dataset",
            revision=MODEL_REVISION if model else DATA_REVISION,
            cache_dir=cache_dir,
        )
    )


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path, record):
    if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"Checksum mismatch: {path}")


def checkpoints(tasks, methods, seeds, directory, cache_dir):
    for name, record in sorted(JOBS["checkpoints"].items()):
        task, method, seed = name.split("/")
        if task not in tasks or method not in methods or int(seed[4:]) not in seeds:
            continue
        target = directory / record["file"]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            source = hub_file(record["hub_file"], model=True, cache_dir=cache_dir)
            verify(source, record)
            shutil.copyfile(source, target)
        verify(target, record)
        print("Verified", target)


def data(task, directory, cache_dir):
    """Extract the task's archives into directory/<task> and verify every file."""
    hub = HUB_NAMES.get(task, task)
    prefix = f"datasets/{hub}"
    manifest = json.loads(
        hub_file(f"{prefix}/manifest.json", model=False, cache_dir=cache_dir).read_text()
    )
    destination = directory / task
    if destination.exists():
        raise FileExistsError(f"{destination} exists; remove it or choose another --directory")
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".download-", dir=directory) as temporary:
        staging = Path(temporary)
        for shard in manifest["shards"]:
            archive = hub_file(f"{prefix}/{shard['file']}", model=False, cache_dir=cache_dir)
            verify(archive, shard)
            with tarfile.open(archive, "r:gz") as stream:
                for member in stream:
                    parts = Path(member.name).parts
                    if not member.isfile() or not parts or parts[0] != hub or ".." in parts:
                        raise ValueError(f"Unsafe archive member: {member.name}")
                    target = staging / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with stream.extractfile(member) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
        files = hub_file(
            f"{prefix}/{manifest['files_manifest']['file']}", model=False, cache_dir=cache_dir
        )
        verify(files, manifest["files_manifest"])
        with gzip.open(files, "rt") as stream:
            for line in stream:
                record = json.loads(line)
                verify(staging / record["path"], record)
        (staging / hub).rename(destination)
    print("Installed and verified", destination)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("kind", choices=("checkpoints", "data"))
    parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
    parser.add_argument("--method", choices=("chronos", "dt", "all"), default="chronos")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--directory", type=Path, help="Default: checkpoints/ or data/.")
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face download cache.")
    args = parser.parse_args()
    tasks = TASKS if args.task == "all" else [args.task]
    directory = args.directory or Path(args.kind)
    if args.kind == "checkpoints":
        methods = ("chronos", "dt") if args.method == "all" else (args.method,)
        seeds = (0, 1, 2) if args.seed is None else (args.seed,)
        checkpoints(tasks, methods, seeds, directory, args.cache_dir)
    else:
        for task in tasks:
            data(task, directory, args.cache_dir)


if __name__ == "__main__":
    main()
