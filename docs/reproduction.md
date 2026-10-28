# Reproduce the paper

Run these commands from the repository root. Evaluation reruns the paper's
recorded scenes and seeds with the published checkpoints.

## Setup

```bash
pip install -e ".[simulation]"
```

The six DexMimicGen tasks also need the official DexMimicGen, Robosuite and
Robomimic releases used for the paper:

```bash
pip install --no-deps -r simulation/dexmimicgen/requirements.txt
```

Keep `--no-deps`: the simulation extra above supplies the runtime dependencies.

On a headless machine, rendering uses EGL (`MUJOCO_GL=egl`).

## Downloads

```bash
python scripts/download.py checkpoints --task threading --method all
python scripts/download.py data --task threading
```

Tasks: the DexMimicGen tasks `threading`, `assembly`, `transport`, `coffee`,
`drawer_cleanup` and `pouring`, plus `conveyor` and `pivoting`. Download only
the tasks you need; dataset sizes are listed on
[Hugging Face](https://huggingface.co/datasets/changhaowang/chronos-policy-data).

Threading and Assembly also use the original DexMimicGen environment metadata:

```bash
hf download MimicGen/dexmimicgen_datasets --repo-type dataset \
  --revision 181967e10c6277a653e7c9761f3978312af3646a \
  --include generated/two_arm_threading.hdf5 \
  --local-dir data/dexmimicgen-original
ln -s ../dexmimicgen-original/generated/two_arm_threading.hdf5 data/threading/evaluation.hdf5
```

For Assembly, download `two_arm_three_piece_assembly.hdf5` instead and link it
as `data/assembly/evaluation.hdf5`.

## Evaluate

```bash
# Both methods, three seeds and six timing conditions for one task.
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_paper.py --task threading --method all
```

Use `--condition sensor_ood` to run one [timing condition](../simulation/README.md#evaluate),
`--seed 0` for one training seed, or `--dry-run` to print the commands. Results
are written to `results/<task>/<method>/seed<seed>/<batch>/<condition>.json`.

To evaluate your own checkpoint with the paper's scenes and seeds:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_paper.py --task conveyor --method chronos \
  --seed 0 --checkpoint outputs/conveyor/chronos/best.pt --output results/retrained
```

## Results table

```bash
# Published results.
python scripts/paper_table.py --reference

# Your results, after evaluating all eight tasks, both methods and three seeds.
python scripts/paper_table.py --results results --output paper-table.md
```

The table reports the mean and sample standard deviation across the three
training seeds, averaging scene batches within each seed.

## Train

Export a task's data as training windows, then train with
[`scripts/train.py`](../scripts/train.py):

```bash
python -m simulation.export --task threading --data data/threading --output data/threading-windows
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --data data/threading-windows \
  --config data/threading-windows/config.json --method chronos --output outputs/threading/chronos
```

Export resamples sensor rates, sensor delays and inference delays for every
training window (`--timing randomized`, the default), so Chronos Policy learns
from irregular timing. Use `--method dt` for the DT baseline. This trainer uses frozen SigLIP features; evaluate retrained checkpoints
separately.

For loading checkpoints in your own code, see the [policy API](policy.md).

<sub>For DexMimicGen, we replay the original 20 Hz actions in MuJoCo to record 100 Hz observations. Threading and Assembly keep successful replays; the other four tasks keep all episodes.</sub>
