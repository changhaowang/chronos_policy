# Chronos Policy: Learning Robot Policies on Physical Time

[Website](https://changhaowang.github.io/chronospolicy/) ·
[Checkpoints](https://huggingface.co/changhaowang/chronos-policy) ·
[Datasets](https://huggingface.co/datasets/changhaowang/chronos-policy-data) ·
[Simulation](simulation/README.md) · [Reproduction](docs/reproduction.md)

Chronos Policy uses observation and action timestamps to learn robot policies that handle
irregular sensor rates, delayed observations, and inference latency.

<p align="center">
  <img src="assets/teaser.gif" width="640" alt="Chronos Policy teaser">
</p>

## Quickstart

Use Python 3.10–3.12.

```bash
git clone --depth 1 https://github.com/changhaowang/chronos_policy.git
cd chronos_policy
pip install -e .
python scripts/train.py --help
```

For GPU training on your own data, see [Training on a new task](docs/policy.md#train-on-a-new-task).

## Tasks

Eight simulation tasks: six DexMimicGen tasks (Threading, Assembly, Transport,
Coffee, Drawer Cleanup, Pouring) and our MuJoCo Conveyor and Pivoting tasks.
[`simulation/`](simulation/README.md) has their evaluation, replay and
data-export code, and the [reproduction guide](docs/reproduction.md) downloads
the published checkpoints and data, reruns the paper's evaluations and trains
new policies.

`python scripts/paper_table.py --reference` prints the paper table.

Built on [Gated Memory Policy](https://github.com/real-stanford/gated-memory-policy),
whose diffusion transformer is adapted from [RDT](https://github.com/thu-ml/RoboticsDiffusionTransformer).
We find that using L1 loss instead of MSE and leaving 6D rotation coordinates
unnormalized improves the DT baseline's performance.
See the [training guide](docs/policy.md#train-on-a-new-task) for configuration.

## Code

`chronos/` contains the policy, transformer, attention, and checkpoint loader.
Chronos Policy and the DT baseline share the implementation; choose
`positional_encoding="time"` or `"index"`.
`scripts/` has the trainer, downloads, paper evaluation and results table, and
`simulation/` has the eight task environments.

## Citation

```bibtex
@misc{wang2026chronospolicy,
  title = {Chronos Policy: Learning Robot Policies on Physical Time},
  author = {Wang, Changhao and Clark, Jaden and Song, Shuran},
  year = {2026},
  eprint = {2610.13929},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  url = {https://arxiv.org/abs/2610.13929}
}
```
