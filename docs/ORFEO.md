# Orfeo
> This explaination and setup can be applied to pretty much any cluster with little changes

The workflow uses three GPU jobs:

1. Train the checkpoint and run the complete structured comparison;
2. Run the unstructured weight-level comparison;
3. Run the structured-GA sensitivity study.

Jobs 2 and 3 start independently after job 1 succeeds. This keeps the checkpoint shared and consistent while allowing the two remaining experiments to run in parallel. The final outputs are `structured.json`, `unstructured.json`, and `ablation.json`.

## One-time setup

> [!WARNING]
> **V100 compatibility:** Keep `torch~=2.6.0` and `torchvision~=0.21.0`.
> Unbounded minimum constraints such as `torch>=2.6.0` allow `uv` to install
> newer PyTorch binaries that may not support the V100’s Volta architecture
> (compute capability 7.0), depending on their bundled CUDA/cuDNN version.

**On a login node:**

```bash
cp .env.example .env
# Set ARTIFACTS_DIR to an Orfeo scratch directory.
uv sync
uv run python scripts/download.py
```

FashionMNIST must already be present under `$ARTIFACTS_DIR/data` because compute nodes may not have internet access.

## Submit

From the repository root:

```bash
./scripts/run_orfeo.sh
```

The wrapper prints all three job IDs. Follow their state with `squeue -u "$USER"`;
logs are written under `logs/` and begin with the exact Python, PyTorch, Torchvision, CUDA, and GPU versions.
If the core job fails, Slurm does not start either dependent job.

After all three jobs complete, copy the artifact directory back and export the report:

```bash
uv run pytest -q
uv run marimo run notebooks/report.py
uv run marimo export html notebooks/report.py -o report.html --no-include-code -f
```
