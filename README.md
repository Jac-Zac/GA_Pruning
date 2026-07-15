# Genetic algorithms for neural-network pruning

[![Open interactive report](https://img.shields.io/badge/Open_interactive_report-Marimo-3a8a94?logo=github&logoColor=white)](https://jac-zac.github.io/GA_Pruning/)

![showcase](./assets/pruning.png)

This project studies how the mask representation changes genetic-algorithm pruning of an MLP (`784 → 256 → 256 → 256 → 10`) trained on FashionMNIST.
The final structured approach evolves one bit per hidden neuron and performs much better at high sparsity.

## Present the final report

Requirements are Python 3.13 and [uv](https://docs.astral.sh/uv/).
The report reads the three saved JSON files under `artifacts/results/`; it never reruns training or pruning.

```bash
uv sync --frozen
uv run marimo run notebooks/report.py
```

The live app contains the optional sparsity/method explorer. Create a static, non-reactive backup after copying in the final artifacts:

```bash
uv run marimo export html notebooks/report.py -o report.html --no-include-code -f
```

## Run the experiments

```bash
uv run python main.py train
uv run python main.py structured
uv run python main.py unstructured
uv run python main.py ablation
```

These commands produce `structured.json`, `unstructured.json`, and `ablation.json`.
Run the offline tests with `uv run pytest -q`. For the cluster workflow, see [docs/ORFEO.md](docs/ORFEO.md); for the complete methodology and fairness rules, see [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md).

## Repository map

| Path | Purpose |
|---|---|
| `main.py` | training and the three final experiment commands |
| `src/gamo/ga/` | unstructured and structured genetic searches |
| `src/gamo/run/` | canonical experiment protocol and result generation |
| `notebooks/report.py` | artifact-only Marimo presentation |
| `notebooks/report_wasm.py` | self-contained GitHub Pages presentation |
| `scripts/run_orfeo.sh` | three-job final Slurm submission |

## Deploy the interactive report

The full experiment report is available as a Marimo notebook in
[`notebooks/report.py`](notebooks/report.py). A lightweight, browser-compatible
version with the same presentation and interactive result explorer lives in
[`notebooks/report_wasm.py`](notebooks/report_wasm.py). It uses only the curated
result files under `notebooks/public/`; it does not load checkpoints, datasets,
logs, or PyTorch.

To preview the deployable version locally:

```bash
uv run marimo export html-wasm notebooks/report_wasm.py -o dist --mode run --execute
python -m http.server --directory dist
```
