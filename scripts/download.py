#!/usr/bin/env python3
"""Pre-download datasets for offline use on HPC compute nodes.

Run this once on a LOGIN node before submitting SLURM jobs.
Compute nodes are typically offline, so the dataset must already be cached under
``$ARTIFACTS_DIR/data`` (configured via ``.env`` or the environment).

Usage: ``uv run python scripts/download.py``
"""

from torchvision import datasets

from gamo.utils.paths import data_dir


def main() -> None:
    root = data_dir()
    print(f"Downloading FashionMNIST to {root}/")
    datasets.FashionMNIST(root=root, train=True, download=True)
    datasets.FashionMNIST(root=root, train=False, download=True)
    print("FashionMNIST ready")


if __name__ == "__main__":
    main()
