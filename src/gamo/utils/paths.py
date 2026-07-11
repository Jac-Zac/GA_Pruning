"""Centralised output locations.

Everything the code reads or writes lives under a single top-level directory
controlled by the ``ARTIFACTS_DIR`` environment variable (default ``artifacts``).
On HPC clusters set it to the scratch partition filesystem, e.g. in ``.env``::

The layout under the root::

    <ARTIFACTS_DIR>/
        data/            -- downloaded datasets (e.g. FashionMNIST)
        checkpoints/     -- trained model weights
        results/         -- plots and experiment JSONs

Build paths with the helpers rather than hardcoding strings, e.g.::

    data_dir()                                              -> artifacts/data/
    checkpoint_path("mlp.safetensors")                      -> artifacts/checkpoints/mlp.safetensors
    results_path("structured.json", mkdir=True)             -> artifacts/results/structured.json
"""

import json
import os
import tempfile
from collections.abc import Mapping
from typing import Any

from dotenv import load_dotenv

# Pull env vars from .env when running locally / in notebooks.
load_dotenv()


def get_artifacts_dir() -> str:
    """Root directory for all project outputs. Override with ``ARTIFACTS_DIR`` env var."""
    return os.getenv("ARTIFACTS_DIR", "artifacts")


def data_dir() -> str:
    """Root dir for downloaded datasets (always under the artifacts root)."""
    return os.path.join(get_artifacts_dir(), "data")


def _join(base: str, parts: tuple, mkdir: bool) -> str:
    path = os.path.join(base, *parts)
    if mkdir:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return path


def checkpoint_path(*parts: str, mkdir: bool = False) -> str:
    """Path under artifacts/checkpoints/. Pass mkdir=True to create its parent dir."""
    return _join(os.path.join(get_artifacts_dir(), "checkpoints"), parts, mkdir)


def results_path(*parts: str, mkdir: bool = False) -> str:
    """Path under artifacts/results/. Pass mkdir=True to create its parent dir."""
    return _join(os.path.join(get_artifacts_dir(), "results"), parts, mkdir)


def write_json_atomic(path: str, data: Mapping[str, Any]) -> None:
    """Atomically replace ``path`` with indented JSON data."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    # A temporary file in the same directory makes the final replace atomic.
    fd, tmp_path = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=directory)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        # Remove partial output even when the write is interrupted.
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise
