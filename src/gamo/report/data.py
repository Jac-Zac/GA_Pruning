"""Artifact loading and transformation for the Marimo report."""

import json
import os
from collections.abc import Iterable, Mapping, Sequence

import torch

from gamo.methods import (
    HILL_CLIMBING,
    MAGNITUDE,
    MAGNITUDE_PER_LAYER,
    RANDOM,
    RANDOM_SEARCH,
    STRUCTURED_METHODS,
)
from gamo.utils.paths import results_path
from gamo.utils.pruning import neuron_to_weight_masks

STRUCTURED_REPORT_METHODS = STRUCTURED_METHODS

STRUCTURED_METHOD_LABELS = {
    "GA-uniform": "GA · uniform",
    "GA-two_point": "GA · two-point",
    HILL_CLIMBING: "Hill climbing",
    RANDOM_SEARCH: "Random search",
    MAGNITUDE: "Magnitude · global",
    MAGNITUDE_PER_LAYER: "Magnitude · per layer",
    RANDOM: "Random mask",
}


def _read_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def load_report_artifacts() -> tuple[dict | None, dict | None, dict | None, str | None]:
    """Load the three artifacts required by the final report."""
    sweep = _read_json(results_path("structured.json"))
    unstructured = _read_json(results_path("unstructured.json"))
    ablation = _read_json(results_path("ablation.json"))
    error = None
    required = {
        "sparsities",
        "architecture",
        "methods",
        "dense_acc",
        "dense_val_acc",
        "config",
    }
    if sweep is None:
        error = "The structured result is missing. Run the three experiments first."
    elif not required.issubset(sweep):
        error = "The structured result is incomplete; missing: " + ", ".join(
            sorted(required - set(sweep))
        )
    elif validation_error := _structured_artifact_error(sweep):
        error = validation_error
    elif unstructured is None or not {
        "sparsities",
        "dense_acc",
        "parameter_counts",
        "config",
    }.issubset(unstructured):
        error = "The unstructured result is missing or incomplete."
    elif ablation is None or not {"sensitivity", "config"}.issubset(ablation):
        error = "The final run is incomplete: all three result artifacts are required."
    elif unstructured["dense_acc"] != sweep["dense_acc"]:
        error = "Structured and unstructured artifacts use different checkpoints."
    elif set(unstructured["sparsities"]) != {
        str(value) for value in sweep["sparsities"]
    }:
        error = "Structured and unstructured artifacts use different sparsity sweeps."

    return sweep, unstructured, ablation, error


def _structured_artifact_error(sweep: dict) -> str | None:
    """Return a clear report error for malformed structured-result contents."""
    sparsities = sweep["sparsities"]
    architecture = sweep["architecture"]
    hidden_sizes = architecture.get("hidden_sizes", [])
    if (
        not sparsities
        or architecture.get("input_size", 0) <= 0
        or architecture.get("output_size", 0) <= 0
        or not hidden_sizes
        or any(size <= 0 for size in hidden_sizes)
    ):
        return (
            "The structured result has an empty sparsity sweep or invalid architecture."
        )

    missing_methods = set(STRUCTURED_REPORT_METHODS) - set(sweep["methods"])
    if missing_methods:
        return "The structured result is missing methods: " + ", ".join(
            sorted(missing_methods)
        )

    genome_length = sum(hidden_sizes)
    for method in STRUCTURED_REPORT_METHODS:
        result = sweep["methods"][method]
        if len(result.get("accuracy", [])) != len(sparsities):
            return f"The structured result has the wrong series length for {method}."
        if len(result.get("accuracy_std", [])) != len(sparsities):
            return (
                f"The structured result has the wrong std-series length for {method}."
            )
        for sparsity in sparsities:
            runs = result.get("runs", {}).get(f"{sparsity:.4f}", [])
            if not runs:
                return f"The structured result is missing runs for {method}."
            if any(len(run.get("mask", [])) != genome_length for run in runs):
                return f"The structured result has an invalid mask for {method}."
    return None


def mean_runs(runs: Iterable[Sequence[float]]) -> list[float]:
    """Pointwise mean of equal-length runs."""
    values = list(runs)
    if values and any(len(value) != len(values[0]) for value in values[1:]):
        raise ValueError("artifact runs must have equal lengths")
    return [sum(point) / len(point) for point in zip(*values)] if values else []


def _runs(sweep: dict, method: str, sparsity: float) -> list[dict]:
    try:
        return sweep["methods"][method]["runs"][f"{sparsity:.4f}"]
    except KeyError as exc:
        raise ValueError(f"missing runs for {method} at {sparsity:.0%}") from exc


def effective_sparsity_rows(
    sweep: dict, primary_method: str
) -> tuple[list[dict[str, str]], list[float]]:
    """Compute mean effective sparsity from per-seed artifact statistics."""
    rows = []
    weight_sparsities = []
    for index, sparsity in enumerate(sweep["sparsities"]):
        method = sweep["methods"][primary_method]
        stats = [
            run["sparsity_stats"] for run in _runs(sweep, primary_method, sparsity)
        ]
        weight_mean = sum(item["weight_sparsity"] for item in stats) / len(stats)
        parameter_mean = sum(item["parameter_sparsity"] for item in stats) / len(stats)
        weight_sparsities.append(weight_mean)
        rows.append(
            {
                "Neuron sparsity": f"{sparsity:.0%}",
                "Weight sparsity": f"{weight_mean:.1%}",
                "Parameter sparsity": f"{parameter_mean:.1%}",
                "Test accuracy": f"{method['accuracy'][index]:.1f}%",
            }
        )
    return rows, weight_sparsities


def weight_layer_rows(weight_masks: Mapping[str, torch.Tensor]) -> list[dict[str, str]]:
    """Summarize exact per-matrix weight removal for a mask visualization."""
    rows = []
    for layer, mask in weight_masks.items():
        total = mask.numel()
        kept = int(mask.sum())
        rows.append(
            {
                "Weight matrix": layer,
                "Shape": f"{mask.shape[0]} × {mask.shape[1]}",
                "Weights removed": f"{total - kept:,}",
                "Weight sparsity": f"{1 - kept / total:.1%}",
            }
        )
    return rows


def neuron_layer_rows(
    neuron_mask: torch.Tensor, hidden_sizes: Sequence[int]
) -> list[dict[str, str]]:
    """Summarize the chromosome as one keep/remove count per hidden layer."""
    flat_mask = neuron_mask.detach().cpu().bool().flatten()
    if flat_mask.numel() != sum(hidden_sizes):
        raise ValueError("neuron mask length does not match hidden_sizes")

    rows = []
    for index, layer_mask in enumerate(flat_mask.split(list(hidden_sizes)), start=1):
        kept = int(layer_mask.sum())
        total = layer_mask.numel()
        rows.append(
            {
                "Hidden layer": f"Layer {index}",
                "Neurons kept": f"{kept:,} / {total:,}",
                "Neurons removed": f"{total - kept:,}",
                "Neuron sparsity": f"{1 - kept / total:.1%}",
            }
        )
    return rows


def structured_explorer_view(
    sweep: dict, method: str, sparsity: float
) -> dict[str, object]:
    """Collect one artifact-only method/sparsity view for the report explorer."""
    if method not in sweep["methods"]:
        raise ValueError(f"unknown structured method: {method}")
    try:
        index = sweep["sparsities"].index(sparsity)
    except ValueError as exc:
        raise ValueError(f"unknown structured sparsity: {sparsity}") from exc

    result = sweep["methods"][method]
    runs = _runs(sweep, method, sparsity)
    representative = next(
        (run for run in runs if run["seed"] == sweep["config"]["seed"]), runs[0]
    )
    architecture = sweep["architecture"]
    raw_mask = representative["mask"]
    weight_masks = neuron_to_weight_masks(
        torch.tensor(raw_mask, dtype=torch.bool),
        architecture["hidden_sizes"],
        input_size=architecture["input_size"],
        output_size=architecture["output_size"],
    )

    def average(field: str) -> float:
        return sum(float(run["sparsity_stats"][field]) for run in runs) / len(runs)

    return {
        "accuracy": result["accuracy"][index],
        "std": result["accuracy_std"][index],
        "weight_sparsity": average("weight_sparsity"),
        "curve": mean_runs(run["curve"] for run in runs if run["curve"]),
        "distribution": mean_runs(run["layer_sparsity"] for run in runs),
        "neuron_mask": torch.tensor(raw_mask, dtype=torch.bool),
        "representative_seed": representative["seed"],
        "weight_masks": weight_masks,
    }
