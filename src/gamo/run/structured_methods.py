"""Reusable method runners for the structured-pruning comparison."""

import statistics
from collections.abc import Callable

import torch

from gamo.ga.search import SearchResult
from gamo.ga.structured import (
    evaluate_masks,
    neuron_magnitude_mask,
    neuron_magnitude_mask_per_layer,
    num_neurons,
    per_layer_prune_fractions,
    random_neuron_mask,
)
from gamo.methods import (
    MAGNITUDE,
    MAGNITUDE_PER_LAYER,
    RANDOM,
    STATIC_METHODS,
)
from gamo.run.experiment import ExperimentContext, SearchConfig
from gamo.utils.environment import set_seed
from gamo.utils.pruning import structured_sparsity_stats


def mask_accuracy(context: ExperimentContext, test_evaluator, mask) -> float:
    test_model, test_batches = test_evaluator
    accuracies, _ = evaluate_masks(
        test_model,
        test_batches,
        mask.unsqueeze(0).to(context.device),
    )
    return accuracies.item() * 100


def _run_record(
    context: ExperimentContext,
    mask,
    accuracy: float,
    seed: int | None = None,
    curve: list[float] | None = None,
) -> dict:
    return {
        "seed": seed,
        "accuracy": accuracy,
        "mask": mask.tolist(),
        "curve": curve or [],
        "layer_sparsity": per_layer_prune_fractions(context.model, mask),
        "sparsity_stats": structured_sparsity_stats(
            mask,
            list(context.model.hidden_sizes),
            input_size=context.model.input_size,
            output_size=context.model.num_classes,
        ),
    }


def _empty_method() -> dict:
    return {"accuracy": [], "accuracy_std": [], "runs": {}}


def run_stochastic_method(
    label: str,
    context: ExperimentContext,
    test_evaluator,
    search: Callable[[float], SearchResult],
) -> dict[str, dict]:
    """Run any mask-search method across the shared seed/sparsity grid."""
    config = context.config
    method = _empty_method()

    for sparsity in config.sparsities:
        runs = []
        for seed in config.seeds:
            set_seed(seed)
            search_result = search(sparsity)
            mask = search_result.mask
            accuracy = mask_accuracy(context, test_evaluator, mask)
            runs.append(_run_record(context, mask, accuracy, seed, search_result.curve))

        accuracies = [run["accuracy"] for run in runs]
        method["runs"][f"{sparsity:.4f}"] = runs
        method["accuracy"].append(statistics.mean(accuracies))
        method["accuracy_std"].append(
            statistics.stdev(accuracies) if len(runs) > 1 else 0.0
        )

    return {label: method}


def run_static_baselines(
    context: ExperimentContext,
    test_evaluator,
) -> dict[str, dict]:
    """Run random masks and deterministic structured importance baselines."""
    config = context.config
    results = {label: _empty_method() for label in STATIC_METHODS}
    for sparsity in config.sparsities:
        random_runs = []
        for seed in config.seeds:
            set_seed(seed)
            mask = random_neuron_mask(num_neurons(context.model), sparsity)
            accuracy = mask_accuracy(context, test_evaluator, mask)
            random_runs.append(_run_record(context, mask, accuracy, seed))

        random_accuracies = [run["accuracy"] for run in random_runs]
        random_method = results[RANDOM]
        random_method["runs"][f"{sparsity:.4f}"] = random_runs
        random_method["accuracy"].append(statistics.mean(random_accuracies))
        random_method["accuracy_std"].append(
            statistics.stdev(random_accuracies) if len(random_runs) > 1 else 0.0
        )

        masks = {
            MAGNITUDE: neuron_magnitude_mask(context.model, sparsity),
            MAGNITUDE_PER_LAYER: neuron_magnitude_mask_per_layer(
                context.model, sparsity
            ),
        }
        stacked = torch.stack([mask.to(context.device) for mask in masks.values()])
        test_model, test_batches = test_evaluator
        accuracies, _ = evaluate_masks(test_model, test_batches, stacked)
        for (label, mask), accuracy in zip(masks.items(), accuracies, strict=True):
            value = accuracy.item() * 100
            method = results[label]
            method["accuracy"].append(value)
            method["accuracy_std"].append(0.0)
            method["runs"][f"{sparsity:.4f}"] = [_run_record(context, mask, value)]

    return results


def print_summary(
    config: SearchConfig, results: dict[str, dict], methods: list[str]
) -> None:
    for index, sparsity in enumerate(config.sparsities):
        print(f"\n=== Neuron sparsity {sparsity:.0%} ===")
        for method in methods:
            if method in results:
                mean = results[method]["accuracy"][index]
                std = results[method]["accuracy_std"][index]
                print(f"  {method}: {mean:.2f} ± {std:.2f}")
