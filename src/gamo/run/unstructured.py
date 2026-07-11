"""Unstructured GA primitives and the canonical weight-level comparison."""

import statistics
from dataclasses import dataclass

import torch

from gamo.ga.operators import target_keep_count
from gamo.ga.search import genetic_search
from gamo.ga.unstructured import (
    create_batched_fitness_func,
    expand_weight_masks,
    extract_weight_values,
    magnitude_pruning_mask,
    random_weight_mask,
    weight_parameter_selector,
)
from gamo.model.model import SimpleMLP
from gamo.run.experiment import ExperimentContext, SearchConfig
from gamo.utils.environment import set_seed
from gamo.utils.model_utils import eval_weights
from gamo.utils.paths import results_path, write_json_atomic


@dataclass
class RunResult:
    random_mask: torch.Tensor
    ga_mask: torch.Tensor
    curve: list[float]
    random_accuracy: float
    random_loss: float
    ga_accuracy: float
    ga_loss: float
    random_sparsity: float
    ga_sparsity: float


def run_ga_unstructured(
    model: SimpleMLP,
    original_weights: torch.Tensor,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    sparsity: float,
    pop_size: int = 100,
    n_gen: int = 100,
    tournament_size: int = 4,
) -> RunResult:
    model_kwargs = {
        "input_size": model.input_size,
        "hidden_sizes": model.hidden_sizes,
        "num_classes": model.num_classes,
    }
    weight_selector = weight_parameter_selector(model).to(device)
    prunable_weights = extract_weight_values(original_weights, weight_selector)
    random_mask = expand_weight_masks(
        random_weight_mask(prunable_weights.numel(), sparsity, device), weight_selector
    )

    fitness_fn = create_batched_fitness_func(
        original_weights,
        model,
        device,
        val_loader,
        weight_selector,
    )

    search = genetic_search(
        fitness_fn,
        target_ones=target_keep_count(prunable_weights.numel(), sparsity),
        pop_size=pop_size,
        n=prunable_weights.numel(),
        device=device,
        iterations=n_gen,
        tournament_size=tournament_size,
    )
    best_mask = expand_weight_masks(search.mask, weight_selector)

    random_result = eval_weights(
        SimpleMLP, model_kwargs, original_weights * random_mask, device, test_loader
    )
    ga_result = eval_weights(
        SimpleMLP, model_kwargs, original_weights * best_mask, device, test_loader
    )
    # The unstructured experiment reports sparsity over weights only; biases are not part
    # of its pruning budget and are always kept in the expanded masks above.
    random_sparsity = 1 - random_mask[weight_selector].float().mean().item()
    ga_sparsity = 1 - best_mask[weight_selector].float().mean().item()

    return RunResult(
        random_mask=random_mask,
        ga_mask=best_mask,
        curve=search.curve,
        random_accuracy=random_result["accuracy"],
        random_loss=random_result["loss"],
        random_sparsity=random_sparsity,
        ga_accuracy=ga_result["accuracy"],
        ga_loss=ga_result["loss"],
        ga_sparsity=ga_sparsity,
    )


def run_unstructured_comparison(config: SearchConfig) -> dict:
    """Compare unstructured GA, magnitude, and random pruning."""
    set_seed(config.seed)
    context = ExperimentContext.load(config)
    model = context.model
    selector = weight_parameter_selector(model).to(context.device)
    prunable_weights = extract_weight_values(context.original_weights, selector)
    output_path = results_path("unstructured.json", mkdir=True)

    print(f"Device: {context.device}")
    print(f"Total parameters: {context.original_weights.numel():,}")
    print(f"Dense test accuracy: {context.dense_accuracy:.2f}%\n")

    results = {
        "dataset": "FashionMNIST",
        "hidden_sizes": list(context.model_kwargs["hidden_sizes"]),
        "sparsity_scope": "weights_only",
        "bias_policy": "all biases are kept and excluded from the sparsity budget",
        "parameter_counts": {
            "weights": int(selector.sum().item()),
            "biases": int((~selector).sum().item()),
        },
        "seed_scope": "GA/random-mask stochasticity; model and data split are fixed",
        "seed_schedule": (
            "the same five repeat seeds are reused across sparsities; random and GA "
            "share each run stream"
        ),
        "dense_acc": context.dense_accuracy,
        "config": {
            **config.as_dict(),
            "mutation_rule": "1 / genome length",
        },
        "sparsities": {},
    }

    for sparsity in config.sparsities:
        print(f"{'=' * 60}\nSparsity: {sparsity:.0%}\n{'=' * 60}")
        magnitude_mask = expand_weight_masks(
            magnitude_pruning_mask(prunable_weights, sparsity), selector
        )
        magnitude_sparsity = 1.0 - magnitude_mask[selector].float().mean().item()
        magnitude_result = eval_weights(
            SimpleMLP,
            context.model_kwargs,
            context.original_weights * magnitude_mask.float(),
            context.device,
            context.test_loader,
        )
        print(
            f"  Magnitude:  {magnitude_result['accuracy']:.2f}% "
            f"(sparsity={magnitude_sparsity:.4f})"
        )

        random_accuracies, random_sparsities = [], []
        ga_accuracies, ga_sparsities, ga_curves = [], [], {}
        seed_runs = {}
        for seed in config.seeds:
            set_seed(seed)
            print(
                f"  Running GA seed={seed} "
                f"(pop={config.pop_size}, gen={config.num_iterations})..."
            )
            run = run_ga_unstructured(
                model,
                context.original_weights,
                context.val_loader,
                context.test_loader,
                context.device,
                sparsity=sparsity,
                pop_size=config.pop_size,
                n_gen=config.num_iterations,
                tournament_size=config.tournament_size,
            )
            random_accuracies.append(run.random_accuracy)
            random_sparsities.append(run.random_sparsity)
            ga_accuracies.append(run.ga_accuracy)
            ga_sparsities.append(run.ga_sparsity)
            ga_curves[f"seed={seed}"] = run.curve
            seed_runs[f"seed={seed}"] = {
                "random_test_accuracy": run.random_accuracy,
                "random_test_loss": run.random_loss,
                "ga_test_accuracy": run.ga_accuracy,
                "ga_test_loss": run.ga_loss,
            }
        random_mean, random_std = _mean_std(random_accuracies)
        ga_mean, ga_std = _mean_std(ga_accuracies)
        print(
            f"  Random:     {random_mean:.2f} ± {random_std:.2f}% "
            f"(sparsity={statistics.mean(random_sparsities):.4f})"
        )
        print(
            f"  GA          {ga_mean:.2f} ± {ga_std:.2f}% "
            f"(sparsity={statistics.mean(ga_sparsities):.4f})"
        )

        results["sparsities"][str(sparsity)] = {
            "mag_accuracy": magnitude_result["accuracy"],
            "mag_sparsity": magnitude_sparsity,
            "random_accuracy": random_mean,
            "random_accuracy_std": random_std,
            "random_sparsity": statistics.mean(random_sparsities),
            "ga_accuracy": ga_mean,
            "ga_accuracy_std": ga_std,
            "ga_sparsity": statistics.mean(ga_sparsities),
            "ga_curves": ga_curves,
            "runs": seed_runs,
        }
        write_json_atomic(output_path, results)
        print(f"  → saved checkpoint to {output_path}\n")

    print(f"Done. Full results at {output_path}")
    return results


def _mean_std(values: list[float]) -> tuple[float, float]:
    return (
        statistics.mean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )
