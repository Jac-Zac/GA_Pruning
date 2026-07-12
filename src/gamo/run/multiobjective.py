"""Run and persist Platypus accuracy/sparsity multiobjective pruning."""

import torch

from gamo.ga.multiobjective import optimize_structured_multiobjective
from gamo.ga.structured import (
    evaluate_masks,
    per_layer_prune_fractions,
    prepare_batched_evaluation,
)
from gamo.run.experiment import ExperimentContext, MultiobjectiveConfig
from gamo.utils.environment import set_seed
from gamo.utils.paths import results_path, write_json_atomic
from gamo.utils.pruning import structured_sparsity_stats


def run_multiobjective(config: MultiobjectiveConfig) -> dict:
    """Optimize on validation data and evaluate the final Pareto set on test data."""
    set_seed(config.seed)
    context = ExperimentContext.load(config)
    validation_evaluator = prepare_batched_evaluation(
        context.model, context.val_loader, context.device
    )
    test_evaluator = prepare_batched_evaluation(
        context.model, context.test_loader, context.device
    )

    print(
        f"Device: {context.device} | Platypus NSGA-II | "
        f"population={config.population_size} | budget={config.max_evaluations:,}"
    )
    search = optimize_structured_multiobjective(
        context.model,
        context.val_loader,
        context.device,
        population_size=config.population_size,
        max_evaluations=config.max_evaluations,
        evaluator=validation_evaluator,
    )
    if not search.pareto_front:
        raise RuntimeError("Platypus returned an empty Pareto front")

    masks = torch.stack([point.mask for point in search.pareto_front]).to(
        context.device
    )
    test_model, test_batches = test_evaluator
    test_accuracies, test_losses = evaluate_masks(test_model, test_batches, masks)
    hidden_sizes = list(context.model.hidden_sizes)

    front = []
    for point, test_accuracy, test_loss in zip(
        search.pareto_front, test_accuracies, test_losses, strict=True
    ):
        front.append(
            {
                "validation_accuracy": point.validation_accuracy * 100,
                "test_accuracy": test_accuracy.item() * 100,
                "test_loss": test_loss.item(),
                "parameter_sparsity": point.parameter_sparsity,
                "per_layer_prune_fractions": per_layer_prune_fractions(
                    context.model, point.mask
                ),
                "effective_sparsity": structured_sparsity_stats(
                    point.mask,
                    hidden_sizes,
                    input_size=context.model.input_size,
                    output_size=context.model.num_classes,
                ),
                "mask": point.mask.tolist(),
            }
        )

    output = {
        "dataset": "FashionMNIST",
        "hidden_sizes": hidden_sizes,
        "algorithm": "Platypus NSGA-II",
        "objectives": [
            {"name": "validation_accuracy", "direction": "maximize"},
            {"name": "parameter_sparsity", "direction": "maximize"},
        ],
        "selection_note": (
            "NSGA-II sees validation accuracy only; test metrics are computed once "
            "for the final nondominated set"
        ),
        "dense_test_accuracy": context.dense_accuracy,
        "evaluations": search.evaluations,
        "pareto_front": front,
        "config": config.as_dict(),
    }
    output_path = results_path("multiobjective.json", mkdir=True)
    write_json_atomic(output_path, output)
    print(f"Pareto points: {len(front)} | saved {output_path}")
    return output
