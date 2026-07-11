"""Structured-GA sensitivity study at one pruning level."""

import statistics

from gamo.ga.structured import evolve_structured, prepare_batched_evaluation
from gamo.run.experiment import AblationConfig, ExperimentContext
from gamo.run.structured_methods import mask_accuracy
from gamo.utils.environment import set_seed
from gamo.utils.paths import results_path, write_json_atomic


def sensitivity_variants(config: AblationConfig) -> list[tuple[str, str, dict]]:
    """Return one-factor variants with a fixed evaluation budget."""
    canonical = {
        "pop_size": config.search.pop_size,
        "n_gen": config.search.num_iterations,
        "t_size": config.search.tournament_size,
        "crossover": "uniform",
        "elitism": True,
    }

    variants = [
        ("Canonical", "Operators", canonical),
        ("No crossover", "Operators", {**canonical, "crossover": None}),
        ("No elitism", "Operators", {**canonical, "elitism": False}),
    ]
    for tournament_size in (2, 8):
        if tournament_size != config.search.tournament_size:
            variants.append(
                (
                    f"Tournament {tournament_size}",
                    "Tournament size",
                    {**canonical, "t_size": tournament_size},
                )
            )
    # Population variants adjust generations to preserve the evaluation budget.
    for population_size in (config.search.pop_size // 2, config.search.pop_size * 2):
        population_size = max(2, population_size)
        if population_size != config.search.pop_size:
            generations, remainder = divmod(
                config.search.evaluation_budget, population_size
            )
            if remainder:
                raise ValueError(
                    "evaluation budget must be divisible by ablation population sizes"
                )
            variants.append(
                (
                    f"Population {population_size}",
                    "Population size",
                    {
                        **canonical,
                        "pop_size": population_size,
                        "n_gen": generations,
                    },
                )
            )

    return variants


def run_ablation(config: AblationConfig) -> dict:
    context = ExperimentContext.load(config.search)
    validation_evaluator = prepare_batched_evaluation(
        context.model, context.val_loader, context.device
    )
    test_evaluator = prepare_batched_evaluation(
        context.model, context.test_loader, context.device
    )
    study = {
        "values": [],
        "groups": [],
        "accuracies": [],
        "stds": [],
        "settings": {},
        "runs": {},
    }
    for label, group, settings in sensitivity_variants(config):
        accuracies = []
        for seed in config.search.seeds:
            set_seed(seed)
            search_result = evolve_structured(
                context.model,
                context.val_loader,
                context.device,
                config.focus,
                evaluator=validation_evaluator,
                **settings,
            )
            accuracy = mask_accuracy(context, test_evaluator, search_result.mask)
            accuracies.append(accuracy)
            study["runs"][f"{label}|seed={seed}"] = {
                "test_accuracy": accuracy,
                "validation_curve": search_result.curve,
            }

        study["values"].append(label)
        study["groups"].append(group)
        study["accuracies"].append(statistics.mean(accuracies))
        study["stds"].append(
            statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
        )
        study["settings"][label] = {
            **settings,
            "fitness_evaluations": settings["pop_size"] * settings["n_gen"],
        }

    results = {
        "sensitivity": study,
        "config": {
            **config.search.as_dict(),
            "mutation_rule": "1 / genome length",
            "focus": config.focus,
            "canonical_fitness_evaluations": config.search.evaluation_budget,
        },
    }
    output_path = results_path("ablation.json", mkdir=True)
    write_json_atomic(output_path, results)
    print(f"Saved ablation results to {output_path}")
    return results
