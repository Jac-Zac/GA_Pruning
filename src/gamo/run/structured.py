"""Orchestration for structured (neuron-level) pruning experiments."""

from gamo.ga.structured import (
    evolve_structured,
    hill_climb_structured,
    num_neurons,
    prepare_batched_evaluation,
    random_search_structured,
)
from gamo.methods import HILL_CLIMBING, RANDOM_SEARCH, STATIC_METHODS, ga_label
from gamo.run.experiment import ExperimentContext, StructuredConfig
from gamo.run.structured_methods import (
    print_summary,
    run_static_baselines,
    run_stochastic_method,
)
from gamo.utils.paths import results_path, write_json_atomic


def run_structured(
    config: StructuredConfig,
) -> dict[str, dict]:
    """Run the complete structured-pruning comparison."""
    context = ExperimentContext.load(config.search)
    validation_evaluator = prepare_batched_evaluation(
        context.model, context.val_loader, context.device
    )
    test_evaluator = prepare_batched_evaluation(
        context.model, context.test_loader, context.device
    )
    labels = [ga_label(crossover) for crossover in config.crossovers]

    print(
        f"Device: {context.device} | hidden neurons: {num_neurons(context.model)} | "
        f"dense acc {context.dense_accuracy:.2f}% | seeds: {config.search.num_seeds}"
    )
    if labels:
        print(f"GA configs: {', '.join(labels)}")

    results = run_static_baselines(context, test_evaluator)

    search_methods = (
        (
            RANDOM_SEARCH,
            lambda sparsity: random_search_structured(
                context.model,
                context.val_loader,
                context.device,
                sparsity,
                pop_size=config.search.pop_size,
                n_gen=config.search.num_iterations,
                evaluator=validation_evaluator,
            ),
        ),
        (
            HILL_CLIMBING,
            lambda sparsity: hill_climb_structured(
                context.model,
                context.val_loader,
                context.device,
                sparsity,
                pop_size=config.search.pop_size,
                n_steps=config.search.num_iterations,
                evaluator=validation_evaluator,
            ),
        ),
    )
    for label, search in search_methods:
        print(
            f"{label}: {config.search.evaluation_budget:,} mask evaluations "
            "per seed/sparsity"
        )
        results.update(run_stochastic_method(label, context, test_evaluator, search))

    for crossover in config.crossovers:
        label = ga_label(crossover)

        # Bind this loop value now so each search keeps its own crossover method.
        def search(sparsity, crossover=crossover):
            return evolve_structured(
                context.model,
                context.val_loader,
                context.device,
                sparsity,
                pop_size=config.search.pop_size,
                n_gen=config.search.num_iterations,
                t_size=config.search.tournament_size,
                crossover=crossover,
                evaluator=validation_evaluator,
            )

        results.update(run_stochastic_method(label, context, test_evaluator, search))

    methods = [
        *STATIC_METHODS,
        RANDOM_SEARCH,
        HILL_CLIMBING,
        *labels,
    ]
    print_summary(config.search, results, methods)
    output_path = _save(config, context, results)
    print(f"\nSaved {output_path}")
    return results


def _save(
    config: StructuredConfig,
    context: ExperimentContext,
    results: dict[str, dict],
):
    output = {
        "dataset": "FashionMNIST",
        "architecture": {
            "input_size": context.model.input_size,
            "hidden_sizes": list(context.model.hidden_sizes),
            "output_size": context.model.num_classes,
        },
        "sparsities": list(config.search.sparsities),
        "dense_acc": context.dense_accuracy,
        "methods": results,
        "config": {
            **config.as_dict(),
            "mutation_rule": "1 / genome length",
            "fitness_evaluations_per_full_run": config.search.evaluation_budget,
        },
    }
    output_path = results_path("structured.json", mkdir=True)
    write_json_atomic(output_path, output)
    return output_path
