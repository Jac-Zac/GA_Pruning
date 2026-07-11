"""Focused tests for structured pruning and its genetic search."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from gamo.ga.operators import target_keep_count
from gamo.ga.search import SearchResult
from gamo.ga.structured import (
    evaluate_masks,
    evolve_structured,
    get_linears,
    hill_climb_structured,
    neuron_importance,
    neuron_magnitude_mask,
    neuron_magnitude_mask_per_layer,
    num_neurons,
    prepare_batched_evaluation,
    random_neuron_mask,
    random_search_structured,
)
from gamo.model.model import SimpleMLP
from gamo.run.experiment import SearchConfig
from gamo.run.structured_methods import run_stochastic_method
from gamo.utils.pruning import neuron_to_weight_masks, structured_sparsity_stats

HIDDEN = (5, 2)
CPU = torch.device("cpu")


def _model():
    torch.manual_seed(0)
    return SimpleMLP(hidden_sizes=HIDDEN, num_classes=10)


def _loader(batch_size=16):
    torch.manual_seed(1)
    return DataLoader(
        TensorDataset(torch.randn(32, 784), torch.randint(0, 10, (32,))),
        batch_size=batch_size,
        shuffle=False,
    )


def test_structured_baselines_obey_the_same_budget_and_magnitude_ranking():
    model = _model()
    sparsity = 0.5
    expected = target_keep_count(num_neurons(model), sparsity)
    global_mask = neuron_magnitude_mask(model, sparsity)
    per_layer_mask = neuron_magnitude_mask_per_layer(model, sparsity)
    random_mask = random_neuron_mask(num_neurons(model), sparsity)

    assert {int(mask.sum()) for mask in (global_mask, per_layer_mask, random_mask)} == {
        expected
    }
    scores = neuron_importance(model)
    assert scores[global_mask].min() >= scores[~global_mask].max()

    layer_keeps, offset = [], 0
    for layer in get_linears(model)[:-1]:
        block = per_layer_mask[offset : offset + layer.out_features]
        layer_keeps.append(int(block.sum()))
        offset += layer.out_features
    assert layer_keeps == [3, 1]


@pytest.mark.parametrize(("crossover", "elitism"), [("two_point", True), (None, False)])
def test_evolution_supports_key_variants_without_breaking_budget(crossover, elitism):
    model = _model()
    expected = target_keep_count(num_neurons(model), 0.5)
    result = evolve_structured(
        model,
        _loader(),
        CPU,
        0.5,
        pop_size=5,
        n_gen=2,
        crossover=crossover,
        elitism=elitism,
    )
    assert result.mask.sum() == expected
    assert 0 <= result.fitness <= 1
    assert len(result.curve) == 2


@pytest.mark.parametrize(
    ("search", "iterations"),
    [
        (random_search_structured, {"n_gen": 3}),
        (hill_climb_structured, {"n_steps": 3}),
    ],
)
def test_alternative_searches_preserve_budget_and_best_so_far_curve(search, iterations):
    model = _model()
    expected = target_keep_count(num_neurons(model), 0.7)
    result = search(model, _loader(), CPU, 0.7, pop_size=4, **iterations)
    assert result.mask.sum() == expected
    assert 0 <= result.fitness <= 1
    assert len(result.curve) == 3
    assert result.curve == sorted(result.curve)


def test_batched_mask_evaluation_matches_dense_accuracy_and_handles_population():
    model = _model().eval()
    neuron_model, batches = prepare_batched_evaluation(
        model, _loader(batch_size=16), CPU
    )
    masks = torch.stack(
        [
            torch.ones(num_neurons(model), dtype=torch.bool),
            torch.zeros(num_neurons(model), dtype=torch.bool),
        ]
    )

    fitness, losses = evaluate_masks(neuron_model, batches, masks)

    correct = sum((model(x).argmax(1) == y).sum().item() for x, y in batches)
    assert fitness.shape == losses.shape == (2,)
    assert abs(fitness[0].item() - correct / 32) < 1e-6
    dense_loss = (
        sum(
            F.cross_entropy(model(images), labels, reduction="sum").item()
            for images, labels in batches
        )
        / 32
    )
    assert abs(losses[0].item() - dense_loss) < 1e-6
    assert torch.isfinite(losses).all()


def test_neuron_masks_expand_to_correct_weight_connectivity():
    model = _model()
    layer_0 = torch.arange(HIDDEN[0]) % 2 == 0
    layer_1 = torch.arange(HIDDEN[1]) < HIDDEN[1] // 2
    masks = neuron_to_weight_masks(torch.cat([layer_0, layer_1]), list(HIDDEN))

    assert torch.equal(masks["Input → L0"], layer_0.unsqueeze(1).expand(-1, 784))
    assert torch.equal(masks["L0 → L1"], layer_1.unsqueeze(1) & layer_0.unsqueeze(0))
    assert torch.equal(masks["L1 → Output"], layer_1.unsqueeze(0).expand(10, -1))


def test_structured_sparsity_counts_output_bias_as_unpruned():
    all_pruned = structured_sparsity_stats(
        torch.zeros(sum(HIDDEN), dtype=torch.bool), list(HIDDEN)
    )
    assert all_pruned["weight_sparsity"] == 1.0
    assert all_pruned["kept_parameters"] == 10


def test_structured_method_keeps_each_runs_data_together():
    model = _model()
    config = SearchConfig(
        sparsities=(0.5,),
        num_seeds=1,
        pop_size=2,
        num_iterations=1,
        tournament_size=1,
        batch_size=16,
    )
    evaluator = prepare_batched_evaluation(model, _loader(), CPU)
    context = SimpleNamespace(config=config, model=model, device=CPU)
    mask = random_neuron_mask(num_neurons(model), 0.5)

    results = run_stochastic_method(
        "Test",
        context,
        evaluator,
        lambda _: SearchResult(mask, 0.5, [50.0]),
    )

    method = results["Test"]
    run = method["runs"]["0.5000"][0]
    assert method["accuracy"] == [run["accuracy"]]
    assert set(run) == {
        "seed",
        "accuracy",
        "mask",
        "curve",
        "layer_sparsity",
        "sparsity_stats",
    }
