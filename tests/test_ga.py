"""Focused tests for the unstructured genetic algorithm."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from gamo.ga.operators import (
    bit_flip_mutation,
    crossover_population,
    repair_population,
    target_keep_count,
    tournament_selection,
)
from gamo.ga.unstructured import (
    create_batched_fitness_func,
    expand_weight_masks,
    magnitude_pruning_mask,
    random_weight_mask,
    weight_parameter_selector,
)
from gamo.model.model import SimpleMLP
from gamo.utils.model_utils import flatten_params


def _model():
    torch.manual_seed(0)
    return SimpleMLP(hidden_sizes=(4,), num_classes=3)


def test_sparsity_budget_is_exact_and_validated():
    assert target_keep_count(10, 0.3) == 7
    assert target_keep_count(10, 0.0) == 10
    assert target_keep_count(10, 1.0) == 0
    with pytest.raises(ValueError):
        target_keep_count(10, 1.1)

    random_mask = random_weight_mask(10, 0.3, torch.device("cpu"))
    assert random_mask.dtype == torch.bool
    assert random_mask.sum() == 7


def test_magnitude_pruning_keeps_the_largest_weights():
    weights = torch.tensor([0.1, 0.5, 0.2, 0.8, 0.3])
    assert torch.equal(
        magnitude_pruning_mask(weights, 0.4),
        torch.tensor([False, True, False, True, True]),
    )


def test_weight_masks_prune_only_weights_for_a_population():
    model = _model()
    selector = weight_parameter_selector(model)
    values = flatten_params(model)[selector]
    masks = torch.ones(2, values.numel(), dtype=torch.bool)
    masks[1, : values.numel() // 2] = False

    full_masks = expand_weight_masks(masks, selector)

    assert full_masks.shape == (2, selector.numel())
    assert full_masks[:, ~selector].all()  # biases are never part of the genome
    assert torch.equal(full_masks[:, selector], masks)


def test_batched_fitness_evaluates_every_individual_on_supplied_data():
    model = _model()
    model.eval()
    selector = weight_parameter_selector(model)
    images = torch.randn(7, 784)
    labels = torch.randint(0, 3, (7,))
    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=3,
        shuffle=False,
    )
    fitness = create_batched_fitness_func(
        flatten_params(model),
        model,
        torch.device("cpu"),
        loader,
        selector,
    )
    population = torch.stack(
        [
            torch.ones(int(selector.sum()), dtype=torch.bool),
            torch.zeros(int(selector.sum()), dtype=torch.bool),
        ]
    )

    scores = fitness(population)

    assert scores.shape == (2,)
    assert ((scores >= 0) & (scores <= 1)).all()
    dense_accuracy = (model(images).argmax(1) == labels).float().mean()
    assert torch.equal(scores[0], dense_accuracy)


def test_population_repair_and_mutation_preserve_binary_genomes():
    population = torch.stack(
        [torch.ones(10, dtype=torch.bool), torch.zeros(10, dtype=torch.bool)]
    )
    repaired = repair_population(population, 5)

    assert repaired.shape == population.shape
    assert repaired.sum(dim=1).tolist() == [5, 5]
    assert torch.equal(
        bit_flip_mutation(torch.tensor([[True]])), torch.tensor([[False]])
    )


def test_tournament_selection_returns_each_tournament_winner(monkeypatch):
    population = torch.tensor([[False, False], [True, False], [True, True]])
    fitnesses = torch.tensor([0.1, 0.9, 0.2])
    candidates = torch.tensor([[0, 1], [2, 0]])
    monkeypatch.setattr(
        torch,
        "randint",
        lambda high, size, device: candidates.to(device),
    )

    selected = tournament_selection(population, fitnesses, 2, 2)

    assert torch.equal(selected, population[[1, 2]])


@pytest.mark.parametrize("method", ["uniform", "two_point"])
def test_crossover_produces_complementary_children(method):
    parents = torch.tensor(
        [
            [False, False, False, False],
            [True, True, True, True],
            [True, True, True, True],
            [False, False, False, False],
        ]
    )

    children = crossover_population(parents, method)

    assert children.shape == parents.shape
    assert (children[:2] ^ children[2:]).all()


def test_no_crossover_copies_the_selected_parents():
    parents = torch.tensor([[False, True], [True, False], [True, True], [False, False]])
    assert torch.equal(crossover_population(parents, None), parents)


def test_two_point_crossover_rejects_one_gene_genomes():
    with pytest.raises(ValueError, match="at least two genes"):
        crossover_population(torch.tensor([[False], [True]]), "two_point")
