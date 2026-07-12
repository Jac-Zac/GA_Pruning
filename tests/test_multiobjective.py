"""Focused tests for the Platypus accuracy/sparsity search."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from gamo.ga.multiobjective import optimize_structured_multiobjective
from gamo.ga.structured import num_neurons
from gamo.model.model import SimpleMLP
from gamo.run.experiment import MultiobjectiveConfig
from gamo.utils.environment import set_seed
from gamo.utils.pruning import structured_sparsity_stats

CPU = torch.device("cpu")


def _model():
    torch.manual_seed(0)
    return SimpleMLP(hidden_sizes=(4, 2), num_classes=3)


def _loader():
    torch.manual_seed(1)
    return DataLoader(
        TensorDataset(torch.randn(12, 784), torch.randint(0, 3, (12,))),
        batch_size=6,
        shuffle=False,
    )


def test_platypus_search_returns_a_valid_nondominated_front():
    set_seed(7)
    model = _model()
    result = optimize_structured_multiobjective(
        model,
        _loader(),
        CPU,
        population_size=4,
        max_evaluations=8,
    )

    assert result.evaluations == 8
    assert result.pareto_front
    assert [point.parameter_sparsity for point in result.pareto_front] == sorted(
        point.parameter_sparsity for point in result.pareto_front
    )
    for point in result.pareto_front:
        assert point.mask.dtype == torch.bool
        assert point.mask.numel() == num_neurons(model)
        assert 0 <= point.validation_accuracy <= 1
        assert point.parameter_sparsity == pytest.approx(
            structured_sparsity_stats(
                point.mask,
                list(model.hidden_sizes),
                input_size=model.input_size,
                output_size=model.num_classes,
            )["parameter_sparsity"]
        )

    for left in result.pareto_front:
        for right in result.pareto_front:
            dominates = (
                right.validation_accuracy >= left.validation_accuracy
                and right.parameter_sparsity >= left.parameter_sparsity
                and (
                    right.validation_accuracy > left.validation_accuracy
                    or right.parameter_sparsity > left.parameter_sparsity
                )
            )
            assert not dominates


def test_multiobjective_config_rejects_an_incomplete_initial_budget():
    with pytest.raises(ValueError, match="initial population"):
        MultiobjectiveConfig(population_size=10, max_evaluations=9)
