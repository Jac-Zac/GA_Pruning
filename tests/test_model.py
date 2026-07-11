"""Tests for the non-trivial batched neuron-ablation model."""

import torch

from gamo.model.model import SimpleMLP
from gamo.utils.model_utils import (
    flatten_params,
    set_params_from_tensor,
    unflatten_batched_params,
)


def test_neuron_ablation_applies_each_layer_mask_correctly():
    torch.manual_seed(0)
    dense = SimpleMLP(hidden_sizes=(16, 8), num_classes=10).eval()
    inputs = torch.randn(2, 784)
    partial_mask = torch.cat(
        [
            torch.arange(16) % 2 == 0,
            torch.arange(8) < 4,
        ]
    ).float()

    with torch.inference_mode():
        kept = dense(inputs, torch.ones(1, 24))
        pruned = dense(inputs, torch.zeros(1, 24))
        partial = dense(inputs, partial_mask.unsqueeze(0))

        first = torch.relu(dense.hidden_layers[0](inputs)) * partial_mask[:16]
        second = torch.relu(dense.hidden_layers[2](first)) * partial_mask[16:]
        expected_partial = dense.output_layer(second)

    assert kept.shape == (1, 2, 10)
    assert torch.allclose(kept, dense(inputs).unsqueeze(0), atol=1e-5)
    assert torch.allclose(pruned[0, 0], pruned[0, 1], atol=1e-6)
    assert torch.allclose(partial[0], expected_partial, atol=1e-6)


def test_batched_weights_match_dense_model():
    torch.manual_seed(1)
    dense = SimpleMLP(hidden_sizes=(8, 4), num_classes=3).eval()
    batched_weights = [
        parameter.detach().unsqueeze(0) for parameter in dense.parameters()
    ]
    inputs = torch.randn(5, 784)

    with torch.inference_mode():
        batched_output = dense(inputs, weights=batched_weights)

    assert batched_output.shape == (1, 5, 3)
    assert torch.allclose(batched_output[0], dense(inputs), atol=1e-6)


def test_flat_parameter_round_trip_and_batched_shapes():
    model = SimpleMLP(hidden_sizes=(4,), num_classes=3)
    original = flatten_params(model).clone()
    set_params_from_tensor(model, torch.zeros_like(original))
    set_params_from_tensor(model, original)

    assert torch.equal(flatten_params(model), original)
    batched = unflatten_batched_params(original.repeat(2, 1), model)
    assert [tensor.shape[1:] for tensor in batched] == [
        parameter.shape for parameter in model.parameters()
    ]
