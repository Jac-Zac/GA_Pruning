"""Architecture-aware statistics for structured neuron masks."""

import torch


def structured_parameter_sparsities(
    neuron_masks: torch.Tensor,
    hidden_sizes: list[int],
    input_size: int = 784,
    output_size: int = 10,
) -> torch.Tensor:
    """Return effective parameter sparsity for each structured neuron mask."""
    if neuron_masks.ndim != 2 or neuron_masks.shape[1] != sum(hidden_sizes):
        raise ValueError("neuron_masks must have shape (population, sum(hidden_sizes))")
    if not hidden_sizes:
        raise ValueError("hidden_sizes must not be empty")

    layer_keeps = torch.stack(
        [block.sum(dim=1) for block in neuron_masks.float().split(hidden_sizes, dim=1)],
        dim=1,
    )
    kept_weights = input_size * layer_keeps[:, 0]
    for index in range(1, len(hidden_sizes)):
        kept_weights += layer_keeps[:, index - 1] * layer_keeps[:, index]
    kept_weights += output_size * layer_keeps[:, -1]
    kept_biases = layer_keeps.sum(dim=1) + output_size

    layer_sizes = [input_size, *hidden_sizes, output_size]
    total_weights = sum(
        left * right for left, right in zip(layer_sizes, layer_sizes[1:])
    )
    total_parameters = total_weights + sum(hidden_sizes) + output_size
    return 1.0 - (kept_weights + kept_biases) / total_parameters


def neuron_to_weight_masks(
    neuron_mask: torch.Tensor,
    hidden_sizes: list[int],
    input_size: int = 784,
    output_size: int = 10,
) -> dict[str, torch.Tensor]:
    """Expand a flat neuron mask into masks for each weight matrix."""
    flat_mask = neuron_mask.detach().cpu().bool().flatten()
    if not hidden_sizes or flat_mask.numel() != sum(hidden_sizes):
        raise ValueError("neuron mask length does not match hidden_sizes")

    layers = list(flat_mask.split(hidden_sizes))
    weight_masks = {
        "Input → L0": layers[0].unsqueeze(1).expand(hidden_sizes[0], input_size)
    }
    for index in range(1, len(layers)):
        weight_masks[f"L{index - 1} → L{index}"] = layers[index].unsqueeze(1) & layers[
            index - 1
        ].unsqueeze(0)
    weight_masks[f"L{len(layers) - 1} → Output"] = (
        layers[-1].unsqueeze(0).expand(output_size, hidden_sizes[-1])
    )
    return weight_masks


def structured_sparsity_stats(
    neuron_mask: torch.Tensor,
    hidden_sizes: list[int],
    input_size: int = 784,
    output_size: int = 10,
) -> dict[str, float | int]:
    """Count retained neurons, weights, biases, and parameters."""
    flat_mask = neuron_mask.detach().cpu().bool().flatten()
    if not hidden_sizes or flat_mask.numel() != sum(hidden_sizes):
        raise ValueError("neuron mask length does not match hidden_sizes")

    kept_per_layer = [int(layer.sum()) for layer in flat_mask.split(hidden_sizes)]
    widths = [input_size, *hidden_sizes, output_size]
    kept_widths = [input_size, *kept_per_layer, output_size]
    total_weights = sum(left * right for left, right in zip(widths, widths[1:]))
    kept_weights = sum(
        left * right for left, right in zip(kept_widths, kept_widths[1:])
    )
    total_neurons = sum(hidden_sizes)
    kept_neurons = sum(kept_per_layer)
    total_parameters = total_weights + total_neurons + output_size
    kept_parameters = kept_weights + kept_neurons + output_size
    return {
        "total_neurons": total_neurons,
        "kept_neurons": kept_neurons,
        "neuron_sparsity": 1 - kept_neurons / total_neurons,
        "total_weights": total_weights,
        "kept_weights": kept_weights,
        "weight_sparsity": 1 - kept_weights / total_weights,
        "total_parameters": total_parameters,
        "kept_parameters": kept_parameters,
        "parameter_sparsity": 1 - kept_parameters / total_parameters,
    }
