"""Unstructured (per-weight) genetic pruning used as the negative control.

Unstructured pruning evolves one bit per *weight* (~335k genes), huge, near-flat
search space where the GA does not beat magnitude pruning.
"""

from collections.abc import Callable

import torch
from torch import nn

from gamo.model.model import SimpleMLP
from gamo.utils.model_utils import unflatten_batched_params

from .operators import random_population, target_keep_count


def magnitude_pruning_mask(weights: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Project-owned global L1/magnitude pruning over a flat weight vector.

    This deliberately implements PyTorch's L1-unstructured criterion directly so the
    experiment's pruning rule remains transparent and local.
    """
    keep_count = target_keep_count(weights.numel(), sparsity)
    mask = torch.zeros_like(weights, dtype=torch.bool)
    if keep_count:
        indices = torch.topk(weights.abs(), keep_count, largest=True).indices
        mask[indices] = True
    return mask


def weight_parameter_selector(model: nn.Module) -> torch.Tensor:
    """Return a flat boolean selector for the model's weight tensors only.

    The unstructured experiment deliberately excludes bias vectors from its genome and
    sparsity budget.  Keeping this selector in one place ensures that GA, random, and
    magnitude masks all use the same definition of "weight sparsity".
    """
    return torch.cat(
        [
            torch.full(
                (param.numel(),), param.ndim > 1, dtype=torch.bool, device=param.device
            )
            for param in model.parameters()
        ]
    )


def expand_weight_masks(
    weight_masks: torch.Tensor, weight_selector: torch.Tensor
) -> torch.Tensor:
    """Expand weight-only keep masks to flat parameter masks with biases always kept.

    ``weight_masks`` may contain one mask ``(n_weights,)`` or a population
    ``(population, n_weights)``.  The returned tensor has the matching leading dimensions
    and one entry per model parameter.
    """
    if weight_masks.shape[-1] != int(weight_selector.sum().item()):
        raise ValueError(
            "weight mask length does not match the number of weight parameters"
        )
    selector = weight_selector.to(weight_masks.device)
    full_shape = (*weight_masks.shape[:-1], selector.numel())
    # Start with every parameter kept, then overwrite weight positions only.
    full = torch.ones(full_shape, dtype=torch.bool, device=weight_masks.device)
    full[..., selector] = weight_masks.bool()
    return full


def random_weight_mask(n: int, sparsity: float, device: torch.device) -> torch.Tensor:
    """Create a random keep-mask over ``n`` weights with exact target sparsity."""
    return random_population(1, n, target_keep_count(n, sparsity), device)[0]


def create_batched_fitness_func(
    original_weights: torch.Tensor,
    model_template: SimpleMLP,
    device: torch.device,
    val_loader: torch.utils.data.DataLoader,
    weight_selector: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a vectorised fitness function over an entire population."""
    batches = [(images.to(device), labels.to(device)) for images, labels in val_loader]
    if not batches:
        raise ValueError("validation loader must not be empty")

    @torch.inference_mode()
    def batched_fitness_func(population: torch.Tensor) -> torch.Tensor:
        batched_masks = expand_weight_masks(population.to(device), weight_selector)
        batched_masked_weights = original_weights * batched_masks
        batched_weights = unflatten_batched_params(
            batched_masked_weights, model_template
        )

        pop_size = population.shape[0]
        total_correct = torch.zeros(pop_size, device=device)
        total_samples = 0

        for images, labels in batches:
            logits = model_template(images, weights=batched_weights)
            predictions = logits.argmax(dim=2)
            total_correct += (predictions == labels.unsqueeze(0)).sum(dim=1)
            total_samples += labels.size(0)

        return total_correct / total_samples

    return batched_fitness_func
