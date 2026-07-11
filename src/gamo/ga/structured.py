"""Neuron-level pruning masks and their batched fitness evaluation."""

import copy
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from gamo.model.model import SimpleMLP

from .operators import random_population, target_keep_count
from .search import SearchResult, genetic_search, hill_climb, random_search

PreparedEvaluation = tuple[
    SimpleMLP,
    list[tuple[torch.Tensor, torch.Tensor]],
]


def get_linears(model: SimpleMLP) -> list[nn.Linear]:
    """Return hidden and output linear layers in forward order."""
    hidden = [layer for layer in model.hidden_layers if isinstance(layer, nn.Linear)]
    return [*hidden, model.output_layer]


def num_neurons(model: SimpleMLP) -> int:
    return sum(model.hidden_sizes)


def per_layer_prune_fractions(
    model: SimpleMLP, neuron_mask: torch.Tensor
) -> list[float]:
    """Return the fraction pruned in each hidden layer."""
    if neuron_mask.numel() != num_neurons(model):
        raise ValueError("neuron mask length does not match the model")
    fractions, offset = [], 0
    for layer in get_linears(model)[:-1]:
        block = neuron_mask[offset : offset + layer.out_features]
        fractions.append(1 - block.float().mean().item())
        offset += layer.out_features
    return fractions


def neuron_importance(model: SimpleMLP) -> torch.Tensor:
    """Return incoming-weight L2 norms for every hidden neuron."""
    return torch.cat(
        [layer.weight.detach().norm(dim=1).cpu() for layer in get_linears(model)[:-1]]
    )


def _mask_from_scores(scores: torch.Tensor, keep_count: int) -> torch.Tensor:
    if not 0 <= keep_count <= scores.numel():
        raise ValueError("keep_count must be within the score length")
    mask = torch.zeros(scores.numel(), dtype=torch.bool)
    if keep_count:
        mask[scores.topk(keep_count).indices] = True
    return mask


def neuron_magnitude_mask(model: SimpleMLP, sparsity: float) -> torch.Tensor:
    """Globally keep the neurons with the largest incoming-weight norms."""
    scores = neuron_importance(model)
    return _mask_from_scores(scores, target_keep_count(scores.numel(), sparsity))


def _proportional_layer_keeps(layer_sizes: list[int], total_keep: int) -> list[int]:
    """Allocate an exact global budget proportionally across layers."""
    total = sum(layer_sizes)
    if not 0 <= total_keep <= total:
        raise ValueError("total_keep must be within the total layer size")
    if total == 0:
        return [0] * len(layer_sizes)
    raw = [total_keep * size / total for size in layer_sizes]
    keeps = [int(value) for value in raw]
    remainder = total_keep - sum(keeps)
    # Give leftover slots to layers with the largest fractional shares.
    order = sorted(
        range(len(raw)),
        key=lambda index: (raw[index] - keeps[index], -index),
        reverse=True,
    )
    for index in order[:remainder]:
        keeps[index] += 1
    return keeps


def neuron_magnitude_mask_per_layer(model: SimpleMLP, sparsity: float) -> torch.Tensor:
    """Apply magnitude pruning within each layer using one exact global budget."""
    hidden = get_linears(model)[:-1]
    layer_sizes = [layer.out_features for layer in hidden]
    total_keep = target_keep_count(sum(layer_sizes), sparsity)
    keep_counts = _proportional_layer_keeps(layer_sizes, total_keep)
    return torch.cat(
        [
            _mask_from_scores(layer.weight.detach().norm(dim=1).cpu(), keep_count)
            for layer, keep_count in zip(hidden, keep_counts, strict=True)
        ]
    )


def random_neuron_mask(n: int, sparsity: float) -> torch.Tensor:
    """Sample one exact-sparsity neuron keep-mask on CPU."""
    target = target_keep_count(n, sparsity)
    return random_population(1, n, target, torch.device("cpu"))[0]


def prepare_batched_evaluation(
    model: SimpleMLP,
    loader,
    device: torch.device,
) -> PreparedEvaluation:
    """Copy the model and preload a loader for repeated population evaluation."""
    neuron_model = copy.deepcopy(model).to(device)
    batches = [
        (images.flatten(1).to(device), labels.to(device)) for images, labels in loader
    ]
    if not batches:
        raise ValueError("evaluation loader must not be empty")
    return neuron_model, batches


@torch.inference_mode()
def evaluate_masks(
    neuron_model: SimpleMLP,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return accuracy and average loss for every mask."""
    expected = sum(neuron_model.hidden_sizes)
    if masks.ndim != 2 or masks.shape[1] != expected:
        raise ValueError(f"masks must have shape (population, {expected})")
    population = masks.shape[0]
    correct = torch.zeros(population, device=masks.device)
    total_loss = torch.zeros(population, device=masks.device)
    total = 0
    for images, labels in batches:
        logits = neuron_model(images, masks)
        correct += (logits.argmax(dim=2) == labels.unsqueeze(0)).sum(dim=1)
        repeated_labels = labels.unsqueeze(0).expand(population, -1)
        # Cross-entropy treats population as batch and images as an extra dimension.
        losses = F.cross_entropy(
            logits.transpose(1, 2), repeated_labels, reduction="none"
        )
        total_loss += losses.sum(dim=1)
        total += labels.size(0)
    return correct / total, total_loss / total


def _search_problem(
    model: SimpleMLP,
    loader,
    device: torch.device,
    sparsity: float,
    evaluator: PreparedEvaluation | None,
) -> tuple[int, int, Callable[[torch.Tensor], torch.Tensor]]:
    n = num_neurons(model)
    if evaluator is None:
        evaluator = prepare_batched_evaluation(model, loader, device)
    neuron_model, batches = evaluator

    def fitness(population: torch.Tensor) -> torch.Tensor:
        return evaluate_masks(neuron_model, batches, population)[0]

    return n, target_keep_count(n, sparsity), fitness


def random_search_structured(
    model: SimpleMLP,
    val_loader,
    device: torch.device,
    sparsity: float,
    pop_size: int = 100,
    n_gen: int = 100,
    evaluator: PreparedEvaluation | None = None,
) -> SearchResult:
    n, target, fitness = _search_problem(model, val_loader, device, sparsity, evaluator)
    return random_search(
        fitness,
        n=n,
        target_ones=target,
        device=device,
        pop_size=pop_size,
        iterations=n_gen,
    )


def hill_climb_structured(
    model: SimpleMLP,
    val_loader,
    device: torch.device,
    sparsity: float,
    *,
    pop_size: int = 100,
    n_steps: int = 150,
    evaluator: PreparedEvaluation | None = None,
) -> SearchResult:
    n, target, fitness = _search_problem(model, val_loader, device, sparsity, evaluator)
    return hill_climb(
        fitness,
        n=n,
        target_ones=target,
        device=device,
        pop_size=pop_size,
        iterations=n_steps,
    )


def evolve_structured(
    model: SimpleMLP,
    val_loader,
    device: torch.device,
    sparsity: float,
    pop_size: int = 100,
    n_gen: int = 100,
    t_size: int = 4,
    crossover: str | None = "uniform",
    elitism: bool = True,
    evaluator: PreparedEvaluation | None = None,
) -> SearchResult:
    n, target, fitness = _search_problem(model, val_loader, device, sparsity, evaluator)
    return genetic_search(
        fitness,
        n=n,
        target_ones=target,
        device=device,
        pop_size=pop_size,
        iterations=n_gen,
        tournament_size=t_size,
        crossover=crossover,
        elitism=elitism,
    )
