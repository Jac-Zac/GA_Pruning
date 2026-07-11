"""Representation-agnostic search over fixed-cardinality bit masks."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .operators import (
    bit_flip_mutation,
    crossover_population,
    random_population,
    repair_population,
    tournament_selection,
)

FitnessFunction = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class SearchResult:
    mask: torch.Tensor
    fitness: float
    curve: list[float]


def _fitness(fitness_fn: FitnessFunction, population: torch.Tensor) -> torch.Tensor:
    fitnesses = fitness_fn(population)
    if fitnesses.shape != (population.shape[0],):
        raise ValueError("fitness function must return one score per mask")
    return fitnesses


def random_search(
    fitness_fn: FitnessFunction,
    *,
    n: int,
    target_ones: int,
    device: torch.device,
    pop_size: int,
    iterations: int,
) -> SearchResult:
    """Return the best mask from independent fixed-budget samples."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    best_mask = None
    best_fitness = float("-inf")
    curve = []
    for _ in range(iterations):
        population = random_population(pop_size, n, target_ones, device)
        fitnesses = _fitness(fitness_fn, population)
        index = int(fitnesses.argmax().item())
        candidate = fitnesses[index].item()
        if best_mask is None or candidate > best_fitness:
            best_fitness = candidate
            best_mask = population[index].clone()
        curve.append(best_fitness * 100)
    return SearchResult(best_mask, best_fitness, curve)


def _swap_neighbors(masks: torch.Tensor) -> torch.Tensor:
    # Masked scores make argmax choose one kept and one dropped gene per row.
    keep_scores = torch.rand(masks.shape, device=masks.device).masked_fill(~masks, -1)
    drop_scores = torch.rand(masks.shape, device=masks.device).masked_fill(masks, -1)
    rows = torch.arange(masks.shape[0], device=masks.device)
    proposed = masks.clone()
    proposed[rows, keep_scores.argmax(dim=1)] = False
    proposed[rows, drop_scores.argmax(dim=1)] = True
    return proposed


def hill_climb(
    fitness_fn: FitnessFunction,
    *,
    n: int,
    target_ones: int,
    device: torch.device,
    pop_size: int,
    iterations: int,
) -> SearchResult:
    """Run parallel one-swap hill-climbing chains."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    chains = random_population(pop_size, n, target_ones, device)
    current = _fitness(fitness_fn, chains).clone()
    index = int(current.argmax().item())
    best_mask = chains[index].clone()
    best_fitness = current[index].item()
    curve = [best_fitness * 100]

    for _ in range(1, iterations):
        proposed = _swap_neighbors(chains) if 0 < target_ones < n else chains.clone()
        proposed_fitness = _fitness(fitness_fn, proposed)
        accept = proposed_fitness >= current
        chains[accept] = proposed[accept]
        current[accept] = proposed_fitness[accept]
        index = int(current.argmax().item())
        if current[index].item() > best_fitness:
            best_fitness = current[index].item()
            best_mask = chains[index].clone()
        curve.append(best_fitness * 100)
    return SearchResult(best_mask, best_fitness, curve)


def genetic_search(
    fitness_fn: FitnessFunction,
    *,
    n: int,
    target_ones: int,
    device: torch.device,
    pop_size: int,
    iterations: int,
    tournament_size: int = 4,
    crossover: str | None = "uniform",
    elitism: bool = True,
) -> SearchResult:
    """Evolve a fixed-cardinality bit-mask population."""
    if iterations < 1 or tournament_size < 1:
        raise ValueError("iterations and tournament_size must be positive")
    if crossover not in {None, "uniform", "two_point"}:
        raise ValueError("crossover must be None, 'uniform', or 'two_point'")
    if crossover == "two_point" and n < 2:
        raise ValueError("two-point crossover requires at least two genes")

    population = random_population(pop_size, n, target_ones, device)
    best_mask = population[0].clone()
    best_fitness = float("-inf")
    curve = []

    for generation in range(iterations):
        fitnesses = _fitness(fitness_fn, population)
        generation_best = int(fitnesses.argmax().item())
        candidate = fitnesses[generation_best].item()
        if candidate > best_fitness:
            best_fitness = candidate
            best_mask = population[generation_best].clone()
        curve.append(best_fitness * 100)
        if generation == iterations - 1:
            break

        # Keep pop_size if elitism is enabled
        elite = population[generation_best].unsqueeze(0) if elitism else population[:0]
        offspring_count = pop_size - elite.shape[0]
        if offspring_count == 0:
            population = elite.clone()
            continue

        # Crossover needs pairs, so temporarily round an odd child count up to even.
        draws = offspring_count + offspring_count % 2
        parents = tournament_selection(
            population,
            fitnesses,
            num_parents=draws,
            tournament_size=tournament_size,
        )
        children = crossover_population(parents, crossover)[:offspring_count]
        children = repair_population(bit_flip_mutation(children), target_ones)
        population = torch.cat([elite, children])

    return SearchResult(best_mask, best_fitness, curve)
