"""Shared fixed-cardinality bit-mask operations."""

import torch


def target_keep_count(n: int, sparsity: float) -> int:
    """Return the closest achievable keep budget for ``sparsity``."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError("sparsity must be between 0 and 1")
    return round((1.0 - sparsity) * n)


def random_population(
    pop_size: int,
    n: int,
    target_ones: int,
    device: torch.device,
) -> torch.Tensor:
    """Uniformly sample ``pop_size`` fixed-cardinality masks."""
    if pop_size < 1 or n < 1:
        raise ValueError("pop_size and genome length must be positive")
    if not 0 <= target_ones <= n:
        raise ValueError("target_ones must be within the genome length")
    # Top-k random scores sample an exact size subset.
    scores = torch.rand(pop_size, n, device=device)
    keep = scores.topk(target_ones, dim=1).indices
    masks = torch.zeros(pop_size, n, dtype=torch.bool, device=device)
    masks.scatter_(1, keep, True)
    return masks


def tournament_selection(
    population: torch.Tensor,
    fitnesses: torch.Tensor,
    num_parents: int,
    tournament_size: int,
) -> torch.Tensor:
    """Select parents by keeping the best mask in each random tournament."""
    if population.ndim != 2:
        raise ValueError("population must have shape (population, genes)")
    if fitnesses.shape != (population.shape[0],):
        raise ValueError("fitnesses must contain one score per mask")
    if fitnesses.device != population.device:
        raise ValueError("population and fitnesses must use the same device")
    if num_parents < 1 or tournament_size < 1:
        raise ValueError("num_parents and tournament_size must be positive")

    # NOTE: This samples with replacement also in tournaments to allow for vectorization
    candidates = torch.randint(
        population.shape[0],
        (num_parents, tournament_size),
        device=population.device,
    )
    # Gather converts each tournament's winning column into a population index.
    winner_columns = fitnesses[candidates].argmax(dim=1, keepdim=True)
    winner_indices = candidates.gather(1, winner_columns).squeeze(1)
    return population[winner_indices]


def crossover_population(
    parents: torch.Tensor,
    method: str | None = "uniform",
) -> torch.Tensor:
    """Cross paired parents using uniform, two-point, or no crossover."""
    if parents.ndim != 2 or parents.shape[0] < 2 or parents.shape[0] % 2:
        raise ValueError("parents must contain an even number of 1D genomes")
    if method not in {None, "uniform", "two_point"}:
        raise ValueError("method must be None, 'uniform', or 'two_point'")

    num_pairs = parents.shape[0] // 2
    genome_length = parents.shape[1]
    left, right = parents[:num_pairs], parents[num_pairs:]

    if method is None:
        crossover_mask = torch.zeros_like(left, dtype=torch.bool)
    elif method == "uniform":
        crossover_mask = torch.rand(left.shape, device=parents.device) < 0.5
    else:
        if genome_length < 2:
            raise ValueError("two-point crossover requires at least two genes")
        first = torch.randint(0, genome_length, (num_pairs, 1), device=parents.device)
        # Sample from n-1 positions, then skip `first` so the cut points differ.
        second = torch.randint(
            0, genome_length - 1, (num_pairs, 1), device=parents.device
        )
        second += second >= first
        low, high = torch.min(first, second), torch.max(first, second)
        genes = torch.arange(genome_length, device=parents.device).unsqueeze(0)
        crossover_mask = (genes >= low) & (genes < high)

    return torch.cat(
        [
            torch.where(crossover_mask, right, left),
            torch.where(crossover_mask, left, right),
        ]
    )


def repair_population(masks: torch.Tensor, target_ones: int) -> torch.Tensor:
    """Repair every mask to contain exactly ``target_ones`` kept genes."""
    if masks.ndim != 2:
        raise ValueError("masks must have shape (population, genes)")
    population, genome_length = masks.shape
    if not 0 <= target_ones <= genome_length:
        raise ValueError("target_ones must be within the genome length")
    # Ones score above zeros (since results of rand is always bounuded [0, 1)).
    # This breaks ties throuhg the random offset
    scores = masks.float() + torch.rand(population, genome_length, device=masks.device)
    keep = scores.topk(target_ones, dim=1).indices
    repaired = torch.zeros_like(masks, dtype=torch.bool)
    repaired.scatter_(1, keep, True)
    return repaired


def bit_flip_mutation(masks: torch.Tensor) -> torch.Tensor:
    """Mutate bits at a rate of one expected flip per genome."""
    if masks.ndim != 2 or masks.shape[1] < 1:
        raise ValueError("masks must have shape (population, genes)")
    mutation_rate = 1 / masks.shape[1]
    flips = torch.rand(masks.shape, device=masks.device) < mutation_rate
    return masks.bool() ^ flips
