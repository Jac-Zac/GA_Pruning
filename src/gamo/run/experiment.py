"""Canonical configuration and shared setup for the final experiments."""

from dataclasses import asdict, dataclass

import torch

from gamo.model.model import DEFAULT_HIDDEN_SIZES, SimpleMLP
from gamo.utils.data import get_dataloaders
from gamo.utils.environment import get_device
from gamo.utils.model_utils import (
    CHECKPOINT_NAME,
    eval_weights,
    flatten_params,
    load_pretrained_model,
)
from gamo.utils.paths import checkpoint_path

DEFAULT_SPARSITIES = (0.3, 0.5, 0.7, 0.85, 0.9)
VALID_CROSSOVERS = ("uniform", "two_point")


@dataclass(frozen=True)
class SearchConfig:
    """Settings shared by every mask search."""

    sparsities: tuple[float, ...] = DEFAULT_SPARSITIES
    num_seeds: int = 5
    pop_size: int = 100
    num_iterations: int = 250
    tournament_size: int = 4
    batch_size: int = 1024
    seed: int = 1337

    def __post_init__(self) -> None:
        if not self.sparsities or any(not 0 < value < 1 for value in self.sparsities):
            raise ValueError("sparsities must contain values between 0 and 1")
        if len(set(self.sparsities)) != len(self.sparsities):
            raise ValueError("sparsities must be unique")
        if self.num_seeds < 1:
            raise ValueError("num_seeds must be at least 1")
        if self.pop_size < 2:
            raise ValueError("pop_size must be at least 2")
        if min(self.num_iterations, self.tournament_size, self.batch_size) < 1:
            raise ValueError("search sizes must be positive")

    @property
    def seeds(self) -> range:
        return range(self.seed, self.seed + self.num_seeds)

    @property
    def evaluation_budget(self) -> int:
        return self.pop_size * self.num_iterations

    def as_dict(self) -> dict:
        values = asdict(self)
        values["sparsities"] = list(self.sparsities)
        return values


@dataclass(frozen=True)
class StructuredConfig:
    search: SearchConfig
    crossovers: tuple[str, ...] = VALID_CROSSOVERS

    def __post_init__(self) -> None:
        if not self.crossovers or not set(self.crossovers) <= set(VALID_CROSSOVERS):
            raise ValueError(f"crossovers must be selected from {VALID_CROSSOVERS}")
        if len(set(self.crossovers)) != len(self.crossovers):
            raise ValueError("crossovers must be unique")

    def as_dict(self) -> dict:
        return {**self.search.as_dict(), "crossovers": list(self.crossovers)}


@dataclass(frozen=True)
class AblationConfig:
    search: SearchConfig
    focus: float = 0.85

    def __post_init__(self) -> None:
        if not 0 < self.focus < 1:
            raise ValueError("focus must be between 0 and 1")


@dataclass(frozen=True)
class MultiobjectiveConfig:
    """Settings for one Platypus NSGA-II Pareto search."""

    population_size: int = 100
    max_evaluations: int = 25_000
    batch_size: int = 1024
    seed: int = 1337

    def __post_init__(self) -> None:
        if min(self.population_size, self.max_evaluations, self.batch_size) < 1:
            raise ValueError("multiobjective search sizes must be positive")
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.max_evaluations < self.population_size:
            raise ValueError("max_evaluations must cover the initial population")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentContext:
    config: SearchConfig | MultiobjectiveConfig
    device: torch.device
    model: SimpleMLP
    model_kwargs: dict
    original_weights: torch.Tensor
    val_loader: object
    test_loader: object
    dense_val_accuracy: float
    dense_accuracy: float

    @classmethod
    def load(cls, config: SearchConfig | MultiobjectiveConfig) -> "ExperimentContext":
        device = get_device()
        model_kwargs = {"num_classes": 10, "hidden_sizes": DEFAULT_HIDDEN_SIZES}
        model = load_pretrained_model(
            checkpoint_path(CHECKPOINT_NAME), device, **model_kwargs
        )
        original_weights = flatten_params(model).to(device)
        _, val_loader, test_loader = get_dataloaders(
            batch_size=config.batch_size, seed=config.seed
        )
        dense_val_accuracy = eval_weights(
            SimpleMLP, model_kwargs, original_weights, device, val_loader
        )["accuracy"]
        dense_accuracy = eval_weights(
            SimpleMLP, model_kwargs, original_weights, device, test_loader
        )["accuracy"]
        return cls(
            config=config,
            device=device,
            model=model,
            model_kwargs=model_kwargs,
            original_weights=original_weights,
            val_loader=val_loader,
            test_loader=test_loader,
            dense_val_accuracy=dense_val_accuracy,
            dense_accuracy=dense_accuracy,
        )
