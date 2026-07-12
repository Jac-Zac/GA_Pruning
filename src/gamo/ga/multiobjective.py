"""Platypus NSGA-II search for the accuracy/sparsity Pareto front."""

from dataclasses import dataclass

import torch
from platypus import (
    NSGAII,
    Binary,
    Evaluator,
    InjectedPopulation,
    Problem,
    Solution,
    nondominated,
)

from gamo.model.model import SimpleMLP
from gamo.utils.pruning import structured_parameter_sparsities

from .structured import (
    PreparedEvaluation,
    evaluate_masks,
    num_neurons,
    prepare_batched_evaluation,
)


@dataclass(frozen=True)
class ParetoPoint:
    """One nondominated structured-pruning mask."""

    mask: torch.Tensor
    validation_accuracy: float
    parameter_sparsity: float


@dataclass(frozen=True)
class MultiobjectiveSearchResult:
    """The final nondominated set and the actual Platypus evaluation count."""

    pareto_front: list[ParetoPoint]
    evaluations: int


class StructuredPruningProblem(Problem):
    """Two-objective binary problem evaluated in population-sized torch batches."""

    def __init__(self, evaluator: PreparedEvaluation, device: torch.device):
        self.mask_evaluator = evaluator
        self.device = device
        model = evaluator[0]
        self.hidden_sizes = list(model.hidden_sizes)
        self.input_size = model.hidden_layers[0].in_features
        self.output_size = model.num_classes
        width = sum(self.hidden_sizes)
        super().__init__(1, 2)
        self.types[0] = Binary(width)
        self.directions[:] = Problem.MAXIMIZE

    def evaluate(self, solution: Solution) -> None:
        """Support direct Platypus evaluation in addition to the batched path."""
        masks = torch.tensor(
            [solution.variables[0]], dtype=torch.bool, device=self.device
        )
        accuracy, parameter_sparsity = self.evaluate_mask_batch(masks)
        solution.objectives[:] = [accuracy.item(), parameter_sparsity.item()]

    def evaluate_mask_batch(
        self, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model, batches = self.mask_evaluator
        accuracies, _ = evaluate_masks(model, batches, masks.to(self.device))
        parameter_sparsities = structured_parameter_sparsities(
            masks,
            self.hidden_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
        )
        return accuracies, parameter_sparsities


class BatchedMaskEvaluator(Evaluator):
    """Evaluate all Platypus jobs in one vectorised masked-model forward pass."""

    def evaluate_all(self, jobs, **kwargs):
        jobs = list(jobs)
        if not jobs:
            return jobs

        problems = {id(job.solution.problem) for job in jobs}
        if len(problems) != 1 or not isinstance(
            jobs[0].solution.problem, StructuredPruningProblem
        ):
            raise TypeError("BatchedMaskEvaluator requires StructuredPruningProblem")

        problem = jobs[0].solution.problem
        masks = torch.tensor(
            [job.solution.variables[0] for job in jobs],
            dtype=torch.bool,
            device=problem.device,
        )
        accuracies, parameter_sparsities = problem.evaluate_mask_batch(masks)

        for job, accuracy, parameter_sparsity in zip(
            jobs, accuracies.tolist(), parameter_sparsities.tolist(), strict=True
        ):
            solution = job.solution
            solution.objectives[:] = [accuracy, parameter_sparsity]
            solution.constraint_violation = 0.0
            solution.feasible = True
            solution.evaluated = True
        return jobs


def _injected_extremes(problem: Problem, width: int) -> InjectedPopulation:
    """Seed both ends of the trade-off while leaving other slots random."""
    solutions = []
    for kept in (False, True):
        solution = Solution(problem)
        solution.variables[0] = [kept] * width
        solutions.append(solution)
    return InjectedPopulation(solutions)


def optimize_structured_multiobjective(
    model: SimpleMLP,
    val_loader,
    device: torch.device,
    *,
    population_size: int = 100,
    max_evaluations: int = 25_000,
    evaluator: PreparedEvaluation | None = None,
) -> MultiobjectiveSearchResult:
    """Use Platypus NSGA-II to maximize validation accuracy and neuron sparsity."""
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if max_evaluations < population_size:
        raise ValueError("max_evaluations must cover the initial population")
    if evaluator is None:
        evaluator = prepare_batched_evaluation(model, val_loader, device)

    width = num_neurons(model)
    problem = StructuredPruningProblem(evaluator, device)
    algorithm = NSGAII(
        problem,
        population_size=population_size,
        generator=_injected_extremes(problem, width),
        evaluator=BatchedMaskEvaluator(),
    )
    algorithm.run(max_evaluations)

    points = []
    seen_masks: set[tuple[bool, ...]] = set()
    for solution in nondominated(algorithm.result):
        key = tuple(bool(value) for value in solution.variables[0])
        if key in seen_masks:
            continue
        seen_masks.add(key)
        points.append(
            ParetoPoint(
                mask=torch.tensor(key, dtype=torch.bool),
                validation_accuracy=float(solution.objectives[0]),
                parameter_sparsity=float(solution.objectives[1]),
            )
        )
    points.sort(key=lambda point: (point.parameter_sparsity, point.validation_accuracy))
    return MultiobjectiveSearchResult(points, algorithm.nfe)
