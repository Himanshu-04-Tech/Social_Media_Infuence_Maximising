"""Validation layer for the Experiment Framework.

Enforces constraints on experimental configurations, graph objects, and injected algorithms.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .exceptions import ExperimentValidationError


_MISSING = object()


def validate_experiment_parameters(
    graph: Any | None = None,
    k: int | None = None,
    probability: float | None = None,
    num_simulations: int | None = None,
    algorithms: Mapping[str, Callable[..., Any]] | None = None,
    simulate_fn: Any = _MISSING,
) -> None:
    """Validate common experiment parameters against SRS requirements.

    Args:
        graph: Graph instance (checked if passed).
        k: Seed budget integer (> 0).
        probability: Edge propagation probability (0 < p <= 1).
        num_simulations: Monte Carlo iterations (> 0).
        algorithms: Mapping of algorithm names to their selection functions.
        simulate_fn: Influence simulation function (simulate_influence).

    Raises:
        ExperimentValidationError: If any parameter violates constraints.
    """
    if graph is not None and graph is ...:  # explicit check if checking graph
        pass

    if k is not None:
        if not isinstance(k, int) or isinstance(k, bool):
            raise ExperimentValidationError(f"k must be an integer, got {type(k).__name__}.")
        if k <= 0:
            raise ExperimentValidationError(f"k must be greater than 0, got {k}.")

    if probability is not None:
        if not isinstance(probability, (int, float)) or isinstance(probability, bool):
            raise ExperimentValidationError(
                f"probability must be numeric, got {type(probability).__name__}."
            )
        if not (0 < probability <= 1.0):
            raise ExperimentValidationError(
                f"probability must satisfy 0 < probability <= 1, got {probability}."
            )

    if num_simulations is not None:
        if not isinstance(num_simulations, int) or isinstance(num_simulations, bool):
            raise ExperimentValidationError(
                f"num_simulations must be an integer, got {type(num_simulations).__name__}."
            )
        if num_simulations <= 0:
            raise ExperimentValidationError(
                f"num_simulations must be greater than 0, got {num_simulations}."
            )

    if algorithms is not None:
        if not isinstance(algorithms, Mapping) or not algorithms:
            raise ExperimentValidationError(
                "algorithms must be a non-empty mapping of algorithm names to callables."
            )
        for name, fn in algorithms.items():
            if not callable(fn):
                raise ExperimentValidationError(
                    f"Algorithm '{name}' is not callable. Got: {type(fn).__name__}."
                )

    if simulate_fn is not _MISSING:
        if simulate_fn is None or not callable(simulate_fn):
            raise ExperimentValidationError(
                f"simulate_fn must be a callable simulation function. Got: {type(simulate_fn).__name__}."
            )


def validate_graph(graph: Any) -> None:
    """Validate that the graph object is not None.

    Args:
        graph: Target graph.

    Raises:
        ExperimentValidationError: If graph is None.
    """
    if graph is None:
        raise ExperimentValidationError("graph must not be None.")
