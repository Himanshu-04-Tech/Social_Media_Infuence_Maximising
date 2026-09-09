"""Validation layer for Monte Carlo Simulation Engine inputs.

Provides validation routines to verify all inputs prior to running simulations,
ensuring failure occurs early with informative exception messages.
"""

from __future__ import annotations

from typing import Any, Callable, Collection

from .exceptions import MonteCarloValidationError


def validate_monte_carlo_inputs(
    graph: Any,
    seeds: Collection[Any],
    probability: float,
    num_simulations: int,
    simulate_fn: Callable[..., int] | None = None,
) -> None:
    """Validate all input arguments for the Monte Carlo simulation engine.

    Args:
        graph: NetworkX graph object (or compatible graph representation). Must not be None.
        seeds: Collection of initial seed node identifiers. Must not be empty.
        probability: Edge propagation probability. Must satisfy 0 < probability <= 1.
        num_simulations: Number of Monte Carlo iterations. Must be an integer > 0.
        simulate_fn: Optional dependency-injected simulation function. Must be callable if provided.

    Raises:
        MonteCarloValidationError: If any input fails validation criteria.
    """
    if graph is None:
        raise MonteCarloValidationError("Graph must not be None.")

    if seeds is None:
        raise MonteCarloValidationError("Seeds must not be None.")

    try:
        seeds_len = len(seeds)
    except TypeError as err:
        raise MonteCarloValidationError(
            f"Seeds must be a sized collection (e.g., list, set, tuple). Got: {type(seeds).__name__}"
        ) from err

    if seeds_len == 0:
        raise MonteCarloValidationError("Seeds collection must not be empty. At least one seed node is required.")

    if not isinstance(probability, (int, float)) or isinstance(probability, bool):
        raise MonteCarloValidationError(
            f"Probability must be a numeric value in (0, 1]. Got: {type(probability).__name__}"
        )

    if not (0 < probability <= 1.0):
        raise MonteCarloValidationError(
            f"Probability must satisfy 0 < probability <= 1. Got: {probability}"
        )

    if not isinstance(num_simulations, int) or isinstance(num_simulations, bool):
        raise MonteCarloValidationError(
            f"num_simulations must be an integer. Got: {type(num_simulations).__name__}"
        )

    if num_simulations <= 0:
        raise MonteCarloValidationError(
            f"num_simulations must be greater than 0. Got: {num_simulations}"
        )

    if simulate_fn is not None and not callable(simulate_fn):
        raise MonteCarloValidationError(
            f"simulate_fn must be a callable simulation function. Got: {type(simulate_fn).__name__}"
        )
