"""Validation layer for the Metrics & Analytics Engine.

Enforces parameter constraints prior to computing metrics records.
"""

from __future__ import annotations

from typing import Any

from .exceptions import MetricsValidationError


def validate_metrics_inputs(
    monte_carlo_result: Any,
    algorithm_name: str,
    node_count: int,
    runtime_sec: float,
) -> None:
    """Validate input parameters for metrics calculation.

    Args:
        monte_carlo_result: MonteCarloResult instance or compatible object containing simulation stats.
        algorithm_name: Name of the evaluated algorithm (must be a non-empty string).
        node_count: Total number of nodes in the graph (must be > 0).
        runtime_sec: Execution runtime in seconds (must be >= 0).

    Raises:
        MetricsValidationError: If any parameter violates domain constraints.
    """
    if monte_carlo_result is None:
        raise MetricsValidationError("monte_carlo_result must not be None.")

    if not isinstance(algorithm_name, str) or not algorithm_name.strip():
        raise MetricsValidationError("algorithm_name must be a non-empty string.")

    if not isinstance(node_count, int) or isinstance(node_count, bool):
        raise MetricsValidationError(
            f"node_count must be an integer, got {type(node_count).__name__}."
        )

    if node_count <= 0:
        raise MetricsValidationError(f"node_count must be greater than 0, got {node_count}.")

    if not isinstance(runtime_sec, (int, float)) or isinstance(runtime_sec, bool):
        raise MetricsValidationError(
            f"runtime_sec must be a numeric value, got {type(runtime_sec).__name__}."
        )

    if runtime_sec < 0:
        raise MetricsValidationError(f"runtime_sec must be non-negative, got {runtime_sec}.")

    # Validate attributes from monte_carlo_result
    try:
        mean_influence = getattr(monte_carlo_result, "mean_influence")
        seed_count = getattr(monte_carlo_result, "seed_count")
    except AttributeError as err:
        raise MetricsValidationError(
            f"monte_carlo_result missing required attributes: {err}"
        ) from err

    if not isinstance(mean_influence, (int, float)) or isinstance(mean_influence, bool):
        raise MetricsValidationError(
            f"mean_influence must be numeric, got {type(mean_influence).__name__}."
        )

    if mean_influence < 0:
        raise MetricsValidationError(
            f"mean_influence must be non-negative, got {mean_influence}."
        )

    if not isinstance(seed_count, int) or isinstance(seed_count, bool):
        raise MetricsValidationError(
            f"seed_count must be an integer, got {type(seed_count).__name__}."
        )

    if seed_count <= 0:
        raise MetricsValidationError(
            f"seed_count must be greater than 0, got {seed_count}."
        )
