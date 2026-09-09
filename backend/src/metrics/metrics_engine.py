"""Core calculation engine for influence metrics and performance indicators.

Computes influence percentage, spread efficiency, average reach, and runtime statistics
from Monte Carlo simulation results.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from .models import MetricsRecord
from .validators import validate_metrics_inputs

logger = logging.getLogger(__name__)


def calculate_metrics(
    monte_carlo_result: Any,
    algorithm_name: str,
    node_count: int,
    runtime_sec: float,
    timestamp: str | None = None,
    custom_logger: logging.Logger | None = None,
) -> MetricsRecord:
    """Calculate evaluation metrics from a Monte Carlo simulation result.

    Args:
        monte_carlo_result: Output from Module 1 run_monte_carlo() containing simulation statistics.
        algorithm_name: Name of the algorithm (e.g., 'random', 'degree', 'greedy', 'bfs', 'ic').
        node_count: Total number of nodes in the social network graph.
        runtime_sec: Total execution duration in seconds.
        timestamp: Optional ISO 8601 timestamp string; defaults to current UTC time.
        custom_logger: Optional logger instance; defaults to module logger.

    Returns:
        Populated MetricsRecord dataclass instance.

    Raises:
        MetricsValidationError: If any input fails validation checks.
    """
    log = custom_logger or logger

    # 1. Input validation
    validate_metrics_inputs(
        monte_carlo_result=monte_carlo_result,
        algorithm_name=algorithm_name,
        node_count=node_count,
        runtime_sec=runtime_sec,
    )

    # 2. Structured log: Calculation Started
    log.info(
        "Metrics Calculation Started | Algorithm Name=%s",
        algorithm_name,
        extra={
            "event": "Metrics Calculation Started",
            "algorithm_name": algorithm_name,
        },
    )

    mean_influence = float(monte_carlo_result.mean_influence)
    seed_count = int(monte_carlo_result.seed_count)
    probability = float(getattr(monte_carlo_result, "probability", 0.0))
    min_influence = int(getattr(monte_carlo_result, "min_influence", 0))
    max_influence = int(getattr(monte_carlo_result, "max_influence", 0))
    std_influence = float(getattr(monte_carlo_result, "std_influence", 0.0))

    # 3. Compute metrics
    # Influence percentage: (mean_influence / node_count) * 100
    influence_pct = (mean_influence / node_count) * 100.0

    # Average reach: identical to mean influence
    average_reach = mean_influence

    # Spread efficiency: mean_influence / seed_count
    spread_efficiency = mean_influence / seed_count

    # Runtime with millisecond precision
    runtime_recorded = round(float(runtime_sec), 3)

    record_timestamp = (
        timestamp
        if timestamp is not None
        else datetime.now(timezone.utc).isoformat()
    )

    # 4. Structured log: Metric Values & Completion
    log.info(
        "Algorithm Name=%s | Mean Influence=%.2f | Influence Percentage=%.2f%% | "
        "Spread Efficiency=%.2f | Runtime=%.3fs",
        algorithm_name,
        mean_influence,
        influence_pct,
        spread_efficiency,
        runtime_recorded,
        extra={
            "algorithm_name": algorithm_name,
            "mean_influence": mean_influence,
            "influence_percentage": influence_pct,
            "spread_efficiency": spread_efficiency,
            "runtime": runtime_recorded,
        },
    )

    log.info(
        "Metrics Calculation Completed | Algorithm Name=%s",
        algorithm_name,
        extra={
            "event": "Metrics Calculation Completed",
            "algorithm_name": algorithm_name,
        },
    )

    return MetricsRecord(
        algorithm_name=algorithm_name,
        seed_count=seed_count,
        probability=probability,
        mean_influence=mean_influence,
        min_influence=min_influence,
        max_influence=max_influence,
        std_influence=std_influence,
        influence_pct=influence_pct,
        runtime_sec=runtime_recorded,
        average_reach=average_reach,
        spread_efficiency=spread_efficiency,
        node_count=node_count,
        timestamp=record_timestamp,
    )
