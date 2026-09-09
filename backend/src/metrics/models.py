"""Data models for the Metrics & Analytics Engine.

Defines the MetricsRecord dataclass and dictionary serialization utilities
for storing algorithm performance evaluations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class MetricsRecord:
    """Represents a unified performance evaluation record for an algorithm run.

    Attributes:
        algorithm_name: Identifier for the selection/propagation algorithm.
        seed_count: Number of initial seed nodes evaluated.
        probability: Edge propagation probability used during simulation.
        mean_influence: Mean number of influenced nodes observed.
        min_influence: Minimum number of influenced nodes observed.
        max_influence: Maximum number of influenced nodes observed.
        std_influence: Sample standard deviation of the influence distribution.
        influence_pct: Percentage of total network nodes influenced (0-100%).
        runtime_sec: Execution runtime measured in seconds (millisecond precision).
        average_reach: Expected influence reach (equivalent to mean_influence).
        spread_efficiency: Average influenced nodes per seed node.
        node_count: Total number of nodes present in the graph.
        timestamp: ISO 8601 formatted string indicating calculation time.
    """

    algorithm_name: str
    seed_count: int
    probability: float

    mean_influence: float
    min_influence: int
    max_influence: int
    std_influence: float

    influence_pct: float
    runtime_sec: float
    average_reach: float
    spread_efficiency: float

    node_count: int
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Convert this MetricsRecord instance into a dictionary."""
        return asdict(self)


def metrics_to_dict(record: MetricsRecord) -> dict[str, Any]:
    """Convert a MetricsRecord to a standard Python dictionary.

    Provides a standalone functional serialization helper for CSV export,
    experiment storage layers, and dashboard tables.

    Args:
        record: MetricsRecord instance to serialize.

    Returns:
        Dictionary containing all metrics record attributes.
    """
    if not isinstance(record, MetricsRecord):
        raise TypeError(f"Expected MetricsRecord instance, got {type(record).__name__}")
    return record.to_dict()
