"""Data models for the Monte Carlo Simulation Engine.

Defines the structured output dataclass representing statistical results
and metadata from a completed Monte Carlo simulation run.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MonteCarloResult:
    """Represents the statistical outcome and metadata of a Monte Carlo simulation.

    Attributes:
        mean_influence: Expected (mean) number of influenced nodes across simulations.
        min_influence: Minimum number of influenced nodes observed in any single run.
        max_influence: Maximum number of influenced nodes observed in any single run.
        std_influence: Sample standard deviation of influenced nodes across runs.
        raw_results: Ordered list of influence counts from each simulation run.
        num_simulations: Total number of Monte Carlo runs executed.
        probability: Propagation probability applied to edges during the simulation.
        seed_count: Size of the initial seed node set.
    """

    mean_influence: float
    min_influence: int
    max_influence: int
    std_influence: float
    raw_results: list[int] = field(default_factory=list)
    num_simulations: int = 0
    probability: float = 0.0
    seed_count: int = 0
