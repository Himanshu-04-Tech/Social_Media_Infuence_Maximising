"""Configuration data models and default benchmarks for experiments.

Provides strongly typed configuration containers and canonical parameter sets
as defined in the SRS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# Canonical parameter sets specified by SRS
DEFAULT_EXPERIMENT_A_K_VALUES: list[int] = [5, 10, 20, 50]
DEFAULT_EXPERIMENT_A_PROBABILITY: float = 0.1

DEFAULT_EXPERIMENT_B_P_VALUES: list[float] = [0.05, 0.10, 0.20, 0.30]
DEFAULT_EXPERIMENT_B_K: int = 10

DEFAULT_EXPERIMENT_C_SIZES: list[int] = [500, 1000, 2000, 4000]

DEFAULT_EXPERIMENT_D_SIMULATION_COUNTS: list[int] = [10, 50, 100, 500]

DEFAULT_NUM_SIMULATIONS: int = 50


@dataclass
class ExperimentConfig:
    """Configuration parameters for a single experiment run.

    Attributes:
        experiment_name: Identifier or title for the experiment suite.
        k: Seed budget (number of seeds to select).
        probability: Edge propagation probability.
        num_simulations: Number of Monte Carlo iterations per configuration.
        node_count: Total number of nodes in the graph being evaluated.
        random_seed: Optional integer for deterministic reproducibility.
    """

    experiment_name: str
    k: int
    probability: float
    num_simulations: int
    node_count: int
    random_seed: int | None = None
