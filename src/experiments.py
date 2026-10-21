"""src/experiments.py

Root-level forwarder for the Automated Experiment Framework.
Bridges direct imports to backend.src.experiments.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure backend/src is on sys.path
_BACKEND_SRC = Path(__file__).resolve().parent.parent / "backend" / "src"
if str(_BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(_BACKEND_SRC))

from experiments.exceptions import ExperimentError, ExperimentValidationError
from experiments.experiment_configs import (
    DEFAULT_EXPERIMENT_A_K_VALUES,
    DEFAULT_EXPERIMENT_A_PROBABILITY,
    DEFAULT_EXPERIMENT_B_K,
    DEFAULT_EXPERIMENT_B_P_VALUES,
    DEFAULT_EXPERIMENT_C_SIZES,
    DEFAULT_EXPERIMENT_D_SIMULATION_COUNTS,
    DEFAULT_NUM_SIMULATIONS,
    ExperimentConfig,
)
from experiments.experiment_models import ExperimentResult
from experiments.experiment_runner import (
    EXPERIMENT_DF_COLUMNS,
    run_all_experiments,
    run_convergence_experiment,
    run_probability_experiment,
    run_scalability_experiment,
    run_seed_budget_experiment,
)
from experiments.utilities import default_graph_sampler, generate_experiment_id, get_node_count
from experiments.validators import validate_experiment_parameters, validate_graph

__all__ = [
    "run_all_experiments",
    "run_seed_budget_experiment",
    "run_probability_experiment",
    "run_scalability_experiment",
    "run_convergence_experiment",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentError",
    "ExperimentValidationError",
    "generate_experiment_id",
    "get_node_count",
    "default_graph_sampler",
    "validate_experiment_parameters",
    "validate_graph",
    "EXPERIMENT_DF_COLUMNS",
    "DEFAULT_EXPERIMENT_A_K_VALUES",
    "DEFAULT_EXPERIMENT_A_PROBABILITY",
    "DEFAULT_EXPERIMENT_B_K",
    "DEFAULT_EXPERIMENT_B_P_VALUES",
    "DEFAULT_EXPERIMENT_C_SIZES",
    "DEFAULT_EXPERIMENT_D_SIMULATION_COUNTS",
    "DEFAULT_NUM_SIMULATIONS",
]
