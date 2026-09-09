"""Experiment Framework Package for Influence Maximization.

Exposes experiment runners, configuration containers, result models,
and experiment utilities.
"""

from .exceptions import ExperimentError, ExperimentValidationError
from .experiment_configs import (
    DEFAULT_EXPERIMENT_A_K_VALUES,
    DEFAULT_EXPERIMENT_A_PROBABILITY,
    DEFAULT_EXPERIMENT_B_K,
    DEFAULT_EXPERIMENT_B_P_VALUES,
    DEFAULT_EXPERIMENT_C_SIZES,
    DEFAULT_EXPERIMENT_D_SIMULATION_COUNTS,
    DEFAULT_NUM_SIMULATIONS,
    ExperimentConfig,
)
from .experiment_models import ExperimentResult
from .experiment_runner import (
    EXPERIMENT_DF_COLUMNS,
    run_all_experiments,
    run_convergence_experiment,
    run_probability_experiment,
    run_scalability_experiment,
    run_seed_budget_experiment,
)
from .utilities import default_graph_sampler, generate_experiment_id, get_node_count
from .validators import validate_experiment_parameters, validate_graph

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
