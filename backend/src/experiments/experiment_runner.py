"""Experiment Runner for Social Network Influence Maximization.

Orchestrates controlled experimental studies (Seed Budget, Probability,
Scalability, and Convergence) by combining Role 1 algorithms with
Module 1 (Monte Carlo Engine) and Module 2 (Metrics Engine).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from metrics import calculate_metrics
from monte_carlo import run_monte_carlo

from .exceptions import ExperimentValidationError
from .experiment_configs import (
    DEFAULT_EXPERIMENT_A_K_VALUES,
    DEFAULT_EXPERIMENT_A_PROBABILITY,
    DEFAULT_EXPERIMENT_B_K,
    DEFAULT_EXPERIMENT_B_P_VALUES,
    DEFAULT_EXPERIMENT_C_SIZES,
    DEFAULT_EXPERIMENT_D_SIMULATION_COUNTS,
    DEFAULT_NUM_SIMULATIONS,
)
from .experiment_models import ExperimentResult
from .utilities import default_graph_sampler, generate_experiment_id, get_node_count
from .validators import validate_experiment_parameters, validate_graph

logger = logging.getLogger(__name__)

# Standard columns required in all experiment DataFrames by SRS
EXPERIMENT_DF_COLUMNS: list[str] = [
    "experiment_id",
    "algorithm_name",
    "k",
    "probability",
    "num_simulations",
    "mean_influence",
    "std_influence",
    "influence_pct",
    "runtime_sec",
    "spread_efficiency",
]


def _build_algorithms_dict(
    algorithms: Mapping[str, Callable[..., Any]] | None = None,
    random_selection: Callable[..., Any] | None = None,
    degree_selection: Callable[..., Any] | None = None,
    greedy_selection: Callable[..., Any] | None = None,
    bfs_selection: Callable[..., Any] | None = None,
    ic_selection: Callable[..., Any] | None = None,
) -> dict[str, Callable[..., Any]]:
    """Resolve and bundle seed selection algorithms into a standardized dictionary."""
    resolved: dict[str, Callable[..., Any]] = {}
    if algorithms:
        resolved.update(algorithms)

    if random_selection is not None:
        resolved["random"] = random_selection
    if degree_selection is not None:
        resolved["degree"] = degree_selection
    if greedy_selection is not None:
        resolved["greedy"] = greedy_selection
    if bfs_selection is not None:
        resolved["bfs"] = bfs_selection
    if ic_selection is not None:
        resolved["ic"] = ic_selection

    if not resolved:
        raise ExperimentValidationError(
            "No seed selection algorithms provided. Pass algorithms mapping or individual selection callables."
        )

    return resolved


# ---------------------------------------------------------------------------
# EXPERIMENT A: Seed Budget Comparison
# ---------------------------------------------------------------------------
def run_seed_budget_experiment(
    graph: Any,
    algorithms: Mapping[str, Callable[..., Any]],
    simulate_fn: Callable[..., int],
    k_values: Sequence[int] = DEFAULT_EXPERIMENT_A_K_VALUES,
    probability: float = DEFAULT_EXPERIMENT_A_PROBABILITY,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    random_seed: int | None = None,
    custom_logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Execute Experiment A: Seed Budget Comparison across multiple k values.

    Fixed: probability = 0.1
    Varied: k in [5, 10, 20, 50]

    Args:
        graph: Target social network graph.
        algorithms: Mapping of algorithm names to seed selection functions.
        simulate_fn: Simulation function (Role 1 simulate_influence).
        k_values: Sequence of seed budget integers.
        probability: Edge cascade probability.
        num_simulations: Monte Carlo iterations per test point.
        random_seed: Optional reproducibility seed.
        custom_logger: Optional logger instance.

    Returns:
        pd.DataFrame containing comparison metrics for all algorithms across k values.
    """
    log = custom_logger or logger
    validate_graph(graph)
    validate_experiment_parameters(
        graph=graph,
        probability=probability,
        num_simulations=num_simulations,
        algorithms=algorithms,
        simulate_fn=simulate_fn,
    )
    for k in k_values:
        validate_experiment_parameters(k=k)

    node_count = get_node_count(graph)

    log.info(
        "Experiment Started | Experiment Type=Seed Budget Comparison (Exp A) | "
        "k_values=%s | probability=%.2f",
        list(k_values),
        probability,
        extra={"event": "Experiment Started", "experiment_type": "Seed Budget Comparison"},
    )

    exp_start_time = time.perf_counter()
    rows: list[dict[str, Any]] = []

    for k in k_values:
        exp_id = generate_experiment_id("expA", k=k, probability=probability)

        for algo_name, algo_fn in algorithms.items():
            log.info(
                "Current Algorithm=%s | Current Configuration=k=%d, p=%.2f, sims=%d",
                algo_name,
                k,
                probability,
                num_simulations,
                extra={"current_algorithm": algo_name, "k": k, "probability": probability},
            )

            # Measure total algorithm runtime (selection + simulation)
            algo_start = time.perf_counter()
            seeds = algo_fn(graph, k)
            mc_result = run_monte_carlo(
                graph=graph,
                seeds=seeds,
                probability=probability,
                num_simulations=num_simulations,
                simulate_fn=simulate_fn,
                random_seed=random_seed,
                custom_logger=log,
            )
            algo_runtime = time.perf_counter() - algo_start

            record = calculate_metrics(
                monte_carlo_result=mc_result,
                algorithm_name=algo_name,
                node_count=node_count,
                runtime_sec=algo_runtime,
                custom_logger=log,
            )

            row_data = {
                "experiment_id": exp_id,
                "algorithm_name": algo_name,
                "k": k,
                "probability": probability,
                "num_simulations": num_simulations,
                "mean_influence": record.mean_influence,
                "std_influence": record.std_influence,
                "influence_pct": record.influence_pct,
                "runtime_sec": record.runtime_sec,
                "spread_efficiency": record.spread_efficiency,
            }
            rows.append(row_data)

    total_time = time.perf_counter() - exp_start_time
    log.info(
        "Experiment Completed | Experiment Type=Seed Budget Comparison | Runtime=%.4fs",
        total_time,
        extra={"event": "Experiment Completed", "runtime": total_time},
    )

    return pd.DataFrame(rows, columns=EXPERIMENT_DF_COLUMNS)


# ---------------------------------------------------------------------------
# EXPERIMENT B: Propagation Probability Comparison
# ---------------------------------------------------------------------------
def run_probability_experiment(
    graph: Any,
    algorithms: Mapping[str, Callable[..., Any]],
    simulate_fn: Callable[..., int],
    p_values: Sequence[float] = DEFAULT_EXPERIMENT_B_P_VALUES,
    k: int = DEFAULT_EXPERIMENT_B_K,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    random_seed: int | None = None,
    custom_logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Execute Experiment B: Propagation Probability Comparison.

    Fixed: k = 10
    Varied: probability in [0.05, 0.10, 0.20, 0.30]

    Args:
        graph: Target social network graph.
        algorithms: Mapping of algorithm names to seed selection functions.
        simulate_fn: Simulation function (Role 1 simulate_influence).
        p_values: Sequence of propagation probabilities to evaluate.
        k: Fixed seed budget.
        num_simulations: Monte Carlo iterations per test point.
        random_seed: Optional reproducibility seed.
        custom_logger: Optional logger instance.

    Returns:
        pd.DataFrame containing comparison metrics across propagation probabilities.
    """
    log = custom_logger or logger
    validate_graph(graph)
    validate_experiment_parameters(
        graph=graph,
        k=k,
        num_simulations=num_simulations,
        algorithms=algorithms,
        simulate_fn=simulate_fn,
    )
    for p in p_values:
        validate_experiment_parameters(probability=p)

    node_count = get_node_count(graph)

    log.info(
        "Experiment Started | Experiment Type=Propagation Probability Comparison (Exp B) | "
        "k=%d | p_values=%s",
        k,
        list(p_values),
        extra={"event": "Experiment Started", "experiment_type": "Propagation Probability Comparison"},
    )

    exp_start_time = time.perf_counter()
    rows: list[dict[str, Any]] = []

    for p in p_values:
        exp_id = generate_experiment_id("expB", probability=p)

        for algo_name, algo_fn in algorithms.items():
            log.info(
                "Current Algorithm=%s | Current Configuration=k=%d, p=%.2f, sims=%d",
                algo_name,
                k,
                p,
                num_simulations,
                extra={"current_algorithm": algo_name, "k": k, "probability": p},
            )

            algo_start = time.perf_counter()
            seeds = algo_fn(graph, k)
            mc_result = run_monte_carlo(
                graph=graph,
                seeds=seeds,
                probability=p,
                num_simulations=num_simulations,
                simulate_fn=simulate_fn,
                random_seed=random_seed,
                custom_logger=log,
            )
            algo_runtime = time.perf_counter() - algo_start

            record = calculate_metrics(
                monte_carlo_result=mc_result,
                algorithm_name=algo_name,
                node_count=node_count,
                runtime_sec=algo_runtime,
                custom_logger=log,
            )

            row_data = {
                "experiment_id": exp_id,
                "algorithm_name": algo_name,
                "k": k,
                "probability": p,
                "num_simulations": num_simulations,
                "mean_influence": record.mean_influence,
                "std_influence": record.std_influence,
                "influence_pct": record.influence_pct,
                "runtime_sec": record.runtime_sec,
                "spread_efficiency": record.spread_efficiency,
            }
            rows.append(row_data)

    total_time = time.perf_counter() - exp_start_time
    log.info(
        "Experiment Completed | Experiment Type=Propagation Probability Comparison | Runtime=%.4fs",
        total_time,
        extra={"event": "Experiment Completed", "runtime": total_time},
    )

    return pd.DataFrame(rows, columns=EXPERIMENT_DF_COLUMNS)


# ---------------------------------------------------------------------------
# EXPERIMENT C: Network Size Scalability
# ---------------------------------------------------------------------------
def run_scalability_experiment(
    algorithms: Mapping[str, Callable[..., Any]],
    simulate_fn: Callable[..., int],
    graph_sampler: Callable[[int], Any] | None = None,
    sizes: Sequence[int] = DEFAULT_EXPERIMENT_C_SIZES,
    k: int = DEFAULT_EXPERIMENT_B_K,
    probability: float = DEFAULT_EXPERIMENT_A_PROBABILITY,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    random_seed: int | None = None,
    custom_logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Execute Experiment C: Network Size Scalability.

    Varied: network sizes in [500, 1000, 2000, 4000] via graph_sampler(size)
    Evaluates: execution runtime and influence percentage scalability across sizes.

    Args:
        algorithms: Mapping of algorithm names to seed selection functions.
        simulate_fn: Simulation function (Role 1 simulate_influence).
        graph_sampler: Dependency function accepting size: int and returning graph.
        sizes: Sequence of network sizes to evaluate.
        k: Seed count per size evaluation.
        probability: Edge cascade probability.
        num_simulations: Monte Carlo iterations per test point.
        random_seed: Optional reproducibility seed.
        custom_logger: Optional logger instance.

    Returns:
        pd.DataFrame containing scalability metrics across network sizes.
    """
    log = custom_logger or logger
    sampler = graph_sampler or default_graph_sampler

    validate_experiment_parameters(
        k=k,
        probability=probability,
        num_simulations=num_simulations,
        algorithms=algorithms,
        simulate_fn=simulate_fn,
    )
    for s in sizes:
        if s <= 0:
            raise ExperimentValidationError(f"Network size must be > 0, got {s}")

    log.info(
        "Experiment Started | Experiment Type=Network Size Scalability (Exp C) | sizes=%s",
        list(sizes),
        extra={"event": "Experiment Started", "experiment_type": "Network Size Scalability"},
    )

    exp_start_time = time.perf_counter()
    rows: list[dict[str, Any]] = []

    for size in sizes:
        exp_id = generate_experiment_id("expC", node_count=size)
        sampled_graph = sampler(size)
        actual_node_count = get_node_count(sampled_graph)

        for algo_name, algo_fn in algorithms.items():
            log.info(
                "Current Algorithm=%s | Current Configuration=size=%d, k=%d, p=%.2f",
                algo_name,
                size,
                k,
                probability,
                extra={"current_algorithm": algo_name, "size": size, "k": k},
            )

            algo_start = time.perf_counter()
            seeds = algo_fn(sampled_graph, k)
            mc_result = run_monte_carlo(
                graph=sampled_graph,
                seeds=seeds,
                probability=probability,
                num_simulations=num_simulations,
                simulate_fn=simulate_fn,
                random_seed=random_seed,
                custom_logger=log,
            )
            algo_runtime = time.perf_counter() - algo_start

            record = calculate_metrics(
                monte_carlo_result=mc_result,
                algorithm_name=algo_name,
                node_count=actual_node_count,
                runtime_sec=algo_runtime,
                custom_logger=log,
            )

            row_data = {
                "experiment_id": exp_id,
                "algorithm_name": algo_name,
                "k": k,
                "probability": probability,
                "num_simulations": num_simulations,
                "mean_influence": record.mean_influence,
                "std_influence": record.std_influence,
                "influence_pct": record.influence_pct,
                "runtime_sec": record.runtime_sec,
                "spread_efficiency": record.spread_efficiency,
            }
            rows.append(row_data)

    total_time = time.perf_counter() - exp_start_time
    log.info(
        "Experiment Completed | Experiment Type=Network Size Scalability | Runtime=%.4fs",
        total_time,
        extra={"event": "Experiment Completed", "runtime": total_time},
    )

    return pd.DataFrame(rows, columns=EXPERIMENT_DF_COLUMNS)


# ---------------------------------------------------------------------------
# EXPERIMENT D: Monte Carlo Convergence
# ---------------------------------------------------------------------------
def run_convergence_experiment(
    graph: Any,
    seeds: Sequence[Any] | None = None,
    algorithms: Mapping[str, Callable[..., Any]] | None = None,
    simulate_fn: Callable[..., int] | None = None,
    simulation_counts: Sequence[int] = DEFAULT_EXPERIMENT_D_SIMULATION_COUNTS,
    probability: float = DEFAULT_EXPERIMENT_A_PROBABILITY,
    k: int = DEFAULT_EXPERIMENT_B_K,
    random_seed: int | None = None,
    custom_logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Execute Experiment D: Monte Carlo Convergence Analysis.

    Fixed: graph, probability, seed set
    Varied: simulation_counts in [10, 50, 100, 500]
    Computes: relative_standard_error = std_influence / mean_influence

    Args:
        graph: Target graph instance.
        seeds: Optional fixed seed set. If None, uses first algorithm in algorithms.
        algorithms: Optional algorithm mapping used to generate seeds if seeds is None.
        simulate_fn: Simulation function (Role 1 simulate_influence).
        simulation_counts: Sequence of Monte Carlo iteration counts.
        probability: Edge cascade probability.
        k: Seed budget used if generating seeds.
        random_seed: Optional reproducibility seed.
        custom_logger: Optional logger instance.

    Returns:
        pd.DataFrame containing convergence statistics including relative_standard_error.
    """
    log = custom_logger or logger
    validate_graph(graph)

    if simulate_fn is None or not callable(simulate_fn):
        raise ExperimentValidationError("simulate_fn must be a callable simulation function.")

    validate_experiment_parameters(
        probability=probability,
        k=k,
        simulate_fn=simulate_fn,
    )
    for n in simulation_counts:
        validate_experiment_parameters(num_simulations=n)

    node_count = get_node_count(graph)

    # Determine seed set to keep constant across iterations
    resolved_seeds: Sequence[Any]
    algo_name = "fixed_seed_set"
    if seeds is not None and len(seeds) > 0:
        resolved_seeds = seeds
    elif algorithms:
        algo_name, algo_fn = next(iter(algorithms.items()))
        resolved_seeds = algo_fn(graph, k)
    else:
        raise ExperimentValidationError(
            "Experiment D requires either a seed collection or an algorithm to generate seeds."
        )

    log.info(
        "Experiment Started | Experiment Type=Monte Carlo Convergence (Exp D) | "
        "simulation_counts=%s",
        list(simulation_counts),
        extra={"event": "Experiment Started", "experiment_type": "Monte Carlo Convergence"},
    )

    exp_start_time = time.perf_counter()
    rows: list[dict[str, Any]] = []

    columns_with_convergence = list(EXPERIMENT_DF_COLUMNS) + ["relative_standard_error"]

    for n_sims in simulation_counts:
        exp_id = generate_experiment_id("expD", num_simulations=n_sims)

        log.info(
            "Current Algorithm=%s | Current Configuration=num_simulations=%d, p=%.2f",
            algo_name,
            n_sims,
            probability,
            extra={"current_algorithm": algo_name, "num_simulations": n_sims},
        )

        algo_start = time.perf_counter()
        mc_result = run_monte_carlo(
            graph=graph,
            seeds=resolved_seeds,
            probability=probability,
            num_simulations=n_sims,
            simulate_fn=simulate_fn,
            random_seed=random_seed,
            custom_logger=log,
        )
        algo_runtime = time.perf_counter() - algo_start

        record = calculate_metrics(
            monte_carlo_result=mc_result,
            algorithm_name=algo_name,
            node_count=node_count,
            runtime_sec=algo_runtime,
            custom_logger=log,
        )

        # Formula: relative_standard_error = std_influence / mean_influence
        if record.mean_influence > 0:
            rel_std_err = record.std_influence / record.mean_influence
        else:
            rel_std_err = 0.0

        row_data = {
            "experiment_id": exp_id,
            "algorithm_name": algo_name,
            "k": len(resolved_seeds),
            "probability": probability,
            "num_simulations": n_sims,
            "mean_influence": record.mean_influence,
            "std_influence": record.std_influence,
            "influence_pct": record.influence_pct,
            "runtime_sec": record.runtime_sec,
            "spread_efficiency": record.spread_efficiency,
            "relative_standard_error": rel_std_err,
        }
        rows.append(row_data)

    total_time = time.perf_counter() - exp_start_time
    log.info(
        "Experiment Completed | Experiment Type=Monte Carlo Convergence | Runtime=%.4fs",
        total_time,
        extra={"event": "Experiment Completed", "runtime": total_time},
    )

    return pd.DataFrame(rows, columns=columns_with_convergence)


# ---------------------------------------------------------------------------
# MASTER RUNNER: run_all_experiments()
# ---------------------------------------------------------------------------
def run_all_experiments(
    graph: Any,
    random_selection: Callable[..., Any] | None = None,
    degree_selection: Callable[..., Any] | None = None,
    greedy_selection: Callable[..., Any] | None = None,
    bfs_selection: Callable[..., Any] | None = None,
    ic_selection: Callable[..., Any] | None = None,
    simulate_influence: Callable[..., int] | None = None,
    algorithms: Mapping[str, Callable[..., Any]] | None = None,
    graph_sampler: Callable[[int], Any] | None = None,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    random_seed: int | None = None,
    custom_logger: logging.Logger | None = None,
) -> dict[str, pd.DataFrame]:
    """Execute all four benchmark experiments and return comparative DataFrames.

    Orchestrates:
        - Experiment A: Seed Budget Comparison
        - Experiment B: Propagation Probability Comparison
        - Experiment C: Network Size Scalability
        - Experiment D: Monte Carlo Convergence

    Args:
        graph: Base network graph.
        random_selection: Random seed selection function (Role 1).
        degree_selection: Degree-based seed selection function (Role 1).
        greedy_selection: Greedy seed selection function (Role 1).
        bfs_selection: BFS-based seed selection function (Role 1).
        ic_selection: Optional IC seed selection function.
        simulate_influence: Simulation cascade function (Role 1).
        algorithms: Optional dictionary of algorithm callables.
        graph_sampler: Graph generation function for Experiment C.
        num_simulations: Default Monte Carlo iteration count.
        random_seed: Optional seed for pseudo-random reproducibility.
        custom_logger: Optional logger instance.

    Returns:
        Dictionary mapping experiment keys to pandas DataFrames:
        {
            "experiment_a": DataFrame,
            "experiment_b": DataFrame,
            "experiment_c": DataFrame,
            "experiment_d": DataFrame
        }
    """
    log = custom_logger or logger
    validate_graph(graph)

    if simulate_influence is None or not callable(simulate_influence):
        raise ExperimentValidationError(
            "simulate_influence must be provided as a callable simulation function."
        )

    resolved_algorithms = _build_algorithms_dict(
        algorithms=algorithms,
        random_selection=random_selection,
        degree_selection=degree_selection,
        greedy_selection=greedy_selection,
        bfs_selection=bfs_selection,
        ic_selection=ic_selection,
    )

    log.info(
        "Master Runner Initiated | Algorithms=%s | num_simulations=%d",
        list(resolved_algorithms.keys()),
        num_simulations,
    )
    master_start = time.perf_counter()

    # 1. Run Experiment A: Seed Budget
    df_a = run_seed_budget_experiment(
        graph=graph,
        algorithms=resolved_algorithms,
        simulate_fn=simulate_influence,
        num_simulations=num_simulations,
        random_seed=random_seed,
        custom_logger=log,
    )

    # 2. Run Experiment B: Propagation Probability
    df_b = run_probability_experiment(
        graph=graph,
        algorithms=resolved_algorithms,
        simulate_fn=simulate_influence,
        num_simulations=num_simulations,
        random_seed=random_seed,
        custom_logger=log,
    )

    # 3. Run Experiment C: Network Scalability
    df_c = run_scalability_experiment(
        algorithms=resolved_algorithms,
        simulate_fn=simulate_influence,
        graph_sampler=graph_sampler,
        num_simulations=num_simulations,
        random_seed=random_seed,
        custom_logger=log,
    )

    # 4. Run Experiment D: Monte Carlo Convergence
    df_d = run_convergence_experiment(
        graph=graph,
        algorithms=resolved_algorithms,
        simulate_fn=simulate_influence,
        random_seed=random_seed,
        custom_logger=log,
    )

    master_elapsed = time.perf_counter() - master_start
    log.info(
        "Master Runner Completed All 4 Experiments | Total Execution Time=%.4fs",
        master_elapsed,
    )

    return {
        "experiment_a": df_a,
        "experiment_b": df_b,
        "experiment_c": df_c,
        "experiment_d": df_d,
    }
