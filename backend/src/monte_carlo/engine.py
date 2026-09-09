"""Monte Carlo Simulation Engine for Influence Maximization.

Executes repeated independent simulation runs of the independent cascade / influence
propagation process (Role 1 simulate_influence), aggregating results to compute expected
spread metrics with sample statistics.
"""

from __future__ import annotations

import logging
import random
import statistics
import time
from typing import Any, Callable, Collection

from .exceptions import SimulationExecutionError
from .models import MonteCarloResult
from .validators import validate_monte_carlo_inputs

logger = logging.getLogger(__name__)


def run_monte_carlo(
    graph: Any,
    seeds: Collection[Any],
    probability: float,
    num_simulations: int,
    simulate_fn: Callable[[Any, Collection[Any], float], int],
    random_seed: int | None = None,
    custom_logger: logging.Logger | None = None,
) -> MonteCarloResult:
    """Execute Monte Carlo simulation across multiple independent runs.

    Repeatedly evaluates the influence spread using the provided simulate_fn,
    calculating statistical metrics including mean, minimum, maximum, and sample
    standard deviation of the influenced node counts.

    Args:
        graph: NetworkX graph or compatible network representation.
        seeds: Collection of initial seed node IDs.
        probability: Edge propagation probability (0 < probability <= 1).
        num_simulations: Total number of iterations to simulate (> 0).
        simulate_fn: Dependency-injected simulation function (from Role 1) with signature:
                     simulate_fn(graph, seeds, probability) -> int
        random_seed: Optional integer seed for deterministic pseudo-random reproducibility.
        custom_logger: Optional logger instance; defaults to module logger.

    Returns:
        MonteCarloResult containing mean, min, max, std deviation, and raw run results.

    Raises:
        MonteCarloValidationError: If any input fails validation checks.
        SimulationExecutionError: If simulate_fn raises ValueError, TypeError, or RuntimeError.
    """
    log = custom_logger or logger

    # 1. Validate inputs prior to simulation execution
    validate_monte_carlo_inputs(
        graph=graph,
        seeds=seeds,
        probability=probability,
        num_simulations=num_simulations,
        simulate_fn=simulate_fn,
    )

    seed_count = len(seeds)

    # 2. Configure deterministic seed if provided
    if random_seed is not None:
        random.seed(random_seed)

    # 3. Structured log: Simulation Started
    log.info(
        "Simulation Started | num_simulations=%d, seed_count=%d, probability=%.4f, random_seed=%s",
        num_simulations,
        seed_count,
        probability,
        random_seed,
        extra={
            "event": "Simulation Started",
            "num_simulations": num_simulations,
            "seed_count": seed_count,
            "probability": probability,
            "random_seed": random_seed,
        },
    )

    start_time = time.perf_counter()
    raw_results: list[int] = []

    # 4. Monte Carlo iteration loop
    for iteration in range(num_simulations):
        try:
            influenced = simulate_fn(graph, seeds, probability)
        except (ValueError, TypeError, RuntimeError) as err:
            log.error(
                "Role 1 simulation execution failure at run %d/%d: %s",
                iteration + 1,
                num_simulations,
                err,
            )
            raise SimulationExecutionError(
                f"Underlying simulation failed during iteration {iteration + 1}: {err}"
            ) from err

        raw_results.append(int(influenced))

    elapsed_time = time.perf_counter() - start_time

    # 5. Compute statistics using Python's statistics library
    mean_influence = float(statistics.mean(raw_results))
    min_influence = min(raw_results)
    max_influence = max(raw_results)
    # Sample standard deviation requires at least 2 data points; return 0.0 for N=1
    std_influence = (
        float(statistics.stdev(raw_results)) if len(raw_results) > 1 else 0.0
    )

    # 6. Structured log: Simulation Finished
    log.info(
        "Simulation Finished | Execution Time=%.4fs, Mean Influence=%.2f, Std Influence=%.2f",
        elapsed_time,
        mean_influence,
        std_influence,
        extra={
            "event": "Simulation Finished",
            "execution_time": elapsed_time,
            "mean_influence": mean_influence,
            "std_influence": std_influence,
            "min_influence": min_influence,
            "max_influence": max_influence,
        },
    )

    return MonteCarloResult(
        mean_influence=mean_influence,
        min_influence=min_influence,
        max_influence=max_influence,
        std_influence=std_influence,
        raw_results=raw_results,
        num_simulations=num_simulations,
        probability=probability,
        seed_count=seed_count,
    )
