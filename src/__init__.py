"""
Influence Maximization Engine

Authors:
    Zaid Hussain - Graph & Algorithm Engineer
    Himanshu     - Frontend, Visualization & Integration Engineer

Exposes the exact high-level API functions requested for frontend integration:
    - random_selection(graph, k)
    - degree_selection(graph, k)
    - greedy_selection(graph, k, p=0.05, ...)
    - simulate_influence(graph, seeds, p=0.05, ...)
"""

from typing import Iterable, List, Optional, Set, Tuple, Union
import random

from src.graph import Graph
from src.propagation import (
    simulate_independent_cascade,
    estimate_influence,
    estimate_influence_stats,
)
from src.seed_selection import (
    select_seeds_random,
    select_seeds_degree,
    select_seeds_greedy,
)
from src.metrics import run_experiment, compare_algorithms
from src.main import run_algorithm_api, compare_algorithms_api


# =====================================================================
# Direct Frontend API Wrappers (Matched to Himanshu's exact specifications)
# =====================================================================

def random_selection(
    graph: Graph,
    k: int,
    rng: Optional[Union[random.Random, int]] = None,
) -> List[int]:
    """
    Selects k random seed nodes.

    Args:
        graph: The Graph instance.
        k: Number of seeds to select.
        rng: Optional random.Random instance or int seed.

    Returns:
        List[int] containing k unique seed node IDs.
    """
    return select_seeds_random(graph, k, rng=rng)


def degree_selection(
    graph: Graph,
    k: int,
) -> List[int]:
    """
    Selects k seed nodes with the highest degree in the graph.

    Args:
        graph: The Graph instance.
        k: Number of seeds to select.

    Returns:
        List[int] containing k unique seed node IDs sorted descending by degree.
    """
    return select_seeds_degree(graph, k)


def greedy_selection(
    graph: Graph,
    k: int,
    p: float = 0.05,
    simulations: int = 100,
    candidate_nodes: Optional[Iterable[int]] = None,
    rng: Optional[Union[random.Random, int]] = None,
    return_marginal_gains: bool = False,
) -> Union[List[int], Tuple[List[int], List[float]]]:
    """
    Greedy Influence Maximization selecting k seeds based on marginal influence gain.

    Args:
        graph: The Graph instance.
        k: Number of seeds to select.
        p: Propagation probability (default: 0.05).
        simulations: Number of Monte Carlo simulations R (default: 100).
        candidate_nodes: Optional candidate pool restriction (e.g. top 50 degree nodes).
        rng: Optional random.Random instance or int seed.
        return_marginal_gains: If True, returns (seeds, marginal_gains).

    Returns:
        List[int] of k seeds (or tuple of (seeds, marginal_gains) if return_marginal_gains=True).
    """
    return select_seeds_greedy(
        graph=graph,
        k=k,
        p=p,
        simulations=simulations,
        candidate_nodes=candidate_nodes,
        rng=rng,
        return_marginal_gains=return_marginal_gains,
    )


def simulate_influence(
    graph: Graph,
    seeds: Iterable[int],
    p: float = 0.05,
    rng: Optional[Union[random.Random, int]] = None,
    return_history: bool = False,
) -> Union[Set[int], Tuple[Set[int], List[Set[int]]]]:
    """
    Simulates influence diffusion under the Independent Cascade model.

    Args:
        graph: The Graph instance.
        seeds: Iterable of seed node IDs.
        p: Propagation probability per edge (default: 0.05).
        rng: Optional random.Random instance or int seed.
        return_history: If True, returns (active_nodes, history_layers) for step-by-step UI animation.

    Returns:
        Set[int] of all infected nodes (or (active_nodes, history_layers) if return_history=True).
    """
    return simulate_independent_cascade(
        graph=graph,
        seeds=seeds,
        p=p,
        rng=rng,
        return_history=return_history,
    )


__all__ = [
    "Graph",
    "random_selection",
    "degree_selection",
    "greedy_selection",
    "simulate_influence",
    "select_seeds_random",
    "select_seeds_degree",
    "select_seeds_greedy",
    "simulate_independent_cascade",
    "estimate_influence",
    "estimate_influence_stats",
    "run_experiment",
    "compare_algorithms",
    "run_algorithm_api",
    "compare_algorithms_api",
]
