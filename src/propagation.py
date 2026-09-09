"""
src/propagation.py

Independent Cascade (IC) influence diffusion model and Monte Carlo simulation engine.
Designed and analyzed for DAA (Design and Analysis of Algorithms).

Authors:
    Zaid Hussain - Graph & Algorithm Engineer
    Himanshu     - Frontend, Visualization & Integration Engineer
"""

from collections import deque
import math
import random
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

from src.graph import Graph


def validate_probability(p: float) -> None:
    """
    Validates that propagation probability p is a valid float in [0.0, 1.0].

    Raises:
        TypeError: If p is not a numeric float/int.
        ValueError: If p < 0.0 or p > 1.0.
    """
    if not isinstance(p, (int, float)) or isinstance(p, bool):
        raise TypeError(f"Propagation probability must be a float, got {type(p).__name__}")
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Propagation probability p must be in [0.0, 1.0], got {p}")


def validate_seeds(graph: Graph, seeds: Iterable[int]) -> Set[int]:
    """
    Validates and deduplicates a collection of seed node IDs against the graph.

    Rules:
        - Duplicate seeds are automatically deduplicated (mathematical seed set).
        - Any seed ID not existing in the graph raises ValueError.

    Complexity:
        Time: Expected O(|S|)
        Space: O(|S|)

    Returns:
        Deduplicated set of valid seed node IDs.

    Raises:
        TypeError: If any seed ID is not an integer.
        ValueError: If any seed ID does not exist in graph.adj.
    """
    seed_set: Set[int] = set()
    for s in seeds:
        if not isinstance(s, int) or isinstance(s, bool):
            raise TypeError(f"Seed ID must be an integer, got {type(s).__name__}: {s}")
        if not graph.has_node(s):
            raise ValueError(f"Seed node ID {s} does not exist in the graph.")
        seed_set.add(s)
    return seed_set


def _get_rng(rng: Optional[Union[random.Random, int]]) -> random.Random:
    """
    Helper to resolve a random number generator instance.
    Enables deterministic testing while supporting unseeded production execution.
    """
    if rng is None:
        return random.Random()
    if isinstance(rng, int):
        return random.Random(rng)
    if isinstance(rng, random.Random):
        return rng
    raise TypeError(
        f"rng must be None, an int seed, or a random.Random instance, got {type(rng).__name__}"
    )


def simulate_independent_cascade(
    graph: Graph,
    seeds: Iterable[int],
    p: float,
    rng: Optional[Union[random.Random, int]] = None,
    return_history: bool = False,
) -> Union[Set[int], Tuple[Set[int], List[Set[int]]]]:
    """
    Simulates a single run of influence diffusion under the Independent Cascade (IC) model.

    Model Dynamics:
        1. At step t = 0, initial seed set S is activated.
        2. At step t >= 1, each node u activated at step t - 1 gets exactly one chance
           to activate each of its currently inactive neighbors v with independent probability p.
        3. Once active, a node remains active forever (progressive diffusion).
        4. The cascade terminates when no newly activated nodes are produced.

    DAA Complexity Analysis:
        - Time: O(V_A + E_A) where V_A is the set of activated nodes and E_A is the set of
          incident edges scanned. In the worst case (e.g. p = 1.0 or dense cascade), V_A = V
          and every node is activated at most once; when popped, its neighbors are scanned.
          In an undirected graph, an edge can be inspected from both endpoints, so the total
          number of adjacency entries scanned across the entire cascade is at most:
              sum_{v in V_A} deg(v) <= 2E.
          Therefore, worst-case time is strictly O(V + E).
        - Space: O(V) to store active_set, queues/layers, and optional activation history.

    Args:
        graph: The undirected Graph instance.
        seeds: Iterable of seed node IDs (validated and deduplicated).
        p: Propagation probability on each directed attempt in [0.0, 1.0].
        rng: Optional Random instance or integer seed for deterministic simulation.
        return_history: If True, returns (active_set, history) where history is a list
                        of disjoint sets representing newly activated nodes per time step.
                        If False (default), returns only active_set.

    Returns:
        If return_history is False: Set[int] of all active nodes.
        If return_history is True:  Tuple[Set[int], List[Set[int]]] (active_set, history).
    """
    validate_probability(p)
    seed_set = validate_seeds(graph, seeds)
    r = _get_rng(rng)

    if not seed_set:
        return (set(), []) if return_history else set()

    # Boundary case: p == 0.0 -> no propagation can occur beyond seeds
    if p == 0.0:
        if return_history:
            return set(seed_set), [set(seed_set)]
        return set(seed_set)

    active_set: Set[int] = set(seed_set)
    current_layer: List[int] = list(seed_set)
    history: List[Set[int]] = [set(seed_set)] if return_history else []

    while current_layer:
        next_layer: Set[int] = set()

        for u in current_layer:
            for v in graph.neighbors(u):
                if v not in active_set:
                    # Each active node u gets exactly one attempt to activate neighbor v
                    if r.random() < p:
                        active_set.add(v)
                        next_layer.add(v)

        if return_history and next_layer:
            history.append(next_layer)

        current_layer = list(next_layer)

    if return_history:
        return active_set, history
    return active_set


def estimate_influence(
    graph: Graph,
    seeds: Iterable[int],
    p: float,
    simulations: int = 100,
    rng: Optional[Union[random.Random, int]] = None,
) -> float:
    """
    Estimates the expected influence spread sigma(S) using Monte Carlo simulations.

    Mathematical Basis:
        The true influence spread sigma(S) = E[|Active(S)|].
        By the Law of Large Numbers, the sample mean across R independent simulations
        converges to sigma(S):
            hat{sigma}(S) = (1 / R) * sum_{i=1}^R |Active_i(S)|.

    DAA Complexity Analysis:
        - Time: O(R * (V_A + E_A)) on average, bounded by O(R * (V + E)) in the worst case.
        - Space: O(V) — memory is reused across simulations; only one simulation's
          state is maintained at any given time.

    Args:
        graph: The undirected Graph instance.
        seeds: Iterable of seed node IDs.
        p: Propagation probability in [0.0, 1.0].
        simulations: Number of Monte Carlo runs R (must be >= 1).
        rng: Optional Random instance or integer seed for reproducibility.

    Returns:
        float: Monte Carlo sample mean of the number of activated nodes.
    """
    validate_probability(p)
    if not isinstance(simulations, int) or isinstance(simulations, bool) or simulations <= 0:
        raise ValueError(f"Number of simulations must be a positive integer, got {simulations}")

    seed_set = validate_seeds(graph, seeds)
    if not seed_set:
        return 0.0

    # Boundary case: p == 0.0 -> spread is always exactly |seed_set|
    if p == 0.0:
        return float(len(seed_set))

    # Boundary case: p == 1.0 -> deterministic reachability (union of connected components)
    if p == 1.0:
        reachable: Set[int] = set()
        for s in seed_set:
            if s not in reachable:
                comp_nodes, _ = graph.bfs(s)
                reachable.update(comp_nodes)
        return float(len(reachable))

    r = _get_rng(rng)
    total_activated = 0

    for _ in range(simulations):
        active = simulate_independent_cascade(
            graph, seed_set, p, rng=r, return_history=False
        )
        total_activated += len(active)

    return total_activated / simulations


def estimate_influence_stats(
    graph: Graph,
    seeds: Iterable[int],
    p: float,
    simulations: int = 100,
    rng: Optional[Union[random.Random, int]] = None,
) -> Dict[str, float]:
    """
    Detailed statistical reporting of Monte Carlo simulations for reporting and charts.
    Kept separate from the core estimate_influence() API.

    Returns:
        Dictionary containing:
        - mean: float
        - std_dev: float
        - min: int
        - max: int
        - simulations: int
    """
    validate_probability(p)
    if not isinstance(simulations, int) or isinstance(simulations, bool) or simulations <= 0:
        raise ValueError(f"Number of simulations must be a positive integer, got {simulations}")

    seed_set = validate_seeds(graph, seeds)
    if not seed_set:
        return {"mean": 0.0, "std_dev": 0.0, "min": 0, "max": 0, "simulations": simulations}

    r = _get_rng(rng)
    counts: List[int] = []

    for _ in range(simulations):
        active = simulate_independent_cascade(
            graph, seed_set, p, rng=r, return_history=False
        )
        counts.append(len(active))

    mean_val = sum(counts) / simulations
    variance = sum((x - mean_val) ** 2 for x in counts) / (simulations if simulations > 1 else 1)
    std_dev = math.sqrt(variance)

    return {
        "mean": round(mean_val, 4),
        "std_dev": round(std_dev, 4),
        "min": min(counts),
        "max": max(counts),
        "simulations": simulations,
    }
