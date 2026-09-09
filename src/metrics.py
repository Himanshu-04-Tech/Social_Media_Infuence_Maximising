"""
src/metrics.py

Benchmarking, experimentation, and comparative analysis engine for Influence Maximization.
Designed and analyzed for DAA (Design and Analysis of Algorithms).

Responsibilities:
    - High-resolution timing: separates seed-selection runtime from final spread evaluation runtime.
    - Deterministic, decoupled RNG stream derivation from a master seed.
    - JSON-serializable output structures for Himanshu's frontend.
    - Clean comparative benchmarking across Random, Highest-Degree, and Greedy algorithms.

Authors:
    Zaid Hussain - Graph & Algorithm Engineer
    Himanshu     - Frontend, Visualization & Integration Engineer
"""

import math
import random
import statistics
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from src.graph import Graph
from src.propagation import simulate_independent_cascade, validate_probability
from src.seed_selection import (
    _resolve_rng,
    _validate_candidates,
    _validate_graph_and_k,
    select_seeds_degree,
    select_seeds_greedy,
    select_seeds_random,
)


def _derive_seed(master_seed: Optional[int], salt: int) -> Optional[int]:
    """
    Deterministically derives a child RNG seed from a master seed and salt integer.
    If master_seed is None, returns None (enabling independent unseeded random execution).
    """
    if master_seed is None:
        return None
    # Use standard 31-bit integer range for portability
    return (master_seed * 1000003 + salt * 1009) % (2**31 - 1)


def evaluate_seed_spread_detailed(
    graph: Graph,
    seeds: List[int],
    p: float,
    simulations: int = 100,
    rng: Optional[Union[random.Random, int]] = None,
) -> Dict[str, Any]:
    """
    Performs an independent Monte Carlo evaluation of a selected seed set.
    Calculates sample mean, sample standard deviation (N - 1 degrees of freedom), min, and max.

    Returns:
        JSON-serializable dictionary with statistics at full floating-point precision.
    """
    validate_probability(p)
    if simulations <= 0:
        raise ValueError(f"simulations must be a positive integer, got {simulations}")

    r = _resolve_rng(rng)

    if not seeds:
        return {
            "mean": 0.0,
            "std_dev": 0.0,
            "min": 0,
            "max": 0,
            "simulations": simulations,
        }

    # Boundary case: p == 0.0
    if p == 0.0:
        return {
            "mean": float(len(seeds)),
            "std_dev": 0.0,
            "min": len(seeds),
            "max": len(seeds),
            "simulations": simulations,
        }

    # Boundary case: p == 1.0 (deterministic reachability)
    if p == 1.0:
        reachable: Set[int] = set()
        for s in seeds:
            if s not in reachable:
                comp, _ = graph.bfs(s)
                reachable.update(comp)
        reach_len = len(reachable)
        return {
            "mean": float(reach_len),
            "std_dev": 0.0,
            "min": reach_len,
            "max": reach_len,
            "simulations": simulations,
        }

    counts: List[int] = []
    for _ in range(simulations):
        active = simulate_independent_cascade(graph, seeds, p, rng=r, return_history=False)
        counts.append(len(active))

    mean_val = sum(counts) / float(simulations)
    std_dev_val = statistics.stdev(counts) if len(counts) > 1 else 0.0

    return {
        "mean": mean_val,
        "std_dev": std_dev_val,
        "min": min(counts),
        "max": max(counts),
        "simulations": simulations,
    }


def run_experiment(
    algorithm: str,
    graph: Graph,
    k: int,
    p: float,
    simulations: int = 100,
    candidate_nodes: Optional[Iterable[int]] = None,
    master_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Executes a single seed-selection algorithm, measures selection and evaluation runtimes,
    and returns a fully JSON-serializable experiment report.

    Timing separation:
        - selection_runtime_seconds: High-resolution time taken to select the k seeds.
          For Greedy, this includes all candidate influence evaluations.
        - evaluation_runtime_seconds: Time taken by the independent Monte Carlo evaluation
          of the selected seed set.
        - total_runtime_seconds: selection + evaluation runtimes.

    Args:
        algorithm: 'random', 'degree', or 'greedy' (case-insensitive).
        graph: The undirected Graph instance.
        k: Seed budget (1 <= k <= |V|).
        p: Propagation probability in [0.0, 1.0].
        simulations: Number of Monte Carlo simulations R for evaluation.
        candidate_nodes: Optional candidate pool C subseteq V.
        master_seed: Optional integer master seed for reproducible execution.

    Returns:
        JSON-serializable dictionary containing all metadata, seeds, metrics, and runtimes.
    """
    alg = algorithm.strip().lower()
    valid_algs = {"random", "degree", "greedy"}
    if alg not in valid_algs:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Must be one of: {sorted(valid_algs)}")

    _validate_graph_and_k(graph, k)
    validate_probability(p)
    if simulations <= 0:
        raise ValueError(f"simulations must be a positive integer, got {simulations}")

    # Validate candidate nodes if specified
    validated_candidates = (
        _validate_candidates(graph, candidate_nodes, k)
        if candidate_nodes is not None
        else None
    )
    candidate_count = (
        len(validated_candidates)
        if validated_candidates is not None
        else graph.number_of_nodes()
    )

    # Derive separate deterministic RNG streams for selection and final evaluation
    sel_seed = _derive_seed(master_seed, salt=101)
    eval_seed = _derive_seed(master_seed, salt=202)

    rng_selection = random.Random(sel_seed) if sel_seed is not None else random.Random()
    rng_eval = random.Random(eval_seed) if eval_seed is not None else random.Random()

    seed_nodes: List[int] = []
    marginal_gains: List[float] = []

    # 1. Measure Seed Selection Runtime
    t_sel_start = time.perf_counter()

    if alg == "random":
        if validated_candidates is not None:
            seed_nodes = rng_selection.sample(validated_candidates, k)
        else:
            seed_nodes = select_seeds_random(graph, k, rng=rng_selection)

    elif alg == "degree":
        if validated_candidates is not None:
            # Rank candidates by degree in the graph
            cand_degrees = [(c, graph.degree(c)) for c in validated_candidates]
            cand_degrees.sort(key=lambda item: item[1], reverse=True)
            seed_nodes = [node for node, _ in cand_degrees[:k]]
        else:
            seed_nodes = select_seeds_degree(graph, k)

    elif alg == "greedy":
        # Greedy performs its internal evaluations using rng_selection
        seed_nodes, marginal_gains = select_seeds_greedy(
            graph=graph,
            k=k,
            p=p,
            simulations=simulations,
            candidate_nodes=validated_candidates,
            rng=rng_selection,
            return_marginal_gains=True,
        )

    t_sel_end = time.perf_counter()
    selection_runtime = t_sel_end - t_sel_start

    # 2. Measure Independent Final Spread Evaluation Runtime
    t_eval_start = time.perf_counter()
    spread_stats = evaluate_seed_spread_detailed(
        graph=graph,
        seeds=seed_nodes,
        p=p,
        simulations=simulations,
        rng=rng_eval,
    )
    t_eval_end = time.perf_counter()
    evaluation_runtime = t_eval_end - t_eval_start

    total_runtime = selection_runtime + evaluation_runtime

    return {
        "algorithm": alg,
        "k": k,
        "probability": p,
        "simulations": simulations,
        "candidate_count": candidate_count,
        "is_restricted_candidate_pool": candidate_nodes is not None,
        "master_seed": master_seed,
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "seed_nodes": [int(s) for s in seed_nodes],
        "influence_spread": spread_stats["mean"],
        "marginal_gains": [float(g) for g in marginal_gains],
        "selection_runtime_seconds": selection_runtime,
        "evaluation_runtime_seconds": evaluation_runtime,
        "total_runtime_seconds": total_runtime,
        "spread_stats": spread_stats,
    }


def compare_algorithms(
    graph: Graph,
    k: int,
    p: float,
    simulations: int = 100,
    candidate_nodes: Optional[Iterable[int]] = None,
    master_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Runs all three algorithms (Random, Degree, Greedy) under identical experimental conditions.

    RNG Independence:
        Each algorithm receives a deterministic RNG stream derived from master_seed with
        an algorithm-specific salt. Reordering algorithms does not alter the random numbers
        encountered by another algorithm.

    Returns:
        JSON-serializable dictionary with configuration metadata and results list in order:
        [Random, Degree, Greedy].
    """
    _validate_graph_and_k(graph, k)
    validate_probability(p)
    if simulations <= 0:
        raise ValueError(f"simulations must be a positive integer, got {simulations}")

    # Validate candidates once so all algorithms share the exact same candidate pool
    validated_candidates = (
        _validate_candidates(graph, candidate_nodes, k)
        if candidate_nodes is not None
        else None
    )
    candidate_count = (
        len(validated_candidates)
        if validated_candidates is not None
        else graph.number_of_nodes()
    )

    algorithms_in_order = ["random", "degree", "greedy"]
    results: List[Dict[str, Any]] = []

    # Map each algorithm to an independent deterministic master seed offset
    salt_offsets = {"random": 1000, "degree": 2000, "greedy": 3000}

    for alg in algorithms_in_order:
        alg_master_seed = _derive_seed(master_seed, salt_offsets[alg])
        res = run_experiment(
            algorithm=alg,
            graph=graph,
            k=k,
            p=p,
            simulations=simulations,
            candidate_nodes=validated_candidates,
            master_seed=alg_master_seed,
        )
        # Restore the top-level master_seed in the metadata for clarity
        res["master_seed"] = master_seed
        results.append(res)

    return {
        "configuration": {
            "k": k,
            "probability": p,
            "simulations": simulations,
            "candidate_count": candidate_count,
            "is_restricted_candidate_pool": candidate_nodes is not None,
            "master_seed": master_seed,
            "graph_nodes": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges(),
        },
        "results": results,
    }
