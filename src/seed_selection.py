"""
src/seed_selection.py

Seed selection algorithms for Influence Maximization under the Independent Cascade model.
Designed and analyzed for DAA (Design and Analysis of Algorithms).

Algorithms implemented:
    1. Random Seed Selection (Baseline)
    2. Highest-Degree Seed Selection (Fast Heuristic)
    3. Greedy Influence Maximization (Core DAA Approximation Algorithm)

Authors:
    Zaid Hussain - Graph & Algorithm Engineer
    Himanshu     - Frontend, Visualization & Integration Engineer
"""

import random
from typing import Iterable, List, Optional, Set, Tuple, Union

from src.graph import Graph
from src.propagation import estimate_influence, validate_probability


def _resolve_rng(rng: Optional[Union[random.Random, int]]) -> random.Random:
    """
    Resolves an optional RNG instance or integer seed into a dedicated random.Random object.
    Ensures all randomness is contained and 100% reproducible for unit tests and benchmarks.
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


def _validate_graph_and_k(graph: Graph, k: int) -> None:
    """
    Validates general feasibility conditions for seed selection.

    Raises:
        ValueError: If graph has no vertices.
        ValueError: If k <= 0 or k > |V|.
    """
    n = graph.number_of_nodes()
    if n == 0:
        raise ValueError("Graph is empty: cannot select seeds from a graph with 0 nodes.")
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"Budget k must be a positive integer, got {k}")
    if k > n:
        raise ValueError(
            f"Budget k={k} cannot exceed total number of nodes in graph (|V|={n})"
        )


def _validate_candidates(
    graph: Graph, candidate_nodes: Optional[Iterable[int]], k: int
) -> List[int]:
    """
    Validates and deduplicates candidate nodes.

    Rules:
        - If candidate_nodes is None: evaluates the full ground set V.
        - If candidate_nodes is provided: evaluates restricted candidate pool C subseteq V.
        - Every candidate must exist in the graph.
        - Deduplicates candidate IDs.
        - Ensures candidate pool size is >= k.

    Raises:
        ValueError: If any candidate node ID does not exist in graph.
        ValueError: If fewer than k valid candidate nodes are available.
    """
    if candidate_nodes is None:
        return graph.nodes()

    seen: Set[int] = set()
    valid_candidates: List[int] = []

    for c in candidate_nodes:
        if not isinstance(c, int) or isinstance(c, bool):
            raise TypeError(f"Candidate node ID must be an integer, got {type(c).__name__}: {c}")
        if not graph.has_node(c):
            raise ValueError(f"Candidate node ID {c} does not exist in the graph.")
        if c not in seen:
            seen.add(c)
            valid_candidates.append(c)

    if len(valid_candidates) < k:
        raise ValueError(
            f"Budget k={k} exceeds number of available candidate nodes ({len(valid_candidates)})"
        )

    return valid_candidates


def select_seeds_random(
    graph: Graph, k: int, rng: Optional[Union[random.Random, int]] = None
) -> List[int]:
    """
    Randomly selects k distinct seed nodes from the graph without replacement.

    Purpose:
        Serves as the empirical baseline lower bound for algorithm comparisons.

    DAA Complexity Analysis:
        - Time: O(V) to access the node collection + O(k) for sample without replacement.
          Overall: O(V) time.
        - Space: O(V) to materialize candidate node list.

    Args:
        graph: The undirected Graph instance.
        k: Number of seeds to select (1 <= k <= |V|).
        rng: Optional Random instance or int seed for deterministic execution.

    Returns:
        List[int] containing k unique seed node IDs.
    """
    _validate_graph_and_k(graph, k)
    r = _resolve_rng(rng)
    all_nodes = graph.nodes()
    return r.sample(all_nodes, k)


def select_seeds_degree(graph: Graph, k: int) -> List[int]:
    """
    Selects the k nodes with the highest degree in the graph.

    Purpose:
        A fast structural heuristic based on the intuition that highly connected hubs
        initiate larger influence cascades.

    Limitations for DAA Viva:
        Degree selection does NOT account for overlap / redundancy. If top-degree hubs
        cluster in the same dense community (e.g. mutual friends sharing the same audience),
        selecting them redundantly wastes seed budget.

    DAA Complexity Analysis:
        - Time: O(V log V) to rank all nodes by degree via Timsort.
        - Space: O(V) to store the ranked list.

    Args:
        graph: The undirected Graph instance.
        k: Number of seeds to select (1 <= k <= |V|).

    Returns:
        List[int] containing k unique seed node IDs sorted descending by degree.
    """
    _validate_graph_and_k(graph, k)
    ranked = graph.degree_ranking()
    return [node for node, _ in ranked[:k]]


def select_seeds_greedy(
    graph: Graph,
    k: int,
    p: float,
    simulations: int = 100,
    candidate_nodes: Optional[Iterable[int]] = None,
    rng: Optional[Union[random.Random, int]] = None,
    return_marginal_gains: bool = False,
) -> Union[List[int], Tuple[List[int], List[float]]]:
    """
    Greedy Influence Maximization algorithm (Kempe, Kleinberg, Tardos 2003).

    Algorithmic Mechanics:
        Starts with empty seed set S = [].
        At each of the k iterations:
            1. Considers every available candidate v not yet in S.
            2. Computes the marginal gain:
                   Delta(v | S) = sigma(S union {v}) - sigma(S)
            3. Greedily selects the candidate v* maximizing Delta(v | S).
            4. Adds v* to S and records its marginal gain.

    Theoretical Foundation:
        - The true influence function sigma(S) = E[|Active(S)|] is monotone and submodular.
        - Monotonicity: sigma(S) <= sigma(T) for all S subseteq T.
        - Submodularity (Diminishing Returns):
              sigma(S union {v}) - sigma(S) >= sigma(T union {v}) - sigma(T)
              for all S subseteq T and v not in T.
        - By Nemhauser et al. (1978), greedy achieves a (1 - 1/e) approx (approx 63.2%)
          to the optimal seed set when sigma is evaluated exactly.
        - Because sigma(S) is estimated via Monte Carlo simulation hat{sigma}(S),
          Kempe et al. proved a (1 - 1/e - epsilon) guarantee with high probability.

    Restricted Candidate Set vs Full Ground Set:
        - When candidate_nodes is None, evaluates all V \\ S (standard greedy).
        - When candidate_nodes = C subseteq V, performs greedy optimization RESTRICTED to C.
          Restricting C accelerates execution for tests/demos, but optimizes over a smaller
          feasible ground set rather than the global optimum.

    DAA Complexity Analysis:
        - Number of candidate evaluations across k steps:
              sum_{i=0}^{k-1} (|C| - i) = O(k * |C|)
        - Each candidate evaluation runs R Monte Carlo simulations costing O(R * (V_A + E_A)).
        - Total Time Complexity:
              O(k * |C| * R * (V_A + E_A)) typically, bounded by O(k * |V| * R * (V + E)) worst case.
        - Total Space Complexity: O(|V|) to maintain candidate pools and seed sets.

    Args:
        graph: The undirected Graph instance.
        k: Number of seeds to select (1 <= k <= |V|).
        p: Propagation probability in [0.0, 1.0].
        simulations: Number of Monte Carlo simulations R per candidate evaluation.
        candidate_nodes: Optional subset of candidate node IDs C subseteq V.
        rng: Optional Random instance or int seed for deterministic execution.
        return_marginal_gains: If True, returns (seeds, marginal_gains).
                               If False (default), returns seeds.

    Returns:
        If return_marginal_gains is False: List[int] of k selected seed IDs.
        If return_marginal_gains is True:  Tuple[List[int], List[float]] (seeds, marginal_gains).
    """
    _validate_graph_and_k(graph, k)
    validate_probability(p)
    candidates = _validate_candidates(graph, candidate_nodes, k)
    r = _resolve_rng(rng)

    S: List[int] = []
    selected_set: Set[int] = set()
    marginal_gains: List[float] = []
    current_spread: float = 0.0

    for _ in range(k):
        best_candidate: Optional[int] = None
        best_spread: float = -1.0

        for v in candidates:
            if v in selected_set:
                continue

            trial_seeds = S + [v]
            spread = estimate_influence(
                graph, trial_seeds, p, simulations=simulations, rng=r
            )

            if spread > best_spread:
                best_spread = spread
                best_candidate = v

        if best_candidate is None:
            # Fallback if candidates exhausted (guarded by validation)
            break

        gain = best_spread - current_spread
        marginal_gains.append(round(gain, 4))
        S.append(best_candidate)
        selected_set.add(best_candidate)
        current_spread = best_spread

    if return_marginal_gains:
        return S, marginal_gains
    return S
