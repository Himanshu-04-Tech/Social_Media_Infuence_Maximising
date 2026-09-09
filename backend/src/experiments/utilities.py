"""Utilities for Experiment Framework.

Provides unique experiment ID generators, graph introspection helpers,
and graph generation utilities.
"""

from __future__ import annotations

from typing import Any


def generate_experiment_id(
    prefix: str,
    k: int | None = None,
    probability: float | None = None,
    node_count: int | None = None,
    num_simulations: int | None = None,
) -> str:
    """Generate a standardized, unique experiment identifier matching SRS specifications.

    Examples:
        generate_experiment_id("expA", k=10, probability=0.1) -> "expA_k10_p01"
        generate_experiment_id("expB", probability=0.20) -> "expB_p020"
        generate_experiment_id("expC", node_count=1000) -> "expC_v1000"
        generate_experiment_id("expD", num_simulations=100) -> "expD_n100"

    Args:
        prefix: Experiment identifier prefix (e.g. 'expA', 'expB', 'expC', 'expD').
        k: Seed count (used in Experiment A).
        probability: Edge propagation probability (used in Experiment A & B).
        node_count: Total vertices / graph size (used in Experiment C).
        num_simulations: Number of Monte Carlo iterations (used in Experiment D).

    Returns:
        Formatted experiment ID string.
    """
    prefix_clean = prefix.strip()
    parts = [prefix_clean]

    if prefix_clean == "expA" or "expA" in prefix_clean:
        if k is not None:
            parts.append(f"k{k}")
        if probability is not None:
            # Format 0.1 -> p01, 0.05 -> p005, etc.
            p_pct = int(round(probability * 100))
            if p_pct % 10 == 0:
                parts.append(f"p{p_pct // 10:02d}")
            else:
                parts.append(f"p{p_pct:03d}")
    elif prefix_clean == "expB" or "expB" in prefix_clean:
        if probability is not None:
            # Format 0.20 -> p020, 0.05 -> p005, 0.10 -> p010, 0.30 -> p030
            p_val = int(round(probability * 100))
            parts.append(f"p{p_val:03d}")
    elif prefix_clean == "expC" or "expC" in prefix_clean:
        if node_count is not None:
            parts.append(f"v{node_count}")
    elif prefix_clean == "expD" or "expD" in prefix_clean:
        if num_simulations is not None:
            parts.append(f"n{num_simulations}")
    else:
        # Generic fallback appending all present components
        if k is not None:
            parts.append(f"k{k}")
        if probability is not None:
            parts.append(f"p{int(round(probability * 100))}")
        if node_count is not None:
            parts.append(f"v{node_count}")
        if num_simulations is not None:
            parts.append(f"n{num_simulations}")

    return "_".join(parts)


def get_node_count(graph: Any) -> int:
    """Extract node count from any NetworkX-compatible graph representation.

    Args:
        graph: Graph instance.

    Returns:
        Total number of nodes as integer.

    Raises:
        ValueError: If node count cannot be introspected.
    """
    if graph is None:
        raise ValueError("Cannot determine node count of None graph.")

    if hasattr(graph, "number_of_nodes") and callable(graph.number_of_nodes):
        return int(graph.number_of_nodes())

    if hasattr(graph, "nodes"):
        nodes_attr = getattr(graph, "nodes")
        if callable(nodes_attr):
            return len(nodes_attr())
        return len(nodes_attr)

    if hasattr(graph, "node_count"):
        return int(getattr(graph, "node_count"))

    try:
        return len(graph)
    except (TypeError, AttributeError) as err:
        raise ValueError(
            f"Unable to determine node count from graph object: {type(graph).__name__}"
        ) from err


def default_graph_sampler(size: int, seed: int | None = None) -> Any:
    """Default fallback graph generator for scalability experiments.

    Generates a fast synthetic Barabási-Albert scale-free network or Erdos-Renyi network.

    Args:
        size: Target node count.
        seed: Random seed.

    Returns:
        NetworkX graph instance.
    """
    try:
        import networkx as nx

        # Use scale-free Barabási-Albert model typical for social networks
        m = min(3, max(1, size - 1))
        return nx.barabasi_albert_graph(n=size, m=m, seed=seed)
    except ImportError:
        # Lightweight dummy graph fallback if networkx is unavailable
        class FallbackGraph:
            def __init__(self, n: int):
                self.n = n
            def number_of_nodes(self) -> int:
                return self.n
            def __len__(self) -> int:
                return self.n
        return FallbackGraph(size)
