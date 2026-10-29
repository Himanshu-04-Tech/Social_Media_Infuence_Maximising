"""Orchestration and data loading utilities for the Streamlit Dashboard.

Glues together Role 1 algorithms, Module 1 (Monte Carlo), Module 2 (Metrics),
Module 4 (Storage), Module 5 (Charts), and Module 6 (Network Visualization)
without containing any simulation or statistical logic itself.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import streamlit as st

# Module 1
from monte_carlo.engine import run_monte_carlo

# Module 2
from metrics.metrics_engine import calculate_metrics

# Module 4
from storage.storage_manager import StorageManager

# Module 5
from visualization.chart_builder import VisualizationFactory

# Module 6
from network_visualization.network_renderer import NetworkVisualizationFactory

from .config import DATASETS

logger = logging.getLogger(__name__)


def to_networkx(graph: Any) -> nx.Graph:
    """Standardize graph object into a NetworkX Graph instance.

    Works seamlessly with both custom Role 1 Graph classes and native NetworkX graphs.
    """
    if isinstance(graph, nx.Graph):
        return graph

    G = nx.Graph()
    if hasattr(graph, "adj"):
        for u, neighbors in graph.adj.items():
            G.add_node(u)
            for v in neighbors:
                if u < v:
                    G.add_edge(u, v)
    elif hasattr(graph, "nodes") and hasattr(graph, "edges"):
        G.add_nodes_from(graph.nodes())
        G.add_edges_from(graph.edges())

    return G


def to_role1_graph(graph: Any) -> Any:
    """Convert NetworkX or other graph representations into a Role 1 Graph."""
    if hasattr(graph, "degree_ranking") and hasattr(graph, "adj") and hasattr(graph, "bfs"):
        return graph

    try:
        from src.graph import Graph as Role1Graph
        r1_g = Role1Graph()
        if isinstance(graph, nx.Graph):
            for u in graph.nodes():
                r1_g.add_node(int(u))
            for u, v in graph.edges():
                r1_g.add_edge(int(u), int(v))
            return r1_g
    except Exception as err:
        logger.debug("Failed converting to Role 1 Graph: %s", err)

    return graph


@st.cache_resource(show_spinner=False)
def load_dataset(dataset_name: str) -> nx.Graph:
    """Load and cache network graphs with high performance.

    Args:
        dataset_name: Key from config.DATASETS.

    Returns:
        NetworkX Graph instance.
    """
    dataset_info = DATASETS.get(dataset_name, {})
    dataset_type = dataset_info.get("type", "synthetic_ba")
    dataset_path = dataset_info.get("path")

    if dataset_type == "file" and dataset_path and Path(dataset_path).exists():
        logger.info("Loading dataset from file: %s", dataset_path)
        G = nx.read_edgelist(str(dataset_path), nodetype=int)
        return G

    if dataset_type == "builtin_karate":
        logger.info("Loading Zachary's Karate Club dataset.")
        return nx.karate_club_graph()

    logger.info("Loading synthetic Barabási-Albert scale-free network.")
    return nx.barabasi_albert_graph(n=300, m=3, seed=42)


def select_seeds(
    graph: Any,
    algorithm: str,
    k: int,
    p: float = 0.10,
    candidate_mode: str = "Top 50 Candidates (Fast)",
) -> list[int]:
    """Select k seed nodes using Role 1 APIs or standard graph algorithms.

    Args:
        graph: Network graph (Role 1 Graph or NetworkX Graph).
        algorithm: 'Random', 'Degree', 'Greedy', 'BFS', or 'IC'.
        k: Number of seeds to choose.
        p: Propagation probability for greedy simulation.
        candidate_mode: Candidate pool restriction for greedy.

    Returns:
        List of k seed node identifiers.
    """
    algo_clean = algorithm.lower().strip()
    node_list = list(graph.nodes() if hasattr(graph, "nodes") and callable(graph.nodes) else graph.nodes)
    k = min(k, len(node_list))

    # Role 1 compatibility: attempt Role 1 Graph algorithms first
    r1_graph = to_role1_graph(graph)
    is_role1_graph = hasattr(r1_graph, "degree_ranking") and hasattr(r1_graph, "adj")

    if is_role1_graph:
        try:
            from src import (  # type: ignore
                random_selection,
                degree_selection,
                greedy_selection,
            )
            if algo_clean == "random":
                return random_selection(r1_graph, k)
            elif algo_clean == "degree":
                return degree_selection(r1_graph, k)
            elif algo_clean == "greedy":
                is_fast = "fast" in candidate_mode.lower()
                pool = [n for n, _ in r1_graph.degree_ranking()[:max(50, k)]] if is_fast else None
                sims = 10 if is_fast else 50
                return greedy_selection(r1_graph, k, p=p, candidate_nodes=pool, simulations=sims)
            elif algo_clean == "bfs":
                start_node = r1_graph.degree_ranking()[0][0]
                bfs_nodes, _ = r1_graph.bfs(start_node)
                return bfs_nodes[:k]
        except Exception as err:
            logger.debug("Role 1 seed selection fallback: %s", err)

    # General NetworkX graph implementation
    if algo_clean == "random":
        return random.sample(node_list, k)

    elif algo_clean == "degree":
        sorted_by_degree = sorted(node_list, key=lambda n: graph.degree(n), reverse=True)
        return sorted_by_degree[:k]

    elif algo_clean == "greedy":
        # Fast marginal gain greedy selection
        selected: list[int] = []
        candidates = sorted(node_list, key=lambda n: graph.degree(n), reverse=True)[:50] if "fast" in candidate_mode.lower() else node_list[:100]

        for _ in range(k):
            best_node = None
            best_gain = -1.0
            for cand in candidates:
                if cand in selected:
                    continue
                gain = sum(1 for nbr in graph.neighbors(cand) if nbr not in selected)
                if gain > best_gain:
                    best_gain = gain
                    best_node = cand
            if best_node is not None:
                selected.append(best_node)
            else:
                break
        return selected

    elif algo_clean == "bfs":
        start_node = max(node_list, key=lambda n: graph.degree(n))
        bfs_nodes = list(nx.bfs_tree(graph, start_node).nodes()) if isinstance(graph, nx.Graph) else graph.bfs(start_node)[0]
        return bfs_nodes[:k]

    else:  # Default / IC
        sorted_by_degree = sorted(node_list, key=lambda n: graph.degree(n), reverse=True)
        return sorted_by_degree[:k]


def simulate_single_cascade(
    graph: Any,
    seeds: list[int],
    p: float,
) -> tuple[int, set[int]]:
    """Simulate a single Independent Cascade propagation run.

    Returns:
        Tuple of (influenced_count, set_of_all_influenced_nodes).
    """
    r1_graph = to_role1_graph(graph)
    if hasattr(r1_graph, "adj") and hasattr(r1_graph, "degree"):
        try:
            from src import simulate_influence  # type: ignore
            res = simulate_influence(r1_graph, seeds, p)
            if hasattr(res, "nodes"):
                return len(res.nodes), set(res.nodes)
            if isinstance(res, set):
                return len(res), res
            if isinstance(res, (int, float)):
                return int(res), set(seeds)
        except Exception as err:
            logger.debug("Role 1 cascade simulation fallback: %s", err)

    # Standard NetworkX cascade simulation
    active_nodes = set(seeds)
    newly_active = set(seeds)

    while newly_active:
        next_layer = set()
        for u in newly_active:
            for v in graph.neighbors(u):
                if v not in active_nodes:
                    if random.random() <= p:
                        active_nodes.add(v)
                        next_layer.add(v)
        newly_active = next_layer

    return len(active_nodes), active_nodes


def run_full_simulation_pipeline(
    graph: Any,
    algorithm: str,
    k: int,
    probability: float,
    num_simulations: int,
    candidate_mode: str = "Top 50 Candidates (Fast)",
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Orchestrate full Role 1 -> Module 1 -> Module 2 -> Module 4 -> Module 5/6 pipeline."""
    start_time = time.perf_counter()

    nx_graph = to_networkx(graph)

    # Step 1: Seed Selection
    if progress_callback:
        progress_callback(0.15, f"Selecting {k} seeds via {algorithm}...")
    seeds = select_seeds(graph, algorithm, k, p=probability, candidate_mode=candidate_mode)

    # Step 2: Single Cascade for Node-Level Network Visualizer
    if progress_callback:
        progress_callback(0.35, "Propagating single cascade for visualization...")
    _, influenced_nodes = simulate_single_cascade(nx_graph, seeds, probability)

    # Step 3: Monte Carlo Engine (Module 1)
    if progress_callback:
        progress_callback(0.55, f"Running {num_simulations} Monte Carlo iterations...")

    def sim_fn(g: Any, s: Any, p: float) -> int:
        count, _ = simulate_single_cascade(g, list(s), p)
        return count

    mc_result = run_monte_carlo(
        graph=nx_graph,
        seeds=seeds,
        probability=probability,
        num_simulations=num_simulations,
        simulate_fn=sim_fn,
    )

    elapsed_time = time.perf_counter() - start_time

    # Step 4: Metrics Engine (Module 2)
    if progress_callback:
        progress_callback(0.75, "Calculating metrics and statistics...")
    metrics_record = calculate_metrics(
        monte_carlo_result=mc_result,
        algorithm_name=algorithm,
        node_count=nx_graph.number_of_nodes(),
        runtime_sec=elapsed_time,
    )

    # Step 5: Data Storage Layer (Module 4)
    if progress_callback:
        progress_callback(0.85, "Persisting simulation outputs...")
    storage = StorageManager()
    try:
        storage.save_raw_simulation(
            algorithm_name=algorithm,
            k=k,
            probability=probability,
            monte_carlo_result=mc_result,
        )
    except Exception as err:
        logger.warning("Storage save warning: %s", err)

    # Step 6: Render Visualizations (Modules 5 & 6)
    if progress_callback:
        progress_callback(0.95, "Generating network and analytics charts...")
    net_factory = NetworkVisualizationFactory()
    pyvis_net = net_factory.render_pyvis(
        graph=nx_graph,
        seed_nodes=seeds,
        influenced_nodes=influenced_nodes,
    )
    plotly_net = net_factory.render_plotly(
        graph=nx_graph,
        seed_nodes=seeds,
        influenced_nodes=influenced_nodes,
    )

    if progress_callback:
        progress_callback(1.0, "Simulation pipeline complete!")

    inf_pct = getattr(metrics_record, "influence_pct", getattr(metrics_record, "influence_percentage", 0.0))
    rt_sec = getattr(metrics_record, "runtime_sec", getattr(metrics_record, "runtime_seconds", 0.0))

    record_dict = {
        "Algorithm": algorithm,
        "Seeds (k)": k,
        "Probability (p)": probability,
        "Simulations (N)": num_simulations,
        "Mean Influence": metrics_record.mean_influence,
        "Influence %": f"{inf_pct:.2f}%",
        "Spread Efficiency": metrics_record.spread_efficiency,
        "Runtime (s)": round(rt_sec, 3),
        "Min Influence": metrics_record.min_influence,
        "Max Influence": metrics_record.max_influence,
        "Std Influence": metrics_record.std_influence,
    }

    return {
        "metrics_record": metrics_record,
        "seeds": seeds,
        "influenced_nodes": influenced_nodes,
        "pyvis_net": pyvis_net,
        "plotly_net": plotly_net,
        "record_dict": record_dict,
        "mc_result": mc_result,
    }
