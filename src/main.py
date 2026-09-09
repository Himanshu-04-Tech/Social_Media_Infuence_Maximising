"""
src/main.py

Main entry point, Command-Line Interface (CLI), and programmatic API for Influence Maximization.
Acts as the clean architectural bridge between the backend algorithm engine and Himanshu's frontend.

Authors:
    Zaid Hussain - Graph & Algorithm Engineer
    Himanshu     - Frontend, Visualization & Integration Engineer
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional

from src.graph import Graph
from src.metrics import compare_algorithms, run_experiment


# =====================================================================
# Programmatic API for Himanshu's Frontend
# =====================================================================

def run_algorithm_api(
    algorithm: str,
    k: int = 10,
    p: float = 0.05,
    simulations: int = 100,
    candidate_nodes: Optional[Iterable[int]] = None,
    master_seed: Optional[int] = 42,
    dataset_path: str = "data/facebook_combined.txt",
) -> Dict[str, Any]:
    """
    Programmatic API endpoint for running a single algorithm.
    Returns pure JSON-serializable dictionary with seeds, spread, and runtime metrics.
    """
    graph = Graph()
    graph.load_from_edge_list(dataset_path)
    return run_experiment(
        algorithm=algorithm,
        graph=graph,
        k=k,
        p=p,
        simulations=simulations,
        candidate_nodes=candidate_nodes,
        master_seed=master_seed,
    )


def compare_algorithms_api(
    k: int = 10,
    p: float = 0.05,
    simulations: int = 100,
    candidate_nodes: Optional[Iterable[int]] = None,
    master_seed: Optional[int] = 42,
    dataset_path: str = "data/facebook_combined.txt",
) -> Dict[str, Any]:
    """
    Programmatic API endpoint for running a full 3-way algorithm comparison.
    Returns pure JSON-serializable dictionary formatted for frontend charts and tables.
    """
    graph = Graph()
    graph.load_from_edge_list(dataset_path)
    return compare_algorithms(
        graph=graph,
        k=k,
        p=p,
        simulations=simulations,
        candidate_nodes=candidate_nodes,
        master_seed=master_seed,
    )


# =====================================================================
# CLI Formatting Utilities
# =====================================================================

def _format_comparison_table(report: Dict[str, Any]) -> str:
    """Renders a clean human-readable ASCII/Markdown report table."""
    cfg = report["configuration"]
    results = report["results"]

    pool_desc = (
        f"Restricted ({cfg['candidate_count']} candidates)"
        if cfg["is_restricted_candidate_pool"]
        else f"Full Graph ({cfg['graph_nodes']:,} nodes)"
    )

    lines = [
        "=" * 106,
        "INFLUENCE MAXIMIZATION BENCHMARK REPORT (DAA PROJECT)",
        "=" * 106,
        f"Graph: {cfg['graph_nodes']:,} nodes, {cfg['graph_edges']:,} edges  |  k = {cfg['k']}  |  p = {cfg['probability']}  |  Simulations (R) = {cfg['simulations']}",
        f"Candidate Pool: {pool_desc}  |  Master Seed: {cfg['master_seed']}",
        "-" * 106,
        f"{'Algorithm':<12} {'Selected Seeds':<32} {'Avg Spread (± std)':<22} {'Select Time':<14} {'Eval Time':<12} {'Total Time':<12}",
        "-" * 106,
    ]

    for res in results:
        alg_name = res["algorithm"].capitalize()
        seeds_str = str(res["seed_nodes"])
        if len(seeds_str) > 30:
            seeds_str = seeds_str[:27] + "..."

        spread_str = f"{res['influence_spread']:.2f} (±{res['spread_stats']['std_dev']:.2f})"
        sel_time_str = f"{res['selection_runtime_seconds']:.4f}s"
        eval_time_str = f"{res['evaluation_runtime_seconds']:.4f}s"
        tot_time_str = f"{res['total_runtime_seconds']:.4f}s"

        lines.append(
            f"{alg_name:<12} {seeds_str:<32} {spread_str:<22} {sel_time_str:<14} {eval_time_str:<12} {tot_time_str:<12}"
        )

    lines.append("-" * 106)

    # Display Greedy marginal gains if present
    for res in results:
        if res["algorithm"] == "greedy" and res.get("marginal_gains"):
            gains_formatted = [round(g, 2) for g in res["marginal_gains"]]
            lines.append(f"Greedy Marginal Gains per Step: {gains_formatted}")

    lines.append("=" * 106)
    return "\n".join(lines)


# =====================================================================
# CLI Entry Point
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Influence Maximization in Social Networks — DAA Algorithm Engine"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/facebook_combined.txt",
        help="Path to edge list dataset file (default: data/facebook_combined.txt)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of seed nodes to select (default: 10)",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.05,
        help="Propagation probability per edge (default: 0.05)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=100,
        help="Number of Monte Carlo simulations R (default: 100)",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=None,
        help="Optional top-N degree candidate restriction for fast demo mode (e.g. --candidates 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="all",
        choices=["all", "random", "degree", "greedy"],
        help="Algorithm to run ('all' for 3-way comparison, or individual)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON format to stdout for frontend consumption",
    )

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: Dataset file not found at '{args.data}'", file=sys.stderr)
        sys.exit(1)

    # 1. Load Graph
    graph = Graph()
    graph.load_from_edge_list(args.data)

    # 2. Determine Candidate Pool
    candidate_nodes = None
    if args.candidates is not None:
        if args.candidates < args.k:
            print(
                f"Error: --candidates ({args.candidates}) cannot be less than --k ({args.k})",
                file=sys.stderr,
            )
            sys.exit(1)
        ranked = graph.degree_ranking()
        candidate_nodes = [node for node, _ in ranked[: args.candidates]]

    # 3. Execute
    if args.algorithm == "all":
        report = compare_algorithms(
            graph=graph,
            k=args.k,
            p=args.p,
            simulations=args.simulations,
            candidate_nodes=candidate_nodes,
            master_seed=args.seed,
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(_format_comparison_table(report))
    else:
        result = run_experiment(
            algorithm=args.algorithm,
            graph=graph,
            k=args.k,
            p=args.p,
            simulations=args.simulations,
            candidate_nodes=candidate_nodes,
            master_seed=args.seed,
        )
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            # Wrap in mini-table
            wrapped = {
                "configuration": {
                    "k": args.k,
                    "probability": args.p,
                    "simulations": args.simulations,
                    "candidate_count": result["candidate_count"],
                    "is_restricted_candidate_pool": result["is_restricted_candidate_pool"],
                    "master_seed": args.seed,
                    "graph_nodes": graph.number_of_nodes(),
                    "graph_edges": graph.number_of_edges(),
                },
                "results": [result],
            }
            print(_format_comparison_table(wrapped))


if __name__ == "__main__":
    main()
