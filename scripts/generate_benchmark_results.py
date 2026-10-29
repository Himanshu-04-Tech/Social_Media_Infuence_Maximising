"""Script to generate canonical precomputed benchmark experiment CSVs.

Populates:
- expA_results.csv (Seed Budget Analysis)
- expB_results.csv (Probability Sensitivity)
- expC_results.csv (Network Scalability)
- expD_results.csv (Monte Carlo Convergence)
"""

import sys
import shutil
from pathlib import Path

# Add project root and backend/src to path
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_SRC = ROOT_DIR / "backend" / "src"
for p in (str(ROOT_DIR), str(BACKEND_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
from src import (
    Graph,
    random_selection,
    degree_selection,
    greedy_selection,
    simulate_influence,
)
from dashboard.utilities import select_seeds, simulate_single_cascade
from experiments.experiment_runner import (
    run_seed_budget_experiment,
    run_probability_experiment,
    run_scalability_experiment,
    run_convergence_experiment,
)
from storage.storage_manager import StorageManager

def main():
    print("Loading Facebook SNAP dataset...")
    graph = Graph()
    graph.load_from_edge_list(str(ROOT_DIR / "data" / "facebook_combined.txt"))
    print(f"Loaded graph: {len(graph.nodes())} nodes, {graph._num_edges} edges.")

    algorithms = {
        "Random": lambda g, k: select_seeds(g, "Random", k),
        "Degree": lambda g, k: select_seeds(g, "Degree", k),
        "Greedy": lambda g, k: select_seeds(g, "Greedy", k, candidate_mode="fast"),
    }

    def simulate_fn(g, seeds, p):
        count, _ = simulate_single_cascade(g, list(seeds), p)
        return count

    storage = StorageManager()

    print("Running Experiment A (Seed Budget)...", flush=True)
    df_a = run_seed_budget_experiment(
        graph=graph,
        algorithms=algorithms,
        simulate_fn=simulate_fn,
        k_values=[5, 10, 20, 50],
        probability=0.10,
        num_simulations=10,
        random_seed=42,
    )
    storage.save_experiment_dataframe(df_a, "expA")
    print("Experiment A completed.", flush=True)

    print("Running Experiment B (Probability Sensitivity)...", flush=True)
    df_b = run_probability_experiment(
        graph=graph,
        algorithms=algorithms,
        simulate_fn=simulate_fn,
        p_values=[0.05, 0.10, 0.20, 0.30],
        k=10,
        num_simulations=10,
        random_seed=42,
    )
    storage.save_experiment_dataframe(df_b, "expB")
    print("Experiment B completed.", flush=True)

    print("Running Experiment C (Network Scalability)...", flush=True)
    df_c = run_scalability_experiment(
        algorithms=algorithms,
        simulate_fn=simulate_fn,
        sizes=[500, 1000, 2000, 4000],
        k=10,
        probability=0.10,
        num_simulations=10,
        random_seed=42,
    )
    storage.save_experiment_dataframe(df_c, "expC")
    print("Experiment C completed.", flush=True)

    print("Running Experiment D (Monte Carlo Convergence)...", flush=True)
    df_d = run_convergence_experiment(
        graph=graph,
        algorithms=algorithms,
        simulate_fn=simulate_fn,
        simulation_counts=[10, 50, 100, 500],
        probability=0.10,
        k=10,
        random_seed=42,
    )
    storage.save_experiment_dataframe(df_d, "expD")
    print("Experiment D completed.", flush=True)

    # Mirror results to root results/ directory
    root_results_exp = ROOT_DIR / "results" / "experiments"
    root_results_exp.mkdir(parents=True, exist_ok=True)
    backend_results_exp = ROOT_DIR / "backend" / "results" / "experiments"

    for f in backend_results_exp.glob("*.csv"):
        shutil.copy(f, root_results_exp / f.name)
        print(f"Copied {f.name} to root results/experiments/")

    # Also mirror dataset stats and metrics
    for cat in ["dataset_stats", "metrics", "raw"]:
        src_cat = ROOT_DIR / "backend" / "results" / cat
        dst_cat = ROOT_DIR / "results" / cat
        dst_cat.mkdir(parents=True, exist_ok=True)
        if src_cat.exists():
            for f in src_cat.glob("*.*"):
                shutil.copy(f, dst_cat / f.name)

    print("All benchmark experiments successfully generated and synchronized!")

if __name__ == "__main__":
    main()
