"""Page 2: Batch Experimentation Suite.

Executes Module 3 automated benchmark experiments:
- Experiment A: Seed Budget (k = 5, 10, 20, 50)
- Experiment B: Probability Sensitivity (p = 0.05, 0.10, 0.20, 0.30)
- Experiment C: Runtime Scalability vs Network Size (|V| = 500, 1000, 2000, 4000)
- Experiment D: Monte Carlo Convergence (N = 10, 50, 100, 500)
- Run All Benchmark Suite
"""

from __future__ import annotations

import logging
import pandas as pd
import streamlit as st

from experiments.experiment_runner import (
    run_all_experiments,
    run_convergence_experiment,
    run_probability_experiment,
    run_scalability_experiment,
    run_seed_budget_experiment,
)
from storage.storage_manager import StorageManager
from visualization.chart_builder import VisualizationFactory
from ..state_manager import DashboardStateManager
from ..utilities import load_dataset, select_seeds, simulate_single_cascade

logger = logging.getLogger(__name__)


def render_experiments_page(params: dict[str, any]) -> None:
    """Render the Automated Experiments tab."""
    st.title("🔬 Automated Experimentation Framework")
    st.caption("Execute controlled comparative experiments defined in the SRS and visualize outcomes.")

    dataset_name = params.get("dataset", "SNAP Facebook (4,039 nodes)")
    graph = load_dataset(dataset_name)

    # Standard benchmark algorithm mapping
    algorithms = {
        "Random": lambda g, k: select_seeds(g, "Random", k),
        "Degree": lambda g, k: select_seeds(g, "Degree", k),
        "Greedy": lambda g, k: select_seeds(g, "Greedy", k, candidate_mode="fast"),
    }

    def sim_fn(g: any, s: any, p: float) -> int:
        count, _ = simulate_single_cascade(g, list(s), p)
        return count

    # Experiment Triggers (SRS Required)
    st.subheader("⚡ Execute Experiments")
    c1, c2, c3, c4, c5 = st.columns(5)

    exp_to_run = None
    with c1:
        if st.button("🎯 Exp A: Seed Budget", use_container_width=True) is True:
            exp_to_run = "A"
    with c2:
        if st.button("🎲 Exp B: Probability", use_container_width=True) is True:
            exp_to_run = "B"
    with c3:
        if st.button("📈 Exp C: Scalability", use_container_width=True) is True:
            exp_to_run = "C"
    with c4:
        if st.button("🔄 Exp D: Convergence", use_container_width=True) is True:
            exp_to_run = "D"
    with c5:
        if st.button("🚀 Run All Experiments", type="primary", use_container_width=True) is True:
            exp_to_run = "ALL"

    stored_exps = DashboardStateManager.get("experiment_results", {})
    storage = StorageManager()

    if exp_to_run:
        progress_bar = st.progress(0.0, text=f"Executing Experiment {exp_to_run}...")

        try:
            if exp_to_run == "A":
                progress_bar.progress(0.3, "Running Experiment A across k in [5, 10, 20, 50]...")
                df_a = run_seed_budget_experiment(
                    graph=graph,
                    algorithms=algorithms,
                    simulate_fn=sim_fn,
                    k_values=[5, 10, 20, 50],
                    probability=0.10,
                    num_simulations=min(params.get("num_simulations", 50), 30),
                )
                stored_exps["ExpA"] = df_a
                storage.save_experiment_dataframe(df_a, "expA")
                st.success("✅ Experiment A completed and saved to results/experiments/!")

            elif exp_to_run == "B":
                progress_bar.progress(0.3, "Running Experiment B across p in [0.05, 0.10, 0.20, 0.30]...")
                df_b = run_probability_experiment(
                    graph=graph,
                    algorithms=algorithms,
                    simulate_fn=sim_fn,
                    p_values=[0.05, 0.10, 0.20, 0.30],
                    k=params.get("k", 10),
                    num_simulations=min(params.get("num_simulations", 50), 30),
                )
                stored_exps["ExpB"] = df_b
                storage.save_experiment_dataframe(df_b, "expB")
                st.success("✅ Experiment B completed and saved to results/experiments/!")

            elif exp_to_run == "C":
                progress_bar.progress(0.3, "Running Experiment C across network node counts [500, 1000, 2000, 4000]...")
                df_c = run_scalability_experiment(
                    algorithms=algorithms,
                    simulate_fn=sim_fn,
                    sizes=[500, 1000, 2000, 4000],
                    k=params.get("k", 10),
                    probability=params.get("probability", 0.10),
                    num_simulations=min(params.get("num_simulations", 50), 20),
                )
                stored_exps["ExpC"] = df_c
                storage.save_experiment_dataframe(df_c, "expC")
                st.success("✅ Experiment C completed and saved to results/experiments/!")

            elif exp_to_run == "D":
                progress_bar.progress(0.3, "Running Experiment D across simulation counts [10, 50, 100, 500]...")
                df_d = run_convergence_experiment(
                    graph=graph,
                    algorithms=algorithms,
                    simulate_fn=sim_fn,
                    simulation_counts=[10, 50, 100, 500],
                    probability=params.get("probability", 0.10),
                    k=params.get("k", 10),
                )
                stored_exps["ExpD"] = df_d
                storage.save_experiment_dataframe(df_d, "expD")
                st.success("✅ Experiment D completed and saved to results/experiments/!")

            elif exp_to_run == "ALL":
                progress_bar.progress(0.2, "Executing benchmark suite...")
                suite = run_all_experiments(
                    graph=graph,
                    algorithms=algorithms,
                    simulate_influence=sim_fn,
                    num_simulations=min(params.get("num_simulations", 50), 20),
                )
                if "experiment_a" in suite:
                    stored_exps["ExpA"] = suite["experiment_a"]
                    storage.save_experiment_dataframe(suite["experiment_a"], "expA")
                if "experiment_b" in suite:
                    stored_exps["ExpB"] = suite["experiment_b"]
                    storage.save_experiment_dataframe(suite["experiment_b"], "expB")
                if "experiment_c" in suite:
                    stored_exps["ExpC"] = suite["experiment_c"]
                    storage.save_experiment_dataframe(suite["experiment_c"], "expC")
                if "experiment_d" in suite:
                    stored_exps["ExpD"] = suite["experiment_d"]
                    storage.save_experiment_dataframe(suite["experiment_d"], "expD")
                st.success("✅ All benchmark experiments completed and saved successfully!")

            DashboardStateManager.set("experiment_results", stored_exps)
            progress_bar.empty()

        except Exception as err:
            progress_bar.empty()
            logger.error("Experiment execution failure: %s", err, exc_info=True)
            st.error(f"❌ Experiment execution error: {err}")

    # Display Active Experiment Results
    st.divider()
    st.subheader("📊 Experiment Results & Visualizations")

    if not stored_exps:
        st.info("No experiments executed yet. Click one of the buttons above to run.")
        return

    exp_tabs = st.tabs(list(stored_exps.keys()))
    for idx, (exp_key, df) in enumerate(stored_exps.items()):
        with exp_tabs[idx]:
            st.write(f"**Experiment Dataset: {exp_key} ({len(df)} records)**")
            st.dataframe(df, use_container_width=True)

            # Render Matching Module 5 Chart
            try:
                if exp_key == "ExpA":
                    fig = VisualizationFactory.create_seed_budget_chart(df)
                    st.plotly_chart(fig, use_container_width=True)
                elif exp_key == "ExpB":
                    fig = VisualizationFactory.create_probability_sensitivity_chart(df)
                    st.plotly_chart(fig, use_container_width=True)
                elif exp_key == "ExpC":
                    fig = VisualizationFactory.create_runtime_scalability_chart(df, log_scale=True)
                    st.plotly_chart(fig, use_container_width=True)
                elif exp_key == "ExpD":
                    fig = VisualizationFactory.create_convergence_chart(df)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as err:
                logger.debug("Chart render note: %s", err)

            csv_content = df.to_csv(index=False)
            st.download_button(
                label=f"📥 Download {exp_key} CSV",
                data=csv_content,
                file_name=f"{exp_key.lower()}_results.csv",
                mime="text/csv",
                key=f"dl_{exp_key}",
            )
