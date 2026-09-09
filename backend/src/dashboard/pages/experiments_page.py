"""Page 2: Batch Experimentation Suite.

Executes Module 3 automated benchmark experiments:
- Experiment A: Seed Budget (k = 1, 5, 10, 20)
- Experiment B: Probability Sensitivity (p = 0.01, 0.05, 0.10, 0.20)
- Experiment C: Runtime Scalability vs Network Size
- Experiment D: Monte Carlo Convergence
- Run All Benchmark Suite
"""

from __future__ import annotations

import logging
import streamlit as st

from experiments.experiment_runner import (
    run_all_experiments,
    run_convergence_experiment,
    run_probability_experiment,
    run_scalability_experiment,
    run_seed_budget_experiment,
)
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

    # Wrap mock/real simulation functions for experiment framework
    def get_seeds_fn(g, algo, k):
        return select_seeds(g, algo, k)

    def sim_fn(g, s, p):
        count, _ = simulate_single_cascade(g, list(s), p)
        return count

    # Experiment Triggers (SRS Required)
    st.subheader("⚡ Execute Experiments")
    c1, c2, c3, c4, c5 = st.columns(5)

    exp_to_run = None
    with c1:
        if st.button("🎯 Exp A: Seed Budget", use_container_width=True):
            exp_to_run = "A"
    with c2:
        if st.button("🎲 Exp B: Probability", use_container_width=True):
            exp_to_run = "B"
    with c3:
        if st.button("📈 Exp C: Scalability", use_container_width=True):
            exp_to_run = "C"
    with c4:
        if st.button("🔄 Exp D: Convergence", use_container_width=True):
            exp_to_run = "D"
    with c5:
        if st.button("🚀 Run All Experiments", type="primary", use_container_width=True):
            exp_to_run = "ALL"

    stored_exps = DashboardStateManager.get("experiment_results", {})

    if exp_to_run:
        progress_bar = st.progress(0.0, text=f"Executing Experiment {exp_to_run}...")

        try:
            if exp_to_run == "A":
                progress_bar.progress(0.5, "Running Experiment A across k in [1, 5, 10, 20]...")
                res_a = run_seed_budget_experiment(graph, get_seeds_fn=get_seeds_fn, simulate_fn=sim_fn)
                stored_exps["ExpA"] = res_a.results_dataframe
                st.success("✅ Experiment A completed!")

            elif exp_to_run == "B":
                progress_bar.progress(0.5, "Running Experiment B across p in [0.01, 0.05, 0.10, 0.20]...")
                res_b = run_probability_experiment(graph, get_seeds_fn=get_seeds_fn, simulate_fn=sim_fn)
                stored_exps["ExpB"] = res_b.results_dataframe
                st.success("✅ Experiment B completed!")

            elif exp_to_run == "C":
                progress_bar.progress(0.5, "Running Experiment C across network node counts...")
                res_c = run_scalability_experiment(graph, get_seeds_fn=get_seeds_fn, simulate_fn=sim_fn)
                stored_exps["ExpC"] = res_c.results_dataframe
                st.success("✅ Experiment C completed!")

            elif exp_to_run == "D":
                progress_bar.progress(0.5, "Running Experiment D across simulation counts...")
                res_d = run_convergence_experiment(graph, get_seeds_fn=get_seeds_fn, simulate_fn=sim_fn)
                stored_exps["ExpD"] = res_d.results_dataframe
                st.success("✅ Experiment D completed!")

            elif exp_to_run == "ALL":
                progress_bar.progress(0.2, "Executing benchmark suite...")
                suite = run_all_experiments(graph, get_seeds_fn=get_seeds_fn, simulate_fn=sim_fn)
                stored_exps["ExpA"] = suite["expA"].results_dataframe
                stored_exps["ExpB"] = suite["expB"].results_dataframe
                stored_exps["ExpC"] = suite["expC"].results_dataframe
                stored_exps["ExpD"] = suite["expD"].results_dataframe
                st.success("✅ All experiments completed successfully!")

            DashboardStateManager.set("experiment_results", stored_exps)
            progress_bar.empty()

        except Exception as err:
            progress_bar.empty()
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
