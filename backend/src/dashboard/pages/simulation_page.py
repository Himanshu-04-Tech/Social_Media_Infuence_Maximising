"""Page 1: Single Simulation runner.

Executes a single controlled simulation workflow:
Role 1 -> Module 1 -> Module 2 -> Module 4 -> Module 5 & 6.
"""

from __future__ import annotations

import logging
import streamlit as st

from ..components.dataset_stats import render_dataset_stats
from ..components.metrics_panel import render_metrics_panel
from ..components.network_panel import render_network_panel
from ..components.charts_panel import render_charts_panel
from ..components.results_table import render_results_table
from ..state_manager import DashboardStateManager
from ..utilities import load_dataset, run_full_simulation_pipeline

logger = logging.getLogger(__name__)


def render_simulation_page(params: dict[str, any]) -> None:
    """Render the Single Simulation page."""
    st.title("🎯 Single Simulation Engine")
    st.caption("Configure parameters, execute Monte Carlo diffusion, and inspect network dynamics.")

    dataset_name = params.get("dataset", "SNAP Facebook (4,039 nodes)")

    # 1. Load Dataset
    try:
        graph = load_dataset(dataset_name)
    except Exception as err:
        st.error(f"❌ Failed to load dataset '{dataset_name}': {err}")
        return

    # 2. Dataset Statistics Panel (SRS Required)
    render_dataset_stats(graph)

    st.divider()

    # 3. Run Controls (SRS Required)
    st.subheader("▶️ Simulation Controls")
    c_btn1, c_btn2, _ = st.columns([2, 1, 4])

    with c_btn1:
        run_clicked = st.button("🚀 Run Simulation", type="primary", use_container_width=True)
    with c_btn2:
        reset_clicked = st.button("🔄 Reset", use_container_width=True)

    if reset_clicked:
        DashboardStateManager.reset()
        st.success("Session state reset.")
        st.rerun()

    progress_placeholder = st.empty()

    if run_clicked:
        # Validate inputs before running
        k = params["k"]
        p = params["probability"]
        n_sims = params["num_simulations"]

        if k <= 0:
            st.error("❌ Seed Count (k) must be greater than 0.")
            return
        if not (0.0 < p <= 1.0):
            st.error(f"❌ Propagation Probability (p) must be in (0, 1]. Got {p}.")
            return
        if n_sims <= 0:
            st.error(f"❌ Monte Carlo Runs (N) must be greater than 0. Got {n_sims}.")
            return

        progress_bar = progress_placeholder.progress(0.0, text="Initializing simulation pipeline...")

        def on_progress(fraction: float, message: str) -> None:
            progress_bar.progress(fraction, text=message)

        try:
            results = run_full_simulation_pipeline(
                graph=graph,
                algorithm=params["algorithm"],
                k=k,
                probability=p,
                num_simulations=n_sims,
                candidate_mode=params.get("candidate_mode", "Top 50 Candidates (Fast)"),
                progress_callback=on_progress,
            )

            # Update State
            DashboardStateManager.set("latest_metrics", results["metrics_record"])
            DashboardStateManager.set("latest_seeds", results["seeds"])
            DashboardStateManager.set("latest_influenced", results["influenced_nodes"])
            DashboardStateManager.add_simulation_result(results["record_dict"])

            progress_placeholder.empty()
            st.success(f"✅ Simulation for '{params['algorithm']}' completed successfully!")

        except Exception as err:
            progress_placeholder.empty()
            logger.error("Simulation pipeline error: %s", err, exc_info=True)
            st.error(f"❌ Simulation Failed: {err}")
            return

    # 4. Metrics Panel (SRS Required)
    current_metrics = DashboardStateManager.get("latest_metrics")
    render_metrics_panel(current_metrics)

    st.divider()

    # 5. Network Visualization Panel (SRS Required)
    current_seeds = DashboardStateManager.get("latest_seeds", [])
    current_influenced = DashboardStateManager.get("latest_influenced", set())
    render_network_panel(
        graph=graph,
        seed_nodes=current_seeds,
        influenced_nodes=current_influenced,
        preferred_renderer=params.get("renderer", "PyVis (Interactive Physics)"),
    )

    st.divider()

    # 6. Charts Panel (Module 5 Visualizations)
    results_df = DashboardStateManager.get("results_dataframe")
    render_charts_panel(results_df)

    st.divider()

    # 7. Results Table & Export Controls
    render_results_table(results_df)
