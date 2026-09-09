"""Sidebar controls component for the Streamlit Dashboard.

Provides user configuration for dataset selection, algorithm, and simulation parameters.
"""

from __future__ import annotations

import streamlit as st

from ..config import (
    ALGORITHMS,
    DATASETS,
    PARAM_BOUNDS,
    RENDERER_CHOICES,
)
from ..state_manager import DashboardStateManager


def render_sidebar() -> dict[str, any]:
    """Render the sidebar controls and return the active parameter values.

    Returns:
        Dictionary of user-selected parameter settings.
    """
    with st.sidebar:
        st.header("⚙️ Simulation Settings")
        st.caption("Configure dataset and parameters per SRS specifications.")

        # 1. Dataset Selection
        dataset_options = list(DATASETS.keys())
        current_dataset = DashboardStateManager.get("selected_dataset", dataset_options[0])
        selected_dataset = st.selectbox(
            "📁 Select Dataset",
            options=dataset_options,
            index=dataset_options.index(current_dataset) if current_dataset in dataset_options else 0,
            help="Choose between real SNAP social networks or synthetic benchmarks.",
        )
        DashboardStateManager.set("selected_dataset", selected_dataset)

        st.divider()

        # 2. Algorithm Selection
        current_algo = DashboardStateManager.get("selected_algorithm", "Degree")
        selected_algo = st.selectbox(
            "🧠 Seed Selection Algorithm",
            options=ALGORITHMS,
            index=ALGORITHMS.index(current_algo) if current_algo in ALGORITHMS else 1,
            help="Select algorithm to identify initial seed influencers.",
        )
        DashboardStateManager.set("selected_algorithm", selected_algo)

        # Optional Greedy Optimization Candidate Pool
        candidate_mode = "Full Graph"
        if selected_algo == "Greedy":
            candidate_mode = st.radio(
                "⚡ Greedy Candidate Pool",
                options=["Top 50 Candidates (Fast)", "Full Graph (Thorough)"],
                index=0,
                help="Restricting candidates to top-degree hubs enables near-instant interactive response.",
            )
            DashboardStateManager.set("candidate_mode", candidate_mode)

        st.divider()

        # 3. Seed Count (k: 1–100)
        current_k = DashboardStateManager.get("k", PARAM_BOUNDS["k"]["default"])
        k = st.slider(
            "🎯 Seed Count (k)",
            min_value=PARAM_BOUNDS["k"]["min"],
            max_value=PARAM_BOUNDS["k"]["max"],
            value=current_k,
            step=PARAM_BOUNDS["k"]["step"],
            help="Budget of seed nodes to activate initially.",
        )
        DashboardStateManager.set("k", k)

        # 4. Propagation Probability (p: 0.01–1.0)
        current_p = DashboardStateManager.get("probability", PARAM_BOUNDS["probability"]["default"])
        p = st.slider(
            "🎲 Propagation Probability (p)",
            min_value=float(PARAM_BOUNDS["probability"]["min"]),
            max_value=float(PARAM_BOUNDS["probability"]["max"]),
            value=float(current_p),
            step=float(PARAM_BOUNDS["probability"]["step"]),
            format="%.2f",
            help="Probability of influence transmission across each social edge.",
        )
        DashboardStateManager.set("probability", p)

        # 5. Simulation Count (N: 10–1000)
        current_n = DashboardStateManager.get("num_simulations", PARAM_BOUNDS["num_simulations"]["default"])
        num_simulations = st.slider(
            "🔄 Monte Carlo Runs (N)",
            min_value=PARAM_BOUNDS["num_simulations"]["min"],
            max_value=PARAM_BOUNDS["num_simulations"]["max"],
            value=current_n,
            step=PARAM_BOUNDS["num_simulations"]["step"],
            help="Number of independent simulation runs for statistical accuracy.",
        )
        DashboardStateManager.set("num_simulations", num_simulations)

        st.divider()

        # Renderer Selection
        current_renderer = DashboardStateManager.get("network_renderer", RENDERER_CHOICES[0])
        renderer = st.selectbox(
            "🖥️ Network Renderer",
            options=RENDERER_CHOICES,
            index=RENDERER_CHOICES.index(current_renderer) if current_renderer in RENDERER_CHOICES else 0,
        )
        DashboardStateManager.set("network_renderer", renderer)

        return {
            "dataset": selected_dataset,
            "algorithm": selected_algo,
            "candidate_mode": candidate_mode,
            "k": k,
            "probability": p,
            "num_simulations": num_simulations,
            "renderer": renderer,
        }
