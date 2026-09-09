"""Metrics panel component for the Streamlit Dashboard.

Displays statistical summary cards from Module 2 (Metrics Engine) output.
"""

from __future__ import annotations

import streamlit as st

from metrics.models import MetricsRecord


def render_metrics_panel(metrics: MetricsRecord | None) -> None:
    """Render the 7 KPI metric cards required by SRS.

    Attributes displayed:
        - Mean Influence
        - Min Influence
        - Max Influence
        - Std Influence
        - Influence %
        - Spread Efficiency
        - Runtime
    """
    st.subheader("🎯 Influence Performance Metrics")

    if metrics is None:
        st.info("Run a simulation to generate statistical performance metrics.")
        return

    # Extract fields with graceful attribute fallback
    inf_pct = getattr(metrics, "influence_pct", getattr(metrics, "influence_percentage", 0.0))
    rt_sec = getattr(metrics, "runtime_sec", getattr(metrics, "runtime_seconds", 0.0))

    # Row 1: Statistical Spread Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="📈 Mean Influence",
            value=f"{metrics.mean_influence:.1f}",
            help="Expected number of influenced nodes across Monte Carlo simulations.",
        )
    with c2:
        st.metric(
            label="📉 Min Influence",
            value=f"{metrics.min_influence}",
            help="Minimum influenced nodes observed in any single simulation run.",
        )
    with c3:
        st.metric(
            label="🚀 Max Influence",
            value=f"{metrics.max_influence}",
            help="Maximum influenced nodes observed in any single simulation run.",
        )
    with c4:
        st.metric(
            label="📏 Std Deviation",
            value=f"±{metrics.std_influence:.2f}",
            help="Sample standard deviation across simulation runs.",
        )

    # Row 2: Efficiency and Computational Runtime
    c5, c6, c7 = st.columns(3)
    with c5:
        st.metric(
            label="🌐 Influence %",
            value=f"{inf_pct:.2f}%",
            help="Percentage of the total network reached by the cascade.",
        )
    with c6:
        st.metric(
            label="⚡ Spread Efficiency",
            value=f"{metrics.spread_efficiency:.2f}",
            help="Ratio of influenced nodes to initial seed budget (Mean / k).",
        )
    with c7:
        st.metric(
            label="⏱️ Execution Runtime",
            value=f"{rt_sec:.3f} s",
            help="Total computation time for seed selection and Monte Carlo simulations.",
        )
