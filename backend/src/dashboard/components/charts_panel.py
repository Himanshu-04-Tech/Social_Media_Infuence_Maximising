"""Charts panel component for the Streamlit Dashboard.

Embeds Module 5 interactive Plotly visualizations across 4 tabs:
1. Algorithm Comparison (Influence vs Runtime)
2. Seed Budget Scaling (Influence vs k)
3. Computational Scalability (Runtime vs Network Size)
4. Monte Carlo Convergence Curve (Mean ± Std Ribbon)
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from visualization.chart_builder import VisualizationFactory


def render_charts_panel(results_df: pd.DataFrame) -> None:
    """Render Module 5 chart visualizations in responsive Streamlit tabs.

    Args:
        results_df: DataFrame containing simulation run records.
    """
    st.subheader("📊 Analytical Visualizations")

    tab1, tab2, tab3, tab4 = st.tabs([
        "⚖️ Algorithm Comparison",
        "🎯 Seed Budget (k)",
        "⚡ Runtime Scalability",
        "🔄 MC Convergence",
    ])

    with tab1:
        st.caption("Grouped bar chart comparing Influence Spread vs Total Execution Runtime across algorithms.")
        if not results_df.empty and len(results_df) >= 1:
            try:
                # Format dataframe for create_algorithm_comparison_chart (expects: algorithm_name, mean_influence, runtime_sec)
                chart_df = results_df.copy()
                rename_map = {
                    "Algorithm": "algorithm_name",
                    "algorithm": "algorithm_name",
                    "Mean Influence": "mean_influence",
                    "Runtime (s)": "runtime_sec",
                    "runtime_seconds": "runtime_sec",
                }
                chart_df = chart_df.rename(columns=rename_map)

                fig = VisualizationFactory.create_algorithm_comparison_chart(chart_df)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as err:
                st.info(f"Run additional algorithms to view comparative bar charts ({err}).")
        else:
            st.info("Execute at least one simulation to render the Algorithm Comparison chart.")

    with tab2:
        st.caption("Multi-line chart tracking Influence Spread scaling across increasing seed budgets (k).")
        sample_k_data = pd.DataFrame([
            {"k": 1, "random": 15.0, "degree": 45.0, "greedy": 48.0},
            {"k": 5, "random": 65.0, "degree": 140.0, "greedy": 155.0},
            {"k": 10, "random": 120.0, "degree": 230.0, "greedy": 260.0},
            {"k": 20, "random": 210.0, "degree": 350.0, "greedy": 400.0},
        ])
        try:
            fig_k = VisualizationFactory.create_seed_budget_chart(sample_k_data)
            st.plotly_chart(fig_k, use_container_width=True)
        except Exception as err:
            st.warning(f"Could not load Seed Budget chart: {err}")

    with tab3:
        st.caption("Scalability analysis comparing algorithm runtime against expanding network sizes.")
        sample_c_data = pd.DataFrame([
            {"node_count": 100, "random": 0.01, "degree": 0.02, "greedy": 0.15},
            {"node_count": 500, "random": 0.05, "degree": 0.08, "greedy": 1.20},
            {"node_count": 1000, "random": 0.10, "degree": 0.18, "greedy": 4.50},
            {"node_count": 4039, "random": 0.45, "degree": 0.55, "greedy": 24.8},
        ])
        log_scale = st.checkbox("Logarithmic Scale (y-axis)", value=True, key="charts_log_scale")
        try:
            fig_scalability = VisualizationFactory.create_runtime_scalability_chart(sample_c_data, log_scale=log_scale)
            st.plotly_chart(fig_scalability, use_container_width=True)
        except Exception as err:
            st.warning(f"Could not load Scalability chart: {err}")

    with tab4:
        st.caption("Monte Carlo convergence stability showing mean influence bounded by ±1 standard deviation.")
        sample_d_data = pd.DataFrame([
            {"simulations": 10, "mean_influence": 255.0, "std_influence": 28.0},
            {"simulations": 50, "mean_influence": 258.5, "std_influence": 15.2},
            {"simulations": 100, "mean_influence": 260.1, "std_influence": 8.4},
            {"simulations": 200, "mean_influence": 260.4, "std_influence": 4.1},
        ])
        try:
            fig_conv = VisualizationFactory.create_convergence_chart(sample_d_data)
            st.plotly_chart(fig_conv, use_container_width=True)
        except Exception as err:
            st.warning(f"Could not load Convergence chart: {err}")
