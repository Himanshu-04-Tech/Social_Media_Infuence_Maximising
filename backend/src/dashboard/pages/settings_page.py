"""Page 4: Simulator Settings & Customization.

Allows customizing default simulation hyperparameters, visual theme,
chart styling, and primary network rendering engines.
"""

from __future__ import annotations

import streamlit as st

from ..config import PARAM_BOUNDS, RENDERER_CHOICES
from ..state_manager import DashboardStateManager


def render_settings_page(params: dict[str, any]) -> None:
    """Render the Settings and preferences configuration page."""
    st.title("⚙️ Dashboard Settings & Preferences")
    st.caption("Configure global simulator hyperparameters, rendering options, and presentation styles.")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("🎨 Appearance & Styling")

        theme = st.selectbox(
            "Color Theme",
            options=["Light (Default)", "Dark Modern"],
            index=0 if DashboardStateManager.get("theme", "Light") == "Light" else 1,
            help="Select dashboard theme.",
        )
        DashboardStateManager.set("theme", "Light" if "light" in theme.lower() else "Dark")

        chart_style = st.selectbox(
            "Plotly Chart Style Template",
            options=["plotly_white", "plotly_dark", "seaborn", "ggplot2"],
            index=0,
            help="Default styling template applied to all analytical figures.",
        )
        DashboardStateManager.set("chart_style", chart_style)

        network_renderer = st.selectbox(
            "Default Network Renderer",
            options=RENDERER_CHOICES,
            index=0 if "pyvis" in DashboardStateManager.get("network_renderer", "").lower() else 1,
            help="Choose between physics-enabled PyVis or native Plotly 2D canvas.",
        )
        DashboardStateManager.set("network_renderer", network_renderer)

    with c2:
        st.subheader("⚡ Default Simulation Hyperparameters")

        default_k = st.number_input(
            "Default Seed Count (k)",
            min_value=PARAM_BOUNDS["k"]["min"],
            max_value=PARAM_BOUNDS["k"]["max"],
            value=DashboardStateManager.get("k", PARAM_BOUNDS["k"]["default"]),
            step=1,
        )
        DashboardStateManager.set("k", default_k)

        default_p = st.number_input(
            "Default Propagation Probability (p)",
            min_value=float(PARAM_BOUNDS["probability"]["min"]),
            max_value=float(PARAM_BOUNDS["probability"]["max"]),
            value=float(DashboardStateManager.get("probability", PARAM_BOUNDS["probability"]["default"])),
            step=0.01,
            format="%.2f",
        )
        DashboardStateManager.set("probability", default_p)

        default_n = st.number_input(
            "Default Monte Carlo Iterations (N)",
            min_value=PARAM_BOUNDS["num_simulations"]["min"],
            max_value=PARAM_BOUNDS["num_simulations"]["max"],
            value=DashboardStateManager.get("num_simulations", PARAM_BOUNDS["num_simulations"]["default"]),
            step=10,
        )
        DashboardStateManager.set("num_simulations", default_n)

    st.divider()

    col_btn1, col_btn2, _ = st.columns([2, 2, 4])
    with col_btn1:
        if st.button("💾 Save Preferences", type="primary", use_container_width=True):
            st.success("✅ Preferences saved successfully!")
    with col_btn2:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            DashboardStateManager.set("k", PARAM_BOUNDS["k"]["default"])
            DashboardStateManager.set("probability", PARAM_BOUNDS["probability"]["default"])
            DashboardStateManager.set("num_simulations", PARAM_BOUNDS["num_simulations"]["default"])
            DashboardStateManager.set("network_renderer", RENDERER_CHOICES[0])
            st.success("Preferences reset to factory defaults.")
            st.rerun()
