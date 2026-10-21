"""Main application entry point for the Streamlit Dashboard.

Social Network Influence Maximization Simulator (Role 2)
Authors: Himanshu (Role 2) & Zaid Hussain (Role 1)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure backend/src is available in sys.path
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Ensure root dir and project root are available for Role 1 and package imports
ROOT_DIR = SRC_DIR.parent
PROJECT_ROOT = ROOT_DIR.parent
for p in (str(SRC_DIR), str(ROOT_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st

try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except ImportError:
    HAS_OPTION_MENU = False

from dashboard.components.sidebar import render_sidebar
from dashboard.config import LAYOUT, PAGE_ICON, PAGE_TITLE
from dashboard.pages.experiments_page import render_experiments_page
from dashboard.pages.results_page import render_results_page
from dashboard.pages.settings_page import render_settings_page
from dashboard.pages.simulation_page import render_simulation_page
from dashboard.state_manager import DashboardStateManager


def main() -> None:
    """Entry point for the Streamlit application."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=LAYOUT,
        initial_sidebar_state="expanded",
    )

    # Initialize State
    DashboardStateManager.initialize()

    # Render Sidebar Controls
    params = render_sidebar()

    # Top Navigation Menu
    pages = ["Single Simulation", "Experiments", "Stored Results", "Settings"]
    icons = ["play-circle-fill", "graph-up", "database-fill", "gear-fill"]

    if HAS_OPTION_MENU:
        selected_page = option_menu(
            menu_title=None,
            options=pages,
            icons=icons,
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#FAFAFA", "border-radius": "8px"},
                "icon": {"color": "#3B82F6", "font-size": "18px"},
                "nav-link": {
                    "font-size": "15px",
                    "text-align": "center",
                    "margin": "0px",
                    "--hover-color": "#EFF6FF",
                },
                "nav-link-selected": {"background-color": "#2563EB", "color": "white", "font-weight": "600"},
            },
        )
    else:
        selected_page = st.radio("Navigation", options=pages, horizontal=True, label_visibility="collapsed")

    # Route to Selected Page
    if selected_page == "Single Simulation":
        render_simulation_page(params)
    elif selected_page == "Experiments":
        render_experiments_page(params)
    elif selected_page == "Stored Results":
        render_results_page(params)
    elif selected_page == "Settings":
        render_settings_page(params)


if __name__ == "__main__":
    main()
