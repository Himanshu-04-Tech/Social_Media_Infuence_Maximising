"""Session state manager for the Streamlit Dashboard.

Encapsulates all session state reads, writes, and resets, providing a clean
facade and preventing state leakage or untyped dictionary errors.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import streamlit as st

from .config import (
    DEFAULT_ALGORITHM,
    DEFAULT_DATASET,
    DEFAULT_RENDERER,
    PARAM_BOUNDS,
)

logger = logging.getLogger(__name__)


class DashboardStateManager:
    """Central manager for Streamlit session state."""

    @staticmethod
    def initialize() -> None:
        """Initialize all session state variables with default values."""
        defaults: dict[str, Any] = {
            "selected_dataset": DEFAULT_DATASET,
            "selected_algorithm": DEFAULT_ALGORITHM,
            "k": PARAM_BOUNDS["k"]["default"],
            "probability": PARAM_BOUNDS["probability"]["default"],
            "num_simulations": PARAM_BOUNDS["num_simulations"]["default"],
            "network_renderer": DEFAULT_RENDERER,
            "candidate_mode": "Top 50 Candidates (Fast)",
            "latest_metrics": None,
            "latest_seeds": [],
            "latest_influenced": [],
            "history_records": [],
            "results_dataframe": pd.DataFrame(),
            "latest_network_html": None,
            "latest_plotly_fig": None,
            "experiment_results": {},
            "theme": "Light",
            "chart_style": "Plotly White",
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """Get a value from session state with optional fallback."""
        return st.session_state.get(key, default)

    @staticmethod
    def set(key: str, value: Any) -> None:
        """Set a value in session state."""
        st.session_state[key] = value

    @classmethod
    def add_simulation_result(cls, record_dict: dict[str, Any]) -> None:
        """Append a newly completed simulation run to the cumulative results table."""
        history = cls.get("history_records", [])
        history.append(record_dict)
        cls.set("history_records", history)
        cls.set("results_dataframe", pd.DataFrame(history))

    @classmethod
    def reset(cls) -> None:
        """Reset simulation results while keeping user configurations."""
        cls.set("latest_metrics", None)
        cls.set("latest_seeds", [])
        cls.set("latest_influenced", [])
        cls.set("history_records", [])
        cls.set("results_dataframe", pd.DataFrame())
        cls.set("latest_network_html", None)
        cls.set("latest_plotly_fig", None)
        logger.info("Dashboard state reset by user.")
