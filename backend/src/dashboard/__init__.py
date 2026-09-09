"""Dashboard package for Social Network Influence Maximization Simulator."""

from __future__ import annotations

from .app import main
from .state_manager import DashboardStateManager
from .utilities import load_dataset, run_full_simulation_pipeline

__all__ = [
    "main",
    "DashboardStateManager",
    "load_dataset",
    "run_full_simulation_pipeline",
]
