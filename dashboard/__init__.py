"""Streamlit Dashboard Package.

Extends package search path to include backend/src/dashboard.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_SRC = PROJECT_ROOT / "backend" / "src"
BACKEND_DASHBOARD = BACKEND_SRC / "dashboard"

if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

# Extend __path__ so dashboard.components, dashboard.pages, dashboard.config resolve seamlessly
__path__: list[str] = list(__path__) if "__path__" in locals() or "__path__" in globals() else [str(Path(__file__).resolve().parent)]
if str(BACKEND_DASHBOARD) not in __path__:
    __path__.append(str(BACKEND_DASHBOARD))

try:
    from backend.src.dashboard.app import main
    from backend.src.dashboard.state_manager import DashboardStateManager
    from backend.src.dashboard.utilities import load_dataset, run_full_simulation_pipeline
except Exception:
    try:
        from .streamlit_app import main  # type: ignore[no-redef]
        DashboardStateManager = None
        load_dataset = None
        run_full_simulation_pipeline = None
    except Exception:
        main = None  # type: ignore[assignment]
        DashboardStateManager = None
        load_dataset = None
        run_full_simulation_pipeline = None

__all__ = [
    "main",
    "DashboardStateManager",
    "load_dataset",
    "run_full_simulation_pipeline",
]
