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
if str(BACKEND_DASHBOARD) not in __path__:
    __path__.append(str(BACKEND_DASHBOARD))

try:
    from backend.src.dashboard.app import main
except Exception:
    try:
        from .streamlit_app import main
    except Exception:
        main = None

__all__ = ["main"]
