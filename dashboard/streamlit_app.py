"""Streamlit Dashboard application forwarder.

Bridges root-level dashboard execution with the backend dashboard engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure backend/src and project root are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_SRC = PROJECT_ROOT / "backend" / "src"

if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from backend.src.dashboard.app import main
except (ImportError, ModuleNotFoundError):
    try:
        from dashboard.app import main  # type: ignore
    except (ImportError, ModuleNotFoundError):
        from app import main  # type: ignore

__all__ = ["main"]

if __name__ == "__main__":
    if callable(main):
        main()

