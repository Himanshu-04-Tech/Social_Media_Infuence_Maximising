"""Streamlit Dashboard application runner forwarder."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure backend/src and project root are in sys.path
SRC_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = SRC_DIR.parent
PROJECT_ROOT = ROOT_DIR.parent
for p in (str(SRC_DIR), str(ROOT_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from backend.src.dashboard.app import main
except (ImportError, ModuleNotFoundError):
    try:
        from dashboard.app import main  # type: ignore
    except (ImportError, ModuleNotFoundError):
        from app import main  # type: ignore

__all__ = ["main"]

if __name__ == "__main__":
    main()

