"""Streamlit Dashboard Entry Point."""

from __future__ import annotations

import sys
from pathlib import Path

# Set up module resolution paths
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_SRC = ROOT_DIR / "backend" / "src"

# Ensure backend/src and project root are at the head of sys.path
for p in (str(ROOT_DIR), str(BACKEND_SRC)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

try:
    from dashboard.streamlit_app import main
except (ImportError, ModuleNotFoundError):
    from backend.src.dashboard.app import main

if __name__ == "__main__":
    main()
