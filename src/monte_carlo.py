"""src/monte_carlo.py

Root-level forwarder for the Monte Carlo Simulation Engine.
Bridges direct imports to backend.src.monte_carlo.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure backend/src is on sys.path
_BACKEND_SRC = Path(__file__).resolve().parent.parent / "backend" / "src"
if str(_BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(_BACKEND_SRC))

from monte_carlo.engine import run_monte_carlo
from monte_carlo.exceptions import (
    MonteCarloError,
    MonteCarloValidationError,
    SimulationExecutionError,
)
from monte_carlo.models import MonteCarloResult
from monte_carlo.validators import validate_monte_carlo_inputs

__all__ = [
    "run_monte_carlo",
    "MonteCarloResult",
    "MonteCarloError",
    "MonteCarloValidationError",
    "SimulationExecutionError",
    "validate_monte_carlo_inputs",
]
