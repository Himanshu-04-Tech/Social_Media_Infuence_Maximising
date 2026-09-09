"""Configuration constants and default parameters for the Streamlit Dashboard.

Defines supported datasets, algorithms, simulation ranges, and visualization defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR.parent / "data" if (BASE_DIR.parent / "data").exists() else BASE_DIR / "data"

# Supported Datasets
DATASETS: dict[str, dict[str, Any]] = {
    "SNAP Facebook (4,039 nodes)": {
        "path": DATA_DIR / "facebook_combined.txt",
        "description": "Real Stanford Network Analysis Project (SNAP) Facebook social network.",
        "type": "file",
    },
    "Zachary's Karate Club (34 nodes)": {
        "path": None,
        "description": "Classic benchmark social network of 34 club members.",
        "type": "builtin_karate",
    },
    "Synthetic Scale-Free (300 nodes)": {
        "path": None,
        "description": "Synthetic Barabási–Albert scale-free network modeling preferential attachment.",
        "type": "synthetic_ba",
    },
}

DEFAULT_DATASET = "SNAP Facebook (4,039 nodes)"

# Supported Algorithms
ALGORITHMS: list[str] = [
    "Random",
    "Degree",
    "Greedy",
    "BFS",
    "IC",
]

DEFAULT_ALGORITHM = "Degree"

# Simulation Parameter Bounds (SRS Specified)
PARAM_BOUNDS = {
    "k": {"min": 1, "max": 100, "default": 10, "step": 1},
    "probability": {"min": 0.01, "max": 1.0, "default": 0.10, "step": 0.01},
    "num_simulations": {"min": 10, "max": 1000, "default": 100, "step": 10},
}

# Network Renderer Choices
RENDERER_CHOICES = [
    "PyVis (Interactive Physics)",
    "Plotly (Streamlit Native)",
]
DEFAULT_RENDERER = "PyVis (Interactive Physics)"

# Theme & Styling
PAGE_TITLE = "Social Network Influence Maximization Simulator"
PAGE_ICON = "🌐"
LAYOUT = "wide"
