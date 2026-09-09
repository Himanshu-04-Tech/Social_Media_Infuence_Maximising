"""Data models and configuration containers for the Visualization Engine.

Defines ChartConfig specifying dimensions, titles, labels, and rendering themes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChartConfig:
    """Configuration descriptor for chart appearance and formatting.

    Attributes:
        title: Chart header title.
        x_label: Horizontal axis title.
        y_label: Vertical axis title.
        width: Canvas width in pixels (default: 1200).
        height: Canvas height in pixels (default: 700).
        template: Plotly theme template (default: 'plotly_white').
    """

    title: str
    x_label: str
    y_label: str
    width: int = 1200
    height: int = 700
    template: str = "plotly_white"
