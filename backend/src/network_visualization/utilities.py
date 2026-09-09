"""Export and persistence utilities for network visualizations.

Provides file export functions for both PyVis interactive HTML documents
and Plotly vector/raster and interactive HTML representations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pyvis.network import Network
import plotly.graph_objects as go

from .exceptions import NetworkVisualizationValidationError

logger = logging.getLogger(__name__)


def export_html(network: Network, filepath: Path | str) -> Path:
    """Export a PyVis Network object to a standalone interactive HTML file.

    Args:
        network: Configured pyvis.network.Network instance.
        filepath: Destination file path (e.g., 'network_graph.html').

    Returns:
        Path object pointing to the written HTML file.

    Raises:
        NetworkVisualizationValidationError: If network object or filepath is invalid.
    """
    if network is None or not isinstance(network, Network):
        raise NetworkVisualizationValidationError("Expected a valid PyVis Network instance.")

    target_path = Path(filepath)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    network.write_html(str(target_path))
    logger.info("Network Exported: PyVis HTML saved to %s", target_path)
    return target_path


def export_plotly(figure: go.Figure, filepath: Path | str) -> Path:
    """Export a Plotly Figure to an HTML or static image file.

    Supports:
        - '.html': Full interactive standalone web document.
        - '.png', '.jpg', '.jpeg', '.svg': High-resolution static image.

    Args:
        figure: Configured plotly.graph_objects.Figure instance.
        filepath: Destination file path (e.g., 'network_graph.html' or 'network_graph.png').

    Returns:
        Path object pointing to the written file.

    Raises:
        NetworkVisualizationValidationError: If figure is invalid.
    """
    if figure is None or not isinstance(figure, go.Figure):
        raise NetworkVisualizationValidationError("Expected a valid Plotly Figure instance.")

    target_path = Path(filepath)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = target_path.suffix.lower()

    if suffix in (".html", ".htm"):
        figure.write_html(str(target_path))
        logger.info("Network Exported: Plotly HTML saved to %s", target_path)
    elif suffix in (".png", ".jpg", ".jpeg", ".svg", ".webp"):
        try:
            figure.write_image(str(target_path))
            logger.info("Network Exported: Plotly static image saved to %s", target_path)
        except Exception as err:
            fallback_html = target_path.with_suffix(".html")
            figure.write_html(str(fallback_html))
            logger.warning(
                "Static image export failed (%s). Fallback HTML created at: %s",
                err,
                fallback_html,
            )
            return fallback_html
    else:
        # Default fallback to HTML
        target_path = target_path.with_suffix(".html")
        figure.write_html(str(target_path))
        logger.info("Network Exported: Plotly HTML saved to %s", target_path)

    return target_path
