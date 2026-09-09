"""Utility functions for dataset summary cards and chart export.

Provides summary metrics computation and multi-format figure export handlers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .exceptions import VisualizationError

logger = logging.getLogger(__name__)


def create_dataset_summary(
    node_count: int,
    edge_count: int,
) -> dict[str, Any]:
    """Compute network topology summary metrics for Streamlit KPI cards.

    Formula:
        avg_degree = (2 * edge_count) / node_count (for undirected graphs)

    Args:
        node_count: Total number of network vertices.
        edge_count: Total number of network edges.

    Returns:
        Dictionary containing:
            - 'node_count': int
            - 'edge_count': int
            - 'avg_degree': float (rounded to 2 decimal places)
    """
    n = int(node_count)
    e = int(edge_count)

    if n > 0:
        avg_degree = round((2.0 * e) / n, 2)
    else:
        avg_degree = 0.0

    return {
        "node_count": n,
        "edge_count": e,
        "avg_degree": avg_degree,
    }


def export_plotly_chart(
    figure: Any,
    filepath: Path | str,
) -> Path:
    """Export a Plotly Figure to HTML, PNG, or SVG format.

    Args:
        figure: Plotly graph_objects.Figure instance.
        filepath: Target output file path (.html, .png, .svg).

    Returns:
        Resolved Path of the saved export.

    Raises:
        VisualizationError: If export format is unsupported or write fails.
    """
    target = Path(filepath).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix.lower()

    try:
        if suffix == ".html":
            figure.write_html(str(target), include_plotlyjs="cdn")
        elif suffix in [".png", ".svg", ".jpeg", ".webp"]:
            try:
                figure.write_image(str(target))
            except (ValueError, RuntimeError) as err:
                # If image rendering engine (Kaleido) is missing, write HTML with message
                logger.warning(
                    "Image engine unavailable (%s); saving interactive HTML alongside: %s",
                    err,
                    target.with_suffix(".html"),
                )
                html_path = target.with_suffix(".html")
                figure.write_html(str(html_path), include_plotlyjs="cdn")
                return html_path
        else:
            raise VisualizationError(
                f"Unsupported Plotly export format '{suffix}'. Supported formats: .html, .png, .svg"
            )

        logger.info(
            "Chart Exported | Format=%s | Path=%s",
            suffix,
            str(target),
            extra={"event": "Chart Exported", "format": suffix, "path": str(target)},
        )
        return target
    except Exception as err:
        logger.error("Failed to export Plotly chart to %s: %s", str(target), err)
        raise VisualizationError(f"Plotly export failed: {err}") from err


def export_matplotlib_chart(
    figure: Any,
    filepath: Path | str,
) -> Path:
    """Export a Matplotlib Figure to PNG or PDF format.

    Args:
        figure: Matplotlib Figure instance.
        filepath: Target output file path (.png, .pdf).

    Returns:
        Resolved Path of the saved export.

    Raises:
        VisualizationError: If export format is unsupported or write fails.
    """
    target = Path(filepath).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix.lower()

    if suffix not in [".png", ".pdf", ".jpg", ".jpeg", ".svg"]:
        raise VisualizationError(
            f"Unsupported Matplotlib export format '{suffix}'. Supported formats: .png, .pdf, .svg"
        )

    try:
        figure.savefig(str(target), bbox_inches="tight", dpi=300)
        logger.info(
            "Chart Exported | Format=%s | Path=%s",
            suffix,
            str(target),
            extra={"event": "Chart Exported", "format": suffix, "path": str(target)},
        )
        return target
    except Exception as err:
        logger.error("Failed to export Matplotlib chart to %s: %s", str(target), err)
        raise VisualizationError(f"Matplotlib export failed: {err}") from err
