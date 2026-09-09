"""Unified Visualization Factory and Chart Builder.

Serves as the primary façade for Module 5, exposing a single entry point
to construct Plotly and Matplotlib figures.
Supports both class-level static calls and configured instance calls.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from .matplotlib_charts import (
    create_matplotlib_algorithm_comparison,
    create_matplotlib_convergence,
    create_matplotlib_degree_distribution,
    create_matplotlib_seed_budget,
)
from .models import ChartConfig
from .plotly_charts import (
    create_algorithm_comparison_chart,
    create_convergence_chart,
    create_degree_distribution_chart,
    create_runtime_scalability_chart,
    create_seed_budget_chart,
)
from .utilities import (
    create_dataset_summary,
    export_matplotlib_chart,
    export_plotly_chart,
)

logger = logging.getLogger(__name__)


class VisualizationFactory:
    """Central Factory for producing visualization artifacts."""

    def __init__(
        self,
        default_config: ChartConfig | None = None,
        custom_logger: logging.Logger | None = None,
    ) -> None:
        """Initialize VisualizationFactory.

        Args:
            default_config: Default ChartConfig to apply when not specified per call.
            custom_logger: Optional logger instance.
        """
        self.default_config = default_config
        self.log = custom_logger or logger

    @classmethod
    def create_algorithm_comparison_chart(
        cls,
        metrics_df: pd.DataFrame,
        config: ChartConfig | None = None,
        custom_logger: logging.Logger | None = None,
    ) -> go.Figure:
        """Create interactive Plotly algorithm comparison grouped bar chart."""
        return create_algorithm_comparison_chart(
            metrics_df=metrics_df,
            config=config,
            custom_logger=custom_logger or logger,
        )

    @classmethod
    def create_seed_budget_chart(
        cls,
        experiment_a_df: pd.DataFrame,
        config: ChartConfig | None = None,
        custom_logger: logging.Logger | None = None,
    ) -> go.Figure:
        """Create interactive Plotly influence vs seed budget multi-line chart."""
        return create_seed_budget_chart(
            experiment_a_df=experiment_a_df,
            config=config,
            custom_logger=custom_logger or logger,
        )

    @classmethod
    def create_runtime_scalability_chart(
        cls,
        experiment_c_df: pd.DataFrame,
        log_scale: bool = False,
        config: ChartConfig | None = None,
        custom_logger: logging.Logger | None = None,
    ) -> go.Figure:
        """Create interactive Plotly runtime vs network size scalability chart."""
        return create_runtime_scalability_chart(
            experiment_c_df=experiment_c_df,
            log_scale=log_scale,
            config=config,
            custom_logger=custom_logger or logger,
        )

    @classmethod
    def create_convergence_chart(
        cls,
        experiment_d_df: pd.DataFrame,
        config: ChartConfig | None = None,
        custom_logger: logging.Logger | None = None,
    ) -> go.Figure:
        """Create interactive Plotly Monte Carlo convergence chart with confidence bands."""
        return create_convergence_chart(
            experiment_d_df=experiment_d_df,
            config=config,
            custom_logger=custom_logger or logger,
        )

    @classmethod
    def create_degree_distribution_chart(
        cls,
        degree_distribution: Any,
        config: ChartConfig | None = None,
        custom_logger: logging.Logger | None = None,
    ) -> go.Figure:
        """Create interactive Plotly degree distribution histogram."""
        return create_degree_distribution_chart(
            degree_distribution=degree_distribution,
            config=config,
            custom_logger=custom_logger or logger,
        )

    @staticmethod
    def create_dataset_summary(
        node_count: int,
        edge_count: int,
    ) -> dict[str, Any]:
        """Compute network summary KPI metrics (node_count, edge_count, avg_degree)."""
        return create_dataset_summary(node_count=node_count, edge_count=edge_count)

    @staticmethod
    def export_plotly_chart(
        figure: go.Figure,
        filepath: Path | str,
    ) -> Path:
        """Export Plotly figure to HTML, PNG, or SVG."""
        return export_plotly_chart(figure, filepath)

    @staticmethod
    def export_matplotlib_chart(
        figure: Any,
        filepath: Path | str,
    ) -> Path:
        """Export Matplotlib figure to PNG or PDF."""
        return export_matplotlib_chart(figure, filepath)
