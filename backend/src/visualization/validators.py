"""Validation layer for the Visualization Engine.

Verifies DataFrame non-emptiness, column requirements, and degree distributions
prior to generating figures.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from .exceptions import VisualizationValidationError
from .models import ChartConfig


def validate_chart_dataframe(
    df: Any,
    required_columns: Sequence[str] | None = None,
    chart_name: str = "Chart",
) -> None:
    """Validate that a DataFrame is populated and contains the necessary columns for charting.

    Args:
        df: Input DataFrame.
        required_columns: Sequence of column names required for rendering.
        chart_name: Name of chart used for descriptive error messages.

    Raises:
        VisualizationValidationError: If DataFrame is None, empty, or missing columns.
    """
    if df is None:
        raise VisualizationValidationError(f"{chart_name}: Input DataFrame must not be None.")

    if not isinstance(df, pd.DataFrame):
        raise VisualizationValidationError(
            f"{chart_name}: Expected pandas DataFrame, got: {type(df).__name__}."
        )

    if df.empty:
        raise VisualizationValidationError(
            f"{chart_name}: Input DataFrame is empty. Cannot generate visual chart from empty data."
        )

    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise VisualizationValidationError(
                f"{chart_name}: DataFrame missing required columns: {missing}. Present: {list(df.columns)}"
            )


def validate_degree_distribution(degrees: Any) -> Sequence[int | float]:
    """Validate that degree distribution data is non-empty and convert to numeric sequence.

    Supports:
        - List/sequence of degrees: [1, 2, 2, 3, 5, ...]
        - Dictionary mapping node ID to degree: {0: 3, 1: 5, ...}
        - Dictionary mapping degree to frequency: {'1': 20, '2': 15, ...}

    Args:
        degrees: Degree distribution container.

    Returns:
        Flattened sequence of degree values.

    Raises:
        VisualizationValidationError: If container is None or empty.
    """
    if degrees is None:
        raise VisualizationValidationError("Degree distribution must not be None.")

    if isinstance(degrees, Mapping):
        if not degrees:
            raise VisualizationValidationError("Degree distribution mapping must not be empty.")
        # If values are frequencies or node degrees, extract values
        vals = list(degrees.values())
        return vals

    try:
        deg_len = len(degrees)
    except TypeError as err:
        raise VisualizationValidationError(
            f"Degree distribution must be a sequence or mapping. Got: {type(degrees).__name__}"
        ) from err

    if deg_len == 0:
        raise VisualizationValidationError("Degree distribution list must not be empty.")

    return list(degrees)


def validate_chart_config(config: Any) -> None:
    """Validate ChartConfig container.

    Args:
        config: ChartConfig instance or None.

    Raises:
        VisualizationValidationError: If config is invalid.
    """
    if config is not None and not isinstance(config, ChartConfig):
        raise VisualizationValidationError(
            f"Expected ChartConfig instance, got {type(config).__name__}."
        )
