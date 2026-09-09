"""Metrics & Analytics Engine Package for Influence Maximization.

Exposes metrics calculation, cross-algorithm aggregation, ranking, models,
and validation tools.
"""

from .aggregators import (
    METRICS_COLUMNS,
    aggregate_metrics,
    dataframe_from_metrics,
    get_best_algorithm,
    sort_by_metric,
)
from .exceptions import MetricsError, MetricsValidationError
from .metrics_engine import calculate_metrics
from .models import MetricsRecord, metrics_to_dict
from .validators import validate_metrics_inputs

__all__ = [
    "calculate_metrics",
    "aggregate_metrics",
    "dataframe_from_metrics",
    "sort_by_metric",
    "get_best_algorithm",
    "metrics_to_dict",
    "MetricsRecord",
    "MetricsError",
    "MetricsValidationError",
    "validate_metrics_inputs",
    "METRICS_COLUMNS",
]
