"""Visualization Engine Package for Social Network Influence Maximization Simulator.

Exposes the VisualizationFactory, chart builders, configuration models, export tools,
and custom validation exceptions.
"""

from .chart_builder import VisualizationFactory
from .exceptions import VisualizationError, VisualizationValidationError
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
    create_probability_sensitivity_chart,
    create_runtime_scalability_chart,
    create_seed_budget_chart,
)
from .utilities import (
    create_dataset_summary,
    export_matplotlib_chart,
    export_plotly_chart,
)
from .validators import (
    validate_chart_config,
    validate_chart_dataframe,
    validate_degree_distribution,
)

__all__ = [
    "VisualizationFactory",
    "ChartConfig",
    "create_algorithm_comparison_chart",
    "create_seed_budget_chart",
    "create_probability_sensitivity_chart",
    "create_runtime_scalability_chart",
    "create_convergence_chart",
    "create_degree_distribution_chart",
    "create_dataset_summary",
    "export_plotly_chart",
    "export_matplotlib_chart",
    "create_matplotlib_algorithm_comparison",
    "create_matplotlib_seed_budget",
    "create_matplotlib_convergence",
    "create_matplotlib_degree_distribution",
    "VisualizationError",
    "VisualizationValidationError",
    "validate_chart_dataframe",
    "validate_degree_distribution",
    "validate_chart_config",
]
