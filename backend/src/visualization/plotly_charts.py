"""Plotly interactive chart builders for influence maximization experiments.

Generates standalone, interactive Plotly Figure objects for algorithm comparisons,
seed budget curves, scalability benchmarks, convergence bands, and degree distributions.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .models import ChartConfig
from .validators import (
    validate_chart_config,
    validate_chart_dataframe,
    validate_degree_distribution,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CHART 1: Algorithm Comparison Bar Chart
# ---------------------------------------------------------------------------
def create_algorithm_comparison_chart(
    metrics_df: pd.DataFrame,
    config: ChartConfig | None = None,
    custom_logger: logging.Logger | None = None,
) -> go.Figure:
    """Create an interactive grouped bar chart comparing algorithm influence and runtime.

    Required columns:
        - 'algorithm_name'
        - 'mean_influence'
        - 'runtime_sec'

    Args:
        metrics_df: Aggregated algorithm metrics DataFrame.
        config: Optional ChartConfig overrides.
        custom_logger: Optional logger instance.

    Returns:
        Interactive plotly.graph_objects.Figure.
    """
    log = custom_logger or logger
    chart_name = "Algorithm Comparison Bar Chart"

    log.info(
        "Chart Creation Started | Chart Type=%s | Rows Processed=%d",
        chart_name,
        len(metrics_df) if metrics_df is not None and hasattr(metrics_df, "__len__") else 0,
        extra={"event": "Chart Creation Started", "chart_type": chart_name},
    )

    validate_chart_dataframe(
        metrics_df,
        required_columns=["algorithm_name", "mean_influence", "runtime_sec"],
        chart_name=chart_name,
    )
    validate_chart_config(config)

    cfg = config or ChartConfig(
        title="Algorithm Performance Comparison: Influence Spread & Runtime",
        x_label="Algorithm",
        y_label="Mean Influence Spread (Nodes)",
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar 1: Mean Influence
    fig.add_trace(
        go.Bar(
            x=metrics_df["algorithm_name"],
            y=metrics_df["mean_influence"],
            name="Mean Influence",
            marker_color="#2563EB",  # Vibrant Indigo/Blue
            hovertemplate="<b>%{x}</b><br>Influence: %{y:,.1f} nodes<extra></extra>",
        ),
        secondary_y=False,
    )

    # Bar 2: Runtime
    fig.add_trace(
        go.Bar(
            x=metrics_df["algorithm_name"],
            y=metrics_df["runtime_sec"],
            name="Runtime (Seconds)",
            marker_color="#F59E0B",  # Vibrant Amber/Orange
            hovertemplate="<b>%{x}</b><br>Runtime: %{y:.3f} sec<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=dict(text=f"<b>{cfg.title}</b>", font=dict(size=18)),
        barmode="group",
        template=cfg.template,
        width=cfg.width,
        height=cfg.height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=60, r=60, t=80, b=60),
    )

    fig.update_xaxes(title_text=cfg.x_label)
    fig.update_yaxes(title_text=cfg.y_label, secondary_y=False)
    fig.update_yaxes(title_text="Execution Runtime (Seconds)", secondary_y=True, showgrid=False)

    log.info("Chart Creation Finished | Chart Type=%s", chart_name)
    return fig


# ---------------------------------------------------------------------------
# CHART 2: Influence vs Seed Budget
# ---------------------------------------------------------------------------
def create_seed_budget_chart(
    experiment_a_df: pd.DataFrame,
    config: ChartConfig | None = None,
    custom_logger: logging.Logger | None = None,
) -> go.Figure:
    """Create a multi-line chart displaying influence spread vs seed budget (k).

    Required columns:
        - 'algorithm_name'
        - 'k'
        - 'mean_influence'

    Args:
        experiment_a_df: Experiment A output DataFrame.
        config: Optional ChartConfig overrides.
        custom_logger: Optional logger instance.

    Returns:
        Interactive plotly.graph_objects.Figure.
    """
    log = custom_logger or logger
    chart_name = "Influence vs Seed Budget"

    log.info(
        "Chart Creation Started | Chart Type=%s | Rows Processed=%d",
        chart_name,
        len(experiment_a_df) if experiment_a_df is not None and hasattr(experiment_a_df, "__len__") else 0,
        extra={"event": "Chart Creation Started", "chart_type": chart_name},
    )

    validate_chart_dataframe(
        experiment_a_df,
        required_columns=["algorithm_name", "k", "mean_influence"],
        chart_name=chart_name,
    )
    validate_chart_config(config)

    cfg = config or ChartConfig(
        title="Influence Spread vs Seed Budget (k)",
        x_label="Seed Budget (k)",
        y_label="Expected Influence Spread (Nodes)",
    )

    fig = go.Figure()
    algorithms = experiment_a_df["algorithm_name"].unique()

    for algo in algorithms:
        sub = experiment_a_df[experiment_a_df["algorithm_name"] == algo].sort_values("k")
        fig.add_trace(
            go.Scatter(
                x=sub["k"],
                y=sub["mean_influence"],
                mode="lines+markers",
                name=str(algo),
                hovertemplate=f"<b>{algo}</b><br>k: %{{x}}<br>Spread: %{{y:,.1f}}<extra></extra>",
                line=dict(width=2.5),
                marker=dict(size=8),
            )
        )

    fig.update_layout(
        title=dict(text=f"<b>{cfg.title}</b>", font=dict(size=18)),
        xaxis=dict(title=cfg.x_label, tickmode="linear"),
        yaxis=dict(title=cfg.y_label),
        template=cfg.template,
        width=cfg.width,
        height=cfg.height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=80, b=60),
    )

    log.info("Chart Creation Finished | Chart Type=%s", chart_name)
    return fig


# ---------------------------------------------------------------------------
# CHART 3: Runtime vs Network Size
# ---------------------------------------------------------------------------
def create_runtime_scalability_chart(
    experiment_c_df: pd.DataFrame,
    log_scale: bool = False,
    config: ChartConfig | None = None,
    custom_logger: logging.Logger | None = None,
) -> go.Figure:
    """Create a multi-line chart displaying algorithm execution runtime vs network size.

    Required columns:
        - 'algorithm_name'
        - 'runtime_sec'
        - and either 'node_count' or 'size' or 'k'

    Args:
        experiment_c_df: Experiment C output DataFrame.
        log_scale: If True, sets y-axis to logarithmic scale.
        config: Optional ChartConfig overrides.
        custom_logger: Optional logger instance.

    Returns:
        Interactive plotly.graph_objects.Figure.
    """
    log = custom_logger or logger
    chart_name = "Runtime vs Network Size"

    log.info(
        "Chart Creation Started | Chart Type=%s | Rows Processed=%d",
        chart_name,
        len(experiment_c_df) if experiment_c_df is not None and hasattr(experiment_c_df, "__len__") else 0,
        extra={"event": "Chart Creation Started", "chart_type": chart_name},
    )

    # Determine size column name
    size_col = "node_count"
    if "node_count" not in experiment_c_df.columns:
        if "size" in experiment_c_df.columns:
            size_col = "size"
        elif "k" in experiment_c_df.columns:
            size_col = "k"

    validate_chart_dataframe(
        experiment_c_df,
        required_columns=["algorithm_name", size_col, "runtime_sec"],
        chart_name=chart_name,
    )
    validate_chart_config(config)

    cfg = config or ChartConfig(
        title="Algorithm Scalability: Runtime vs Network Size",
        x_label="Network Size (|V| Nodes)",
        y_label="Execution Runtime (Seconds)" + (" [Log Scale]" if log_scale else ""),
    )

    fig = go.Figure()
    algorithms = experiment_c_df["algorithm_name"].unique()

    for algo in algorithms:
        sub = experiment_c_df[experiment_c_df["algorithm_name"] == algo].sort_values(size_col)
        fig.add_trace(
            go.Scatter(
                x=sub[size_col],
                y=sub["runtime_sec"],
                mode="lines+markers",
                name=str(algo),
                hovertemplate=f"<b>{algo}</b><br>Size: %{{x:,}}<br>Runtime: %{{y:.4f}}s<extra></extra>",
                line=dict(width=2.5),
                marker=dict(size=8),
            )
        )

    yaxis_dict = dict(title=cfg.y_label)
    if log_scale:
        yaxis_dict["type"] = "log"

    fig.update_layout(
        title=dict(text=f"<b>{cfg.title}</b>", font=dict(size=18)),
        xaxis=dict(title=cfg.x_label),
        yaxis=yaxis_dict,
        template=cfg.template,
        width=cfg.width,
        height=cfg.height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=80, b=60),
    )

    log.info("Chart Creation Finished | Chart Type=%s", chart_name)
    return fig


# ---------------------------------------------------------------------------
# CHART 4: Monte Carlo Convergence
# ---------------------------------------------------------------------------
def create_convergence_chart(
    experiment_d_df: pd.DataFrame,
    config: ChartConfig | None = None,
    custom_logger: logging.Logger | None = None,
) -> go.Figure:
    """Create a convergence plot displaying mean influence with confidence bands.

    Confidence Band Formula:
        upper = mean_influence + std_influence
        lower = mean_influence - std_influence

    Required columns:
        - 'num_simulations'
        - 'mean_influence'
        - 'std_influence'

    Args:
        experiment_d_df: Experiment D output DataFrame.
        config: Optional ChartConfig overrides.
        custom_logger: Optional logger instance.

    Returns:
        Interactive plotly.graph_objects.Figure with shaded confidence envelope.
    """
    log = custom_logger or logger
    chart_name = "Monte Carlo Convergence"

    log.info(
        "Chart Creation Started | Chart Type=%s | Rows Processed=%d",
        chart_name,
        len(experiment_d_df) if experiment_d_df is not None and hasattr(experiment_d_df, "__len__") else 0,
        extra={"event": "Chart Creation Started", "chart_type": chart_name},
    )

    validate_chart_dataframe(
        experiment_d_df,
        required_columns=["num_simulations", "mean_influence", "std_influence"],
        chart_name=chart_name,
    )
    validate_chart_config(config)

    cfg = config or ChartConfig(
        title="Monte Carlo Simulation Convergence & Stability",
        x_label="Number of Simulations (Iterations)",
        y_label="Influence Spread (Mean ± Std Dev)",
    )

    df_sorted = experiment_d_df.sort_values("num_simulations")
    x = df_sorted["num_simulations"].tolist()
    mean = df_sorted["mean_influence"].tolist()
    std = df_sorted["std_influence"].tolist()

    upper = [m + s for m, s in zip(mean, std)]
    lower = [max(0.0, m - s) for m, s in zip(mean, std)]

    fig = go.Figure()

    # 1. Upper Bound (invisible line)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=upper,
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
            name="Upper Bound (+1 Std Dev)",
        )
    )

    # 2. Lower Bound with shaded fill to Upper Bound
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.18)",  # Semi-transparent primary blue
            hoverinfo="skip",
            name="Confidence Band (±1 Std Dev)",
        )
    )

    # 3. Mean Influence Line
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean,
            mode="lines+markers",
            name="Mean Influence",
            line=dict(color="#1D4ED8", width=3),
            marker=dict(size=8, color="#1D4ED8"),
            hovertemplate="Simulations: %{x}<br>Mean: %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=f"<b>{cfg.title}</b>", font=dict(size=18)),
        xaxis=dict(title=cfg.x_label),
        yaxis=dict(title=cfg.y_label),
        template=cfg.template,
        width=cfg.width,
        height=cfg.height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=80, b=60),
    )

    log.info("Chart Creation Finished | Chart Type=%s", chart_name)
    return fig


# ---------------------------------------------------------------------------
# CHART 5: Degree Distribution Histogram
# ---------------------------------------------------------------------------
def create_degree_distribution_chart(
    degree_distribution: Any,
    config: ChartConfig | None = None,
    custom_logger: logging.Logger | None = None,
) -> go.Figure:
    """Create an interactive histogram of network node degree distributions.

    Args:
        degree_distribution: Sequence of node degrees or dictionary.
        config: Optional ChartConfig overrides.
        custom_logger: Optional logger instance.

    Returns:
        Interactive plotly.graph_objects.Figure.
    """
    log = custom_logger or logger
    chart_name = "Degree Distribution Histogram"

    log.info("Chart Creation Started | Chart Type=%s", chart_name)

    degrees = validate_degree_distribution(degree_distribution)
    validate_chart_config(config)

    cfg = config or ChartConfig(
        title="Social Network Topology: Node Degree Distribution",
        x_label="Node Degree (Connections / Edges)",
        y_label="Frequency (Number of Nodes)",
    )

    fig = go.Figure(
        data=[
            go.Histogram(
                x=degrees,
                marker_color="#3B82F6",
                opacity=0.85,
                hovertemplate="Degree: %{x}<br>Count: %{y:,}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=dict(text=f"<b>{cfg.title}</b>", font=dict(size=18)),
        xaxis=dict(title=cfg.x_label),
        yaxis=dict(title=cfg.y_label),
        template=cfg.template,
        width=cfg.width,
        height=cfg.height,
        bargap=0.08,
        margin=dict(l=60, r=60, t=80, b=60),
    )

    log.info("Chart Creation Finished | Chart Type=%s", chart_name)
    return fig
