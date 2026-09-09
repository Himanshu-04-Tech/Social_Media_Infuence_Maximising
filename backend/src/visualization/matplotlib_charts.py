"""Matplotlib secondary static chart generators.

Provides static figures for scientific publications, PDF exports, and headless reporting.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .models import ChartConfig
from .validators import (
    validate_chart_config,
    validate_chart_dataframe,
    validate_degree_distribution,
)

logger = logging.getLogger(__name__)


def create_matplotlib_algorithm_comparison(
    metrics_df: pd.DataFrame,
    config: ChartConfig | None = None,
) -> plt.Figure:
    """Generate a grouped bar chart using Matplotlib."""
    validate_chart_dataframe(
        metrics_df,
        required_columns=["algorithm_name", "mean_influence", "runtime_sec"],
        chart_name="Matplotlib Algorithm Comparison",
    )
    validate_chart_config(config)

    cfg = config or ChartConfig(
        title="Algorithm Performance Comparison",
        x_label="Algorithm",
        y_label="Mean Influence Spread",
    )

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    x = np.arange(len(metrics_df))
    width = 0.35

    ax1.bar(x - width / 2, metrics_df["mean_influence"], width, label="Influence Spread", color="#2563EB")
    ax2.bar(x + width / 2, metrics_df["runtime_sec"], width, label="Runtime (s)", color="#F59E0B")

    ax1.set_xlabel(cfg.x_label)
    ax1.set_ylabel(cfg.y_label, color="#2563EB")
    ax2.set_ylabel("Runtime (Seconds)", color="#F59E0B")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_df["algorithm_name"])
    ax1.set_title(cfg.title, fontsize=14, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig


def create_matplotlib_seed_budget(
    experiment_a_df: pd.DataFrame,
    config: ChartConfig | None = None,
) -> plt.Figure:
    """Generate a multi-line seed budget chart using Matplotlib."""
    validate_chart_dataframe(
        experiment_a_df,
        required_columns=["algorithm_name", "k", "mean_influence"],
        chart_name="Matplotlib Seed Budget",
    )
    validate_chart_config(config)

    cfg = config or ChartConfig(
        title="Influence Spread vs Seed Budget (k)",
        x_label="Seed Budget (k)",
        y_label="Expected Influence Spread",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in experiment_a_df["algorithm_name"].unique():
        sub = experiment_a_df[experiment_a_df["algorithm_name"] == algo].sort_values("k")
        ax.plot(sub["k"], sub["mean_influence"], marker="o", linewidth=2, label=str(algo))

    ax.set_xlabel(cfg.x_label)
    ax.set_ylabel(cfg.y_label)
    ax.set_title(cfg.title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig


def create_matplotlib_convergence(
    experiment_d_df: pd.DataFrame,
    config: ChartConfig | None = None,
) -> plt.Figure:
    """Generate a convergence plot with shaded confidence interval using Matplotlib."""
    validate_chart_dataframe(
        experiment_d_df,
        required_columns=["num_simulations", "mean_influence", "std_influence"],
        chart_name="Matplotlib Convergence",
    )
    validate_chart_config(config)

    cfg = config or ChartConfig(
        title="Monte Carlo Convergence & Stability",
        x_label="Number of Simulations",
        y_label="Influence Spread",
    )

    df_sorted = experiment_d_df.sort_values("num_simulations")
    x = df_sorted["num_simulations"].to_numpy()
    mean = df_sorted["mean_influence"].to_numpy()
    std = df_sorted["std_influence"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, mean, color="#1D4ED8", linewidth=2.5, marker="o", label="Mean Influence")
    ax.fill_between(x, np.maximum(0, mean - std), mean + std, color="#3B82F6", alpha=0.25, label="±1 Std Dev")

    ax.set_xlabel(cfg.x_label)
    ax.set_ylabel(cfg.y_label)
    ax.set_title(cfg.title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig


def create_matplotlib_degree_distribution(
    degree_distribution: Any,
    config: ChartConfig | None = None,
) -> plt.Figure:
    """Generate a node degree histogram using Matplotlib."""
    degrees = validate_degree_distribution(degree_distribution)
    validate_chart_config(config)

    cfg = config or ChartConfig(
        title="Node Degree Distribution",
        x_label="Degree",
        y_label="Frequency",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(degrees, bins="auto", color="#3B82F6", edgecolor="white", alpha=0.85)

    ax.set_xlabel(cfg.x_label)
    ax.set_ylabel(cfg.y_label)
    ax.set_title(cfg.title, fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig
