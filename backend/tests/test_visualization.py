"""Unit test suite for Module 5: Visualization Engine.

Verifies:
1. Algorithm comparison grouped bar chart creation.
2. Seed budget multi-line chart trace generation per algorithm.
3. Scalability chart with network size axis.
4. Convergence chart with shaded confidence band.
5. Node degree distribution histogram.
6. Dataset summary KPI metrics (avg_degree).
7. Validation exceptions for empty or malformed DataFrames.
8. Chart file export to HTML/PNG.
9. VisualizationFactory facade integration.
10. Matplotlib secondary static charts.
11. Structured logging emission during chart rendering.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from visualization import (
    ChartConfig,
    VisualizationError,
    VisualizationFactory,
    VisualizationValidationError,
    create_algorithm_comparison_chart,
    create_convergence_chart,
    create_dataset_summary,
    create_degree_distribution_chart,
    create_matplotlib_algorithm_comparison,
    create_runtime_scalability_chart,
    create_seed_budget_chart,
    export_matplotlib_chart,
    export_plotly_chart,
)


@pytest.fixture
def sample_metrics_df() -> pd.DataFrame:
    """Sample DataFrame matching Module 2 aggregate_metrics output."""
    return pd.DataFrame([
        {"algorithm_name": "greedy", "mean_influence": 842.5, "runtime_sec": 2.45},
        {"algorithm_name": "degree", "mean_influence": 620.0, "runtime_sec": 0.85},
        {"algorithm_name": "random", "mean_influence": 310.0, "runtime_sec": 0.12},
    ])


@pytest.fixture
def sample_exp_a_df() -> pd.DataFrame:
    """Sample DataFrame matching Module 3 Experiment A output."""
    rows = []
    for k in [5, 10, 20, 50]:
        rows.append({"algorithm_name": "greedy", "k": k, "mean_influence": k * 45.0})
        rows.append({"algorithm_name": "degree", "k": k, "mean_influence": k * 35.0})
        rows.append({"algorithm_name": "random", "k": k, "mean_influence": k * 18.0})
    return pd.DataFrame(rows)


@pytest.fixture
def sample_exp_c_df() -> pd.DataFrame:
    """Sample DataFrame matching Module 3 Experiment C output."""
    rows = []
    for n in [500, 1000, 2000, 4000]:
        rows.append({"algorithm_name": "greedy", "node_count": n, "runtime_sec": n * 0.001})
        rows.append({"algorithm_name": "random", "node_count": n, "runtime_sec": n * 0.0002})
    return pd.DataFrame(rows)


@pytest.fixture
def sample_exp_d_df() -> pd.DataFrame:
    """Sample DataFrame matching Module 3 Experiment D output."""
    return pd.DataFrame([
        {"num_simulations": 10, "mean_influence": 830.0, "std_influence": 95.0},
        {"num_simulations": 50, "mean_influence": 840.0, "std_influence": 42.0},
        {"num_simulations": 100, "mean_influence": 841.5, "std_influence": 30.0},
        {"num_simulations": 500, "mean_influence": 842.1, "std_influence": 12.0},
    ])


# ---------------------------------------------------------------------------
# Test 1: Algorithm Comparison Chart Created Successfully
# ---------------------------------------------------------------------------
def test_create_algorithm_comparison_chart(sample_metrics_df):
    """Verify algorithm comparison chart creates a valid Plotly Figure."""
    fig = create_algorithm_comparison_chart(sample_metrics_df)
    assert fig is not None
    assert isinstance(fig, go.Figure)
    # Grouped bars should contain 2 traces: Influence and Runtime
    assert len(fig.data) == 2
    assert fig.data[0].name == "Mean Influence"
    assert fig.data[1].name == "Runtime (Seconds)"


# ---------------------------------------------------------------------------
# Test 2: Seed Budget Chart (One Trace Per Algorithm)
# ---------------------------------------------------------------------------
def test_create_seed_budget_chart_traces(sample_exp_a_df):
    """Verify seed budget chart generates exactly one line trace per algorithm."""
    fig = create_seed_budget_chart(sample_exp_a_df)
    assert fig is not None
    assert isinstance(fig, go.Figure)

    # 3 distinct algorithms in sample_exp_a_df -> 3 traces
    unique_algos = sample_exp_a_df["algorithm_name"].unique()
    assert len(fig.data) == len(unique_algos)
    trace_names = [trace.name for trace in fig.data]
    for algo in unique_algos:
        assert algo in trace_names


# ---------------------------------------------------------------------------
# Test 3: Scalability Chart (Contains node_count Axis)
# ---------------------------------------------------------------------------
def test_create_runtime_scalability_chart(sample_exp_c_df):
    """Verify scalability chart contains network size on the x-axis."""
    fig = create_runtime_scalability_chart(sample_exp_c_df, log_scale=True)
    assert fig is not None
    assert len(fig.data) == 2
    assert fig.layout.yaxis.type == "log"
    assert "Network Size" in fig.layout.xaxis.title.text


# ---------------------------------------------------------------------------
# Test 4: Convergence Chart (Contains Confidence Band)
# ---------------------------------------------------------------------------
def test_create_convergence_chart_confidence_band(sample_exp_d_df):
    """Verify convergence chart contains mean line and confidence band traces."""
    fig = create_convergence_chart(sample_exp_d_df)
    assert fig is not None
    # 3 traces: Upper bound, Lower bound with fill, and Mean line
    assert len(fig.data) == 3

    # Check shaded band fill property
    lower_band_trace = fig.data[1]
    assert lower_band_trace.fill == "tonexty"
    assert "Confidence Band" in lower_band_trace.name

    # Check mean line
    mean_trace = fig.data[2]
    assert mean_trace.name == "Mean Influence"


# ---------------------------------------------------------------------------
# Test 5: Degree Distribution Histogram
# ---------------------------------------------------------------------------
def test_create_degree_distribution_chart():
    """Verify degree distribution histogram is generated from numeric list."""
    degrees = [1, 2, 2, 3, 3, 3, 4, 5, 8, 12, 15]
    fig = create_degree_distribution_chart(degrees)
    assert fig is not None
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Histogram)
    assert list(fig.data[0].x) == degrees


# ---------------------------------------------------------------------------
# Test 6: Dataset Summary (avg_degree Calculated Correctly)
# ---------------------------------------------------------------------------
def test_create_dataset_summary():
    """Verify dataset summary computes average degree as (2 * edge_count) / node_count."""
    # Example: 1000 nodes, 5000 edges -> avg_degree = (2 * 5000) / 1000 = 10.0
    summary = create_dataset_summary(node_count=1000, edge_count=5000)
    assert summary["node_count"] == 1000
    assert summary["edge_count"] == 5000
    assert summary["avg_degree"] == 10.0

    # Test edge case: node_count = 0
    zero_summary = create_dataset_summary(node_count=0, edge_count=0)
    assert zero_summary["avg_degree"] == 0.0


# ---------------------------------------------------------------------------
# Test 7: Invalid DataFrame Raises Validation Exception
# ---------------------------------------------------------------------------
def test_invalid_dataframe_raises_validation_error():
    """Verify empty DataFrame or missing columns raises VisualizationValidationError."""
    empty_df = pd.DataFrame()
    with pytest.raises(VisualizationValidationError, match="Input DataFrame is empty"):
        create_algorithm_comparison_chart(empty_df)

    incomplete_df = pd.DataFrame([{"algorithm_name": "greedy"}])
    with pytest.raises(VisualizationValidationError, match="missing required columns"):
        create_algorithm_comparison_chart(incomplete_df)

    # Empty degrees
    with pytest.raises(VisualizationValidationError, match="must not be empty"):
        create_degree_distribution_chart([])


# ---------------------------------------------------------------------------
# Test 8: Export Chart (File Created)
# ---------------------------------------------------------------------------
def test_export_plotly_chart(tmp_path: Path, sample_metrics_df):
    """Verify export_plotly_chart creates HTML output file on disk."""
    fig = create_algorithm_comparison_chart(sample_metrics_df)
    export_path = tmp_path / "charts" / "comparison.html"

    result_path = export_plotly_chart(fig, export_path)
    assert result_path.exists()
    assert result_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Test 9: VisualizationFactory Facade Integration
# ---------------------------------------------------------------------------
def test_visualization_factory(sample_metrics_df, sample_exp_a_df, sample_exp_c_df, sample_exp_d_df):
    """Verify VisualizationFactory provides complete access to all charts and cards."""
    viz = VisualizationFactory()

    c1 = viz.create_algorithm_comparison_chart(sample_metrics_df)
    assert isinstance(c1, go.Figure)

    c2 = viz.create_seed_budget_chart(sample_exp_a_df)
    assert isinstance(c2, go.Figure)

    c3 = viz.create_runtime_scalability_chart(sample_exp_c_df)
    assert isinstance(c3, go.Figure)

    c4 = viz.create_convergence_chart(sample_exp_d_df)
    assert isinstance(c4, go.Figure)

    c5 = viz.create_degree_distribution_chart([1, 2, 3, 4])
    assert isinstance(c5, go.Figure)

    card = viz.create_dataset_summary(4039, 88234)
    assert card["node_count"] == 4039


# ---------------------------------------------------------------------------
# Test 10: Matplotlib Charts and Export
# ---------------------------------------------------------------------------
def test_matplotlib_charts_and_export(tmp_path: Path, sample_metrics_df):
    """Verify Matplotlib static figure generation and PNG export."""
    mpl_fig = create_matplotlib_algorithm_comparison(sample_metrics_df)
    assert mpl_fig is not None

    export_path = tmp_path / "mpl_chart.png"
    out = export_matplotlib_chart(mpl_fig, export_path)
    assert out.exists()
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Test 11: Structured Logging
# ---------------------------------------------------------------------------
def test_visualization_structured_logging(caplog, sample_metrics_df):
    """Verify structured logs are emitted during chart creation."""
    with caplog.at_level(logging.INFO):
        create_algorithm_comparison_chart(sample_metrics_df)

    messages = [rec.message for rec in caplog.records]
    assert any("Chart Creation Started" in msg for msg in messages)
    assert any("Chart Type=Algorithm Comparison Bar Chart" in msg for msg in messages)
    assert any("Chart Creation Finished" in msg for msg in messages)
