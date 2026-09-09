"""Unit test suite for Module 2: Metrics & Analytics Engine.

Verifies:
1. Influence percentage calculation (mean_influence=1000, node_count=4000 -> 25%)
2. Spread efficiency calculation (mean_influence=1000, seed_count=10 -> 100)
3. DataFrame generation schema and required columns
4. Cross-algorithm aggregation for 5 algorithms (random, degree, greedy, bfs, ic)
5. get_best_algorithm() identification of top performer
6. Validation errors for invalid node_count (e.g. node_count = 0)
7. Validation errors for invalid runtime, algorithm name, mean influence, seed count
8. Sorting DataFrame by arbitrary metrics
9. Serialization via metrics_to_dict() and record.to_dict()
10. Structured logging output
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metrics import (
    METRICS_COLUMNS,
    MetricsRecord,
    MetricsValidationError,
    aggregate_metrics,
    calculate_metrics,
    dataframe_from_metrics,
    get_best_algorithm,
    metrics_to_dict,
    sort_by_metric,
)
from monte_carlo import MonteCarloResult


@dataclass
class MockMonteCarloResult:
    """Mock result matching MonteCarloResult attributes for lightweight testing."""
    mean_influence: float
    seed_count: int
    probability: float = 0.1
    min_influence: int = 0
    max_influence: int = 0
    std_influence: float = 0.0
    num_simulations: int = 50
    raw_results: list[int] | None = None


# ---------------------------------------------------------------------------
# Test 1: Influence Percentage Calculation
# ---------------------------------------------------------------------------
def test_influence_percentage_calculation():
    """Verify that influence_pct = (mean_influence / node_count) * 100.

    mean_influence = 1000, node_count = 4000 -> expected influence_pct = 25.0%
    """
    mc_result = MockMonteCarloResult(mean_influence=1000.0, seed_count=10)
    record = calculate_metrics(
        monte_carlo_result=mc_result,
        algorithm_name="greedy",
        node_count=4000,
        runtime_sec=1.234,
        timestamp="2026-09-09T12:00:00Z",
    )

    assert isinstance(record, MetricsRecord)
    assert record.influence_pct == pytest.approx(25.0)
    assert record.average_reach == pytest.approx(1000.0)
    assert record.runtime_sec == 1.234


# ---------------------------------------------------------------------------
# Test 2: Spread Efficiency Calculation
# ---------------------------------------------------------------------------
def test_spread_efficiency_calculation():
    """Verify that spread_efficiency = mean_influence / seed_count.

    mean_influence = 1000, seed_count = 10 -> expected spread_efficiency = 100.0
    """
    mc_result = MockMonteCarloResult(mean_influence=1000.0, seed_count=10)
    record = calculate_metrics(
        monte_carlo_result=mc_result,
        algorithm_name="degree",
        node_count=5000,
        runtime_sec=0.5,
    )

    assert record.spread_efficiency == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Test 3: Validate DataFrame Generation & Required Columns
# ---------------------------------------------------------------------------
def test_dataframe_generation_has_all_required_columns():
    """Verify that aggregate_metrics returns a DataFrame containing all required SRS columns."""
    mc_result = MockMonteCarloResult(mean_influence=500.0, seed_count=5, probability=0.05)
    record = calculate_metrics(
        monte_carlo_result=mc_result,
        algorithm_name="random",
        node_count=2000,
        runtime_sec=0.123,
    )

    df = aggregate_metrics([record])

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == METRICS_COLUMNS
    assert len(df) == 1
    assert df.loc[0, "algorithm_name"] == "random"
    assert df.loc[0, "mean_influence"] == 500.0
    assert df.loc[0, "spread_efficiency"] == 100.0


# ---------------------------------------------------------------------------
# Test 4: Cross Algorithm Aggregation (5 algorithms)
# ---------------------------------------------------------------------------
def test_aggregate_five_algorithms():
    """Verify that aggregating 5 distinct algorithms results in len(df) == 5."""
    algorithms = ["random", "degree", "greedy", "bfs", "ic"]
    records: list[MetricsRecord] = []

    for i, algo in enumerate(algorithms, start=1):
        mc_result = MockMonteCarloResult(mean_influence=float(i * 100), seed_count=10)
        rec = calculate_metrics(
            monte_carlo_result=mc_result,
            algorithm_name=algo,
            node_count=1000,
            runtime_sec=0.05 * i,
        )
        records.append(rec)

    df = aggregate_metrics(records)

    assert len(df) == 5
    assert set(df["algorithm_name"]) == set(algorithms)
    # Also test dataframe_from_metrics alias
    df_alias = dataframe_from_metrics(records)
    pd.testing.assert_frame_equal(df, df_alias)


# ---------------------------------------------------------------------------
# Test 5: get_best_algorithm()
# ---------------------------------------------------------------------------
def test_get_best_algorithm():
    """Verify get_best_algorithm() returns the algorithm with highest mean influence."""
    algorithms_data = [
        ("random", 150.0),
        ("degree", 420.0),
        ("greedy", 850.0),  # Top performer
        ("bfs", 300.0),
        ("ic", 600.0),
    ]

    records = [
        calculate_metrics(
            monte_carlo_result=MockMonteCarloResult(mean_influence=mean_inf, seed_count=10),
            algorithm_name=name,
            node_count=1000,
            runtime_sec=1.0,
        )
        for name, mean_inf in algorithms_data
    ]

    best = get_best_algorithm(records, metric="mean_influence")
    assert best == {
        "algorithm": "greedy",
        "metric": "mean_influence",
        "value": 850.0,
    }

    # Also test with pd.DataFrame directly
    df = aggregate_metrics(records)
    best_from_df = get_best_algorithm(df, metric="mean_influence")
    assert best_from_df == best


# ---------------------------------------------------------------------------
# Test 6: node_count = 0 must raise validation exception
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("invalid_node_count", [0, -1, -500])
def test_invalid_node_count_raises_validation_error(invalid_node_count):
    """Verify that node_count <= 0 raises MetricsValidationError."""
    mc_result = MockMonteCarloResult(mean_influence=100.0, seed_count=5)

    with pytest.raises(MetricsValidationError, match="node_count must be greater than 0"):
        calculate_metrics(
            monte_carlo_result=mc_result,
            algorithm_name="greedy",
            node_count=invalid_node_count,
            runtime_sec=1.0,
        )


# ---------------------------------------------------------------------------
# Additional Test: Invalid Parameter Validations
# ---------------------------------------------------------------------------
def test_invalid_runtime_raises_validation_error():
    """Verify that runtime_sec < 0 raises MetricsValidationError."""
    mc_result = MockMonteCarloResult(mean_influence=100.0, seed_count=5)
    with pytest.raises(MetricsValidationError, match="runtime_sec must be non-negative"):
        calculate_metrics(
            monte_carlo_result=mc_result,
            algorithm_name="greedy",
            node_count=1000,
            runtime_sec=-0.01,
        )


@pytest.mark.parametrize("empty_algo", ["", "   ", None])
def test_empty_algorithm_name_raises_validation_error(empty_algo):
    """Verify that empty or whitespace algorithm_name raises MetricsValidationError."""
    mc_result = MockMonteCarloResult(mean_influence=100.0, seed_count=5)
    with pytest.raises(MetricsValidationError, match="algorithm_name must be a non-empty string"):
        calculate_metrics(
            monte_carlo_result=mc_result,
            algorithm_name=empty_algo,  # type: ignore[arg-type]
            node_count=1000,
            runtime_sec=1.0,
        )


def test_invalid_mean_influence_raises_validation_error():
    """Verify that negative mean_influence raises MetricsValidationError."""
    mc_result = MockMonteCarloResult(mean_influence=-10.0, seed_count=5)
    with pytest.raises(MetricsValidationError, match="mean_influence must be non-negative"):
        calculate_metrics(
            monte_carlo_result=mc_result,
            algorithm_name="greedy",
            node_count=1000,
            runtime_sec=1.0,
        )


def test_invalid_seed_count_raises_validation_error():
    """Verify that seed_count <= 0 raises MetricsValidationError."""
    mc_result = MockMonteCarloResult(mean_influence=10.0, seed_count=0)
    with pytest.raises(MetricsValidationError, match="seed_count must be greater than 0"):
        calculate_metrics(
            monte_carlo_result=mc_result,
            algorithm_name="greedy",
            node_count=1000,
            runtime_sec=1.0,
        )


# ---------------------------------------------------------------------------
# Additional Test: Sorting DataFrame
# ---------------------------------------------------------------------------
def test_sort_by_metric():
    """Verify sort_by_metric() correctly reorders rows in descending/ascending order."""
    records = [
        calculate_metrics(
            monte_carlo_result=MockMonteCarloResult(mean_influence=100.0, seed_count=5),
            algorithm_name="algo_low",
            node_count=1000,
            runtime_sec=0.1,
        ),
        calculate_metrics(
            monte_carlo_result=MockMonteCarloResult(mean_influence=500.0, seed_count=5),
            algorithm_name="algo_high",
            node_count=1000,
            runtime_sec=0.5,
        ),
    ]

    df = aggregate_metrics(records)

    # Descending sort (highest first)
    df_desc = sort_by_metric(df, metric="mean_influence", ascending=False)
    assert df_desc.loc[0, "algorithm_name"] == "algo_high"
    assert df_desc.loc[1, "algorithm_name"] == "algo_low"

    # Ascending sort (lowest first)
    df_asc = sort_by_metric(df, metric="mean_influence", ascending=True)
    assert df_asc.loc[0, "algorithm_name"] == "algo_low"
    assert df_asc.loc[1, "algorithm_name"] == "algo_high"


def test_sort_by_invalid_metric_raises_value_error():
    """Verify that sorting by a non-existent column raises ValueError."""
    df = aggregate_metrics([])
    with pytest.raises(ValueError, match="Metric 'unknown_col' not found"):
        sort_by_metric(df, metric="unknown_col")


# ---------------------------------------------------------------------------
# Additional Test: metrics_to_dict() serialization
# ---------------------------------------------------------------------------
def test_metrics_to_dict():
    """Verify dictionary serialization for export and dashboard tables."""
    mc_result = MockMonteCarloResult(mean_influence=200.0, seed_count=4)
    record = calculate_metrics(
        monte_carlo_result=mc_result,
        algorithm_name="degree",
        node_count=1000,
        runtime_sec=0.456,
        timestamp="2026-09-09T00:00:00Z",
    )

    d = metrics_to_dict(record)
    assert isinstance(d, dict)
    assert d["algorithm_name"] == "degree"
    assert d["mean_influence"] == 200.0
    assert d["influence_pct"] == 20.0
    assert d["spread_efficiency"] == 50.0
    assert d["runtime_sec"] == 0.456
    assert d["timestamp"] == "2026-09-09T00:00:00Z"


# ---------------------------------------------------------------------------
# Additional Test: Structured Logging
# ---------------------------------------------------------------------------
def test_metrics_structured_logging(caplog):
    """Verify structured log messages are emitted during calculate_metrics."""
    mc_result = MockMonteCarloResult(mean_influence=350.0, seed_count=5)

    with caplog.at_level(logging.INFO):
        calculate_metrics(
            monte_carlo_result=mc_result,
            algorithm_name="greedy",
            node_count=1000,
            runtime_sec=1.5,
        )

    log_messages = [rec.message for rec in caplog.records]
    assert any("Metrics Calculation Started" in msg for msg in log_messages)
    assert any("Algorithm Name=greedy" in msg for msg in log_messages)
    assert any("Mean Influence=350.00" in msg for msg in log_messages)
    assert any("Influence Percentage=35.00%" in msg for msg in log_messages)
    assert any("Spread Efficiency=70.00" in msg for msg in log_messages)
    assert any("Runtime=1.500s" in msg for msg in log_messages)
    assert any("Metrics Calculation Completed" in msg for msg in log_messages)
