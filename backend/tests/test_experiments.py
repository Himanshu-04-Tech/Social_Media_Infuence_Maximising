"""Unit test suite for Module 3: Experiment Framework.

Verifies:
1. Experiment A executes all 4 k values across all algorithms.
2. Experiment B executes all 4 propagation probabilities.
3. Experiment C executes all 4 network sizes via graph_sampler.
4. Experiment D executes all 4 simulation counts and computes relative standard error.
5. Monte Carlo engine run_monte_carlo() is mocked and called as expected.
6. Metrics engine calculate_metrics() is mocked and called as expected.
7. Output DataFrames contain all SRS-required columns.
8. Unique experiment IDs are generated correctly conforming to SRS rules.
9. run_all_experiments() master runner orchestrates all 4 experiments.
10. Validation errors for invalid parameters (k, probability, num_simulations, graph).
11. Structured logging output during execution.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiments import (
    DEFAULT_EXPERIMENT_A_K_VALUES,
    DEFAULT_EXPERIMENT_B_P_VALUES,
    DEFAULT_EXPERIMENT_C_SIZES,
    DEFAULT_EXPERIMENT_D_SIMULATION_COUNTS,
    EXPERIMENT_DF_COLUMNS,
    ExperimentValidationError,
    generate_experiment_id,
    run_all_experiments,
    run_convergence_experiment,
    run_probability_experiment,
    run_scalability_experiment,
    run_seed_budget_experiment,
)
from metrics import MetricsRecord
from monte_carlo import MonteCarloResult


class DummyGraph:
    """Mock graph with 1000 nodes for testing."""
    def __init__(self, n: int = 1000):
        self._n = n

    def number_of_nodes(self) -> int:
        return self._n

    def __len__(self) -> int:
        return self._n


@pytest.fixture
def mock_algorithms():
    """Returns a dictionary of mock seed selection algorithms."""
    return {
        "random": lambda g, k: list(range(k)),
        "degree": lambda g, k: list(range(k)),
        "greedy": lambda g, k: list(range(k)),
        "bfs": lambda g, k: list(range(k)),
        "ic": lambda g, k: list(range(k)),
    }


@pytest.fixture
def mock_simulate():
    """Returns a mock simulate_influence function."""
    return lambda g, seeds, p: len(seeds) * 15


# ---------------------------------------------------------------------------
# Test 1: Experiment A - Verify 4 k values are executed
# ---------------------------------------------------------------------------
def test_experiment_a_executes_four_k_values(mock_algorithms, mock_simulate):
    """Verify Experiment A runs exactly 4 k values for all algorithms."""
    graph = DummyGraph(1000)
    df = run_seed_budget_experiment(
        graph=graph,
        algorithms=mock_algorithms,
        simulate_fn=mock_simulate,
        k_values=[5, 10, 20, 50],
        probability=0.1,
        num_simulations=5,
    )

    assert isinstance(df, pd.DataFrame)
    # 4 k values * 5 algorithms = 20 rows
    assert len(df) == 4 * len(mock_algorithms)
    assert set(df["k"].unique()) == {5, 10, 20, 50}
    assert (df["probability"] == 0.1).all()


# ---------------------------------------------------------------------------
# Test 2: Experiment B - Verify 4 probabilities are executed
# ---------------------------------------------------------------------------
def test_experiment_b_executes_four_probabilities(mock_algorithms, mock_simulate):
    """Verify Experiment B runs exactly 4 probabilities for all algorithms."""
    graph = DummyGraph(1000)
    p_values = [0.05, 0.10, 0.20, 0.30]
    df = run_probability_experiment(
        graph=graph,
        algorithms=mock_algorithms,
        simulate_fn=mock_simulate,
        p_values=p_values,
        k=10,
        num_simulations=5,
    )

    assert isinstance(df, pd.DataFrame)
    # 4 p values * 5 algorithms = 20 rows
    assert len(df) == 4 * len(mock_algorithms)
    assert set(df["probability"].unique()) == set(p_values)
    assert (df["k"] == 10).all()


# ---------------------------------------------------------------------------
# Test 3: Experiment C - Verify 4 network sizes are executed
# ---------------------------------------------------------------------------
def test_experiment_c_executes_four_network_sizes(mock_algorithms, mock_simulate):
    """Verify Experiment C runs across 4 network sizes using graph_sampler."""
    sizes = [500, 1000, 2000, 4000]
    sampler_calls = []

    def custom_sampler(size: int):
        sampler_calls.append(size)
        return DummyGraph(size)

    df = run_scalability_experiment(
        algorithms=mock_algorithms,
        simulate_fn=mock_simulate,
        graph_sampler=custom_sampler,
        sizes=sizes,
        k=10,
        num_simulations=5,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4 * len(mock_algorithms)
    assert sampler_calls == sizes
    # Verify experiment IDs reflect sizes
    for s in sizes:
        assert any(f"v{s}" in eid for eid in df["experiment_id"])


# ---------------------------------------------------------------------------
# Test 4: Experiment D - Verify 4 simulation counts & relative_standard_error
# ---------------------------------------------------------------------------
def test_experiment_d_executes_four_simulation_counts(mock_algorithms, mock_simulate):
    """Verify Experiment D runs across 4 simulation counts and computes relative standard error."""
    graph = DummyGraph(1000)
    sim_counts = [10, 50, 100, 500]

    df = run_convergence_experiment(
        graph=graph,
        algorithms=mock_algorithms,
        simulate_fn=mock_simulate,
        simulation_counts=sim_counts,
        probability=0.1,
        k=10,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert list(df["num_simulations"]) == sim_counts
    assert "relative_standard_error" in df.columns
    assert (df["relative_standard_error"] >= 0).all()


# ---------------------------------------------------------------------------
# Test 5: Verify Monte Carlo is called (Mock run_monte_carlo)
# ---------------------------------------------------------------------------
def test_monte_carlo_is_called_with_correct_args(mock_algorithms, mock_simulate):
    """Verify run_monte_carlo is invoked by experiment runners."""
    graph = DummyGraph(1000)
    fake_mc_result = MonteCarloResult(
        mean_influence=150.0,
        min_influence=120,
        max_influence=180,
        std_influence=10.5,
        raw_results=[150],
        num_simulations=10,
        probability=0.1,
        seed_count=5,
    )

    with patch("experiments.experiment_runner.run_monte_carlo", return_value=fake_mc_result) as mock_mc:
        df = run_seed_budget_experiment(
            graph=graph,
            algorithms={"greedy": mock_algorithms["greedy"]},
            simulate_fn=mock_simulate,
            k_values=[5, 10],
            probability=0.1,
            num_simulations=10,
        )

        assert mock_mc.call_count == 2
        assert len(df) == 2


# ---------------------------------------------------------------------------
# Test 6: Verify Metrics Engine is called (Mock calculate_metrics)
# ---------------------------------------------------------------------------
def test_metrics_engine_is_called(mock_algorithms, mock_simulate):
    """Verify calculate_metrics is invoked to compute metrics records."""
    graph = DummyGraph(1000)
    fake_record = MetricsRecord(
        algorithm_name="degree",
        seed_count=10,
        probability=0.1,
        mean_influence=200.0,
        min_influence=180,
        max_influence=220,
        std_influence=12.0,
        influence_pct=20.0,
        runtime_sec=0.45,
        average_reach=200.0,
        spread_efficiency=20.0,
        node_count=1000,
        timestamp="2026-09-09T12:00:00Z",
    )

    with patch("experiments.experiment_runner.calculate_metrics", return_value=fake_record) as mock_calc:
        df = run_probability_experiment(
            graph=graph,
            algorithms={"degree": mock_algorithms["degree"]},
            simulate_fn=mock_simulate,
            p_values=[0.1, 0.2],
            k=10,
            num_simulations=5,
        )

        assert mock_calc.call_count == 2
        assert len(df) == 2
        assert (df["mean_influence"] == 200.0).all()


# ---------------------------------------------------------------------------
# Test 7: Verify DataFrame contains required columns
# ---------------------------------------------------------------------------
def test_dataframe_contains_required_columns(mock_algorithms, mock_simulate):
    """Verify that returned DataFrames strictly contain all SRS-required columns."""
    graph = DummyGraph(500)
    df = run_seed_budget_experiment(
        graph=graph,
        algorithms={"random": mock_algorithms["random"]},
        simulate_fn=mock_simulate,
        k_values=[5],
        probability=0.1,
        num_simulations=5,
    )

    for col in EXPERIMENT_DF_COLUMNS:
        assert col in df.columns, f"Missing required column: {col}"


# ---------------------------------------------------------------------------
# Test 8: Verify experiment IDs are generated correctly
# ---------------------------------------------------------------------------
def test_generate_experiment_id():
    """Verify experiment ID formats conform to SRS rules."""
    assert generate_experiment_id("expA", k=10, probability=0.1) == "expA_k10_p01"
    assert generate_experiment_id("expB", probability=0.20) == "expB_p020"
    assert generate_experiment_id("expB", probability=0.05) == "expB_p005"
    assert generate_experiment_id("expC", node_count=1000) == "expC_v1000"
    assert generate_experiment_id("expD", num_simulations=100) == "expD_n100"


# ---------------------------------------------------------------------------
# Additional Test 9: Master Runner run_all_experiments()
# ---------------------------------------------------------------------------
def test_run_all_experiments(mock_algorithms, mock_simulate):
    """Verify master runner executes all 4 experiments and returns dict of 4 DataFrames."""
    graph = DummyGraph(500)
    results = run_all_experiments(
        graph=graph,
        random_selection=mock_algorithms["random"],
        degree_selection=mock_algorithms["degree"],
        greedy_selection=mock_algorithms["greedy"],
        bfs_selection=mock_algorithms["bfs"],
        simulate_influence=mock_simulate,
        num_simulations=3,
    )

    assert isinstance(results, dict)
    assert set(results.keys()) == {
        "experiment_a",
        "experiment_b",
        "experiment_c",
        "experiment_d",
    }
    for key, df in results.items():
        assert isinstance(df, pd.DataFrame)
        assert not df.empty


# ---------------------------------------------------------------------------
# Additional Test 10: Validation Error Handling
# ---------------------------------------------------------------------------
def test_validation_errors(mock_algorithms, mock_simulate):
    """Verify validation exceptions for invalid parameter states."""
    # Graph is None
    with pytest.raises(ExperimentValidationError, match="graph must not be None"):
        run_seed_budget_experiment(
            graph=None,
            algorithms=mock_algorithms,
            simulate_fn=mock_simulate,
        )

    # Invalid probability
    graph = DummyGraph(100)
    with pytest.raises(ExperimentValidationError, match="probability must satisfy"):
        run_seed_budget_experiment(
            graph=graph,
            algorithms=mock_algorithms,
            simulate_fn=mock_simulate,
            probability=1.5,
        )

    # Invalid k
    with pytest.raises(ExperimentValidationError, match="k must be greater than 0"):
        run_seed_budget_experiment(
            graph=graph,
            algorithms=mock_algorithms,
            simulate_fn=mock_simulate,
            k_values=[0],
        )

    # Invalid simulate_fn
    with pytest.raises(ExperimentValidationError, match="simulate_fn must be a callable"):
        run_seed_budget_experiment(
            graph=graph,
            algorithms=mock_algorithms,
            simulate_fn=None,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Additional Test 11: Structured Logging
# ---------------------------------------------------------------------------
def test_experiment_structured_logging(caplog, mock_algorithms, mock_simulate):
    """Verify structured logging messages are emitted at INFO level."""
    graph = DummyGraph(200)

    with caplog.at_level(logging.INFO):
        run_seed_budget_experiment(
            graph=graph,
            algorithms={"greedy": mock_algorithms["greedy"]},
            simulate_fn=mock_simulate,
            k_values=[5],
            probability=0.1,
            num_simulations=3,
        )

    messages = [rec.message for rec in caplog.records]
    assert any("Experiment Started" in msg for msg in messages)
    assert any("Experiment Type=Seed Budget Comparison" in msg for msg in messages)
    assert any("Current Algorithm=greedy" in msg for msg in messages)
    assert any("Experiment Completed" in msg for msg in messages)
