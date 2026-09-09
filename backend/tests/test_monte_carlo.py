"""Unit test suite for the Monte Carlo Simulation Engine.

Verifies:
1. Statistical calculations on known datasets (mean, min, max, stdev)
2. Validation rules for probability boundaries
3. Validation rules for non-empty seed collections
4. Validation rules for simulation counts
5. Dependency injection and exact invocation count of simulate_fn
6. Deterministic reproducibility with random_seed
7. Graph existence validation
8. Role 1 error re-raising as SimulationExecutionError
9. Edge case with N=1 simulation
10. Structured logging output
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure backend/src is in sys.path for direct test execution
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monte_carlo import (
    MonteCarloError,
    MonteCarloResult,
    MonteCarloValidationError,
    SimulationExecutionError,
    run_monte_carlo,
)


class DummyGraph:
    """Mock graph object compatible with network specifications."""
    def __init__(self, node_count: int = 100):
        self.node_count = node_count


# ---------------------------------------------------------------------------
# Test 1: Known results [10, 20, 30, 40, 50]
# ---------------------------------------------------------------------------
def test_monte_carlo_known_results():
    """Verify statistical calculations against a known sequence of simulation outputs.

    Known values: [10, 20, 30, 40, 50]
    Expected:
      mean = 30.0
      min = 10
      max = 50
      std = sqrt(250) ≈ 15.811388
    """
    known_values = [10, 20, 30, 40, 50]
    iterator = iter(known_values)

    def mock_simulate(graph, seeds, probability):
        return next(iterator)

    graph = DummyGraph()
    seeds = [1, 2, 3]
    probability = 0.2
    num_simulations = len(known_values)

    result = run_monte_carlo(
        graph=graph,
        seeds=seeds,
        probability=probability,
        num_simulations=num_simulations,
        simulate_fn=mock_simulate,
    )

    assert isinstance(result, MonteCarloResult)
    assert result.mean_influence == pytest.approx(30.0)
    assert result.min_influence == 10
    assert result.max_influence == 50
    assert result.std_influence == pytest.approx(15.811388, rel=1e-4)
    assert result.raw_results == known_values
    assert result.num_simulations == 5
    assert result.probability == 0.2
    assert result.seed_count == 3


# ---------------------------------------------------------------------------
# Test 2: Invalid probability (-0.1, 0.0, 1.5) must raise exception
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("invalid_prob", [-0.1, 0.0, -1.0, 1.0001, 2.0])
def test_invalid_probability_raises_validation_error(invalid_prob):
    """Verify that probability <= 0 or > 1 raises MonteCarloValidationError."""
    graph = DummyGraph()
    seeds = [1]
    mock_simulate = MagicMock(return_value=10)

    with pytest.raises(MonteCarloValidationError, match="Probability must satisfy 0 < probability <= 1"):
        run_monte_carlo(
            graph=graph,
            seeds=seeds,
            probability=invalid_prob,
            num_simulations=10,
            simulate_fn=mock_simulate,
        )


# ---------------------------------------------------------------------------
# Test 3: Empty seeds must fail
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("empty_seeds", [[], set(), ()])
def test_empty_seeds_raises_validation_error(empty_seeds):
    """Verify that an empty seed collection raises MonteCarloValidationError."""
    graph = DummyGraph()
    mock_simulate = MagicMock(return_value=10)

    with pytest.raises(MonteCarloValidationError, match="Seeds collection must not be empty"):
        run_monte_carlo(
            graph=graph,
            seeds=empty_seeds,
            probability=0.1,
            num_simulations=10,
            simulate_fn=mock_simulate,
        )


def test_none_seeds_raises_validation_error():
    """Verify that None for seeds raises MonteCarloValidationError."""
    graph = DummyGraph()
    mock_simulate = MagicMock(return_value=10)

    with pytest.raises(MonteCarloValidationError, match="Seeds must not be None"):
        run_monte_carlo(
            graph=graph,
            seeds=None,  # type: ignore[arg-type]
            probability=0.1,
            num_simulations=10,
            simulate_fn=mock_simulate,
        )


# ---------------------------------------------------------------------------
# Test 4: num_simulations <= 0 must fail
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("invalid_sims", [0, -1, -50])
def test_invalid_num_simulations_raises_validation_error(invalid_sims):
    """Verify that num_simulations <= 0 raises MonteCarloValidationError."""
    graph = DummyGraph()
    seeds = [1, 2]
    mock_simulate = MagicMock(return_value=10)

    with pytest.raises(MonteCarloValidationError, match="num_simulations must be greater than 0"):
        run_monte_carlo(
            graph=graph,
            seeds=seeds,
            probability=0.1,
            num_simulations=invalid_sims,
            simulate_fn=mock_simulate,
        )


# ---------------------------------------------------------------------------
# Test 5: Mock simulate_fn called exactly N times
# ---------------------------------------------------------------------------
def test_simulate_fn_called_exactly_n_times():
    """Verify that simulate_fn is called exactly num_simulations times with matching args."""
    mock_simulate = MagicMock(side_effect=[12, 15, 14, 13, 16, 11, 15])
    graph = DummyGraph()
    seeds = [10, 20]
    probability = 0.25
    n = 7

    result = run_monte_carlo(
        graph=graph,
        seeds=seeds,
        probability=probability,
        num_simulations=n,
        simulate_fn=mock_simulate,
    )

    assert mock_simulate.call_count == n
    for call in mock_simulate.call_args_list:
        args, _ = call
        assert args == (graph, seeds, probability)
    assert len(result.raw_results) == n


# ---------------------------------------------------------------------------
# Test 6: Deterministic reproducibility with random_seed
# ---------------------------------------------------------------------------
def test_deterministic_reproducibility():
    """Verify that providing random_seed results in reproducible simulation outcomes."""
    def stochastic_simulate(graph, seeds, probability):
        # Relies on random.random() which should be seeded
        return sum(1 for _ in range(100) if random.random() < probability)

    graph = DummyGraph()
    seeds = [1, 2, 3]

    run1 = run_monte_carlo(
        graph=graph,
        seeds=seeds,
        probability=0.3,
        num_simulations=20,
        simulate_fn=stochastic_simulate,
        random_seed=42,
    )

    run2 = run_monte_carlo(
        graph=graph,
        seeds=seeds,
        probability=0.3,
        num_simulations=20,
        simulate_fn=stochastic_simulate,
        random_seed=42,
    )

    assert run1.raw_results == run2.raw_results
    assert run1.mean_influence == run2.mean_influence
    assert run1.std_influence == run2.std_influence


# ---------------------------------------------------------------------------
# Test 7: Graph is None raises validation error
# ---------------------------------------------------------------------------
def test_none_graph_raises_validation_error():
    """Verify that passing graph=None raises MonteCarloValidationError."""
    mock_simulate = MagicMock(return_value=10)
    with pytest.raises(MonteCarloValidationError, match="Graph must not be None"):
        run_monte_carlo(
            graph=None,
            seeds=[1],
            probability=0.1,
            num_simulations=5,
            simulate_fn=mock_simulate,
        )


# ---------------------------------------------------------------------------
# Test 8: Role 1 error handling (ValueError, TypeError, RuntimeError)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "role1_exc",
    [
        ValueError("Seed node does not exist in graph"),
        TypeError("Graph is not subscriptable"),
        RuntimeError("Max recursion depth exceeded in cascade"),
    ],
)
def test_role1_exceptions_rethrown_as_simulation_execution_error(role1_exc):
    """Verify Role 1 exceptions are converted into SimulationExecutionError."""
    mock_simulate = MagicMock(side_effect=role1_exc)
    graph = DummyGraph()
    seeds = [1]

    with pytest.raises(SimulationExecutionError) as exc_info:
        run_monte_carlo(
            graph=graph,
            seeds=seeds,
            probability=0.1,
            num_simulations=5,
            simulate_fn=mock_simulate,
        )

    # Check exception chaining and message content
    assert exc_info.value.__cause__ is role1_exc
    assert "Underlying simulation failed during iteration 1" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 9: Edge case - num_simulations = 1
# ---------------------------------------------------------------------------
def test_single_simulation_returns_zero_std():
    """Verify that num_simulations=1 produces std_influence=0.0 gracefully."""
    mock_simulate = MagicMock(return_value=42)
    graph = DummyGraph()
    seeds = [1, 2]

    result = run_monte_carlo(
        graph=graph,
        seeds=seeds,
        probability=0.5,
        num_simulations=1,
        simulate_fn=mock_simulate,
    )

    assert result.mean_influence == 42.0
    assert result.min_influence == 42
    assert result.max_influence == 42
    assert result.std_influence == 0.0
    assert result.raw_results == [42]


# ---------------------------------------------------------------------------
# Test 10: Structured logging emission
# ---------------------------------------------------------------------------
def test_structured_logging(caplog):
    """Verify that structured log messages are emitted at INFO level."""
    mock_simulate = MagicMock(return_value=25)
    graph = DummyGraph()

    with caplog.at_level(logging.INFO):
        run_monte_carlo(
            graph=graph,
            seeds=[1, 2],
            probability=0.15,
            num_simulations=3,
            simulate_fn=mock_simulate,
        )

    log_records = [rec.message for rec in caplog.records]
    assert any("Simulation Started" in msg for msg in log_records)
    assert any("Simulation Finished" in msg for msg in log_records)
    assert any("Mean Influence" in msg for msg in log_records)
    assert any("Std Influence" in msg for msg in log_records)
    assert any("Execution Time" in msg for msg in log_records)
