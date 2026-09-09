"""Unit test suite for Module 4: Data Storage Layer.

Verifies:
1. SRS Directory structure created successfully.
2. Experiment DataFrame saved and file exists with expX_results.csv naming.
3. Metrics summary saved and CSV readable.
4. Raw Monte Carlo simulation JSON saved and content verified.
5. Dataset statistics JSON saved and loaded with identical values.
6. Load experiment results verifies DataFrame shape and data preservation.
7. File deletion removes target file.
8. Invalid DataFrame raises validation exception.
9. FR-33 Reproducibility: load_experiment_configuration recovers exact parameters.
10. Cataloging via list_all_saved_results.
11. Structured logging emission for storage operations.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monte_carlo import MonteCarloResult
from storage import (
    FileManager,
    StorageConfig,
    StorageFileNotFoundError,
    StorageManager,
    StorageValidationError,
)


@pytest.fixture
def temp_storage(tmp_path: Path) -> StorageManager:
    """Fixture providing a StorageManager with isolated temporary directory."""
    config = StorageConfig(base_path=tmp_path / "results", create_directories=True)
    return StorageManager(config=config)


@pytest.fixture
def sample_experiment_df() -> pd.DataFrame:
    """Sample DataFrame conforming to SRS experiment columns."""
    return pd.DataFrame([
        {
            "experiment_id": "expA_k10_p01",
            "algorithm_name": "greedy",
            "k": 10,
            "probability": 0.1,
            "num_simulations": 50,
            "mean_influence": 842.5,
            "std_influence": 88.3,
            "influence_pct": 20.8,
            "runtime_sec": 2.45,
            "spread_efficiency": 84.25,
        },
        {
            "experiment_id": "expA_k10_p01",
            "algorithm_name": "degree",
            "k": 10,
            "probability": 0.1,
            "num_simulations": 50,
            "mean_influence": 620.0,
            "std_influence": 70.1,
            "influence_pct": 15.3,
            "runtime_sec": 0.85,
            "spread_efficiency": 62.0,
        },
    ])


# ---------------------------------------------------------------------------
# Test 1: Directory Structure Created Successfully
# ---------------------------------------------------------------------------
def test_directory_structure_created(tmp_path: Path):
    """Verify that all required SRS subdirectories are auto-created."""
    base_dir = tmp_path / "custom_results"
    fm = FileManager(StorageConfig(base_path=base_dir, create_directories=True))

    assert base_dir.exists()
    assert (base_dir / "raw").is_dir()
    assert (base_dir / "experiments").is_dir()
    assert (base_dir / "metrics").is_dir()
    assert (base_dir / "dataset_stats").is_dir()


# ---------------------------------------------------------------------------
# Test 2: Experiment DataFrame Saved (Verify File Exists)
# ---------------------------------------------------------------------------
def test_save_experiment_dataframe_creates_file(temp_storage, sample_experiment_df):
    """Verify experiment DataFrame is saved with canonical filename expA_results.csv."""
    path = temp_storage.save_experiment_dataframe(sample_experiment_df, "expA")

    assert path.exists()
    assert path.name == "expA_results.csv"
    assert path.parent.name == "experiments"
    assert temp_storage.file_exists(path)


# ---------------------------------------------------------------------------
# Test 3: Metrics Summary Saved (Verify CSV Readable)
# ---------------------------------------------------------------------------
def test_save_metrics_summary_is_readable(temp_storage):
    """Verify metrics summary CSV is written and readable."""
    df_metrics = pd.DataFrame([
        {"algorithm_name": "greedy", "mean_influence": 842.5, "influence_pct": 20.8},
        {"algorithm_name": "random", "mean_influence": 310.0, "influence_pct": 7.6},
    ])

    saved_path = temp_storage.save_metrics_summary(df_metrics)

    assert saved_path.exists()
    assert "metrics_summary_" in saved_path.name
    loaded_df = pd.read_csv(saved_path)
    assert len(loaded_df) == 2
    assert list(loaded_df["algorithm_name"]) == ["greedy", "random"]


# ---------------------------------------------------------------------------
# Test 4: Raw Monte Carlo JSON Saved (Verify Content Matches)
# ---------------------------------------------------------------------------
def test_save_raw_simulation_content_matches(temp_storage):
    """Verify raw Monte Carlo simulation JSON is persisted with full metadata."""
    mc_result = MonteCarloResult(
        mean_influence=842.5,
        min_influence=701,
        max_influence=1003,
        std_influence=88.3,
        raw_results=[750, 840, 930],
        num_simulations=3,
        probability=0.1,
        seed_count=10,
    )

    saved_path = temp_storage.save_raw_simulation(
        algorithm_name="greedy",
        k=10,
        probability=0.1,
        monte_carlo_result=mc_result,
    )

    assert saved_path.exists()
    data = temp_storage.load_raw_simulation(saved_path.name)
    assert data["algorithm"] == "greedy"
    assert data["k"] == 10
    assert data["probability"] == 0.1
    assert data["mean"] == 842.5
    assert data["min"] == 701
    assert data["max"] == 1003
    assert data["std"] == 88.3
    assert data["raw_results"] == [750, 840, 930]


# ---------------------------------------------------------------------------
# Test 5: Dataset Statistics JSON Saved (Verify Load Same Values)
# ---------------------------------------------------------------------------
def test_save_dataset_statistics_values_match(temp_storage):
    """Verify dataset topology stats JSON is saved and loaded accurately."""
    degrees = {"1": 15, "2": 42, "3": 108}
    saved_path = temp_storage.save_dataset_statistics(
        node_count=4039,
        edge_count=88234,
        degree_distribution=degrees,
    )

    assert saved_path.exists()
    loaded_data = temp_storage.load_dataset_statistics(saved_path.name)
    assert loaded_data["node_count"] == 4039
    assert loaded_data["edge_count"] == 88234
    assert loaded_data["degree_distribution"] == degrees
    assert "timestamp" in loaded_data


# ---------------------------------------------------------------------------
# Test 6: Load Experiment File (Verify DataFrame Shape Preserved)
# ---------------------------------------------------------------------------
def test_load_experiment_results_preserves_shape(temp_storage, sample_experiment_df):
    """Verify that loaded experiment DataFrame preserves dimensions and values."""
    temp_storage.save_experiment_dataframe(sample_experiment_df, "expA")
    loaded_df = temp_storage.load_experiment_results("expA")

    assert loaded_df.shape == sample_experiment_df.shape
    pd.testing.assert_frame_equal(loaded_df, sample_experiment_df)


# ---------------------------------------------------------------------------
# Test 7: Delete File (Verify File Removed)
# ---------------------------------------------------------------------------
def test_delete_file_removes_file(temp_storage, sample_experiment_df):
    """Verify that deleting a file removes it from disk."""
    saved_path = temp_storage.save_experiment_dataframe(sample_experiment_df, "expB")
    assert saved_path.exists()

    deleted = temp_storage.delete_file(saved_path)
    assert deleted is True
    assert not saved_path.exists()
    assert not temp_storage.file_exists(saved_path)


# ---------------------------------------------------------------------------
# Test 8: Invalid DataFrame Raises Validation Exception
# ---------------------------------------------------------------------------
def test_invalid_dataframe_raises_validation_error(temp_storage):
    """Verify that empty DataFrames or None raise StorageValidationError."""
    empty_df = pd.DataFrame()

    with pytest.raises(StorageValidationError, match="DataFrame is empty"):
        temp_storage.save_experiment_dataframe(empty_df, "expA")

    with pytest.raises(StorageValidationError, match="must not be None"):
        temp_storage.save_experiment_dataframe(None, "expA")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test 9: FR-33 Reproducibility Configuration Retrieval
# ---------------------------------------------------------------------------
def test_load_experiment_configuration_fr33(temp_storage, sample_experiment_df):
    """Verify FR-33 requirement: recovering k, probability, num_simulations from stored data."""
    temp_storage.save_experiment_dataframe(sample_experiment_df, "expA")

    config = temp_storage.load_experiment_configuration("expA_k10_p01")
    assert config == {
        "k": 10,
        "probability": 0.1,
        "num_simulations": 50,
        "algorithm_name": "greedy",
    }


def test_load_nonexistent_file_raises_error(temp_storage):
    """Verify that attempting to load a missing file raises StorageFileNotFoundError."""
    with pytest.raises(StorageFileNotFoundError):
        temp_storage.load_experiment_results("non_existent_exp")

    with pytest.raises(StorageFileNotFoundError):
        temp_storage.load_metrics_summary("missing_metrics.csv")


# ---------------------------------------------------------------------------
# Test 10: Cataloging via list_all_saved_results
# ---------------------------------------------------------------------------
def test_list_all_saved_results(temp_storage, sample_experiment_df):
    """Verify list_all_saved_results catalogs all saved files with metadata."""
    temp_storage.save_experiment_dataframe(sample_experiment_df, "expA")
    temp_storage.save_metrics_summary(sample_experiment_df)
    temp_storage.save_dataset_statistics(100, 200, {})

    catalog = temp_storage.list_all_saved_results()
    assert len(catalog["experiments"]) == 1
    assert catalog["experiments"][0].file_name == "expA_results.csv"
    assert catalog["experiments"][0].file_size > 0
    assert len(catalog["metrics"]) == 1
    assert len(catalog["dataset_stats"]) == 1


# ---------------------------------------------------------------------------
# Test 11: Structured Logging
# ---------------------------------------------------------------------------
def test_storage_structured_logging(caplog, temp_storage, sample_experiment_df):
    """Verify structured logs are emitted for storage operations."""
    with caplog.at_level(logging.INFO):
        p = temp_storage.save_experiment_dataframe(sample_experiment_df, "expC")
        temp_storage.load_experiment_results("expC")
        temp_storage.delete_file(p)

    messages = [rec.message for rec in caplog.records]
    assert any("File Saved" in msg for msg in messages)
    assert any("CSV Export Success" in msg for msg in messages)
    assert any("File Loaded" in msg for msg in messages)
    assert any("File Deleted" in msg for msg in messages)
