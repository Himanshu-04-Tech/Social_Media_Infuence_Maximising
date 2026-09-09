"""Validation layer for the Data Storage Layer.

Verifies DataFrame integrity, file paths, and serialization safety before persistence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .exceptions import StorageValidationError
from .utilities import json_default_converter


def validate_dataframe(
    df: Any,
    required_columns: Sequence[str] | None = None,
    check_experiment_integrity: bool = False,
) -> None:
    """Validate that a DataFrame is populated and conforms to integrity requirements.

    Args:
        df: Target DataFrame.
        required_columns: Optional sequence of column names that must exist.
        check_experiment_integrity: If True, validates experiment_id and timestamp integrity.

    Raises:
        StorageValidationError: If DataFrame is None, not a DataFrame, empty, or missing required columns.
    """
    if df is None:
        raise StorageValidationError("DataFrame must not be None.")

    if not isinstance(df, pd.DataFrame):
        raise StorageValidationError(
            f"Expected pandas DataFrame, got: {type(df).__name__}."
        )

    if df.empty:
        raise StorageValidationError("DataFrame is empty. Cannot persist empty datasets.")

    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise StorageValidationError(
                f"DataFrame is missing required columns: {missing}. Present: {list(df.columns)}"
            )

    if check_experiment_integrity:
        if "experiment_id" not in df.columns:
            raise StorageValidationError(
                "Experiment DataFrame integrity check failed: missing 'experiment_id' column."
            )
        if df["experiment_id"].isnull().all():
            raise StorageValidationError(
                "Experiment DataFrame integrity check failed: 'experiment_id' column contains only null values."
            )


def validate_filename(filename: str) -> None:
    """Validate that proposed filename is non-empty and well-formed.

    Args:
        filename: Proposed filename string.

    Raises:
        StorageValidationError: If filename is empty or invalid.
    """
    if not isinstance(filename, str) or not filename.strip():
        raise StorageValidationError("Filename must be a non-empty string.")


def validate_json_serializable(data: Any) -> None:
    """Verify that a data structure is valid for JSON persistence.

    Args:
        data: Arbitrary data structure.

    Raises:
        StorageValidationError: If data cannot be serialized to JSON.
    """
    try:
        json.dumps(data, default=json_default_converter)
    except (TypeError, ValueError, OverflowError) as err:
        raise StorageValidationError(f"Data is not JSON serializable: {err}") from err


def validate_directory_exists(directory: Path | str) -> None:
    """Verify that a directory path exists on the filesystem.

    Args:
        directory: Path to verify.

    Raises:
        StorageValidationError: If path does not exist or is not a directory.
    """
    p = Path(directory)
    if not p.exists():
        raise StorageValidationError(f"Directory does not exist: {p}")
    if not p.is_dir():
        raise StorageValidationError(f"Path is not a directory: {p}")
