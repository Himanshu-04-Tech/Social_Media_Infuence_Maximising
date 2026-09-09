"""Data models for the Experiment Framework.

Encapsulates complete experiment execution results including DataFrames,
execution timings, and provenance metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ExperimentResult:
    """Represents the complete result of an executed experiment batch.

    Attributes:
        experiment_id: Unique identifier for this experiment execution.
        experiment_name: Human-readable name of the experiment (e.g. 'Seed Budget Comparison').
        metrics_df: Pandas DataFrame containing all evaluated algorithm metric records.
        execution_time_sec: Total duration taken to execute the experiment in seconds.
        created_at: ISO 8601 formatted timestamp indicating when the experiment concluded.
    """

    experiment_id: str
    experiment_name: str
    metrics_df: pd.DataFrame
    execution_time_sec: float
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary (excluding raw DataFrame)."""
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "row_count": len(self.metrics_df),
            "execution_time_sec": self.execution_time_sec,
            "created_at": self.created_at,
        }
