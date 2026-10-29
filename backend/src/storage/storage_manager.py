"""Core Storage Manager for the Social Network Influence Maximization Simulator.

Serves as the unified persistence and retrieval gateway for all experiment outputs,
metrics summaries, raw simulation JSONs, and dataset topology statistics.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .exceptions import StorageFileNotFoundError, StorageValidationError
from .file_manager import FileManager
from .models import StorageConfig, StorageRecord
from .utilities import generate_timestamp, json_default_converter, sanitize_filename
from .validators import (
    validate_dataframe,
    validate_filename,
    validate_json_serializable,
)

logger = logging.getLogger(__name__)


class StorageManager:
    """Central entry point for persisting and loading simulation and experiment artifacts."""

    def __init__(
        self,
        config: StorageConfig | None = None,
        custom_logger: logging.Logger | None = None,
    ) -> None:
        """Initialize StorageManager.

        Args:
            config: Optional StorageConfig; defaults to base_path='backend/results'.
            custom_logger: Optional logger instance.
        """
        self.config = config or StorageConfig()
        self.log = custom_logger or logger
        self.file_manager = FileManager(config=self.config, custom_logger=self.log)

    # -----------------------------------------------------------------------
    # Helper path resolvers
    # -----------------------------------------------------------------------
    def _resolve_file_path(self, category: str, filename_or_path: Path | str | StorageRecord) -> Path:
        """Resolve a filename or path to an absolute path within a category directory."""
        if isinstance(filename_or_path, StorageRecord):
            return Path(filename_or_path.file_path)
        p = Path(filename_or_path)
        if p.is_absolute():
            return p
        return self.file_manager.get_subdirectory(category) / p.name

    def _normalize_experiment_filename(self, experiment_name: str | Path | StorageRecord) -> str:
        """Normalize experiment identifier to canonical expX_results.csv."""
        if isinstance(experiment_name, StorageRecord):
            clean = experiment_name.file_name.strip()
        elif isinstance(experiment_name, Path):
            clean = experiment_name.name.strip()
        else:
            clean = str(experiment_name).strip()

        if clean.endswith(".csv"):
            return clean
        if clean.endswith("_results"):
            return f"{clean}.csv"
        return f"{clean}_results.csv"

    # -----------------------------------------------------------------------
    # EXPERIMENT RESULT STORAGE
    # -----------------------------------------------------------------------
    def save_experiment_dataframe(
        self,
        df: pd.DataFrame,
        experiment_name: str,
    ) -> Path:
        """Save an experiment comparison DataFrame to results/experiments/.

        File names follow the SRS canonical format:
            expA_results.csv, expB_results.csv, expC_results.csv, expD_results.csv

        Args:
            df: Populated experiment DataFrame.
            experiment_name: Identifier for experiment (e.g. 'expA', 'expB', 'expA_results').

        Returns:
            Path of the written CSV file.

        Raises:
            StorageValidationError: If DataFrame is empty or invalid.
        """
        if isinstance(df, (str, Path)) and isinstance(experiment_name, pd.DataFrame):
            df, experiment_name = experiment_name, str(df)

        validate_filename(experiment_name)
        validate_dataframe(df, check_experiment_integrity=True)

        filename = self._normalize_experiment_filename(experiment_name)
        target_path = self.file_manager.get_subdirectory("experiments") / filename

        try:
            df.to_csv(target_path, index=False)
            self.log.info(
                "File Saved | Category=experiments | Path=%s | Rows=%d",
                str(target_path),
                len(df),
                extra={
                    "event": "File Saved",
                    "file_type": "experiment",
                    "path": str(target_path),
                    "rows": len(df),
                },
            )
            self.log.info(
                "CSV Export Success | Path=%s",
                str(target_path),
                extra={"event": "CSV Export Success", "path": str(target_path)},
            )
            return target_path
        except Exception as err:
            self.log.error("Storage Error | Failed saving experiment DataFrame: %s", err)
            raise

    def load_experiment_results(
        self,
        experiment_name: str,
    ) -> pd.DataFrame:
        """Load an experiment DataFrame from results/experiments/.

        Args:
            experiment_name: Identifier (e.g. 'expA', 'expA_results', 'expA_results.csv').

        Returns:
            pd.DataFrame matching stored experiment metrics.

        Raises:
            StorageFileNotFoundError: If the target file does not exist.
        """
        filename = self._normalize_experiment_filename(experiment_name)
        target_path = self._resolve_file_path("experiments", filename)

        if not target_path.exists() or not target_path.is_file():
            self.log.error("Storage Error | Experiment results file not found: %s", str(target_path))
            raise StorageFileNotFoundError(f"Experiment results file not found: {target_path}")

        try:
            df = pd.read_csv(target_path)
            self.log.info(
                "File Loaded | Category=experiments | Path=%s | Rows=%d",
                str(target_path),
                len(df),
                extra={"event": "File Loaded", "path": str(target_path), "rows": len(df)},
            )
            return df
        except Exception as err:
            self.log.error("Storage Error | Failed to load experiment results: %s", err)
            raise

    # -----------------------------------------------------------------------
    # METRICS SUMMARY STORAGE
    # -----------------------------------------------------------------------
    def save_metrics_summary(
        self,
        df: pd.DataFrame,
        filename: str | None = None,
    ) -> Path:
        """Save a metrics summary DataFrame to results/metrics/.

        Naming convention: metrics_summary_TIMESTAMP.csv (unless filename provided).

        Args:
            df: Populated metrics comparison DataFrame.
            filename: Optional custom filename.

        Returns:
            Path of the written CSV file.

        Raises:
            StorageValidationError: If DataFrame is empty or invalid.
        """
        validate_dataframe(df)

        if filename is None:
            ts = generate_timestamp()
            target_name = f"metrics_summary_{ts}.csv"
        else:
            validate_filename(filename)
            target_name = filename if filename.endswith(".csv") else f"{filename}.csv"

        target_path = self.file_manager.get_subdirectory("metrics") / target_name

        try:
            df.to_csv(target_path, index=False)
            self.log.info(
                "File Saved | Category=metrics | Path=%s | Rows=%d",
                str(target_path),
                len(df),
                extra={
                    "event": "File Saved",
                    "file_type": "metrics",
                    "path": str(target_path),
                    "rows": len(df),
                },
            )
            self.log.info(
                "CSV Export Success | Path=%s",
                str(target_path),
                extra={"event": "CSV Export Success", "path": str(target_path)},
            )
            return target_path
        except Exception as err:
            self.log.error("Storage Error | Failed saving metrics summary: %s", err)
            raise

    def load_metrics_summary(
        self,
        filename: str | Path,
    ) -> pd.DataFrame:
        """Load a metrics summary DataFrame from results/metrics/.

        Args:
            filename: Filename or path to load.

        Returns:
            pd.DataFrame matching stored metrics summary.

        Raises:
            StorageFileNotFoundError: If file cannot be found.
        """
        target_path = self._resolve_file_path("metrics", filename)

        if not target_path.exists() or not target_path.is_file():
            self.log.error("Storage Error | Metrics summary file not found: %s", str(target_path))
            raise StorageFileNotFoundError(f"Metrics summary file not found: {target_path}")

        try:
            df = pd.read_csv(target_path)
            self.log.info(
                "File Loaded | Category=metrics | Path=%s | Rows=%d",
                str(target_path),
                len(df),
                extra={"event": "File Loaded", "path": str(target_path), "rows": len(df)},
            )
            return df
        except Exception as err:
            self.log.error("Storage Error | Failed to load metrics summary: %s", err)
            raise

    # -----------------------------------------------------------------------
    # RAW MONTE CARLO STORAGE
    # -----------------------------------------------------------------------
    def save_raw_simulation(
        self,
        algorithm_name: str,
        k: int,
        probability: float,
        monte_carlo_result: Any,
        filename: str | None = None,
    ) -> Path:
        """Save individual Monte Carlo raw simulation result to results/raw/.

        JSON structure:
        {
          "algorithm": "...",
          "k": 10,
          "probability": 0.1,
          "mean": ...,
          "min": ...,
          "max": ...,
          "std": ...,
          "raw_results": [...]
        }

        Filename format: raw_algorithm_k_probability_timestamp.json

        Args:
            algorithm_name: Algorithm name.
            k: Seed count.
            probability: Propagation probability.
            monte_carlo_result: MonteCarloResult instance or dict containing simulation stats.
            filename: Optional custom filename.

        Returns:
            Path of the written JSON file.
        """
        validate_filename(algorithm_name)

        # Extract values
        if isinstance(monte_carlo_result, dict):
            mean_val = monte_carlo_result.get("mean", monte_carlo_result.get("mean_influence", 0.0))
            min_val = monte_carlo_result.get("min", monte_carlo_result.get("min_influence", 0))
            max_val = monte_carlo_result.get("max", monte_carlo_result.get("max_influence", 0))
            std_val = monte_carlo_result.get("std", monte_carlo_result.get("std_influence", 0.0))
            raw_res = monte_carlo_result.get("raw_results", [])
        else:
            mean_val = getattr(monte_carlo_result, "mean_influence", 0.0)
            min_val = getattr(monte_carlo_result, "min_influence", 0)
            max_val = getattr(monte_carlo_result, "max_influence", 0)
            std_val = getattr(monte_carlo_result, "std_influence", 0.0)
            raw_res = getattr(monte_carlo_result, "raw_results", [])

        data = {
            "algorithm": algorithm_name,
            "k": k,
            "probability": probability,
            "mean": float(mean_val),
            "min": int(min_val),
            "max": int(max_val),
            "std": float(std_val),
            "raw_results": list(raw_res),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        validate_json_serializable(data)

        if filename is None:
            ts = generate_timestamp()
            safe_algo = sanitize_filename(algorithm_name)
            target_name = f"raw_{safe_algo}_{k}_{probability}_{ts}.json"
        else:
            validate_filename(filename)
            target_name = filename if filename.endswith(".json") else f"{filename}.json"

        target_path = self.file_manager.get_subdirectory("raw") / target_name

        try:
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=json_default_converter)

            self.log.info(
                "File Saved | Category=raw | Path=%s",
                str(target_path),
                extra={"event": "File Saved", "file_type": "raw", "path": str(target_path)},
            )
            self.log.info(
                "JSON Export Success | Path=%s",
                str(target_path),
                extra={"event": "JSON Export Success", "path": str(target_path)},
            )
            return target_path
        except Exception as err:
            self.log.error("Storage Error | Failed to save raw simulation: %s", err)
            raise

    def load_raw_simulation(
        self,
        filename: str | Path,
    ) -> dict[str, Any]:
        """Load raw Monte Carlo simulation record from results/raw/.

        Args:
            filename: Filename or path to load.

        Returns:
            Dictionary matching the stored JSON structure.

        Raises:
            StorageFileNotFoundError: If the file does not exist.
        """
        target_path = self._resolve_file_path("raw", filename)

        if not target_path.exists() or not target_path.is_file():
            self.log.error("Storage Error | Raw simulation file not found: %s", str(target_path))
            raise StorageFileNotFoundError(f"Raw simulation file not found: {target_path}")

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.log.info(
                "File Loaded | Category=raw | Path=%s",
                str(target_path),
                extra={"event": "File Loaded", "path": str(target_path)},
            )
            return data
        except Exception as err:
            self.log.error("Storage Error | Failed to load raw simulation: %s", err)
            raise

    # -----------------------------------------------------------------------
    # DATASET STATISTICS STORAGE
    # -----------------------------------------------------------------------
    def save_dataset_statistics(
        self,
        node_count: int,
        edge_count: int,
        degree_distribution: Any,
        filename: str | None = None,
    ) -> Path:
        """Save network graph topology statistics to results/dataset_stats/.

        Filename: dataset_stats_TIMESTAMP.json

        Args:
            node_count: Total vertices.
            edge_count: Total edges.
            degree_distribution: Degree counts or histogram mapping/list.
            filename: Optional custom filename.

        Returns:
            Path of the written JSON file.
        """
        data = {
            "node_count": int(node_count),
            "edge_count": int(edge_count),
            "degree_distribution": degree_distribution,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        validate_json_serializable(data)

        if filename is None:
            ts = generate_timestamp()
            target_name = f"dataset_stats_{ts}.json"
        else:
            validate_filename(filename)
            target_name = filename if filename.endswith(".json") else f"{filename}.json"

        target_path = self.file_manager.get_subdirectory("dataset_stats") / target_name

        try:
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=json_default_converter)

            self.log.info(
                "File Saved | Category=dataset_stats | Path=%s",
                str(target_path),
                extra={"event": "File Saved", "file_type": "dataset_stats", "path": str(target_path)},
            )
            self.log.info(
                "JSON Export Success | Path=%s",
                str(target_path),
                extra={"event": "JSON Export Success", "path": str(target_path)},
            )
            return target_path
        except Exception as err:
            self.log.error("Storage Error | Failed saving dataset statistics: %s", err)
            raise

    def load_dataset_statistics(
        self,
        filename: str | Path,
    ) -> dict[str, Any]:
        """Load network topology statistics from results/dataset_stats/.

        Args:
            filename: Filename or path to load.

        Returns:
            Dictionary matching stored dataset statistics JSON.

        Raises:
            StorageFileNotFoundError: If the file does not exist.
        """
        target_path = self._resolve_file_path("dataset_stats", filename)

        if not target_path.exists() or not target_path.is_file():
            self.log.error("Storage Error | Dataset stats file not found: %s", str(target_path))
            raise StorageFileNotFoundError(f"Dataset stats file not found: {target_path}")

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.log.info(
                "File Loaded | Category=dataset_stats | Path=%s",
                str(target_path),
                extra={"event": "File Loaded", "path": str(target_path)},
            )
            return data
        except Exception as err:
            self.log.error("Storage Error | Failed to load dataset statistics: %s", err)
            raise

    # -----------------------------------------------------------------------
    # REPRODUCIBILITY SUPPORT (FR-33)
    # -----------------------------------------------------------------------
    def load_experiment_configuration(
        self,
        experiment_id: str,
        experiment_name: str | None = None,
    ) -> dict[str, Any]:
        """Recover exact configuration parameters used for an experiment run (FR-33).

        Queries stored experiment CSVs to extract:
            k, probability, num_simulations, algorithm_name

        Args:
            experiment_id: Experiment ID to find (e.g. 'expA_k10_p01', 'expB_p020').
            experiment_name: Optional specific experiment name to search.

        Returns:
            Dictionary with keys: 'k', 'probability', 'num_simulations', 'algorithm_name'.

        Raises:
            StorageFileNotFoundError: If no stored experiment contains the experiment_id.
        """
        target_files: list[Path] = []
        exp_dir = self.file_manager.get_subdirectory("experiments")

        if experiment_name:
            target_files.append(exp_dir / self._normalize_experiment_filename(experiment_name))
        else:
            target_files = self.file_manager.list_files(exp_dir, pattern="*.csv")

        for csv_path in target_files:
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue

            if "experiment_id" in df.columns:
                matches = df[df["experiment_id"] == experiment_id]
                if not matches.empty:
                    row = matches.iloc[0]
                    self.log.info(
                        "Found experiment configuration for %s in %s",
                        experiment_id,
                        csv_path.name,
                    )
                    return {
                        "k": int(row["k"]) if "k" in row and pd.notnull(row["k"]) else None,
                        "probability": float(row["probability"]) if "probability" in row and pd.notnull(row["probability"]) else None,
                        "num_simulations": int(row["num_simulations"]) if "num_simulations" in row and pd.notnull(row["num_simulations"]) else None,
                        "algorithm_name": str(row["algorithm_name"]) if "algorithm_name" in row and pd.notnull(row["algorithm_name"]) else None,
                    }

        raise StorageFileNotFoundError(
            f"No stored experiment configuration found matching experiment_id='{experiment_id}'."
        )

    # -----------------------------------------------------------------------
    # CATALOG & UTILITIES
    # -----------------------------------------------------------------------
    def list_all_saved_results(self) -> dict[str, list[StorageRecord]]:
        """List all saved file records across all categories.

        Returns:
            Dictionary mapping category names to lists of StorageRecord objects.
        """
        catalog: dict[str, list[StorageRecord]] = {}

        for category in ["experiments", "metrics", "raw", "dataset_stats"]:
            dir_path = self.file_manager.get_subdirectory(category)
            files = self.file_manager.list_files(dir_path)
            records: list[StorageRecord] = []

            for f in files:
                stat = f.stat()
                created_dt = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat()
                records.append(
                    StorageRecord(
                        file_name=f.name,
                        file_path=str(f.resolve()),
                        created_at=created_dt,
                        file_size=stat.st_size,
                        file_type=category,
                    )
                )

            catalog[category] = records

        return catalog

    def delete_file(self, path: Path | str) -> bool:
        """Delete a file using the FileManager."""
        return self.file_manager.delete_file(path)

    def delete_saved_result(self, target: Path | str | StorageRecord, category: str | None = None) -> bool:
        """Delete a saved result artifact by StorageRecord, Path, or filename with optional category."""
        if isinstance(target, StorageRecord):
            return self.delete_file(target.file_path)
        if category and not Path(str(target)).is_absolute():
            resolved = self._resolve_file_path(category, target)
            return self.delete_file(resolved)
        return self.delete_file(target)

    def file_exists(self, path: Path | str) -> bool:
        """Check if a file exists using the FileManager."""
        return self.file_manager.file_exists(path)
