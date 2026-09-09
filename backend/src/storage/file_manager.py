"""File Manager for low-level filesystem operations in the Data Storage Layer.

Handles directory structure creation, file presence checks, deletion,
file listing, size querying, and timestamping.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from .exceptions import StorageFileNotFoundError
from .models import StorageConfig
from .utilities import generate_timestamp

logger = logging.getLogger(__name__)

# Standard subdirectories required by SRS under results/
REQUIRED_SUBDIRECTORIES: list[str] = [
    "raw",
    "experiments",
    "metrics",
    "dataset_stats",
]


class FileManager:
    """Manages filesystem directory structures and raw file I/O operations."""

    def __init__(
        self,
        config: StorageConfig | None = None,
        custom_logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the FileManager.

        Args:
            config: Optional StorageConfig; defaults to base_path='backend/results'.
            custom_logger: Optional logger instance.
        """
        self.config = config or StorageConfig()
        self.base_path = Path(self.config.base_path).resolve()
        self.log = custom_logger or logger

        if self.config.create_directories:
            self.create_directory_structure()

    def get_subdirectory(self, category: str) -> Path:
        """Retrieve the resolved Path for a specific results category.

        Args:
            category: One of 'raw', 'experiments', 'metrics', 'dataset_stats'.

        Returns:
            Resolved Path object.
        """
        sub_dir = self.base_path / category
        sub_dir.mkdir(parents=True, exist_ok=True)
        return sub_dir

    def create_directory_structure(self) -> dict[str, Path]:
        """Create the SRS-required directory tree under the base storage path.

        Creates:
            results/
            results/raw/
            results/experiments/
            results/metrics/
            results/dataset_stats/

        Returns:
            Dictionary mapping category names to created Path instances.
        """
        created_paths: dict[str, Path] = {"root": self.base_path}
        self.base_path.mkdir(parents=True, exist_ok=True)

        for subdir_name in REQUIRED_SUBDIRECTORIES:
            path = self.base_path / subdir_name
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                self.log.info(
                    "Directory Created | Path=%s",
                    str(path),
                    extra={"event": "Directory Created", "path": str(path)},
                )
            created_paths[subdir_name] = path

        return created_paths

    def file_exists(self, path: Path | str) -> bool:
        """Check if a file exists on the filesystem.

        Args:
            path: Target file path.

        Returns:
            True if file exists and is a regular file, False otherwise.
        """
        p = Path(path)
        return p.exists() and p.is_file()

    def delete_file(self, path: Path | str) -> bool:
        """Delete a file from the filesystem.

        Args:
            path: Target file path.

        Returns:
            True if file was deleted, False if file did not exist.

        Raises:
            StorageFileNotFoundError: If path does not exist when strict deletion expected.
        """
        p = Path(path)
        if not p.exists():
            self.log.warning("Attempted to delete non-existent file: %s", str(p))
            return False

        try:
            p.unlink()
            self.log.info(
                "File Deleted | Path=%s",
                str(p),
                extra={"event": "File Deleted", "path": str(p)},
            )
            return True
        except OSError as err:
            self.log.error("Storage Error | Failed to delete file %s: %s", str(p), err)
            raise

    def list_files(
        self,
        directory: Path | str,
        pattern: str = "*",
    ) -> list[Path]:
        """List files matching a glob pattern within a directory.

        Args:
            directory: Directory to search.
            pattern: Glob pattern (default: '*').

        Returns:
            Sorted list of matching Path objects.
        """
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            return []
        return sorted([p for p in dir_path.glob(pattern) if p.is_file()])

    def get_file_size(self, path: Path | str) -> int:
        """Get the size of a file in bytes.

        Args:
            path: Target file path.

        Returns:
            Size in bytes.

        Raises:
            StorageFileNotFoundError: If the file does not exist.
        """
        p = Path(path)
        if not p.exists() or not p.is_file():
            raise StorageFileNotFoundError(f"File not found: {p}")
        return p.stat().st_size

    def generate_timestamp(self) -> str:
        """Generate a formatted timestamp string."""
        return generate_timestamp()
