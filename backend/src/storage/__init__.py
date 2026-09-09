"""Data Storage Layer Package for Social Network Influence Maximization Simulator.

Exposes StorageManager, FileManager, storage configurations, models, and custom exceptions.
"""

from .exceptions import (
    StorageError,
    StorageFileNotFoundError,
    StorageValidationError,
)
from .file_manager import FileManager
from .models import StorageConfig, StorageRecord
from .storage_manager import StorageManager
from .validators import (
    validate_dataframe,
    validate_directory_exists,
    validate_filename,
    validate_json_serializable,
)

__all__ = [
    "StorageManager",
    "FileManager",
    "StorageConfig",
    "StorageRecord",
    "StorageError",
    "StorageValidationError",
    "StorageFileNotFoundError",
    "validate_dataframe",
    "validate_filename",
    "validate_json_serializable",
    "validate_directory_exists",
]
