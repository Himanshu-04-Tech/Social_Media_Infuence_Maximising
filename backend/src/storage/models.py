"""Data models for the Data Storage Layer.

Defines configuration containers and metadata records for stored artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StorageConfig:
    """Configuration container for storage root and directory provisioning.

    Attributes:
        base_path: Root filesystem path under which result directories are created.
        create_directories: If True, auto-provisions subdirectories upon initialization.
    """

    base_path: Path = field(default_factory=lambda: Path("backend/results"))
    create_directories: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.base_path, str):
            self.base_path = Path(self.base_path)


@dataclass
class StorageRecord:
    """Metadata descriptor for a persisted file artifact.

    Attributes:
        file_name: Base filename including extension.
        file_path: Absolute or canonical filesystem path string.
        created_at: ISO 8601 creation timestamp string.
        file_size: File size in bytes.
        file_type: Categorical artifact type (e.g. 'experiment', 'metrics', 'raw', 'dataset_stats').
    """

    file_name: str
    file_path: str
    created_at: str
    file_size: int
    file_type: str

    def __str__(self) -> str:
        return self.file_name

    def __fspath__(self) -> str:
        return str(self.file_path)

    def encode(self, *args: any, **kwargs: any) -> bytes:
        return self.file_name.encode(*args, **kwargs)

    def strip(self, *args: any, **kwargs: any) -> str:
        return self.file_name.strip(*args, **kwargs)

    def endswith(self, *args: any, **kwargs: any) -> bool:
        return self.file_name.endswith(*args, **kwargs)

    def startswith(self, *args: any, **kwargs: any) -> bool:
        return self.file_name.startswith(*args, **kwargs)

    def replace(self, *args: any, **kwargs: any) -> str:
        return self.file_name.replace(*args, **kwargs)

    def lower(self) -> str:
        return self.file_name.lower()

