"""Utility functions for the Data Storage Layer.

Provides timestamp generation, filename sanitization, and JSON serialization converters.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def generate_timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Generate a sanitized UTC timestamp string suitable for filenames.

    Args:
        fmt: Datetime formatting string (default: '%Y%m%d_%H%M%S').

    Returns:
        Formatted UTC timestamp string.
    """
    return datetime.now(timezone.utc).strftime(fmt)


def sanitize_filename(name: str) -> str:
    """Sanitize a string for safe usage as a filesystem filename.

    Args:
        name: Proposed filename.

    Returns:
        Sanitized filename with unsafe characters replaced.
    """
    # Replace spaces with underscores and remove non-alphanumeric characters except dot, dash, underscore
    cleaned = re.sub(r"[^\w\-.]", "_", name.strip())
    # Collapse consecutive underscores
    return re.sub(r"_+", "_", cleaned)


def json_default_converter(obj: Any) -> Any:
    """JSON serializer fallback converter for non-primitive types.

    Handles Path objects, numpy scalar types, dataclasses, sets, and iterables.

    Args:
        obj: Object to serialize.

    Returns:
        JSON-compatible primitive or raises TypeError.
    """
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and callable(obj.item):  # numpy scalars
        return obj.item()
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
