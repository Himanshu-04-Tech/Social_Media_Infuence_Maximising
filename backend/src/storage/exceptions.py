"""Custom exceptions for the Data Storage Layer.

Defines domain-specific exceptions for filesystem validation, serialization,
and persistence errors.
"""

from __future__ import annotations


class StorageError(Exception):
    """Base exception for all errors raised within the data storage module."""


class StorageValidationError(StorageError, ValueError):
    """Exception raised when data validation fails before storage (e.g., empty DataFrame).

    Inherits from both StorageError and ValueError for standard error handling.
    """


class StorageFileNotFoundError(StorageError, FileNotFoundError):
    """Exception raised when a requested file or resource cannot be located.

    Inherits from both StorageError and FileNotFoundError for flexible catch blocks.
    """
