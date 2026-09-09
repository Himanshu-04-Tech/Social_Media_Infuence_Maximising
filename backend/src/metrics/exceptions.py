"""Custom exceptions for the Metrics & Analytics Engine.

Provides domain-specific exceptions for parameter validation errors
during metrics calculation and aggregation.
"""

from __future__ import annotations


class MetricsError(Exception):
    """Base exception for all errors originating from the metrics module."""


class MetricsValidationError(MetricsError, ValueError):
    """Exception raised when input arguments fail metrics validation rules.

    Inherits from both MetricsError and ValueError for flexible exception handling.
    """
