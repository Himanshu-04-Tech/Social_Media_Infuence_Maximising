"""Custom exceptions for the Visualization Engine.

Defines domain-specific errors for chart data validation and figure rendering.
"""

from __future__ import annotations


class VisualizationError(Exception):
    """Base exception for all errors raised by the visualization engine."""


class VisualizationValidationError(VisualizationError, ValueError):
    """Exception raised when input data or chart configurations fail validation.

    Inherits from both VisualizationError and ValueError for standard error handling.
    """
