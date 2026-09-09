"""Custom exceptions for the Experiment Framework.

Defines domain-specific errors for experiment parameter validation and runner failures.
"""

from __future__ import annotations


class ExperimentError(Exception):
    """Base exception for all errors raised by the experiment framework."""


class ExperimentValidationError(ExperimentError, ValueError):
    """Exception raised when experiment configuration or inputs fail validation.

    Inherits from both ExperimentError and ValueError for standard error handling.
    """
