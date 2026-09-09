"""Custom exceptions for the Monte Carlo Simulation Engine.

This module defines domain-specific exceptions for parameter validation
and execution errors arising during Monte Carlo simulations.
"""

from __future__ import annotations


class MonteCarloError(Exception):
    """Base exception class for all Monte Carlo module errors."""


class MonteCarloValidationError(MonteCarloError, ValueError):
    """Exception raised when input parameters fail validation checks.

    Inherits from both MonteCarloError and ValueError for flexible exception handling.
    """


class SimulationExecutionError(MonteCarloError, RuntimeError):
    """Exception raised when an underlying simulation function (Role 1) fails.

    Wraps Role 1 exceptions (e.g. ValueError, TypeError, RuntimeError) to decouple
    the engine from downstream failures while preserving error context.
    """
