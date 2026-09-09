"""Custom exceptions for the Network Visualization module.

Defines domain-specific exceptions for network graph validation, rendering,
and querying operations.
"""

from __future__ import annotations


class NetworkVisualizationError(Exception):
    """Base exception for all network visualization errors."""


class NetworkVisualizationValidationError(NetworkVisualizationError, ValueError):
    """Raised when an input graph, configuration, or node collection fails validation."""


class NodeNotFoundError(NetworkVisualizationError, KeyError):
    """Raised when a specified node ID does not exist in the graph."""
