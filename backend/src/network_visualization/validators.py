"""Validation layer for Network Visualization inputs and configurations.

Ensures that graphs, node collections, and visualization options conform
to module requirements before processing or rendering.
"""

from __future__ import annotations

from typing import Any, Collection

import networkx as nx

from .exceptions import NetworkVisualizationValidationError, NodeNotFoundError
from .models import NetworkVisualizationConfig


def validate_graph(graph: Any) -> None:
    """Validate that the graph object is valid and non-empty.

    Args:
        graph: NetworkX graph (Graph, DiGraph, etc.) or compatible object.

    Raises:
        NetworkVisualizationValidationError: If graph is None, not a graph structure,
                                             or contains 0 nodes.
    """
    if graph is None:
        raise NetworkVisualizationValidationError("Graph must not be None.")

    if not hasattr(graph, "nodes") or not hasattr(graph, "edges"):
        raise NetworkVisualizationValidationError(
            f"Expected a NetworkX graph or compatible structure with 'nodes' and 'edges'. Got: {type(graph).__name__}"
        )

    if len(graph.nodes) == 0:
        raise NetworkVisualizationValidationError("Graph must not be empty (contains 0 nodes).")


def validate_node_id(graph: Any, node_id: Any) -> None:
    """Validate that a specific node exists within the graph.

    Args:
        graph: Validated NetworkX graph.
        node_id: Identifier of the node to check.

    Raises:
        NodeNotFoundError: If node_id is not present in the graph.
    """
    if node_id not in graph:
        raise NodeNotFoundError(f"Node '{node_id}' does not exist in the graph.")


def validate_node_collections(
    graph: Any,
    seed_nodes: Collection[Any] | None = None,
    influenced_nodes: Collection[Any] | None = None,
) -> tuple[set[Any], set[Any]]:
    """Validate and normalize seed and influenced node collections.

    Args:
        graph: Validated NetworkX graph.
        seed_nodes: Collection of seed nodes (optional).
        influenced_nodes: Collection of influenced nodes (optional).

    Returns:
        Tuple of (seed_set, influenced_set).

    Raises:
        NetworkVisualizationValidationError: If collections are not iterables.
    """
    try:
        seeds = set(seed_nodes) if seed_nodes is not None else set()
    except TypeError as err:
        raise NetworkVisualizationValidationError(
            f"Seed nodes must be an iterable collection. Got: {type(seed_nodes).__name__}"
        ) from err

    try:
        influenced = set(influenced_nodes) if influenced_nodes is not None else set()
    except TypeError as err:
        raise NetworkVisualizationValidationError(
            f"Influenced nodes must be an iterable collection. Got: {type(influenced_nodes).__name__}"
        ) from err

    return seeds, influenced


def validate_config(config: Any) -> NetworkVisualizationConfig:
    """Validate and return a valid NetworkVisualizationConfig object.

    Args:
        config: Optional NetworkVisualizationConfig or None.

    Returns:
        Valid NetworkVisualizationConfig instance.

    Raises:
        NetworkVisualizationValidationError: If config values are invalid.
    """
    if config is None:
        return NetworkVisualizationConfig()

    if not isinstance(config, NetworkVisualizationConfig):
        raise NetworkVisualizationValidationError(
            f"Expected NetworkVisualizationConfig instance, got: {type(config).__name__}"
        )

    if config.max_nodes <= 0:
        raise NetworkVisualizationValidationError(
            f"max_nodes must be greater than 0. Got: {config.max_nodes}"
        )

    return config
