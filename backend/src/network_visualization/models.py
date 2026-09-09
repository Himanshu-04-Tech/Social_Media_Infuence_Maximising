"""Data models for the Network Visualization module.

Defines configuration options and node metadata dataclasses used by
graph processors and renderers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class NetworkVisualizationConfig:
    """Configuration options for interactive network visualization rendering.

    Attributes:
        max_nodes: Maximum number of nodes to render. Graphs exceeding this
                   limit will be automatically sampled according to the selected strategy.
        physics_enabled: Whether physics force simulation (e.g. Barnes-Hut) is active.
        show_labels: Whether node ID labels are displayed directly on the graph.
        width: Canvas width (CSS string or pixel dimension).
        height: Canvas height (CSS string or pixel dimension).
    """

    max_nodes: int = 1000
    physics_enabled: bool = True
    show_labels: bool = False
    width: str = "100%"
    height: str = "800px"


@dataclass
class NodeMetadata:
    """Encapsulates visualization metadata and simulation state for a single graph node.

    Attributes:
        node_id: Unique identifier of the node (int or comparable key).
        degree: Degree (number of connected incident edges) of the node.
        is_seed: Whether this node is an initial seed node (SRS color: red).
        is_influenced: Whether this node was reached/influenced during simulation (SRS color: green).
        color: Visual color string corresponding to SRS requirements ("red", "green", or "blue").
        size: Visual radius/size computed according to degree thresholds (10, 20, or 30).
    """

    node_id: Any
    degree: int
    is_seed: bool
    is_influenced: bool
    color: str
    size: int
