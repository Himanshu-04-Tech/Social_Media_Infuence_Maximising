"""Central Network Visualization Factory interface.

Provides a unified facade exposing both PyVis and Plotly rendering engines,
graph sampling routines, topological statistics, and node querying capabilities.
"""

from __future__ import annotations

import logging
from typing import Any, Collection

import networkx as nx
import plotly.graph_objects as go
from pyvis.network import Network

from .graph_processor import GraphProcessor
from .models import NetworkVisualizationConfig, NodeMetadata
from .plotly_renderer import PlotlyRenderer
from .pyvis_renderer import PyVisRenderer

logger = logging.getLogger(__name__)


class NetworkVisualizationFactory:
    """Unified facade and factory for network visualization and analytics."""

    def __init__(self, default_config: NetworkVisualizationConfig | None = None) -> None:
        self.config = default_config or NetworkVisualizationConfig()
        self.pyvis_renderer = PyVisRenderer(self.config)
        self.plotly_renderer = PlotlyRenderer(self.config)

    def render_pyvis(
        self,
        graph: Any,
        seed_nodes: Collection[Any] | None = None,
        influenced_nodes: Collection[Any] | None = None,
        config: NetworkVisualizationConfig | None = None,
    ) -> Network:
        """Render graph into an interactive PyVis Network object.

        Args:
            graph: NetworkX graph structure.
            seed_nodes: Optional collection of initial seed node IDs (SRS Red).
            influenced_nodes: Optional collection of influenced node IDs (SRS Green).
            config: Optional override configuration.

        Returns:
            pyvis.network.Network instance.
        """
        active_config = config or self.config
        return self.pyvis_renderer.render_network(
            graph=graph,
            seed_nodes=seed_nodes,
            influenced_nodes=influenced_nodes,
            config=active_config,
        )

    def render_plotly(
        self,
        graph: Any,
        seed_nodes: Collection[Any] | None = None,
        influenced_nodes: Collection[Any] | None = None,
        config: NetworkVisualizationConfig | None = None,
    ) -> go.Figure:
        """Render graph into an interactive Plotly Figure object.

        Args:
            graph: NetworkX graph structure.
            seed_nodes: Optional collection of initial seed node IDs (SRS Red).
            influenced_nodes: Optional collection of influenced node IDs (SRS Green).
            config: Optional override configuration.

        Returns:
            plotly.graph_objects.Figure instance.
        """
        active_config = config or self.config
        return self.plotly_renderer.render_plotly_network(
            graph=graph,
            seed_nodes=seed_nodes,
            influenced_nodes=influenced_nodes,
            config=active_config,
        )

    def sample_graph(
        self,
        graph: Any,
        max_nodes: int = 1000,
        method: str = "highest_degree",
        seed_nodes: Collection[Any] | None = None,
    ) -> nx.Graph:
        """Sample graph to limit size while preserving topological structure.

        Args:
            graph: NetworkX graph.
            max_nodes: Maximum nodes in sample.
            method: 'highest_degree', 'bfs', or 'random'.
            seed_nodes: Optional seed nodes to preserve in BFS.

        Returns:
            Sampled induced subgraph.
        """
        return GraphProcessor.sample_graph(
            graph=graph,
            max_nodes=max_nodes,
            method=method,
            seed_nodes=seed_nodes,
        )

    def get_network_statistics(self, graph: Any) -> dict[str, Any]:
        """Compute network summary statistics.

        Returns:
            Dictionary with nodes, edges, avg_degree, density, connected_components.
        """
        return GraphProcessor.get_network_statistics(graph)

    def find_node(
        self,
        graph: Any,
        node_id: Any,
        seed_nodes: Collection[Any] | None = None,
        influenced_nodes: Collection[Any] | None = None,
    ) -> NodeMetadata:
        """Locate a specific node and return its NodeMetadata."""
        return GraphProcessor.find_node(
            graph=graph,
            node_id=node_id,
            seed_nodes=seed_nodes,
            influenced_nodes=influenced_nodes,
        )

    def filter_nodes(
        self,
        graph: Any,
        node_type: str,
        seed_nodes: Collection[Any] | None = None,
        influenced_nodes: Collection[Any] | None = None,
    ) -> list[Any]:
        """Filter nodes by status: 'seed', 'influenced', or 'normal'."""
        return GraphProcessor.filter_nodes(
            graph=graph,
            node_type=node_type,
            seed_nodes=seed_nodes,
            influenced_nodes=influenced_nodes,
        )
