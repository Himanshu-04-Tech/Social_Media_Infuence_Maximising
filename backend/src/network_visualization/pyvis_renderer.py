"""Interactive network renderer using PyVis (Vis.js).

Produces zoomable, pannable, physics-enabled network graphs with rich tooltips,
SRS color highlights, and interactive node drag and selection capabilities.
"""

from __future__ import annotations

import logging
from typing import Any, Collection

import networkx as nx
from pyvis.network import Network

from .graph_processor import GraphProcessor
from .models import NetworkVisualizationConfig
from .validators import validate_config, validate_graph

logger = logging.getLogger(__name__)


class PyVisRenderer:
    """Renders NetworkX graphs into interactive PyVis Network objects."""

    def __init__(self, default_config: NetworkVisualizationConfig | None = None) -> None:
        self.default_config = default_config or NetworkVisualizationConfig()

    def render_network(
        self,
        graph: Any,
        seed_nodes: Collection[Any] | None = None,
        influenced_nodes: Collection[Any] | None = None,
        config: NetworkVisualizationConfig | None = None,
    ) -> Network:
        """Render a graph into an interactive PyVis Network visualization.

        Features:
            - SRS Color Coding: Red = Seeds, Green = Influenced, Blue = Normal
            - Degree-scaled node sizing (10, 20, 30)
            - Rich HTML tooltips (Node ID, Degree, Seed Status, Influenced Status)
            - Interactive physics simulation via Barnes-Hut
            - Full zoom, pan, drag, and selection interaction

        Args:
            graph: NetworkX graph structure.
            seed_nodes: Optional collection of initial seed node IDs.
            influenced_nodes: Optional collection of influenced node IDs.
            config: Optional NetworkVisualizationConfig override.

        Returns:
            Configured pyvis.network.Network object.
        """
        validate_graph(graph)
        active_config = validate_config(config or self.default_config)

        logger.info("Network Rendering Started (PyVis)")

        # Handle automated graph sampling if node count exceeds maximum threshold
        effective_graph = graph
        if graph.number_of_nodes() > active_config.max_nodes:
            logger.info("Node Sampling Applied: %d nodes > %d max_nodes", graph.number_of_nodes(), active_config.max_nodes)
            effective_graph = GraphProcessor.sample_graph(
                graph,
                max_nodes=active_config.max_nodes,
                method="highest_degree",
                seed_nodes=seed_nodes,
            )

        # Build node metadata with SRS color assignments and sizing
        node_metadata = GraphProcessor.build_node_metadata(
            effective_graph,
            seed_nodes=seed_nodes,
            influenced_nodes=influenced_nodes,
        )

        is_directed = nx.is_directed(effective_graph)
        net = Network(
            height=active_config.height,
            width=active_config.width,
            directed=is_directed,
            notebook=False,
            cdn_resources="remote",
        )

        # Configure physics layout according to config
        if active_config.physics_enabled:
            net.barnes_hut()
        else:
            net.toggle_physics(False)

        # Add Nodes with rich tooltip and SRS attributes
        for node_id, meta in node_metadata.items():
            status_label = "Seed" if meta.is_seed else ("Influenced" if meta.color == "green" else "Normal")
            tooltip = (
                f"Node ID: {node_id}\n"
                f"Degree: {meta.degree}\n"
                f"Seed Status: {'Yes' if meta.is_seed else 'No'}\n"
                f"Influenced Status: {'Yes' if meta.is_influenced else 'No'}"
            )
            label = str(node_id) if active_config.show_labels else ""

            net.add_node(
                node_id,
                label=label,
                title=tooltip,
                color=meta.color,
                size=meta.size,
                status=status_label,
                degree=meta.degree,
            )

        logger.info("Nodes Processed: %d nodes added to PyVis network", len(node_metadata))

        # Add Edges between present nodes
        edges_added = 0
        nodes_in_net = set(node_metadata.keys())
        for u, v in effective_graph.edges():
            if u in nodes_in_net and v in nodes_in_net:
                net.add_edge(u, v)
                edges_added += 1

        logger.info("Edges Processed: %d edges added to PyVis network", edges_added)

        # Enable interaction controls (zoom, pan, drag, selection)
        net.set_options("""
        var options = {
            "interaction": {
                "dragNodes": true,
                "dragView": true,
                "zoomView": true,
                "hover": true,
                "selectable": true,
                "selectConnectedEdges": true
            },
            "physics": {
                "stabilization": {
                    "iterations": 100
                }
            }
        }
        """)

        logger.info("Network Rendering Finished (PyVis)")
        return net
