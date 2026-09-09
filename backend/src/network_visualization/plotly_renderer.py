"""Embedded network renderer using Plotly (graph_objects).

Generates lightweight, interactive Plotly Figure objects designed for zero-latency
native embedding within Streamlit dashboards and web applications.
"""

from __future__ import annotations

import logging
from typing import Any, Collection

import networkx as nx
import plotly.graph_objects as go

from .graph_processor import GraphProcessor
from .models import NetworkVisualizationConfig
from .validators import validate_config, validate_graph

logger = logging.getLogger(__name__)


class PlotlyRenderer:
    """Renders NetworkX graphs into interactive Plotly Figure objects."""

    def __init__(self, default_config: NetworkVisualizationConfig | None = None) -> None:
        self.default_config = default_config or NetworkVisualizationConfig()

    def render_plotly_network(
        self,
        graph: Any,
        seed_nodes: Collection[Any] | None = None,
        influenced_nodes: Collection[Any] | None = None,
        config: NetworkVisualizationConfig | None = None,
    ) -> go.Figure:
        """Render a NetworkX graph into an interactive Plotly Figure.

        Features:
            - SRS Color Categories: Red (Seeds), Green (Influenced), Blue (Normal)
            - Proportional node sizing (10, 20, 30)
            - Rich hover inspection tooltips
            - Minimalist canvas without distracting axes or gridlines

        Args:
            graph: NetworkX graph structure.
            seed_nodes: Optional collection of initial seed node IDs.
            influenced_nodes: Optional collection of influenced node IDs.
            config: Optional NetworkVisualizationConfig override.

        Returns:
            plotly.graph_objects.Figure instance.
        """
        validate_graph(graph)
        active_config = validate_config(config or self.default_config)

        logger.info("Network Rendering Started (Plotly)")

        # Handle automated graph sampling
        effective_graph = graph
        if graph.number_of_nodes() > active_config.max_nodes:
            logger.info("Node Sampling Applied: %d nodes > %d max_nodes", graph.number_of_nodes(), active_config.max_nodes)
            effective_graph = GraphProcessor.sample_graph(
                graph,
                max_nodes=active_config.max_nodes,
                method="highest_degree",
                seed_nodes=seed_nodes,
            )

        # Compute 2D node layout positions
        pos = nx.spring_layout(effective_graph, seed=42)

        # Build node metadata
        node_metadata = GraphProcessor.build_node_metadata(
            effective_graph,
            seed_nodes=seed_nodes,
            influenced_nodes=influenced_nodes,
        )

        # Construct Edge Trace
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        edges_added = 0

        for u, v in effective_graph.edges():
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edges_added += 1

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.8, color="rgba(160, 174, 192, 0.6)"),
            hoverinfo="none",
            mode="lines",
            name="Edges",
            showlegend=False,
        )
        logger.info("Edges Processed: %d edges added to Plotly network", edges_added)

        # Separate nodes into SRS categorical groups for clear legends and styling
        categories = {
            "Seed Nodes (Red)": {"color": "red", "nodes": []},
            "Influenced Nodes (Green)": {"color": "green", "nodes": []},
            "Normal Nodes (Blue)": {"color": "blue", "nodes": []},
        }

        for node_id, meta in node_metadata.items():
            if meta.is_seed:
                categories["Seed Nodes (Red)"]["nodes"].append(meta)
            elif meta.color == "green":
                categories["Influenced Nodes (Green)"]["nodes"].append(meta)
            else:
                categories["Normal Nodes (Blue)"]["nodes"].append(meta)

        node_traces: list[go.Scatter] = []
        for cat_name, cat_data in categories.items():
            meta_list = cat_data["nodes"]
            if not meta_list:
                continue

            xs = [pos[m.node_id][0] for m in meta_list]
            ys = [pos[m.node_id][1] for m in meta_list]
            sizes = [m.size for m in meta_list]
            customdata = [
                (
                    m.node_id,
                    m.degree,
                    "Yes" if m.is_seed else "No",
                    "Yes" if m.is_influenced else "No",
                )
                for m in meta_list
            ]
            texts = [str(m.node_id) if active_config.show_labels else "" for m in meta_list]

            trace = go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text" if active_config.show_labels else "markers",
                text=texts,
                textposition="top center",
                customdata=customdata,
                hovertemplate=(
                    "<b>Node ID:</b> %{customdata[0]}<br>"
                    "<b>Degree:</b> %{customdata[1]}<br>"
                    "<b>Seed Status:</b> %{customdata[2]}<br>"
                    "<b>Influenced Status:</b> %{customdata[3]}"
                    "<extra></extra>"
                ),
                marker=dict(
                    color=cat_data["color"],
                    size=sizes,
                    line=dict(width=1.5, color="#2D3748"),
                    opacity=0.9,
                ),
                name=cat_name,
                showlegend=True,
            )
            node_traces.append(trace)

        logger.info("Nodes Processed: %d nodes added across %d categories", len(node_metadata), len(node_traces))

        # Parse dimensions
        height_px = 750
        if isinstance(active_config.height, str) and active_config.height.endswith("px"):
            try:
                height_px = int(active_config.height.replace("px", "").strip())
            except ValueError:
                pass

        fig = go.Figure(
            data=[edge_trace, *node_traces],
            layout=go.Layout(
                title=dict(
                    text="Social Network Influence Propagation Graph",
                    font=dict(size=18, color="#1A202C"),
                    x=0.5,
                    xanchor="center",
                ),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                ),
                hovermode="closest",
                margin=dict(b=20, l=20, r=20, t=60),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                height=height_px,
            ),
        )

        logger.info("Network Rendering Finished (Plotly)")
        return fig
