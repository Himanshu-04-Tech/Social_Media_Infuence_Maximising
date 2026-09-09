"""Dataset statistics panel component for the Streamlit Dashboard.

Displays network topology metrics (nodes, edges, avg degree, density)
and the degree distribution histogram from Module 5.
"""

from __future__ import annotations

import networkx as nx
import streamlit as st

from visualization.chart_builder import VisualizationFactory
from network_visualization.graph_processor import GraphProcessor


def render_dataset_stats(graph: nx.Graph) -> None:
    """Render the Dataset Statistics Panel and Degree Distribution Chart.

    Args:
        graph: NetworkX graph structure.
    """
    stats = GraphProcessor.get_network_statistics(graph)

    st.subheader("📊 Dataset Topology Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Nodes (|V|)", value=f"{stats['nodes']:,}")
    with col2:
        st.metric(label="Edges (|E|)", value=f"{stats['edges']:,}")
    with col3:
        st.metric(label="Avg Degree (⟨k⟩)", value=f"{stats['avg_degree']:.2f}")
    with col4:
        st.metric(label="Density", value=f"{stats['density']:.5f}")

    # Degree Distribution Chart from Module 5
    with st.expander("📈 View Degree Distribution Histogram", expanded=False):
        degrees = [deg for _, deg in graph.degree()]
        try:
            fig = VisualizationFactory.create_degree_distribution_chart(degrees)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as err:
            st.warning(f"Could not render degree distribution: {err}")
