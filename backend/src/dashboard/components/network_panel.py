"""Network visualization panel component for the Streamlit Dashboard.

Embeds Module 6 graph visualizations (interactive PyVis or native Plotly)
with zoom, pan, node search, and attribute inspection.
"""

from __future__ import annotations

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components

from network_visualization.graph_processor import GraphProcessor
from network_visualization.network_renderer import NetworkVisualizationFactory


def render_network_panel(
    graph: nx.Graph,
    seed_nodes: list[int],
    influenced_nodes: set[int] | list[int],
    preferred_renderer: str = "PyVis (Interactive Physics)",
) -> None:
    """Render interactive network graph visualization and node inspection tools.

    SRS Color Coding:
        - 🔴 Red: Initial Seed Nodes
        - 🟢 Green: Influenced Nodes
        - 🔵 Blue: Normal Inactive Nodes
    """
    st.subheader("🌐 Network Cascade Visualization")

    # Legend Display
    st.markdown(
        """
        <div style="display: flex; gap: 20px; align-items: center; margin-bottom: 12px; font-weight: 500;">
            <span>🎨 <b>SRS Status Legend:</b></span>
            <span><span style="color: #E74C3C; font-size: 1.2em;">●</span> Seed Nodes (Red)</span>
            <span><span style="color: #2ECC71; font-size: 1.2em;">●</span> Influenced Nodes (Green)</span>
            <span><span style="color: #3498DB; font-size: 1.2em;">●</span> Normal Inactive Nodes (Blue)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    renderer_choice = st.radio(
        "Display Mode",
        options=["PyVis (Interactive Physics)", "Plotly (Streamlit Native)"],
        index=0 if "pyvis" in preferred_renderer.lower() else 1,
        horizontal=True,
        label_visibility="collapsed",
    )

    if not seed_nodes and not influenced_nodes:
        st.info("Click **🚀 Run Simulation** above to initiate cascade diffusion and render the active network.")
        preview_graph = st.checkbox("🔍 Preview Network Topology Prior to Simulation", value=False)
        if not preview_graph:
            return

    net_factory = NetworkVisualizationFactory()

    with st.spinner("Rendering network topology..."):
        # If network has >300 nodes, sample for interactive UI smoothness
        display_graph = graph
        if graph.number_of_nodes() > 300:
            display_graph = net_factory.sample_graph(
                graph,
                max_nodes=300,
                method="highest_degree",
                seed_nodes=seed_nodes,
            )

        if "pyvis" in renderer_choice.lower():
            try:
                pyvis_net = net_factory.render_pyvis(
                    graph=display_graph,
                    seed_nodes=seed_nodes,
                    influenced_nodes=influenced_nodes,
                )
                html_str = pyvis_net.generate_html()
                components.html(html_str, height=650, scrolling=True)
            except Exception as err:
                st.error(f"PyVis rendering error: {err}. Falling back to Plotly.")
                fig = net_factory.render_plotly(
                    graph=display_graph,
                    seed_nodes=seed_nodes,
                    influenced_nodes=influenced_nodes,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            fig = net_factory.render_plotly(
                graph=display_graph,
                seed_nodes=seed_nodes,
                influenced_nodes=influenced_nodes,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Node Search and Inspection Tool
    with st.expander("🔍 Node Inspection & Query Tool", expanded=False):
        c1, c2 = st.columns([2, 3])
        with c1:
            all_nodes = list(graph.nodes())
            if all_nodes:
                target_node = st.selectbox("Select Node ID to Inspect", options=all_nodes[:500])
                if target_node is not None:
                    try:
                        node_meta = GraphProcessor.find_node(
                            graph=graph,
                            node_id=target_node,
                            seed_nodes=seed_nodes,
                            influenced_nodes=influenced_nodes,
                        )
                        with c2:
                            st.write(f"**Node ID:** `{node_meta.node_id}`")
                            st.write(f"**Degree:** `{node_meta.degree}` connections")
                            st.write(f"**Seed Node:** `{'Yes (Red)' if node_meta.is_seed else 'No'}`")
                            st.write(f"**Influenced:** `{'Yes (Green)' if node_meta.color == 'green' else ('Yes (Seed)' if node_meta.is_seed else 'No (Blue)')}`")
                            st.write(f"**Visual Radius:** `{node_meta.size}px`")
                    except Exception as err:
                        st.warning(f"Inspection error: {err}")
