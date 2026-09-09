"""Unit test suite for the Network Visualization module.

Verifies:
1. SRS color assignments (Red = Seed, Green = Influenced, Blue = Normal).
2. Degree-based node sizing logic (10, 20, 30).
3. PyVis network generation (object instantiation, node and edge presence).
4. Plotly network figure generation (traces, layout, hover data).
5. Graph sampling algorithms (highest degree, bfs, random) with size constraints.
6. Network topological statistics calculation (nodes, edges, avg degree, density, connected components).
7. Export utilities (PyVis HTML export and Plotly export).
8. Node search and filter methods.
9. Validation error handling for invalid graphs, nodes, and configurations.
10. Structured logging output.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

import networkx as nx
import pytest

# Ensure backend/src is available on path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from network_visualization import (
    GraphProcessor,
    NetworkVisualizationConfig,
    NetworkVisualizationFactory,
    NetworkVisualizationValidationError,
    NodeNotFoundError,
    PlotlyRenderer,
    PyVisRenderer,
    export_html,
    export_plotly,
)


@pytest.fixture
def sample_network():
    """Create a reproducible test graph with 10 nodes and distinct degrees."""
    G = nx.Graph()
    # Star topology centered on 0, with a chain from 1 to 5
    # Center node 0 will have degree 6
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
        (1, 7), (2, 8), (3, 9), (7, 8)
    ]
    G.add_edges_from(edges)
    return G


# ---------------------------------------------------------------------------
# Test 1: Seed nodes = red
# ---------------------------------------------------------------------------
def test_seed_nodes_are_red(sample_network):
    """Verify that all seed nodes receive the 'red' color attribute."""
    seeds = [0, 1]
    influenced = [2, 3]

    metadata = GraphProcessor.build_node_metadata(
        sample_network,
        seed_nodes=seeds,
        influenced_nodes=influenced,
    )

    for seed_id in seeds:
        assert metadata[seed_id].is_seed is True
        assert metadata[seed_id].color == "red"


# ---------------------------------------------------------------------------
# Test 2: Influenced nodes = green
# ---------------------------------------------------------------------------
def test_influenced_nodes_are_green(sample_network):
    """Verify that non-seed influenced nodes receive the 'green' color attribute."""
    seeds = [0]
    influenced = [1, 2, 0]  # Note: seed 0 is in influenced, but seed priority should make it red!

    metadata = GraphProcessor.build_node_metadata(
        sample_network,
        seed_nodes=seeds,
        influenced_nodes=influenced,
    )

    assert metadata[1].color == "green"
    assert metadata[2].color == "green"
    assert metadata[1].is_influenced is True
    # Seed takes precedence as red
    assert metadata[0].color == "red"
    assert metadata[0].is_seed is True


# ---------------------------------------------------------------------------
# Test 3: Normal nodes = blue
# ---------------------------------------------------------------------------
def test_normal_nodes_are_blue(sample_network):
    """Verify that inactive/unreached nodes receive the 'blue' color attribute."""
    seeds = [0]
    influenced = [1]

    metadata = GraphProcessor.build_node_metadata(
        sample_network,
        seed_nodes=seeds,
        influenced_nodes=influenced,
    )

    for node_id in [2, 3, 4, 5, 6, 7, 8, 9]:
        assert metadata[node_id].color == "blue"
        assert metadata[node_id].is_seed is False
        assert metadata[node_id].is_influenced is False


# ---------------------------------------------------------------------------
# Test 4: PyVis network generated
# ---------------------------------------------------------------------------
def test_pyvis_network_generated(sample_network):
    """Verify that PyVis renderer produces a valid Network object with nodes and edges."""
    renderer = PyVisRenderer()
    net = renderer.render_network(
        sample_network,
        seed_nodes=[0],
        influenced_nodes=[1, 2],
    )

    assert net is not None
    assert len(net.nodes) == sample_network.number_of_nodes()
    assert len(net.edges) == sample_network.number_of_edges()

    # Check node attributes in PyVis network
    node_0 = next(n for n in net.nodes if n["id"] == 0)
    assert node_0["color"] == "red"
    assert "Seed Status: Yes" in node_0["title"]


# ---------------------------------------------------------------------------
# Test 5: Plotly network generated
# ---------------------------------------------------------------------------
def test_plotly_network_generated(sample_network):
    """Verify that Plotly renderer produces a valid Figure with line and scatter traces."""
    renderer = PlotlyRenderer()
    fig = renderer.render_plotly_network(
        sample_network,
        seed_nodes=[0],
        influenced_nodes=[1, 2],
    )

    assert fig is not None
    # Expect 1 edge trace and at least 3 node traces (Red, Green, Blue categories)
    assert len(fig.data) >= 2
    trace_names = [trace.name for trace in fig.data]
    assert "Edges" in trace_names
    assert "Seed Nodes (Red)" in trace_names
    assert "Influenced Nodes (Green)" in trace_names
    assert "Normal Nodes (Blue)" in trace_names


# ---------------------------------------------------------------------------
# Test 6: Graph sampling
# ---------------------------------------------------------------------------
def test_graph_sampling():
    """Verify that graph sampling limits node count to <= max_nodes across all methods."""
    large_graph = nx.erdos_renyi_graph(n=100, p=0.05, seed=42)
    max_nodes = 25

    # 1. Highest Degree Sampling
    sampled_hd = GraphProcessor.sample_graph(large_graph, max_nodes=max_nodes, method="highest_degree")
    assert sampled_hd.number_of_nodes() == max_nodes
    assert sampled_hd.number_of_edges() > 0

    # 2. BFS Sampling
    sampled_bfs = GraphProcessor.sample_graph(large_graph, max_nodes=max_nodes, method="bfs")
    assert sampled_bfs.number_of_nodes() == max_nodes

    # 3. Random Sampling
    sampled_rand = GraphProcessor.sample_graph(large_graph, max_nodes=max_nodes, method="random")
    assert sampled_rand.number_of_nodes() == max_nodes

    # If max_nodes >= total nodes, returns full graph
    full_sample = GraphProcessor.sample_graph(large_graph, max_nodes=200)
    assert full_sample.number_of_nodes() == 100


# ---------------------------------------------------------------------------
# Test 7: Network statistics
# ---------------------------------------------------------------------------
def test_network_statistics(sample_network):
    """Verify that topological statistics are calculated accurately."""
    stats = GraphProcessor.get_network_statistics(sample_network)

    assert stats["nodes"] == 10
    assert stats["edges"] == 10
    assert stats["avg_degree"] == 2.0  # 2 * 10 / 10 = 2.0
    expected_density = (2 * 10) / (10 * 9)
    assert pytest.approx(stats["density"], rel=1e-3) == round(expected_density, 6)
    assert stats["connected_components"] == 1


# ---------------------------------------------------------------------------
# Test 8: Export HTML
# ---------------------------------------------------------------------------
def test_export_html(sample_network, tmp_path):
    """Verify that PyVis and Plotly exports generate non-empty HTML files on disk."""
    renderer = PyVisRenderer()
    net = renderer.render_network(sample_network, seed_nodes=[0])

    pyvis_html_file = tmp_path / "network_graph.html"
    exported_path = export_html(net, pyvis_html_file)

    assert exported_path.exists()
    assert exported_path.stat().st_size > 0

    # Also test Plotly HTML export
    plotly_renderer = PlotlyRenderer()
    fig = plotly_renderer.render_plotly_network(sample_network, seed_nodes=[0])

    plotly_html_file = tmp_path / "plotly_network.html"
    exported_plotly = export_plotly(fig, plotly_html_file)

    assert exported_plotly.exists()
    assert exported_plotly.stat().st_size > 0


# ---------------------------------------------------------------------------
# Test 9: Node sizing thresholds (SRS: <=5 -> 10, <=20 -> 20, >20 -> 30)
# ---------------------------------------------------------------------------
def test_node_sizing_thresholds():
    """Verify degree to size mapping according to SRS specification."""
    assert GraphProcessor.compute_node_size(1) == 10
    assert GraphProcessor.compute_node_size(5) == 10
    assert GraphProcessor.compute_node_size(6) == 20
    assert GraphProcessor.compute_node_size(20) == 20
    assert GraphProcessor.compute_node_size(21) == 30
    assert GraphProcessor.compute_node_size(100) == 30


# ---------------------------------------------------------------------------
# Test 10: Node search and filtering
# ---------------------------------------------------------------------------
def test_node_search_and_filter(sample_network):
    """Verify find_node and filter_nodes methods."""
    factory = NetworkVisualizationFactory()

    # Find node
    node_0 = factory.find_node(sample_network, node_id=0, seed_nodes=[0], influenced_nodes=[1])
    assert node_0.node_id == 0
    assert node_0.degree == 6
    assert node_0.is_seed is True
    assert node_0.color == "red"

    # Non-existent node raises error
    with pytest.raises(NodeNotFoundError):
        factory.find_node(sample_network, node_id=999)

    # Filter nodes
    seeds = factory.filter_nodes(sample_network, "seed", seed_nodes=[0, 1])
    assert set(seeds) == {0, 1}

    influenced = factory.filter_nodes(sample_network, "influenced", seed_nodes=[0], influenced_nodes=[1, 2])
    assert set(influenced) == {1, 2}

    normal = factory.filter_nodes(sample_network, "normal", seed_nodes=[0], influenced_nodes=[1, 2])
    assert 0 not in normal
    assert 1 not in normal
    assert 2 not in normal
    assert len(normal) == 7


# ---------------------------------------------------------------------------
# Test 11: Validation and error handling
# ---------------------------------------------------------------------------
def test_validation_errors():
    """Verify validation layers reject None or empty graphs."""
    factory = NetworkVisualizationFactory()

    with pytest.raises(NetworkVisualizationValidationError, match="Graph must not be None"):
        factory.render_pyvis(None)

    empty_graph = nx.Graph()
    with pytest.raises(NetworkVisualizationValidationError, match="must not be empty"):
        factory.render_plotly(empty_graph)

    with pytest.raises(NetworkVisualizationValidationError, match="max_nodes must be greater than 0"):
        factory.sample_graph(nx.path_graph(5), max_nodes=0)


# ---------------------------------------------------------------------------
# Test 12: Structured logging
# ---------------------------------------------------------------------------
def test_structured_logging(sample_network, caplog):
    """Verify that required structured log messages are emitted during rendering."""
    caplog.set_level(logging.INFO)
    factory = NetworkVisualizationFactory()

    factory.render_pyvis(sample_network, seed_nodes=[0])
    assert any("Network Rendering Started" in record.message for record in caplog.records)
    assert any("Nodes Processed" in record.message for record in caplog.records)
    assert any("Edges Processed" in record.message for record in caplog.records)
    assert any("Network Rendering Finished" in record.message for record in caplog.records)
