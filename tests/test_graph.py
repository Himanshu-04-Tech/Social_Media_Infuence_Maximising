"""
tests/test_graph.py

Comprehensive test suite for src/graph.py.
Validates all invariants, BFS traversal & distances, connected components,
the Handshaking Lemma, and the Stanford SNAP Facebook dataset.

Authors:
    Zaid Hussain - Graph & Algorithm Engineer
    Himanshu     - Frontend, Visualization & Integration Engineer
"""

import os
import math
import unittest
from src.graph import Graph


class TestGraphBasics(unittest.TestCase):
    """Unit tests for basic graph construction and edge invariants."""

    def test_01_empty_graph(self):
        """Test 1: Empty graph invariants."""
        g = Graph()
        self.assertEqual(g.nodes(), [])
        self.assertEqual(g.number_of_nodes(), 0)
        self.assertEqual(g.number_of_edges(), 0)
        self.assertEqual(g.average_degree(), 0.0)
        self.assertEqual(g.connected_components(), [])
        self.assertEqual(g.degree_ranking(), [])

    def test_02_single_node(self):
        """Test 2: Single node graph."""
        g = Graph()
        added = g.add_node(42)
        self.assertTrue(added)
        self.assertEqual(g.number_of_nodes(), 1)
        self.assertEqual(g.number_of_edges(), 0)
        self.assertEqual(g.degree(42), 0)
        self.assertEqual(g.neighbors(42), set())
        self.assertEqual(g.connected_components(), [{42}])

    def test_03_adding_nodes(self):
        """Test 3: Adding distinct nodes and duplicate node rejection."""
        g = Graph()
        self.assertTrue(g.add_node(1))
        self.assertTrue(g.add_node(2))
        self.assertTrue(g.add_node(3))
        # Adding existing node should return False without altering state
        self.assertFalse(g.add_node(1))
        self.assertEqual(g.number_of_nodes(), 3)
        self.assertTrue(g.has_node(1))
        self.assertTrue(g.has_node(2))
        self.assertTrue(g.has_node(3))
        self.assertFalse(g.has_node(99))

    def test_04_adding_undirected_edge(self):
        """Test 4: Adding an undirected edge establishes symmetry."""
        g = Graph()
        added = g.add_edge(1, 2)
        self.assertTrue(added)
        self.assertEqual(g.number_of_nodes(), 2)
        self.assertEqual(g.number_of_edges(), 1)
        self.assertTrue(g.has_edge(1, 2))
        self.assertTrue(g.has_edge(2, 1))
        self.assertEqual(g.neighbors(1), {2})
        self.assertEqual(g.neighbors(2), {1})
        self.assertEqual(g.degree(1), 1)
        self.assertEqual(g.degree(2), 1)

    def test_05_duplicate_edge(self):
        """Test 5: Duplicate edge (u, v) is rejected and not double-counted."""
        g = Graph()
        self.assertTrue(g.add_edge(1, 2))
        self.assertFalse(g.add_edge(1, 2))
        self.assertEqual(g.number_of_edges(), 1)
        self.assertEqual(g.neighbors(1), {2})

    def test_06_reverse_duplicate_edge(self):
        """Test 6: Reverse duplicate edge (v, u) is recognized and rejected."""
        g = Graph()
        self.assertTrue(g.add_edge(1, 2))
        self.assertFalse(g.add_edge(2, 1))
        self.assertEqual(g.number_of_edges(), 1)
        self.assertEqual(g.neighbors(2), {1})

    def test_07_self_loop(self):
        """Test 7: Self-loops (u == v) are strictly rejected."""
        g = Graph()
        added = g.add_edge(5, 5)
        self.assertFalse(added)
        self.assertEqual(g.number_of_edges(), 0)
        self.assertFalse(g.has_node(5))

    def test_08_node_degree(self):
        """Test 8: Degree calculation on star graph."""
        g = Graph()
        # Star graph with center 0 and leaves 1, 2, 3, 4
        for leaf in [1, 2, 3, 4]:
            g.add_edge(0, leaf)
        self.assertEqual(g.degree(0), 4)
        for leaf in [1, 2, 3, 4]:
            self.assertEqual(g.degree(leaf), 1)
        # Nonexistent node degree is 0
        self.assertEqual(g.degree(999), 0)

    def test_09_edge_count(self):
        """Test 9: Explicit edge counter correctly tracks insertions."""
        g = Graph()
        edges_to_add = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
        for u, v in edges_to_add:
            g.add_edge(u, v)
        self.assertEqual(g.number_of_edges(), 5)
        self.assertEqual(g.num_edges(), 5)
        # Attempting duplicates should not affect count
        g.add_edge(2, 0)
        g.add_edge(3, 2)
        g.add_edge(9, 9)
        self.assertEqual(g.number_of_edges(), 5)


class TestGraphAlgorithms(unittest.TestCase):
    """Unit tests for BFS, Connected Components, and Degree Ranking."""

    def setUp(self):
        """
        Set up a known test graph:
            0 -- 1 -- 2
            |    |
            3 -- 4
        """
        self.g = Graph()
        self.g.add_edge(0, 1)
        self.g.add_edge(1, 2)
        self.g.add_edge(0, 3)
        self.g.add_edge(1, 4)
        self.g.add_edge(3, 4)

    def test_10_bfs_traversal(self):
        """Test 10: BFS traversal explores all reachable nodes."""
        order, distances = self.g.bfs(0)
        self.assertEqual(order[0], 0)
        self.assertEqual(set(order), {0, 1, 2, 3, 4})
        self.assertEqual(len(order), 5)

    def test_11_bfs_shortest_distances(self):
        """Test 11: BFS computes exact minimum hop shortest path distances."""
        _, distances = self.g.bfs(0)
        expected_distances = {
            0: 0,
            1: 1,
            3: 1,
            2: 2,  # 0 -> 1 -> 2
            4: 2,  # 0 -> 1 -> 4 or 0 -> 3 -> 4
        }
        self.assertEqual(distances, expected_distances)

    def test_12_bfs_nonexistent_node(self):
        """Test 12: BFS from nonexistent node returns clean empty structures."""
        order, distances = self.g.bfs(999)
        self.assertEqual(order, [])
        self.assertEqual(distances, {})

    def test_13_connected_components(self):
        """Test 13: Connected components across disconnected subgraphs."""
        disjoint_g = Graph()
        # Component 1: Triangle {1, 2, 3}
        disjoint_g.add_edge(1, 2)
        disjoint_g.add_edge(2, 3)
        disjoint_g.add_edge(3, 1)

        # Component 2: Single edge {4, 5}
        disjoint_g.add_edge(4, 5)

        # Component 3: Isolated node {6}
        disjoint_g.add_node(6)

        comps = disjoint_g.connected_components()
        self.assertEqual(len(comps), 3)
        # Sorted descending by size
        self.assertEqual(comps[0], {1, 2, 3})
        self.assertEqual(comps[1], {4, 5})
        self.assertEqual(comps[2], {6})

    def test_14_average_degree(self):
        """Test 14: Average degree calculation."""
        # self.g has 5 nodes and 5 edges -> avg degree = 2*5 / 5 = 2.0
        self.assertEqual(self.g.average_degree(), 2.0)

    def test_15_degree_ranking(self):
        """Test 15: Degree ranking sorted descending."""
        ranking = self.g.degree_ranking()
        # Node degrees: 1:3, 0:2, 3:2, 4:2, 2:1
        degrees = [deg for _, deg in ranking]
        # Verify strictly descending order
        for i in range(len(degrees) - 1):
            self.assertGreaterEqual(degrees[i], degrees[i + 1])
        # Center node 1 has degree 3
        self.assertEqual(ranking[0], (1, 3))
        # Node 2 has degree 1
        self.assertEqual(ranking[-1], (2, 1))

    def test_16_handshaking_lemma(self):
        """Test 16: First Theorem of Graph Theory (sum of degrees == 2 * |E|)."""
        sum_deg = sum(self.g.degree(u) for u in self.g.nodes())
        self.assertEqual(sum_deg, 2 * self.g.number_of_edges())


class TestSNAPDataset(unittest.TestCase):
    """Validation tests against the real Stanford SNAP Facebook dataset."""

    DATASET_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "facebook_combined.txt"
    )

    @classmethod
    def setUpClass(cls):
        """Load the SNAP dataset once for all dataset validation tests."""
        cls.graph = Graph()
        if os.path.exists(cls.DATASET_PATH):
            cls.graph.load_from_edge_list(cls.DATASET_PATH)

    def test_17_real_snap_facebook_validation(self):
        """Test 17: Invariant validation on real SNAP Facebook dataset."""
        if not os.path.exists(self.DATASET_PATH):
            self.skipTest(f"Dataset not found at {self.DATASET_PATH}")

        # 1. Node count: 4,039
        self.assertEqual(self.graph.number_of_nodes(), 4039)

        # 2. Edge count: 88,234
        self.assertEqual(self.graph.number_of_edges(), 88234)

        # 3. Connected components: Exactly 1
        comps = self.graph.connected_components()
        self.assertEqual(len(comps), 1)

        # 4. Largest component size: 4,039
        self.assertEqual(len(comps[0]), 4039)

        # 5. Handshaking Lemma: sum(deg) == 2 * 88,234 = 176,468
        total_degree = sum(self.graph.degree(u) for u in self.graph.nodes())
        self.assertEqual(total_degree, 2 * 88234)
        self.assertEqual(total_degree, 176468)

        # 6. Average degree ≈ 43.69
        avg_deg = self.graph.average_degree()
        expected_avg_deg = 176468 / 4039  # 43.6910126...
        self.assertAlmostEqual(avg_deg, expected_avg_deg, places=4)
        self.assertTrue(math.isclose(avg_deg, 43.69, abs_tol=0.01))

        # 7. Summary statistics API structure
        stats = self.graph.get_statistics()
        self.assertEqual(stats["num_nodes"], 4039)
        self.assertEqual(stats["num_edges"], 88234)
        self.assertEqual(stats["num_connected_components"], 1)
        self.assertEqual(stats["largest_component_size"], 4039)
        self.assertEqual(stats["average_degree"], 43.691)
        # Verify highest degree node exists and has high connectivity
        self.assertGreater(stats["max_degree"], 1000)


if __name__ == "__main__":
    unittest.main()
