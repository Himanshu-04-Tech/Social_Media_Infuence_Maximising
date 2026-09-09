"""
tests/test_seed_selection.py

Unit tests for Random, Highest-Degree, and Greedy Seed Selection algorithms.
Validates input constraints, edge cases, candidate pool restrictions,
marginal gains, submodularity on toy models, and deterministic execution.

Authors:
    Zaid Hussain - Graph & Algorithm Engineer
    Himanshu     - Frontend, Visualization & Integration Engineer
"""

import random
import unittest
from src.graph import Graph
from src.propagation import estimate_influence
from src.seed_selection import (
    select_seeds_random,
    select_seeds_degree,
    select_seeds_greedy,
)


class TestSeedSelectionValidation(unittest.TestCase):
    """Validation tests for input parameters, empty graphs, and candidate pools."""

    def test_01_empty_graph_raises_value_error(self):
        """All seed selection algorithms must raise ValueError on an empty graph (|V|=0)."""
        empty_g = Graph()
        with self.assertRaises(ValueError):
            select_seeds_random(empty_g, k=1)
        with self.assertRaises(ValueError):
            select_seeds_degree(empty_g, k=1)
        with self.assertRaises(ValueError):
            select_seeds_greedy(empty_g, k=1, p=0.1)

    def test_02_invalid_k_raises_value_error(self):
        """k <= 0 and k > |V| must raise ValueError across all algorithms."""
        g = Graph()
        g.add_edge(0, 1)  # |V| = 2

        for alg_func in [
            lambda k: select_seeds_random(g, k=k),
            lambda k: select_seeds_degree(g, k=k),
            lambda k: select_seeds_greedy(g, k=k, p=0.1),
        ]:
            with self.assertRaises(ValueError):
                alg_func(0)
            with self.assertRaises(ValueError):
                alg_func(-1)
            with self.assertRaises(ValueError):
                alg_func(3)  # Exceeds |V|=2

    def test_03_candidate_nodes_validation(self):
        """Restricted candidate pool validation: existence, deduplication, and size >= k."""
        g = Graph()
        for i in range(5):
            g.add_edge(i, i + 1)  # Nodes 0..5

        # Nonexistent candidate ID raises ValueError
        with self.assertRaises(ValueError):
            select_seeds_greedy(g, k=2, p=0.1, candidate_nodes=[0, 999])

        # Candidate pool size smaller than k raises ValueError
        with self.assertRaises(ValueError):
            select_seeds_greedy(g, k=3, p=0.1, candidate_nodes=[0, 1])

        # Deduplication works: [0, 0, 1, 1] has size 2, so k=3 should raise ValueError
        with self.assertRaises(ValueError):
            select_seeds_greedy(g, k=3, p=0.1, candidate_nodes=[0, 0, 1, 1])

        # Valid candidate subset restricts chosen seeds strictly to that subset
        seeds = select_seeds_greedy(g, k=2, p=0.1, candidate_nodes=[0, 1, 2], rng=42)
        self.assertEqual(len(seeds), 2)
        self.assertTrue(set(seeds).issubset({0, 1, 2}))


class TestRandomAndDegreeSelection(unittest.TestCase):
    """Unit tests for Random and Highest-Degree seed selection."""

    def setUp(self):
        """
        Line graph: 0 - 1 - 2 - 3 - 4
        Degrees: 0:1, 1:2, 2:2, 3:2, 4:1
        """
        self.g = Graph()
        for i in range(4):
            self.g.add_edge(i, i + 1)

    def test_04_random_selection_properties(self):
        """Random selection returns k unique nodes and respects deterministic RNG."""
        rng1 = random.Random(42)
        seeds1 = select_seeds_random(self.g, k=3, rng=rng1)
        self.assertEqual(len(seeds1), 3)
        self.assertEqual(len(set(seeds1)), 3)

        # Reproducibility check
        rng2 = random.Random(42)
        seeds2 = select_seeds_random(self.g, k=3, rng=rng2)
        self.assertEqual(seeds1, seeds2)

    def test_05_degree_selection_properties(self):
        """Degree selection returns top k highest-degree nodes."""
        # Top 3 nodes should have degree 2 (nodes 1, 2, 3 in some order)
        seeds = select_seeds_degree(self.g, k=3)
        self.assertEqual(len(seeds), 3)
        self.assertEqual(len(set(seeds)), 3)
        for s in seeds:
            self.assertEqual(self.g.degree(s), 2)


class TestGreedyCorrectnessAndSubmodularity(unittest.TestCase):
    """Unit tests for Greedy Influence Maximization on synthetic structural graphs."""

    def test_06_greedy_star_graph(self):
        """
        Test A (Star Graph): Center node 0 connected to 5 leaves.
        Greedy with k=1 must select the center hub.
        """
        star = Graph()
        for leaf in [1, 2, 3, 4, 5]:
            star.add_edge(0, leaf)

        seeds = select_seeds_greedy(star, k=1, p=0.5, simulations=50, rng=42)
        self.assertEqual(seeds, [0])

    def test_07_greedy_disconnected_components(self):
        """
        Test B (Disconnected Components):
        Component 1: Triangle {0, 1, 2}
        Component 2: Triangle {3, 4, 5}
        For k=2, p=1.0:
        Optimal strategy MUST select one node from Component 1 and one from Component 2
        to achieve total spread = 6.0.
        """
        disjoint_g = Graph()
        # Triangle 1
        disjoint_g.add_edge(0, 1)
        disjoint_g.add_edge(1, 2)
        disjoint_g.add_edge(2, 0)
        # Triangle 2
        disjoint_g.add_edge(3, 4)
        disjoint_g.add_edge(4, 5)
        disjoint_g.add_edge(5, 3)

        seeds = select_seeds_greedy(disjoint_g, k=2, p=1.0, simulations=10, rng=42)
        self.assertEqual(len(seeds), 2)
        # One seed in {0, 1, 2} and one seed in {3, 4, 5}
        has_comp1 = any(s in {0, 1, 2} for s in seeds)
        has_comp2 = any(s in {3, 4, 5} for s in seeds)
        self.assertTrue(has_comp1)
        self.assertTrue(has_comp2)

        # Spread must be full graph (6 nodes)
        spread = estimate_influence(disjoint_g, seeds, p=1.0, simulations=5)
        self.assertEqual(spread, 6.0)

    def test_08_greedy_overcomes_degree_overlap_flaw(self):
        """
        Test C (Redundancy / Overlap Flaw of Degree):
        Community 1: Dense 4-clique {0, 1, 2, 3}. All nodes have degree 3.
        Community 2: Hub node 4 with 3 leaves {5, 6, 7}. Node 4 has degree 3.
        Nodes 0, 1, 2, 3, 4 all have degree 3.

        If Degree Heuristic ties by picking two nodes from Community 1 (e.g. [0, 1]),
        its spread at p=1.0 is only 4 nodes.
        Greedy with k=2 recognizes Community 2, picking one node from {0, 1, 2, 3}
        and node 4, achieving total spread = 8 nodes!
        """
        overlap_g = Graph()
        # Clique 1 (nodes 0, 1, 2, 3)
        for u in range(4):
            for v in range(u + 1, 4):
                overlap_g.add_edge(u, v)

        # Star 2 (hub 4 connected to 5, 6, 7)
        overlap_g.add_edge(4, 5)
        overlap_g.add_edge(4, 6)
        overlap_g.add_edge(4, 7)

        seeds_greedy = select_seeds_greedy(overlap_g, k=2, p=1.0, simulations=10, rng=42)
        spread_greedy = estimate_influence(overlap_g, seeds_greedy, p=1.0, simulations=5)
        # Greedy must achieve the full 8 nodes reach
        self.assertEqual(spread_greedy, 8.0)

    def test_09_marginal_gains_and_diminishing_returns(self):
        """
        Verifies return_marginal_gains structure and diminishing returns on deterministic setting.
        For p=1.0 on a connected path graph:
        Delta(s1) >= Delta(s2) >= ... >= 0.
        """
        path_g = Graph()
        for i in range(7):
            path_g.add_edge(i, i + 1)  # 8 nodes: 0..7

        seeds, gains = select_seeds_greedy(
            path_g, k=3, p=1.0, simulations=5, rng=42, return_marginal_gains=True
        )

        self.assertEqual(len(seeds), 3)
        self.assertEqual(len(gains), 3)

        # Under p=1.0, the first seed reaches all 8 nodes because path is connected!
        # Gain 1 = 8.0, Gain 2 = 0.0, Gain 3 = 0.0
        self.assertEqual(gains[0], 8.0)
        self.assertEqual(gains[1], 0.0)
        self.assertEqual(gains[2], 0.0)

        # Verify diminishing returns: gain[0] >= gain[1] >= gain[2] >= 0
        for i in range(len(gains) - 1):
            self.assertGreaterEqual(gains[i], gains[i + 1])
            self.assertGreaterEqual(gains[i + 1], 0.0)

    def test_10_deterministic_greedy_rng(self):
        """Passing identical RNG state yields identical seed selection trajectories."""
        path_g = Graph()
        for i in range(5):
            path_g.add_edge(i, i + 1)

        rng1 = random.Random(123)
        seeds1 = select_seeds_greedy(path_g, k=2, p=0.3, simulations=30, rng=rng1)

        rng2 = random.Random(123)
        seeds2 = select_seeds_greedy(path_g, k=2, p=0.3, simulations=30, rng=rng2)

        self.assertEqual(seeds1, seeds2)


if __name__ == "__main__":
    unittest.main()
