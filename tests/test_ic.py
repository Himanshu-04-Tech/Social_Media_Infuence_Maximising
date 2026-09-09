"""
tests/test_ic.py

Unit tests for Independent Cascade (IC) diffusion and Monte Carlo estimation.
Validates boundary conditions, deterministic behavior, layer history,
input validation, and statistical monotonicity.

Authors:
    Zaid Hussain - Graph & Algorithm Engineer
    Himanshu     - Frontend, Visualization & Integration Engineer
"""

import random
import unittest
from src.graph import Graph
from src.propagation import (
    simulate_independent_cascade,
    estimate_influence,
    estimate_influence_stats,
    validate_probability,
    validate_seeds,
)


class TestICBasicsAndBoundaries(unittest.TestCase):
    """Tests for IC boundary conditions and input validation."""

    def setUp(self):
        """
        Set up a simple 6-node graph with two disconnected triangles:
            Component 1: 0 - 1 - 2 - 0
            Component 2: 3 - 4 - 5 - 3
        """
        self.g = Graph()
        self.g.add_edge(0, 1)
        self.g.add_edge(1, 2)
        self.g.add_edge(2, 0)
        self.g.add_edge(3, 4)
        self.g.add_edge(4, 5)
        self.g.add_edge(5, 3)

    def test_01_empty_seed_set(self):
        """Empty seed set produces 0 influence and empty active set."""
        active = simulate_independent_cascade(self.g, [], 0.5)
        self.assertEqual(active, set())

        active_hist, history = simulate_independent_cascade(
            self.g, [], 0.5, return_history=True
        )
        self.assertEqual(active_hist, set())
        self.assertEqual(history, [])

        influence = estimate_influence(self.g, [], 0.5, simulations=50)
        self.assertEqual(influence, 0.0)

    def test_02_p_zero_boundary(self):
        """When p = 0.0, influence spread is strictly the seed set itself."""
        seeds = [0, 1]
        active = simulate_independent_cascade(self.g, seeds, p=0.0)
        self.assertEqual(active, {0, 1})

        influence = estimate_influence(self.g, seeds, p=0.0, simulations=20)
        self.assertEqual(influence, 2.0)

    def test_03_p_one_boundary_single_component(self):
        """When p = 1.0, cascade deterministically reaches all nodes in the component."""
        seeds = [0]
        active = simulate_independent_cascade(self.g, seeds, p=1.0)
        self.assertEqual(active, {0, 1, 2})

        influence = estimate_influence(self.g, seeds, p=1.0, simulations=10)
        self.assertEqual(influence, 3.0)

    def test_04_p_one_multiple_disconnected_components(self):
        """When p = 1.0 with seeds in multiple components, all nodes in those components activate."""
        seeds = [0, 3]  # One seed in each disconnected component
        active = simulate_independent_cascade(self.g, seeds, p=1.0)
        self.assertEqual(active, {0, 1, 2, 3, 4, 5})

        influence = estimate_influence(self.g, seeds, p=1.0, simulations=10)
        self.assertEqual(influence, 6.0)

    def test_05_disconnected_components_propagation(self):
        """Influence never leaks across disconnected components regardless of p."""
        seeds = [0]
        # Run multiple simulations with high probability
        for _ in range(10):
            active = simulate_independent_cascade(self.g, seeds, p=0.99)
            # Component 2 nodes (3, 4, 5) must never become active
            self.assertTrue(active.isdisjoint({3, 4, 5}))

    def test_06_duplicate_seeds(self):
        """Duplicate seeds are deduplicated without raising errors."""
        seeds_with_dups = [0, 0, 1, 1, 1]
        deduped = validate_seeds(self.g, seeds_with_dups)
        self.assertEqual(deduped, {0, 1})

        active = simulate_independent_cascade(self.g, seeds_with_dups, p=0.0)
        self.assertEqual(active, {0, 1})

    def test_07_invalid_seed_id_raises_value_error(self):
        """Seed node IDs not present in the graph must raise ValueError."""
        with self.assertRaises(ValueError):
            validate_seeds(self.g, [0, 999999])

        with self.assertRaises(ValueError):
            simulate_independent_cascade(self.g, [999999], p=0.1)

        with self.assertRaises(ValueError):
            estimate_influence(self.g, [0, 999999], p=0.1)

    def test_08_invalid_probability_raises_value_error(self):
        """Propagation probabilities outside [0.0, 1.0] must raise ValueError."""
        with self.assertRaises(ValueError):
            validate_probability(-0.01)
        with self.assertRaises(ValueError):
            validate_probability(1.01)
        with self.assertRaises(ValueError):
            simulate_independent_cascade(self.g, [0], p=-0.1)
        with self.assertRaises(ValueError):
            estimate_influence(self.g, [0], p=1.5)

    def test_09_invalid_simulation_count_raises_value_error(self):
        """Non-positive simulation counts must raise ValueError."""
        with self.assertRaises(ValueError):
            estimate_influence(self.g, [0], p=0.1, simulations=0)
        with self.assertRaises(ValueError):
            estimate_influence(self.g, [0], p=0.1, simulations=-10)


class TestICMechanicsAndHistory(unittest.TestCase):
    """Tests for cascade history layers, deterministic RNG, and statistical properties."""

    def setUp(self):
        """
        Line graph: 0 - 1 - 2 - 3 - 4
        Allows precise layer-by-layer step verification.
        """
        self.g = Graph()
        for i in range(4):
            self.g.add_edge(i, i + 1)

    def test_10_deterministic_rng_behavior(self):
        """Identical RNG seeds produce identical simulation trajectories."""
        seeds = [0]
        p = 0.5

        # Two simulations initialized with same random seed
        run1 = simulate_independent_cascade(self.g, seeds, p, rng=42)
        run2 = simulate_independent_cascade(self.g, seeds, p, rng=42)
        self.assertEqual(run1, run2)

        # Monte Carlo estimates with same RNG
        est1 = estimate_influence(self.g, seeds, p, simulations=50, rng=12345)
        est2 = estimate_influence(self.g, seeds, p, simulations=50, rng=12345)
        self.assertEqual(est1, est2)

    def test_11_cascade_history_optional_flag(self):
        """return_history toggles between set output and (set, list[set]) output."""
        res_default = simulate_independent_cascade(self.g, [0], p=0.5, return_history=False)
        self.assertIsInstance(res_default, set)

        res_history = simulate_independent_cascade(self.g, [0], p=0.5, return_history=True)
        self.assertIsInstance(res_history, tuple)
        self.assertEqual(len(res_history), 2)
        active_set, history = res_history
        self.assertIsInstance(active_set, set)
        self.assertIsInstance(history, list)

    def test_12_cascade_history_structure(self):
        """
        Verify that history layers:
        1. Start with seeds at t = 0.
        2. Contain ONLY newly activated nodes at each time step (non-cumulative).
        3. Are mutually disjoint across all steps.
        4. Union of all history layers exactly equals active_set.
        """
        seeds = [0]
        # Deterministic propagation on line graph with p = 1.0:
        # t=0: {0}
        # t=1: {1}
        # t=2: {2}
        # t=3: {3}
        # t=4: {4}
        active_set, history = simulate_independent_cascade(
            self.g, seeds, p=1.0, return_history=True
        )
        self.assertEqual(active_set, {0, 1, 2, 3, 4})
        self.assertEqual(len(history), 5)
        self.assertEqual(history[0], {0})
        self.assertEqual(history[1], {1})
        self.assertEqual(history[2], {2})
        self.assertEqual(history[3], {3})
        self.assertEqual(history[4], {4})

        # Verify disjointness
        for i in range(len(history)):
            for j in range(i + 1, len(history)):
                self.assertTrue(
                    history[i].isdisjoint(history[j]),
                    f"Layer {i} and Layer {j} must be mutually disjoint.",
                )

        # Verify union equals active_set
        union_layers = set()
        for layer in history:
            union_layers.update(layer)
        self.assertEqual(union_layers, active_set)

    def test_13_statistical_monotonicity_of_probability(self):
        """
        Theoretical property: True expected spread sigma(S, p) is monotonically
        non-decreasing with respect to p.
        We test with a clear separation (p1 = 0.05 vs p2 = 0.80) over sufficient
        simulations (R = 200) to ensure high statistical confidence.
        """
        seeds = [0]
        p_low = 0.05
        p_high = 0.80

        spread_low = estimate_influence(self.g, seeds, p=p_low, simulations=200, rng=100)
        spread_high = estimate_influence(self.g, seeds, p=p_high, simulations=200, rng=100)

        self.assertGreater(
            spread_high,
            spread_low,
            f"Expected spread for p={p_high} ({spread_high}) to exceed p={p_low} ({spread_low})",
        )

    def test_14_statistical_monotonicity_of_seed_sets(self):
        """
        Theoretical property: True expected spread sigma(S) is monotone:
        adding seeds never decreases expected influence spread: sigma(S1) <= sigma(S2) for S1 subset of S2.
        """
        s_small = [0]
        s_large = [0, 4]  # Both ends of line graph
        p = 0.4

        spread_small = estimate_influence(self.g, s_small, p=p, simulations=200, rng=200)
        spread_large = estimate_influence(self.g, s_large, p=p, simulations=200, rng=200)

        self.assertGreaterEqual(
            spread_large,
            spread_small,
            f"Expected spread for S2={s_large} ({spread_large}) to be >= S1={s_small} ({spread_small})",
        )

    def test_15_estimate_influence_stats(self):
        """Verify detailed statistical reporting utility."""
        seeds = [0]
        stats = estimate_influence_stats(self.g, seeds, p=0.5, simulations=50, rng=42)
        self.assertIn("mean", stats)
        self.assertIn("std_dev", stats)
        self.assertIn("min", stats)
        self.assertIn("max", stats)
        self.assertIn("simulations", stats)
        self.assertEqual(stats["simulations"], 50)
        self.assertGreaterEqual(stats["min"], 1)
        self.assertLessEqual(stats["max"], 5)
        self.assertGreaterEqual(stats["mean"], stats["min"])
        self.assertLessEqual(stats["mean"], stats["max"])


if __name__ == "__main__":
    unittest.main()
