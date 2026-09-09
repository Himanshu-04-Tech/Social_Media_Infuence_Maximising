"""
tests/test_metrics.py

Unit tests for benchmarking, experimentation, and comparative analysis engine (src/metrics.py).
Validates runtime separation, fair RNG stream derivation, JSON serializability,
and the programmatic API.

Authors:
    Zaid Hussain - Graph & Algorithm Engineer
    Himanshu     - Frontend, Visualization & Integration Engineer
"""

import json
import unittest
from src.graph import Graph
from src.metrics import compare_algorithms, evaluate_seed_spread_detailed, run_experiment


class TestMetricsBasics(unittest.TestCase):
    """Unit tests for experiment runners and data schemas."""

    def setUp(self):
        """
        Set up a small 8-node test network:
            0 - 1 - 2 - 3 - 4 - 5 - 6 - 7
        """
        self.g = Graph()
        for i in range(7):
            self.g.add_edge(i, i + 1)

    def test_01_random_experiment_structure(self):
        """Random experiment report contains all required keys and valid types."""
        res = run_experiment("random", self.g, k=2, p=0.1, simulations=20, master_seed=42)

        expected_keys = {
            "algorithm",
            "k",
            "probability",
            "simulations",
            "candidate_count",
            "is_restricted_candidate_pool",
            "master_seed",
            "graph_nodes",
            "graph_edges",
            "seed_nodes",
            "influence_spread",
            "marginal_gains",
            "selection_runtime_seconds",
            "evaluation_runtime_seconds",
            "total_runtime_seconds",
            "spread_stats",
        }
        self.assertTrue(expected_keys.issubset(res.keys()))
        self.assertEqual(res["algorithm"], "random")
        self.assertEqual(len(res["seed_nodes"]), 2)
        self.assertEqual(res["marginal_gains"], [])
        self.assertGreaterEqual(res["selection_runtime_seconds"], 0.0)
        self.assertGreaterEqual(res["evaluation_runtime_seconds"], 0.0)
        self.assertGreaterEqual(res["total_runtime_seconds"], 0.0)
        self.assertGreaterEqual(res["influence_spread"], 2.0)

    def test_02_degree_experiment_structure(self):
        """Degree experiment report contains valid structural metrics."""
        res = run_experiment("degree", self.g, k=2, p=0.1, simulations=20, master_seed=42)
        self.assertEqual(res["algorithm"], "degree")
        self.assertEqual(len(res["seed_nodes"]), 2)
        self.assertEqual(res["marginal_gains"], [])
        self.assertGreaterEqual(res["selection_runtime_seconds"], 0.0)
        self.assertGreaterEqual(res["evaluation_runtime_seconds"], 0.0)

    def test_03_greedy_experiment_structure(self):
        """Greedy experiment includes marginal gains array of length k."""
        res = run_experiment("greedy", self.g, k=2, p=0.1, simulations=20, master_seed=42)
        self.assertEqual(res["algorithm"], "greedy")
        self.assertEqual(len(res["seed_nodes"]), 2)
        self.assertEqual(len(res["marginal_gains"]), 2)
        self.assertGreaterEqual(res["selection_runtime_seconds"], 0.0)
        self.assertGreaterEqual(res["evaluation_runtime_seconds"], 0.0)

    def test_04_json_serializability(self):
        """All experiment reports must serialize to valid JSON without custom encoders."""
        res = run_experiment("greedy", self.g, k=2, p=0.1, simulations=10, master_seed=42)
        json_str = json.dumps(res)
        self.assertIsInstance(json_str, str)
        reloaded = json.loads(json_str)
        self.assertEqual(reloaded["algorithm"], "greedy")

        comparison = compare_algorithms(self.g, k=2, p=0.1, simulations=10, master_seed=42)
        comp_json = json.dumps(comparison)
        self.assertIsInstance(comp_json, str)
        reloaded_comp = json.loads(comp_json)
        self.assertIn("configuration", reloaded_comp)
        self.assertEqual(len(reloaded_comp["results"]), 3)

    def test_05_reproducibility(self):
        """Same master seed yields identical seed sets and spread metrics."""
        res1 = run_experiment("greedy", self.g, k=2, p=0.2, simulations=30, master_seed=1234)
        res2 = run_experiment("greedy", self.g, k=2, p=0.2, simulations=30, master_seed=1234)

        self.assertEqual(res1["seed_nodes"], res2["seed_nodes"])
        self.assertEqual(res1["marginal_gains"], res2["marginal_gains"])
        self.assertEqual(res1["influence_spread"], res2["influence_spread"])

    def test_06_rng_independence_across_algorithms(self):
        """
        Deriving seeds independently ensures that running one algorithm does not mutate
        the random stream seen by another algorithm under the same master seed.
        """
        # Run comparison (Random -> Degree -> Greedy)
        report = compare_algorithms(self.g, k=2, p=0.2, simulations=30, master_seed=999)
        greedy_from_compare = report["results"][2]

        # Run greedy standalone using the exact derived seed used inside compare_algorithms
        from src.metrics import _derive_seed

        derived_greedy_seed = _derive_seed(999, salt=3000)
        greedy_standalone = run_experiment(
            "greedy", self.g, k=2, p=0.2, simulations=30, master_seed=derived_greedy_seed
        )

        self.assertEqual(greedy_from_compare["seed_nodes"], greedy_standalone["seed_nodes"])
        self.assertEqual(greedy_from_compare["influence_spread"], greedy_standalone["influence_spread"])

    def test_07_compare_algorithms_order(self):
        """compare_algorithms maintains strict [random, degree, greedy] ordering."""
        report = compare_algorithms(self.g, k=2, p=0.1, simulations=10, master_seed=42)
        algs = [r["algorithm"] for r in report["results"]]
        self.assertEqual(algs, ["random", "degree", "greedy"])

    def test_08_invalid_parameters_raise_value_error(self):
        """Invalid inputs to run_experiment and compare_algorithms raise ValueError."""
        with self.assertRaises(ValueError):
            run_experiment("unknown_alg", self.g, k=2, p=0.1)
        with self.assertRaises(ValueError):
            run_experiment("random", self.g, k=-1, p=0.1)
        with self.assertRaises(ValueError):
            run_experiment("random", self.g, k=2, p=-0.1)
        with self.assertRaises(ValueError):
            run_experiment("random", self.g, k=2, p=0.1, simulations=0)
        with self.assertRaises(ValueError):
            compare_algorithms(self.g, k=2, p=0.1, simulations=0)

    def test_09_frontend_direct_wrappers(self):
        """Verify Himanshu's exact requested wrapper functions and signatures."""
        from src import (
            random_selection,
            degree_selection,
            greedy_selection,
            simulate_influence,
        )

        # 1. random_selection(graph, k)
        seeds_rnd = random_selection(self.g, k=2)
        self.assertEqual(len(seeds_rnd), 2)

        # 2. degree_selection(graph, k)
        seeds_deg = degree_selection(self.g, k=2)
        self.assertEqual(len(seeds_deg), 2)

        # 3. greedy_selection(graph, k) with default p
        seeds_grd = greedy_selection(self.g, k=2)
        self.assertEqual(len(seeds_grd), 2)

        # 4. simulate_influence(graph, seeds, p) -> int count (InfluenceResult)
        active = simulate_influence(self.g, seeds_deg, p=0.05)
        self.assertIsInstance(active, int)
        self.assertGreaterEqual(int(active), 2)

        # 5. simulate_influence with return_history=True -> (int count, history_layers)
        active, history = simulate_influence(self.g, seeds_deg, p=0.05, return_history=True)
        self.assertIsInstance(active, int)
        self.assertIsInstance(history, list)
        self.assertGreaterEqual(len(history), 1)



if __name__ == "__main__":
    unittest.main()

