"""Graph processing and metadata generation engine.

Extracts graph metrics, assigns SRS visual aesthetics (colors, sizes),
manages subgraph sampling strategies, and powers node queries and filtering.
"""

from __future__ import annotations

import collections
import logging
import random
from typing import Any, Collection

import networkx as nx

from .exceptions import NetworkVisualizationValidationError, NodeNotFoundError
from .models import NodeMetadata
from .validators import (
    validate_graph,
    validate_node_collections,
    validate_node_id,
)

logger = logging.getLogger(__name__)


class GraphProcessor:
    """Provides analytical processing, metadata assignment, and sampling on network graphs."""

    @staticmethod
    def compute_node_degree(graph: Any) -> dict[Any, int]:
        """Compute the degree for all nodes in the graph.

        Args:
            graph: NetworkX graph object.

        Returns:
            Dictionary mapping node IDs to degree counts.
        """
        validate_graph(graph)
        return {node: graph.degree(node) for node in graph.nodes()}

    @staticmethod
    def compute_node_size(degree: int) -> int:
        """Compute visual node radius based on SRS degree scale thresholds.

        Scale:
            degree <= 5  -> size = 10
            degree <= 20 -> size = 20
            degree > 20  -> size = 30

        Args:
            degree: Integer degree of the node.

        Returns:
            Calculated visual size integer.
        """
        if degree <= 5:
            return 10
        elif degree <= 20:
            return 20
        else:
            return 30

    @classmethod
    def build_node_metadata(
        cls,
        graph: Any,
        seed_nodes: Collection[Any] | None = None,
        influenced_nodes: Collection[Any] | None = None,
    ) -> dict[Any, NodeMetadata]:
        """Build visualization metadata for all nodes in the graph according to SRS specifications.

        Color Assignment:
            - Red   → Seed Nodes (is_seed = True, takes precedence)
            - Green → Influenced Nodes (is_influenced = True, is_seed = False)
            - Blue  → Normal / Inactive Nodes (is_seed = False, is_influenced = False)

        Args:
            graph: NetworkX graph.
            seed_nodes: Collection of seed node identifiers.
            influenced_nodes: Collection of influenced node identifiers.

        Returns:
            Dictionary mapping node ID to NodeMetadata.
        """
        validate_graph(graph)
        seeds, influenced = validate_node_collections(graph, seed_nodes, influenced_nodes)

        metadata_dict: dict[Any, NodeMetadata] = {}
        for node in graph.nodes():
            deg = graph.degree(node)
            is_seed = node in seeds
            is_inf = (node in influenced) and not is_seed

            if is_seed:
                color = "red"
            elif is_inf:
                color = "green"
            else:
                color = "blue"

            size = cls.compute_node_size(deg)

            metadata_dict[node] = NodeMetadata(
                node_id=node,
                degree=deg,
                is_seed=is_seed,
                is_influenced=node in influenced,
                color=color,
                size=size,
            )

        logger.debug(
            "Built metadata for %d nodes (Seeds: %d, Influenced: %d, Normal: %d)",
            len(metadata_dict),
            sum(1 for m in metadata_dict.values() if m.is_seed),
            sum(1 for m in metadata_dict.values() if m.color == "green"),
            sum(1 for m in metadata_dict.values() if m.color == "blue"),
        )
        return metadata_dict

    @classmethod
    def sample_graph(
        cls,
        graph: Any,
        max_nodes: int,
        method: str = "highest_degree",
        seed_nodes: Collection[Any] | None = None,
        random_seed: int | None = 42,
    ) -> nx.Graph:
        """Sample a large graph to limit visual density while preserving structural properties.

        Supported Methods:
            - 'highest_degree': Retains top nodes by degree (default).
            - 'bfs': Breadth-first search traversal starting from highest-degree node.
            - 'random': Uniform random sample of nodes.

        Args:
            graph: NetworkX graph.
            max_nodes: Maximum number of nodes to retain.
            method: Sampling algorithm ('highest_degree', 'bfs', or 'random').
            seed_nodes: Optional seeds to prioritize when method='bfs'.
            random_seed: Random seed for deterministic reproducibility.

        Returns:
            Induced subgraph with at most max_nodes.

        Raises:
            NetworkVisualizationValidationError: If invalid method or non-positive max_nodes.
        """
        validate_graph(graph)
        if max_nodes <= 0:
            raise NetworkVisualizationValidationError(
                f"max_nodes must be greater than 0. Got: {max_nodes}"
            )

        total_nodes = graph.number_of_nodes()
        if total_nodes <= max_nodes:
            return graph.copy()

        logger.info(
            "Node Sampling Applied: Graph has %d nodes > max_nodes %d. Strategy: %s",
            total_nodes,
            max_nodes,
            method,
        )

        normalized_method = method.lower().strip()

        if normalized_method == "highest_degree":
            # Sort by degree descending, tie-break by node identifier
            sorted_nodes = sorted(
                graph.nodes(),
                key=lambda n: (graph.degree(n), str(n)),
                reverse=True,
            )
            selected_nodes = sorted_nodes[:max_nodes]

        elif normalized_method == "random":
            rng = random.Random(random_seed)
            node_list = list(graph.nodes())
            selected_nodes = rng.sample(node_list, max_nodes)

        elif normalized_method == "bfs":
            # Determine start node: prioritized from seeds if present, else max degree
            start_candidates = []
            if seed_nodes:
                valid_seeds = [s for s in seed_nodes if s in graph]
                if valid_seeds:
                    start_candidates = sorted(
                        valid_seeds,
                        key=lambda n: graph.degree(n),
                        reverse=True,
                    )

            if not start_candidates:
                start_candidates = sorted(
                    graph.nodes(),
                    key=lambda n: (graph.degree(n), str(n)),
                    reverse=True,
                )

            selected_set: set[Any] = set()
            queue: collections.deque[Any] = collections.deque()

            for start_node in start_candidates:
                if len(selected_set) >= max_nodes:
                    break
                if start_node in selected_set:
                    continue

                queue.append(start_node)
                selected_set.add(start_node)

                while queue and len(selected_set) < max_nodes:
                    curr = queue.popleft()
                    neighbors = sorted(
                        graph.neighbors(curr),
                        key=lambda n: graph.degree(n),
                        reverse=True,
                    )
                    for nbr in neighbors:
                        if nbr not in selected_set:
                            selected_set.add(nbr)
                            queue.append(nbr)
                            if len(selected_set) >= max_nodes:
                                break

            selected_nodes = list(selected_set)

        else:
            raise NetworkVisualizationValidationError(
                f"Unknown graph sampling method: '{method}'. Supported methods: 'highest_degree', 'bfs', 'random'."
            )

        return graph.subgraph(selected_nodes).copy()

    @classmethod
    def find_node(
        cls,
        graph: Any,
        node_id: Any,
        seed_nodes: Collection[Any] | None = None,
        influenced_nodes: Collection[Any] | None = None,
    ) -> NodeMetadata:
        """Locate a specific node and return its rich NodeMetadata.

        Args:
            graph: NetworkX graph.
            node_id: Target node identifier.
            seed_nodes: Optional collection of seed nodes.
            influenced_nodes: Optional collection of influenced nodes.

        Returns:
            NodeMetadata for the found node.

        Raises:
            NodeNotFoundError: If node_id does not exist in graph.
        """
        validate_graph(graph)
        validate_node_id(graph, node_id)

        seeds, influenced = validate_node_collections(graph, seed_nodes, influenced_nodes)
        deg = graph.degree(node_id)
        is_seed = node_id in seeds
        is_inf = (node_id in influenced) and not is_seed

        if is_seed:
            color = "red"
        elif is_inf:
            color = "green"
        else:
            color = "blue"

        size = cls.compute_node_size(deg)

        return NodeMetadata(
            node_id=node_id,
            degree=deg,
            is_seed=is_seed,
            is_influenced=node_id in influenced,
            color=color,
            size=size,
        )

    @classmethod
    def filter_nodes(
        cls,
        graph: Any,
        node_type: str,
        seed_nodes: Collection[Any] | None = None,
        influenced_nodes: Collection[Any] | None = None,
    ) -> list[Any]:
        """Filter and return node IDs based on their simulation category.

        Supported Categories:
            - 'seed': Initial seed nodes (SRS Red).
            - 'influenced': Non-seed nodes reached by diffusion (SRS Green).
            - 'normal': Inactive nodes not reached (SRS Blue).

        Args:
            graph: NetworkX graph.
            node_type: Node classification category ('seed', 'influenced', 'normal').
            seed_nodes: Collection of seed nodes.
            influenced_nodes: Collection of influenced nodes.

        Returns:
            List of matching node identifiers.

        Raises:
            NetworkVisualizationValidationError: If unsupported node_type provided.
        """
        validate_graph(graph)
        normalized_type = node_type.lower().strip()
        metadata_map = cls.build_node_metadata(graph, seed_nodes, influenced_nodes)

        if normalized_type == "seed":
            return [node_id for node_id, meta in metadata_map.items() if meta.is_seed]
        elif normalized_type == "influenced":
            return [node_id for node_id, meta in metadata_map.items() if meta.color == "green"]
        elif normalized_type in ("normal", "inactive"):
            return [node_id for node_id, meta in metadata_map.items() if meta.color == "blue"]
        else:
            raise NetworkVisualizationValidationError(
                f"Invalid node_type '{node_type}'. Supported types: 'seed', 'influenced', 'normal'."
            )

    @staticmethod
    def get_network_statistics(graph: Any) -> dict[str, Any]:
        """Calculate network topology and connectivity statistics.

        Returns:
            Dictionary containing:
                - nodes: Number of nodes
                - edges: Number of edges
                - avg_degree: Mean node degree
                - density: Graph connectivity density
                - connected_components: Number of connected components

        Args:
            graph: Validated NetworkX graph.
        """
        validate_graph(graph)

        n = graph.number_of_nodes()
        m = graph.number_of_edges()

        if n == 0:
            return {
                "nodes": 0,
                "edges": 0,
                "avg_degree": 0.0,
                "density": 0.0,
                "connected_components": 0,
            }

        is_directed = nx.is_directed(graph)
        avg_degree = round((edges_factor := 1 if is_directed else 2) * m / n, 4)
        density = round(float(nx.density(graph)), 6)

        if is_directed:
            cc = nx.number_weakly_connected_components(graph)
        else:
            cc = nx.number_connected_components(graph)

        stats = {
            "nodes": n,
            "edges": m,
            "avg_degree": avg_degree,
            "density": density,
            "connected_components": cc,
        }
        logger.debug("Computed network statistics: %s", stats)
        return stats
