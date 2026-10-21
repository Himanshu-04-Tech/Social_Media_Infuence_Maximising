"""
src/graph.py

Graph data structure and fundamental graph traversal algorithms for Influence Maximization.
Designed and analyzed for DAA (Design and Analysis of Algorithms).

Authors:
    Zaid Hussain - Graph & Algorithm Engineer
    Himanshu     - Frontend, Visualization & Integration Engineer
"""

from collections import deque
from typing import Dict, List, Set, Tuple, Any
import os


class Graph:
    """
    Undirected Graph implemented via an Adjacency List using hash maps and sets.

    Representation:
        self.adj: Dict[int, Set[int]]
            Maps each vertex ID to a set of its adjacent neighbor vertex IDs.
            Using a set guarantees Expected O(1) duplicate edge elimination and neighbor lookups.
        self._num_edges: int
            Explicitly tracked edge count to allow O(1) retrieval without recalculating sum(degrees)/2.

    DAA Complexity Highlights:
        - Edge insertion: Expected O(1) time
        - Node/Neighbor lookup: Expected O(1) time
        - Degree lookup: O(1) time (via len(self.adj[u]))
        - Total Space: Theta(|V| + |E|) where each undirected edge (u, v) is stored twice.
    """

    def __init__(self) -> None:
        self.adj: Dict[int, Set[int]] = {}
        self._num_edges: int = 0

    def add_node(self, node: int) -> bool:
        """
        Adds a node to the graph if it is not already present.

        Note on isolated / non-consecutive nodes:
            A node is preserved if explicitly registered via add_node() or as an edge endpoint.
            Edge-list ingestion only discovers vertices explicitly present in the dataset.
            We do not assume or invent arbitrary node IDs not in the data.

        Complexity:
            Time: Expected O(1)
            Space: O(1) amortized
        Returns:
            True if the node was newly added, False if it was already present.
        """
        if node not in self.adj:
            self.adj[node] = set()
            return True
        return False

    def add_edge(self, u: int, v: int) -> bool:
        """
        Adds an undirected edge between nodes u and v.

        Invariants maintained:
        1. Self-loops (u == v) are rejected.
        2. Both endpoints u and v are registered in self.adj.
        3. Undirected symmetry: v in adj[u] and u in adj[v].
        4. self._num_edges is incremented ONLY if the edge is genuinely new.
           Because the graph is undirected, (u, v) and (v, u) represent the same edge.

        Complexity:
            Time: Expected O(1)
            Space: Expected O(1) amortized
        Returns:
            True if a new edge was added, False if rejected (self-loop or duplicate).
        """
        if u == v:
            return False

        self.add_node(u)
        self.add_node(v)

        if v in self.adj[u]:
            return False  # Already exists (duplicate)

        self.adj[u].add(v)
        self.adj[v].add(u)
        self._num_edges += 1
        return True

    def has_node(self, node: int) -> bool:
        """Checks if a node exists in the graph. Expected O(1) time."""
        return node in self.adj

    def has_edge(self, u: int, v: int) -> bool:
        """Checks if an undirected edge (u, v) exists. Expected O(1) time."""
        return u in self.adj and v in self.adj[u]

    def neighbors(self, node: int) -> Set[int]:
        """
        Returns the set of neighbors for a given node.
        Returns an empty set if the node does not exist.
        Complexity: Expected O(1) time.
        """
        return self.adj.get(node, set())

    def degree(self, node: int) -> int:
        """
        Returns the degree of a node (number of incident edges).
        Returns 0 if node is isolated or does not exist.
        Complexity: O(1) time.
        """
        if node not in self.adj:
            return 0
        return len(self.adj[node])

    def number_of_nodes(self) -> int:
        """
        Returns total number of vertices |V|.
        Complexity: O(1) time.
        """
        return len(self.adj)

    def num_nodes(self) -> int:
        """Alias for number_of_nodes(). Complexity: O(1) time."""
        return self.number_of_nodes()

    def number_of_edges(self) -> int:
        """
        Returns total number of undirected edges |E|.
        Maintained explicitly in self._num_edges.
        Complexity: O(1) time.
        """
        return self._num_edges

    def num_edges(self) -> int:
        """Alias for number_of_edges(). Complexity: O(1) time."""
        return self.number_of_edges()

    def nodes(self) -> List[int]:
        """
        Returns a list of all vertex IDs in the graph.
        Complexity: O(V) time.
        """
        return list(self.adj.keys())

    def average_degree(self) -> float:
        """
        Computes average degree across all vertices.
        By the Handshaking Lemma:
            sum(deg(u)) = 2 * |E|
        Therefore:
            Avg Degree = 2 * |E| / |V|
        Complexity: O(1) time.
        """
        n = self.number_of_nodes()
        if n == 0:
            return 0.0
        return (2.0 * self._num_edges) / n

    def degree_ranking(self) -> List[Tuple[int, int]]:
        """
        Ranks all nodes by degree in descending order.
        Used as the heuristic baseline for Highest-Degree Seed Selection.

        Complexity:
            Time: O(V log V) to sort the degree list.
            Space: O(V) to store the ranked list.
        Returns:
            List of tuples: [(node_id, degree), ...] sorted descending by degree.
        """
        ranking = [(node, len(nbrs)) for node, nbrs in self.adj.items()]
        ranking.sort(key=lambda item: item[1], reverse=True)
        return ranking

    def bfs(self, start: int) -> Tuple[List[int], Dict[int, int]]:
        """
        Breadth-First Search (BFS) traversal from start node.

        DAA Mechanics:
            - Explores the graph layer by layer using a FIFO queue (deque).
            - Computes shortest path hop-distances from start to all reachable vertices.
            - If start is not in the graph, returns empty traversal order and empty distance dict.

        Complexity:
            Time: O(V + E) — every reachable vertex enters queue at most once,
                  and each incident edge is traversed once per endpoint.
            Space: O(V) — queue, visited traversal order, and distances map.

        Returns:
            (traversal_order, distances)
            - traversal_order: List of node IDs in the order they were popped/visited.
            - distances: Dict mapping reachable node ID -> minimum edge/hop count from start.
        """
        if start not in self.adj:
            return [], {}

        traversal_order: List[int] = []
        distances: Dict[int, int] = {start: 0}
        queue: deque[int] = deque([start])

        while queue:
            u = queue.popleft()
            traversal_order.append(u)
            curr_dist = distances[u]

            for v in self.adj[u]:
                if v not in distances:
                    distances[v] = curr_dist + 1
                    queue.append(v)

        return traversal_order, distances

    def connected_components(self) -> List[Set[int]]:
        """
        Finds all connected components of the graph using BFS.

        DAA Mechanics:
            - Maintains a global `visited` set.
            - Iterates over all vertices in V.
            - When an unvisited vertex is found, triggers BFS to discover all
              nodes in its component, marks them visited, and adds the component.
            - Each vertex and each edge is processed a constant number of times.

        Complexity:
            Time: O(V + E) across the entire computation.
            Space: O(V) to store visited set and component sets.

        Returns:
            List of sets, where each set contains the node IDs in one connected component.
            Sorted by component size in descending order (largest component first).
        """
        visited: Set[int] = set()
        components: List[Set[int]] = []

        for node in self.adj:
            if node not in visited:
                comp_nodes, _ = self.bfs(node)
                comp_set = set(comp_nodes)
                visited.update(comp_set)
                components.append(comp_set)

        components.sort(key=len, reverse=True)
        return components

    def load_from_edge_list(self, filepath: str) -> None:
        """
        Loads an undirected graph from a space- or tab-delimited edge list file.

        Sanitization rules:
            - Strips whitespace.
            - Skips empty lines and comment lines starting with '#' or '%'.
            - Extracts pairs of integers 'u v'.
            - Rejects self-loops (u == v).
            - Deduplicates parallel and reverse edges via set storage.

        Complexity:
            Time: Expected O(V + E)
            Space: O(V + E)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Edge list file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("%"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        u = int(parts[0])
                        v = int(parts[1])
                        self.add_edge(u, v)
                    except ValueError:
                        continue

    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculates key structural metrics of the graph for reporting and frontend display.

        Returns:
            Dictionary containing:
            - num_nodes: int
            - num_edges: int
            - average_degree: float (rounded to 4 decimal places)
            - num_connected_components: int
            - largest_component_size: int
            - min_degree: int
            - max_degree: int
            - connected_components: int
            - highest_degree_nodes: List[Tuple[int, int]] (top 5)
            - degree_distribution: Dict[int, int] (degree -> node frequency)
        """
        n = self.number_of_nodes()
        e = self.number_of_edges()
        avg_deg = self.average_degree()

        comps = self.connected_components()
        num_comps = len(comps)
        largest_comp_size = len(comps[0]) if comps else 0

        ranked = self.degree_ranking()
        max_deg = ranked[0][1] if ranked else 0
        min_deg = ranked[-1][1] if ranked else 0
        top_nodes = ranked[:5]

        degree_dist: Dict[int, int] = {}
        for _, deg in ranked:
            degree_dist[deg] = degree_dist.get(deg, 0) + 1

        return {
            "num_nodes": n,
            "num_edges": e,
            "average_degree": round(avg_deg, 4),
            "min_degree": min_deg,
            "max_degree": max_deg,
            "num_connected_components": num_comps,
            "connected_components": num_comps,
            "largest_component_size": largest_comp_size,
            "highest_degree_nodes": top_nodes,
            "degree_distribution": degree_dist,
        }
