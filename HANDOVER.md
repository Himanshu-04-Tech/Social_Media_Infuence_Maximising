# 🤝 Frontend & Integration Handover Guide (For Himanshu)

Hi Himanshu! 👋 

The entire algorithmic and mathematical backend for our **Influence Maximization** project is complete, fully unit-tested (50/50 passing tests), and optimized. 

You do **not** need to implement any graph math, BFS, diffusion simulations, or seed-selection algorithms yourself. All you need to do is import the high-level Python API functions provided below and connect them to your UI components, charts, and visualizations.

---

## 🛠️ Quick Setup

1. Clone or open the repository:
   ```bash
   cd influence_maximization
   ```
2. Verify that the backend test suite passes:
   ```bash
   python3 -m unittest discover tests -v
   ```
   *(All 50 tests should pass in under 0.1 seconds).*

---

## ⚡ Direct Functions for Role 2 (Exactly What You Asked For!)

You can import and call these 4 functions directly from `src`:

```python
from src import (
    Graph,
    random_selection,
    degree_selection,
    greedy_selection,
    simulate_influence,
)

# 1. Load Graph
graph = Graph()
graph.load_from_edge_list("data/facebook_combined.txt")

# 2. Select Seeds
seeds_random = random_selection(graph, k=10)
seeds_degree = degree_selection(graph, k=10)
seeds_greedy = greedy_selection(graph, k=10, p=0.05)  # or just greedy_selection(graph, k=10)

# 3. Simulate Cascade Spread
# Returns the set of all infected node IDs
infected_users = simulate_influence(graph, seeds=seeds_greedy, p=0.05)

# 4. Optional: Animate Spreading Wavefronts
# Returns (all_infected, [t0_seeds, t1_infected, t2_infected, ...])
all_infected, history = simulate_influence(graph, seeds=seeds_greedy, p=0.05, return_history=True)
```

---

## 🔌 Advanced APIs You Can Also Use


---

### API 1: Graph Topology & Summary Statistics
**Use for:** Displaying the "Dataset Overview" card and rendering the network graph.

```python
from src.graph import Graph

# 1. Load the SNAP dataset
graph = Graph()
graph.load_from_edge_list("data/facebook_combined.txt")

# 2. Get high-level summary metrics
stats = graph.get_statistics()
print(stats)
```

#### What `stats` returns:
```json
{
  "num_nodes": 4039,
  "num_edges": 88234,
  "average_degree": 43.691,
  "num_connected_components": 1,
  "largest_component_size": 4039,
  "max_degree": 1045,
  "highest_degree_nodes": [
    [107, 1045],
    [1684, 792],
    [1912, 755],
    [3437, 547],
    [0, 347]
  ],
  "degree_distribution": {
    "1": 75,
    "2": 94,
    ...
  }
}
```
* **For charts:** Plot `degree_distribution` as a histogram or degree distribution curve.
* **For network nodes:** `graph.nodes()` returns the list of all node IDs.
* **For node degree:** `graph.degree(node_id)` returns the degree of any node.
* **For neighbors:** `graph.neighbors(node_id)` returns the set of adjacent friend IDs.

---

### API 2: Step-by-Step Cascade Diffusion (For UI Animations)
**Use for:** Animating the influence spreading layer-by-layer across the network graph.

```python
from src.graph import Graph
from src.propagation import simulate_independent_cascade

graph = Graph()
graph.load_from_edge_list("data/facebook_combined.txt")

# Run a simulation and request the activation history layers
active_nodes, history = simulate_independent_cascade(
    graph=graph,
    seeds=[107, 1684, 1912],   # Chosen seed nodes
    p=0.05,                    # Propagation probability (e.g. 0.05)
    return_history=True        # MUST be True to get animation steps
)

# Animate the diffusion
print(f"Total infected nodes: {len(active_nodes)}")
for step, newly_active in enumerate(history):
    print(f"Step t={step}: Newly activated {len(newly_active)} nodes -> {newly_active}")
```

#### How to visualize this:
* `history[0]` contains the initial seed nodes at $t=0$ (e.g. color them **Red**).
* `history[1]` contains users infected at step 1 (e.g. transition them to **Orange**).
* `history[2]` contains users infected at step 2 (e.g. transition them to **Yellow**).
* All uninfected nodes remain **Gray**.

---

### API 3: Algorithm Benchmark Comparison (For Dashboard Charts)
**Use for:** Running the comparison between **Random**, **Highest-Degree**, and **Greedy**, and rendering comparison bar charts.

```python
from src.main import compare_algorithms_api

# Run the 3-way benchmark comparison
# TIP: For fast interactive frontend use, set candidate_nodes or use top-50 degree candidates!
graph = Graph()
graph.load_from_edge_list("data/facebook_combined.txt")
top_50 = [node for node, _ in graph.degree_ranking()[:50]]

report = compare_algorithms_api(
    k=10,                      # Number of seeds
    p=0.05,                    # Propagation probability
    simulations=100,           # Monte Carlo runs R
    candidate_nodes=top_50,    # Top 50 candidates for ~25-sec interactive response
    master_seed=42,            # For reproducible results
    dataset_path="data/facebook_combined.txt"
)
```

#### What `report` returns:
```json
{
  "configuration": {
    "k": 10,
    "probability": 0.05,
    "simulations": 100,
    "candidate_count": 50,
    "is_restricted_candidate_pool": true,
    "master_seed": 42,
    "graph_nodes": 4039,
    "graph_edges": 88234
  },
  "results": [
    {
      "algorithm": "random",
      "seed_nodes": [20, 153, 89, ...],
      "influence_spread": 412.5,
      "selection_runtime_seconds": 0.0001,
      "evaluation_runtime_seconds": 0.521,
      "total_runtime_seconds": 0.5211,
      "marginal_gains": [],
      "spread_stats": {
        "mean": 412.5,
        "std_dev": 28.4,
        "min": 350,
        "max": 480
      }
    },
    {
      "algorithm": "degree",
      "seed_nodes": [107, 1684, 1912, ...],
      "influence_spread": 2158.2,
      "selection_runtime_seconds": 0.0005,
      "evaluation_runtime_seconds": 0.504,
      "total_runtime_seconds": 0.5045,
      "marginal_gains": [],
      "spread_stats": {
        "mean": 2158.2,
        "std_dev": 45.1,
        "min": 2051,
        "max": 2270
      }
    },
    {
      "algorithm": "greedy",
      "seed_nodes": [107, 1912, 1684, ...],
      "influence_spread": 2210.8,
      "selection_runtime_seconds": 24.31,
      "evaluation_runtime_seconds": 0.518,
      "total_runtime_seconds": 24.828,
      "marginal_gains": [1045.0, 420.2, 210.5, 140.3, ...],
      "spread_stats": {
        "mean": 2210.8,
        "std_dev": 41.2,
        "min": 2100,
        "max": 2310
      }
    }
  ]
}
```

---

## 📊 Suggested UI Components for the Dashboard

1. **Top Metric Cards:**
   * Nodes: `4,039` | Edges: `88,234` | Avg Degree: `43.69` | Components: `1`
2. **Interactive Controls Sidebar:**
   * Slider for $k$ (Seed Budget): $1$ to $20$ (default: $10$)
   * Slider for $p$ (Propagation Probability): $0.01$ to $0.20$ (default: $0.05$)
   * Slider for Simulations ($R$): $10$ to $200$ (default: $100$)
   * Dropdown for Candidate Mode: "Demo Mode (Top 50 candidates)" vs "Full Graph"
3. **Charts:**
   * **Influence Spread Comparison:** Bar chart comparing `influence_spread` across Random, Degree, and Greedy (with error bars using `spread_stats.std_dev`).
   * **Runtime Comparison:** Bar chart comparing `selection_runtime_seconds` on a log scale (highlighting how fast Degree is vs Greedy).
   * **Diminishing Returns Curve:** Line chart plotting `marginal_gains` of Greedy from step 1 to $k$ (visualizing submodularity!).
4. **Network Visualization:**
   * An interactive 2D graph (using PyVis, D3.js, Cytoscape, or NetworkX) showing the ego-network and coloring the seed nodes and spreading cascade layers.

---

## 💡 Viva Tip to Coordinate On

If the examiner asks about integration:
* **Zaid's defense:** *"I designed a modular, stateless algorithm engine and exposed a decoupled JSON API with deterministic RNG seeding to prevent execution-order side effects."*
* **Himanshu's defense:** *"I consumed Zaid's API without duplicating any algorithmic logic, translating his discrete-time layer histories into real-time wavefront animations and rendering submodular diminishing-returns curves."*
