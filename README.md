# Influence Maximization in Social Networks (DAA Project)

A complete, high-performance algorithmic engine for solving the **Influence Maximization Problem** on real-world social networks under the **Independent Cascade (IC)** diffusion model. Built for the Design and Analysis of Algorithms (DAA) coursework.

---

## 👥 Project Team & Responsibilities

| Team Member | Role | Key Responsibilities |
| :--- | :--- | :--- |
| **Zaid Hussain** | **Graph & Algorithm Engineer** | Graph data structure, dataset ingestion, BFS, Connected Components, Independent Cascade model, Monte Carlo estimation, Seed selection (Random, Degree, Greedy), Complexity & Viva analysis, Backend API. |
| **Himanshu** | **Frontend, Visualization & Integration Engineer** | Interactive network UI, real-time cascade animations, algorithm comparison dashboards, charts, and frontend-backend integration. |

---

## 📁 Repository Structure

```text
influence_maximization/
├── data/
│   ├── .gitkeep
│   └── facebook_combined.txt     # Stanford SNAP Facebook ego-network
├── src/
│   ├── __init__.py
│   ├── graph.py                  # Adjacency-list graph, BFS, components, degree ranking
│   ├── propagation.py            # IC diffusion simulation & Monte Carlo estimation
│   ├── seed_selection.py         # Random, Highest-Degree, and Greedy algorithms
│   ├── metrics.py                # Benchmarking, timing separation & fair RNG streams
│   └── main.py                   # CLI entry point & programmatic API for Himanshu
├── tests/
│   ├── __init__.py
│   ├── test_graph.py             # 17 unit tests (graph invariants, BFS, SNAP stats)
│   ├── test_ic.py                # 15 unit tests (IC boundaries, history layers, RNG)
│   ├── test_seed_selection.py    # 10 unit tests (greedy correctness, submodularity)
│   └── test_metrics.py           # 8 unit tests (timing separation, JSON serialization)
├── .gitignore
└── README.md
```

---

## 📊 Dataset: Stanford SNAP Facebook Combined

* **Source:** Stanford Network Analysis Project (SNAP) `facebook_combined.txt`
* **Nodes ($|V|$):** $4{,}039$
* **Edges ($|E|$):** $88{,}234$ (undirected)
* **Connected Components:** Exactly $1$ (Single Giant Component)
* **Average Degree:** $\approx 43.69$ ($\frac{2E}{V} = \frac{176{,}468}{4{,}039}$)
* **Handshaking Lemma:** $\sum_{u \in V} \text{deg}(u) = 2 \times |E| = 176{,}468$

---

## 🧠 Algorithmic Framework & Complexity Summary

### 1. Graph Primitives (`src/graph.py`)
* **Adjacency List:** Implemented via `dict[int, set[int]]`. Guarantees expected $O(1)$ neighbor lookup and automatic duplicate rejection.
* **Explicit Edge Tracking:** Edge counter `_num_edges` is maintained in $O(1)$ time without recalculating $\sum \text{deg}(u) / 2$.
* **BFS Traversal & Distances:** Computes traversal order and exact shortest hop-distances in $\Theta(V + E)$ time and $O(V)$ space.
* **Connected Components:** Discovers all maximal connected subgraphs using non-redundant BFS exploration in $\Theta(V + E)$ time.

### 2. Influence Diffusion (`src/propagation.py`)
* **Independent Cascade (IC) Model:** Progressive discrete-time diffusion. Each newly active node gets a single chance to activate inactive neighbors with probability $p$.
* **Worst-Case Simulation Complexity:** $O(V + E)$ time, $O(V)$ space. Across the entire cascade, each node is popped at most once, scanning at most $\sum_{u \in V_A} \text{deg}(u) \le 2E$ incident edge entries.
* **Monte Carlo Estimator:** Computes empirical sample mean spread $\hat{\sigma}(S) = \frac{1}{R} \sum_{i=1}^R |A_i(S)|$. Worst-case complexity is $O(R \cdot (V + E))$.
* **Step-by-Step History:** Supports `return_history=True` returning disjoint sets of newly activated nodes per time step for frontend animation.

### 3. Seed Selection Algorithms (`src/seed_selection.py`)

| Algorithm | Type | Selection Complexity | Theoretical Guarantee | Key Trade-off / Limitation |
| :--- | :--- | :--- | :--- | :--- |
| **Random Selection** | Baseline | $O(V)$ | None (lower bound) | Fast, but unguided and produces high variance in spread. |
| **Highest Degree** | Heuristic | $O(V \log V)$ | None (unbounded ratio) | Extremely fast ($< 1\text{ ms}$), but suffers from the **clustering/overlap flaw** (redundantly picks hubs in the same clique). |
| **Greedy Maximization** | Approximation | $O(k \cdot \|C\| \cdot R \cdot (V_A + E_A))$ | $(1 - 1/e)$ approximation ($\approx 63.2\%$) | Optimizes marginal gain $\Delta(v \mid S) = \sigma(S \cup \{v\}) - \sigma(S)$; accounts for overlap; computationally expensive. |

---

## 🎓 DAA Viva Questions & Core Concepts

### Q1: Why is Influence Maximization NP-Hard?
Kempe, Kleinberg, and Tardos (2003) proved that finding the optimal seed set of size $k$ under the Independent Cascade model is **NP-hard** by a reduction from the NP-complete **Set Cover** problem. Furthermore, computing the exact influence $\sigma(S)$ is **$\#\text{P}$-hard** (Chen et al.), necessitating Monte Carlo simulation.

### Q2: What are Monotonicity and Submodularity?
* **Monotonicity:** Adding a node to a seed set never reduces expected influence:
  $$\forall S \subseteq T \subseteq V, \quad \sigma(S) \le \sigma(T)$$
* **Submodularity (Diminishing Marginal Returns):** The additional gain from adding a node to a small set is larger than or equal to adding it to a larger superset:
  $$\forall S \subseteq T \subseteq V, \, v \notin T: \quad \sigma(S \cup \{v\}) - \sigma(S) \ge \sigma(T \cup \{v\}) - \sigma(T)$$

### Q3: What is the Nemhauser et al. (1978) Theorem?
For any non-negative, monotone, submodular function with $f(\emptyset) = 0$, the greedy algorithm choosing the element with maximum marginal gain at each step achieves:
$$\sigma(S_{\text{greedy}}) \ge \left(1 - \frac{1}{e}\right) \sigma(S^*) \approx 0.632 \times \sigma(S^*)$$
Because we estimate spread using $R$ Monte Carlo runs, Kempe et al. established a $(1 - 1/e - \epsilon)$ guarantee with high probability, where $\epsilon \to 0$ as $R \to \infty$.

### Q4: Why does straightforward Greedy take so long on the full Facebook graph?
For $|V| = 4{,}039, k = 10, R = 100$:
$$\text{Total IC simulations} \approx \sum_{i=0}^9 (4{,}039 - i) \times 100 \approx 40{,}000 \times 100 = \mathbf{4{,}000{,}000 \text{ simulations!}}$$
Four million BFS-style graph traversals in Python take $\approx 45\text{ minutes}$. This computational bottleneck motivates our **Fast Demo Mode** (`--candidates 50`) and theoretical research into lazy evaluation (CELF).

---

## 🚀 How to Run the Project

### 1. Run the Complete Test Suite
The project includes 50 comprehensive unit tests:
```bash
python3 -m unittest discover tests -v
```

### 2. Run the Fast Demo Benchmark (Recommended for Interactive Demos)
Restricts candidate evaluations to the top 50 degree nodes so the 3-way comparison finishes in $\approx 25\text{ seconds}$:
```bash
python3 -m src.main --k 10 --p 0.05 --simulations 100 --candidates 50 --seed 42
```

### 3. Run Individual Algorithms
```bash
# Random baseline on full graph
python3 -m src.main --k 10 --p 0.05 --simulations 100 --algorithm random

# Degree heuristic on full graph
python3 -m src.main --k 10 --p 0.05 --simulations 100 --algorithm degree
```

### 4. Output Raw JSON (for Himanshu's Frontend)
```bash
python3 -m src.main --k 10 --p 0.05 --simulations 100 --candidates 50 --json
```

---

## 🔌 Frontend Integration API Reference (for Himanshu)

Himanshu can import the backend API directly:

```python
from src.main import compare_algorithms_api, run_algorithm_api

# 1. Run full 3-way algorithm comparison
report = compare_algorithms_api(
    k=10,
    p=0.05,
    simulations=100,
    candidate_nodes=None,  # or pass list of candidate node IDs
    master_seed=42,
    dataset_path="data/facebook_combined.txt"
)

# 2. Inspect JSON-serializable results
for result in report["results"]:
    print(result["algorithm"])                    # 'random', 'degree', or 'greedy'
    print(result["seed_nodes"])                   # List[int]
    print(result["influence_spread"])              # float (average spread)
    print(result["spread_stats"])                 # {'mean': ..., 'std_dev': ..., 'min': ..., 'max': ...}
    print(result["selection_runtime_seconds"])    # float (time to select seeds)
    print(result["evaluation_runtime_seconds"])   # float (time to evaluate spread)
    print(result["marginal_gains"])               # List[float] (Greedy diminishing returns curve)
```

### Animating Spreading Layers in the UI:
To animate the cascade layer-by-layer:
```python
from src.graph import Graph
from src.propagation import simulate_independent_cascade

g = Graph()
g.load_from_edge_list("data/facebook_combined.txt")

# Returns (all_active_nodes, [layer_0_seeds, layer_1_nodes, layer_2_nodes, ...])
active_set, history = simulate_independent_cascade(
    graph=g,
    seeds=[107, 1684, 1912],
    p=0.05,
    return_history=True
)

for step, newly_active in enumerate(history):
    print(f"Time t={step}: {len(newly_active)} newly infected users")
```
