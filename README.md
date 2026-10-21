# Social Network Influence Maximization Simulator

A complete, high-performance algorithmic and empirical evaluation platform for solving the **Influence Maximization Problem** on large-scale social networks under the **Independent Cascade (IC)** diffusion model. Built for the Design and Analysis of Algorithms (DAA) coursework.

---

## 👥 Role Division & Responsibilities

| Role | Engineer | Core Responsibilities |
| :--- | :--- | :--- |
| **Role 1** | **Graph & Algorithm Engineer** | Graph data structures, SNAP dataset ingestion, BFS shortest paths, Connected Components, Independent Cascade model, Seed Selection algorithms (Random, Degree, Greedy), Complexity analysis, and core unit testing. |
| **Role 2** | **Simulation, Experiments & Dashboard Engineer** | Monte Carlo simulation engine, Metrics & analytics engine, Automated experimental framework (Experiments A–D), Results persistence (CSV/JSON), Interactive 2D Network Visualization (PyVis/Plotly), and Streamlit Dashboard. |

---

## 🚀 Key Features

* **Random Seed Selection:** Unbiased stochastic baseline lower-bound ($O(k)$ sampling).
* **Degree-Based Selection:** High-speed structural heuristic selecting top-$k$ network hubs ($O(V \log V)$).
* **Greedy Influence Maximization:** $(1 - 1/e)$ approximation algorithm optimizing marginal gain $\Delta(v \mid S) = \sigma(S \cup \{v\}) - \sigma(S)$ with submodular diminishing-returns guarantees.
* **Independent Cascade (IC) Model:** Exact discrete-time progressive diffusion simulator with step-by-step wavefront propagation layers.
* **Monte Carlo Simulation Framework:** Law of Large Numbers empirical convergence engine calculating mean reach, variance, standard error, and confidence bounds across $N$ runs.
* **Interactive Streamlit Dashboard:** Faculty-ready web dashboard with 2D physics network rendering, real-time KPI metrics, algorithm comparison bar charts, and 1-click experiment execution.

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Himanshu-04-Tech/Social_Media_Infuence_Maximising.git
   cd Social_Media_Infuence_Maximising
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ Run Dashboard

Launch the interactive Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

> The application will automatically open in your default browser at `http://localhost:8501`.

---

## 📁 Project Structure

```text
Social_Media_Infuence_Maximising/
├── data/
│   ├── .gitkeep
│   └── facebook_combined.txt          # Stanford SNAP Facebook ego-network (4,039 nodes, 88,234 edges)
├── src/                               # Role 1 & Core Engine APIs
│   ├── __init__.py                    # Public API (load_graph, graph_statistics, seed selection, IC)
│   ├── graph.py                       # Adjacency-list Graph, BFS, components, Handshaking Lemma
│   ├── propagation.py                 # Independent Cascade diffusion & wavefront history layers
│   ├── seed_selection.py              # Random, Degree, and Greedy influence maximization
│   ├── metrics.py                     # High-resolution timing separation & fair RNG streams
│   ├── monte_carlo.py                 # Root forwarder to Monte Carlo Engine
│   ├── experiments.py                 # Root forwarder to Automated Experiment Runner
│   └── main.py                        # CLI entry point & JSON benchmark runner
├── dashboard/                         # Role 2 Streamlit Web Application
│   ├── app.py                         # Primary Streamlit entry point (streamlit run dashboard/app.py)
│   └── streamlit_app.py               # Application forwarder & sys.path resolver
├── backend/                           # Role 2 Modular Backend Framework
│   ├── src/
│   │   ├── dashboard/                 # Multi-page controllers, sidebar controls, and UI panels
│   │   ├── experiments/               # Automated benchmark runners (Experiments A, B, C, D)
│   │   ├── metrics/                   # Metrics calculation engine, aggregators, and models
│   │   ├── monte_carlo/               # Monte Carlo simulation engine & validators
│   │   ├── network_visualization/     # PyVis physics force-directed & Plotly 2D renderers
│   │   ├── storage/                   # StorageManager for CSV/JSON persistence & retrieval
│   │   └── visualization/             # Plotly comparison charts, scaling curves & histograms
│   ├── tests/                         # Role 2 Automated Test Suite (95 passing unit tests)
│   └── results/                       # Canonical output directory for CSV/JSON artifacts
├── tests/                             # Role 1 Automated Test Suite (52 passing unit tests)
│   ├── test_graph.py                  # Invariants, BFS shortest paths, SNAP dataset validation
│   ├── test_ic.py                     # IC boundary conditions, history layers, RNG determinism
│   ├── test_seed_selection.py         # Submodularity, overlap flaw counterexample, greedy correctness
│   └── test_metrics.py                # Runtime measurement separation, JSON schemas
├── results/                           # Results storage directory
│   ├── experiments/                   # Automated experiment DataFrames (expA, expB, expC, expD CSVs)
│   ├── metrics/                       # Aggregated metrics comparison summaries
│   ├── raw/                           # Raw Monte Carlo simulation JSON runs
│   └── dataset_stats/                 # Network topology statistics JSON records
├── screenshots/                       # Presentation & Viva Visual Demonstration Artifacts
│   ├── README.md                      # Screenshot capturing guide & viva checklist
│   ├── dashboard_home.png             # Dashboard overview and dataset topology cards
│   ├── graph_visualization.png        # Interactive 2D network cascade (Red/Green/Blue nodes)
│   ├── algorithm_comparison.png       # Reach vs. Runtime grouped comparison bar chart
│   └── influence_results.png          # 7 KPI summary cards and sortable results ledger
├── requirements.txt                   # Universal project dependencies
├── HANDOVER.md                        # Inter-role API contract & frontend handover documentation
└── README.md                          # Master documentation
```

---

## 🎬 Faculty Live Demo Workflow

Experience the full simulation and benchmarking pipeline in 5 intuitive steps:

```
Faculty Live Demo Workflow
┌───────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
│  1. Load Dataset  │ ──> │  2. Select Algorithm  │ ──> │  3. Choose Seed Count │
│  SNAP Facebook    │     │  Degree vs. Greedy    │     │  Slider: k = 10       │
└───────────────────┘     └───────────────────────┘     └───────────────────────┘
                                                                    │
                                                                    ▼
┌───────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
│ 5. Compare Results│ <── │ 4. Run Simulation     │ <── │  Set Probability (p)  │
│ Charts & Exports  │     │ 🚀 Click Run          │     │  Slider: p = 0.05     │
└───────────────────┘     └───────────────────────┘     └───────────────────────┘
```

1. **Load Dataset:** Select `SNAP Facebook (4,039 nodes)` from the sidebar. The overview card instantly displays:
   * **Nodes ($|V|$):** $4{,}039$
   * **Edges ($|E|$):** $88{,}234$
   * **Avg Degree ($\langle k \rangle$):** $43.69$
   * **Density:** $0.01082$
2. **Select Algorithm:** Choose `Degree` (Fast Heuristic) or `Greedy` (Approximation Algorithm). For Greedy, keep **Top 50 Candidates (Fast)** checked to evaluate results interactively in $\approx 20\text{ seconds}$.
3. **Choose Seed Count ($k$) & Probability ($p$):** Adjust the sliders (e.g., $k=10$ seeds, $p=0.05$ transmission probability, $N=100$ Monte Carlo runs).
4. **Run Simulation:** Click **🚀 Run Simulation**. Watch the real-time progress bar execute the diffusion model and render the 2D network:
   * 🔴 **Red Nodes:** Initial Seed Influencers.
   * 🟢 **Green Nodes:** Successfully Influenced Users.
   * 🔵 **Blue Nodes:** Inactive / Unreached Users.
5. **Compare Results & Export:**
   * Review the **7 KPI Metric Cards** (Mean Influence, Min, Max, Std Dev, Influence %, Spread Efficiency, Runtime).
   * Open the **Analytical Visualizations** tab to inspect the Reach vs. Runtime grouped bar chart and seed scaling curves.
   * Export the complete simulation ledger via **📥 Export Table as CSV** or **📦 Export Table as JSON**.

---

## 🔬 Automated Benchmark Experiments (Experiments A–D)

Navigate to the **Experiments** tab in the dashboard or execute via Python:

* **Experiment A (Seed Budget Analysis):** Compares Random, Degree, and Greedy across $k \in \{5, 10, 20, 50\}$ with fixed $p=0.10$. Proves submodular diminishing marginal returns.
* **Experiment B (Probability Sensitivity):** Tests cascade spread across $p \in \{0.05, 0.10, 0.20, 0.30\}$ with fixed $k=10$. Demonstrates percolation phase transitions.
* **Experiment C (Network Scalability):** Evaluates algorithm runtime scaling across synthetic networks of sizes $|V| \in \{500, 1000, 2000, 4000\}$. Demonstrates why Degree ($O(V \log V)$) outperforms Greedy ($O(k \cdot |V| \cdot R \cdot (V+E))$) in speed.
* **Experiment D (Monte Carlo Convergence):** Tracks the Relative Standard Error ($\text{RSE} = s / \bar{x}$) across $N \in \{10, 50, 100, 500\}$ simulation runs, plotting confidence envelopes ($\mu \pm 1\sigma$).

---

## 🧪 Automated Testing Suite (147 Tests)

Run the full dual-role test suite:

```bash
# 1. Run Role 1 Graph & Algorithm Unit Tests (52 tests)
python -m unittest discover tests -v

# 2. Run Role 2 Simulation, Metrics & Dashboard Tests (95 tests)
pytest backend/tests -v
```

All 147 tests pass in under 25 seconds with zero failures.

---

## 💡 Key DAA Theoretical Insights for Viva

1. **Why is Influence Maximization NP-Hard?** Kempe et al. (2003) proved NP-hardness via a reduction from the **Set Cover** problem. Computing exact spread $\sigma(S)$ is $\#\text{P}$-hard (Chen et al.), necessitating Monte Carlo estimation.
2. **What is the Degree Overlap Flaw?** Degree-based selection chooses hubs with the most connections. If these hubs share mutual friends (dense clustering), their reach overlaps heavily, wasting budget.
3. **Why does Greedy achieve $(1 - 1/e)$?** Because influence spread $\sigma(S)$ is **monotone** and **submodular** (diminishing returns). By Nemhauser et al. (1978), greedily selecting the candidate with maximum marginal gain guarantees at least $\approx 63.2\%$ of the optimal spread $S^*$.
