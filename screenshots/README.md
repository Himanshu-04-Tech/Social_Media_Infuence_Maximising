# 📸 Visual Demonstration Artifacts & Screenshot Checklist

This directory contains visual presentation artifacts and screenshots demonstrating the **Social Network Influence Maximization Simulator** Streamlit Dashboard in operation for project defense and viva evaluation.

---

## 📋 Required Screenshots Matrix

| # | File Name | Target Dashboard View | Viva & Evaluation Purpose |
| :-: | :--- | :--- | :--- |
| **1** | `dashboard_home.png` | **Dashboard Home / Dataset Topology** | Demonstrates the loaded social network topology cards (Nodes: 4,039, Edges: 88,234, Avg Degree: 43.69, Density: 0.0108), algorithm selector, and parameter sliders ($k, p, N$). |
| **2** | `graph_visualization.png` | **2D Interactive Network Cascade** | Shows the interactive PyVis / Plotly network graph with color-coded nodes: 🔴 Red (Seed nodes), 🟢 Green (Influenced nodes), and 🔵 Blue (Normal inactive nodes), with degree-proportional sizing. |
| **3** | `algorithm_comparison.png` | **Algorithm Comparison Bar Chart** | Displays the comparative grouped bar chart benchmarking **Random vs. Degree vs. Greedy** across Mean Influence Spread and Computation Runtime (with dual y-axis). |
| **4** | `influence_results.png` | **Metrics KPI Cards & Results Table** | Shows the 7 KPI metric cards (Mean Influence, Min, Max, Std Dev, Influence %, Spread Efficiency, Runtime) and the sortable tabular simulation ledger with CSV/JSON export buttons. |

---

## 🎯 Detailed Guide for Capturing Screenshots

### 1. `dashboard_home.png` (Dashboard Overview)
* **What should be visible:**
  - Sidebar showing dataset selection (`SNAP Facebook (4,039 nodes)`), algorithm dropdown (`Degree` or `Greedy`), and sliders for $k=10$, $p=0.05$, $N=100$.
  - Top Dataset Overview cards displaying **4,039 Nodes**, **88,234 Edges**, **43.69 Average Degree**, and **0.01082 Density**.
  - Simulation Control buttons (`🚀 Run Simulation` and `🔄 Reset`).
* **Why it is needed:** Proves dataset ingestion, parameter validation, and user interface responsiveness without touching code.
* **Viva Value:** Demonstrates to examiners that the tool is faculty-friendly and usable by non-programmers.

---

### 2. `graph_visualization.png` (Network Diffusion)
* **What should be visible:**
  - Interactive force-directed graph canvas (PyVis physics layout).
  - Status Legend: 🔴 **Seed Nodes (Red)**, 🟢 **Influenced Nodes (Green)**, 🔵 **Normal Inactive Nodes (Blue)**.
  - Node Inspection expander showing selected node ID, degree, and activation status.
* **Why it is needed:** Visually verifies the Independent Cascade (IC) discrete diffusion model in action.
* **Viva Value:** Directly proves that seeds propagate influence through topological neighbors and validates the SRS color-coding standard.

---

### 3. `algorithm_comparison.png` (Benchmarking & DAA Analysis)
* **What should be visible:**
  - Grouped bar chart comparing **Random**, **Degree**, and **Greedy**.
  - Blue bars showing Mean Influence Spread (Greedy > Degree >> Random).
  - Amber bars showing Execution Runtime (Degree < 0.5s vs. Greedy ~20s).
  - Dual y-axis showing Influence (left) and Runtime in seconds (right).
* **Why it is needed:** Validates the core algorithmic trade-off studied in DAA coursework.
* **Viva Value:** Demonstrates the **Degree Overlap Flaw** vs. **Greedy Submodular Optimization** ($\approx 63.2\%$ approximation guarantee).

---

### 4. `influence_results.png` (Statistical KPIs & Exports)
* **What should be visible:**
  - 7 KPI cards: Mean Influence, Min Influence, Max Influence, Sample Std Dev ($\pm \sigma$), Influence %, Spread Efficiency ($\bar{I}/k$), and Monotonic Execution Runtime.
  - Cumulative Results Table with multi-column filtering.
  - `📥 Export Table as CSV` and `📦 Export Table as JSON` download buttons.
* **Why it is needed:** Verifies Module 1 (Monte Carlo Engine), Module 2 (Metrics Engine), and Module 4 (Storage Layer).
* **Viva Value:** Proves statistical rigor, Law of Large Numbers convergence, and automated persistence.
