# Influence Maximization in Social Networks (DAA Project)

## Project Overview
This is a **Design and Analysis of Algorithms (DAA)** project focused on solving the **Influence Maximization Problem** on real-world social networks.

Given an undirected social network graph $G = (V, E)$ and an integer $k$, the goal is to select a seed set $S \subseteq V$ of size $|S| = k$ that maximizes the expected spread of influence under the **Independent Cascade (IC)** diffusion model.

---

## Team & Roles

* **Zaid Hussain** — *Graph & Algorithm Engineer*
  * Dataset ingestion & graph modeling
  * Core traversal algorithms (BFS, Connected Components)
  * Independent Cascade (IC) diffusion simulation (Monte Carlo)
  * Seed selection algorithms (Random, Degree Heuristic, Greedy Marginal Gain)
  * Theoretical analysis (Complexity, Submodularity, Monotonicity)
  * Testing & Core Backend API
* **Himanshu** — *Frontend, Visualization & Integration Engineer*
  * UI/Frontend dashboards
  * Interactive network graph visualization
  * Algorithm comparison benchmarks and charts
  * Backend API integration

---

## Dataset
* **Stanford SNAP:** Facebook combined ego-network (`facebook_combined.txt`)
* **Nodes ($V$):** 4,039
* **Edges ($E$):** 88,234 (undirected)

---

## Diffusion Model & Algorithms
1. **Diffusion Model:** Independent Cascade (IC) with Monte Carlo simulations ($R$).
2. **Seed Selection Algorithms:**
   * **Baseline 1:** Random Selection
   * **Baseline 2:** Highest-Degree Heuristic
   * **Core Algorithm:** Greedy Influence Maximization (Marginal Gain)

---

## Project Structure
```text
influence_maximization/
├── data/
│   └── facebook_combined.txt
├── src/
│   ├── __init__.py
│   ├── graph.py
│   ├── propagation.py
│   ├── seed_selection.py
│   ├── metrics.py
│   └── main.py
├── tests/
│   ├── __init__.py
│   ├── test_graph.py
│   ├── test_bfs.py
│   ├── test_ic.py
│   └── test_seed_selection.py
├── .gitignore
└── README.md
```
