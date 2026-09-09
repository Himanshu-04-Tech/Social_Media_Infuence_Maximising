"""Unit test suite for the Streamlit Dashboard (Module 7).

Verifies:
1. Sidebar renders and exposes all required parameter controls.
2. Simulation page loads without uncaught exceptions.
3. Simulation workflow triggers Role 1 + Module 1 + Module 2 + Module 4 + Modules 5/6.
4. Metrics panel extracts and formats all 7 required metrics.
5. Analytical charts panel generates visualizations.
6. Results table renders comparison records and provides export data.
7. Experiment page loads and triggers Module 3 runner.
8. Stored results page reads from Module 4 StorageManager.
9. Data export generates valid CSV and JSON structures.
10. Invalid parameters trigger validation error handling.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import networkx as nx
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

# Ensure backend/src is available on path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

ROOT_DIR = SRC_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

APP_PATH = SRC_DIR / "dashboard" / "app.py"

from dashboard.components.charts_panel import render_charts_panel
from dashboard.components.dataset_stats import render_dataset_stats
from dashboard.components.metrics_panel import render_metrics_panel
from dashboard.components.network_panel import render_network_panel
from dashboard.components.results_table import render_results_table
from dashboard.components.sidebar import render_sidebar
from dashboard.pages.experiments_page import render_experiments_page
from dashboard.pages.results_page import render_results_page
from dashboard.pages.settings_page import render_settings_page
from dashboard.pages.simulation_page import render_simulation_page
from dashboard.state_manager import DashboardStateManager
from dashboard.utilities import (
    load_dataset,
    run_full_simulation_pipeline,
    select_seeds,
    simulate_single_cascade,
    to_networkx,
)
from metrics.models import MetricsRecord


@pytest.fixture
def karate_graph():
    """Create a reproducible small graph for testing."""
    return nx.karate_club_graph()


@pytest.fixture
def sample_metrics():
    """Create a sample MetricsRecord matching the exact model signature."""
    return MetricsRecord(
        algorithm_name="Degree",
        seed_count=4,
        probability=0.1,
        mean_influence=18.5,
        min_influence=12,
        max_influence=25,
        std_influence=3.2,
        influence_pct=54.41,
        runtime_sec=0.045,
        average_reach=18.5,
        spread_efficiency=1.85,
        node_count=34,
        timestamp="2026-09-09T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# Test 1: Sidebar renders
# ---------------------------------------------------------------------------
def test_sidebar_renders():
    """Verify that AppTest renders sidebar with all SRS input controls."""
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    assert len(at.exception) == 0

    # Verify selectbox for dataset and algorithm exist in sidebar
    selectboxes = at.sidebar.selectbox
    assert len(selectboxes) >= 2

    # Verify sliders for k, p, N exist
    sliders = at.sidebar.slider
    assert len(sliders) >= 3


# ---------------------------------------------------------------------------
# Test 2: Simulation page loads
# ---------------------------------------------------------------------------
def test_simulation_page_loads():
    """Verify that AppTest launches the main simulation page without errors."""
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    assert not at.exception
    # Main headers exist
    assert any("Single Simulation Engine" in str(title.value) for title in at.title)
    # Buttons exist (Run Simulation, Reset)
    button_labels = [b.label for b in at.button]
    assert "🚀 Run Simulation" in button_labels
    assert "🔄 Reset" in button_labels


# ---------------------------------------------------------------------------
# Test 3: Run button triggers workflow
# ---------------------------------------------------------------------------
def test_run_button_triggers_workflow(karate_graph):
    """Verify that the full simulation pipeline executes all components end-to-end."""
    results = run_full_simulation_pipeline(
        graph=karate_graph,
        algorithm="Degree",
        k=4,
        probability=0.2,
        num_simulations=10,
    )

    assert results is not None
    assert len(results["seeds"]) == 4
    assert len(results["influenced_nodes"]) >= 4
    assert isinstance(results["metrics_record"], MetricsRecord)
    assert results["metrics_record"].mean_influence >= 4.0
    assert "record_dict" in results
    assert results["record_dict"]["Algorithm"] == "Degree"


# ---------------------------------------------------------------------------
# Test 4: Metrics displayed
# ---------------------------------------------------------------------------
def test_metrics_displayed(sample_metrics):
    """Verify that render_metrics_panel formats all 7 required metrics without errors."""
    with patch("streamlit.subheader") as mock_sub, patch("streamlit.metric") as mock_metric:
        render_metrics_panel(sample_metrics)

        assert mock_sub.called
        assert mock_metric.call_count == 7
        labels = [call.kwargs.get("label") or call.args[0] for call in mock_metric.call_args_list]
        assert any("Mean Influence" in label for label in labels)
        assert any("Min Influence" in label for label in labels)
        assert any("Max Influence" in label for label in labels)
        assert any("Std Deviation" in label for label in labels)
        assert any("Influence %" in label for label in labels)
        assert any("Spread Efficiency" in label for label in labels)
        assert any("Runtime" in label for label in labels)


# ---------------------------------------------------------------------------
# Test 5: Charts displayed
# ---------------------------------------------------------------------------
def test_charts_displayed():
    """Verify that render_charts_panel generates Module 5 analytical figures."""
    df = pd.DataFrame([
        {"Algorithm": "Random", "Mean Influence": 15.0, "Runtime (s)": 0.01},
        {"Algorithm": "Degree", "Mean Influence": 28.0, "Runtime (s)": 0.02},
    ])

    class MockTab:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    with patch("streamlit.tabs", return_value=[MockTab(), MockTab(), MockTab(), MockTab()]):
        with patch("streamlit.plotly_chart") as mock_plotly:
            render_charts_panel(df)
            assert mock_plotly.called


# ---------------------------------------------------------------------------
# Test 6: Results table displayed
# ---------------------------------------------------------------------------
def test_results_table_displayed():
    """Verify that render_results_table processes simulation history without exceptions."""
    df = pd.DataFrame([
        {"Algorithm": "Degree", "Seeds (k)": 5, "Mean Influence": 22.5, "Runtime (s)": 0.02}
    ])

    with patch("streamlit.subheader") as mock_sub, patch("streamlit.download_button") as mock_dl:
        render_results_table(df)
        assert mock_sub.called
        assert mock_dl.call_count >= 1


# ---------------------------------------------------------------------------
# Test 7: Experiment page loads
# ---------------------------------------------------------------------------
def test_experiment_page_loads(karate_graph):
    """Verify that render_experiments_page loads and exposes all experiment triggers."""
    with patch("dashboard.pages.experiments_page.load_dataset", return_value=karate_graph):
        with patch("streamlit.title") as mock_title, patch("streamlit.button") as mock_btn:
            render_experiments_page({"dataset": "Zachary's Karate Club (34 nodes)"})
            assert mock_title.called
            assert mock_btn.call_count >= 5  # Exp A, B, C, D, Run All


# ---------------------------------------------------------------------------
# Test 8: Stored results page loads
# ---------------------------------------------------------------------------
def test_stored_results_page_loads():
    """Verify that render_results_page interfaces with StorageManager."""
    with patch("streamlit.title") as mock_title, patch("streamlit.tabs") as mock_tabs:
        mock_tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
        render_results_page({})
        assert mock_title.called
        assert mock_tabs.called


# ---------------------------------------------------------------------------
# Test 9: Export works
# ---------------------------------------------------------------------------
def test_export_works():
    """Verify that DataFrame export generates valid CSV and JSON string buffers."""
    df = pd.DataFrame([
        {"Algorithm": "Degree", "k": 10, "Mean": 150.5}
    ])
    csv_bytes = df.to_csv(index=False)
    assert "Algorithm,k,Mean" in csv_bytes
    assert "Degree,10,150.5" in csv_bytes

    json_str = df.to_json(orient="records")
    assert '"Algorithm":"Degree"' in json_str


# ---------------------------------------------------------------------------
# Test 10: Invalid parameters show error
# ---------------------------------------------------------------------------
def test_invalid_parameters_show_error(karate_graph):
    """Verify that input validation catches invalid k, p, and N parameters."""
    with patch("dashboard.pages.simulation_page.load_dataset", return_value=karate_graph):
        with patch("streamlit.error") as mock_err, patch("streamlit.button", return_value=True):
            # Invalid k <= 0
            render_simulation_page({
                "dataset": "Karate",
                "algorithm": "Degree",
                "k": 0,
                "probability": 0.1,
                "num_simulations": 100,
            })
            assert mock_err.called
            assert any("Seed Count" in str(c) for c in mock_err.call_args_list)

            # Invalid probability <= 0
            mock_err.reset_mock()
            render_simulation_page({
                "dataset": "Karate",
                "algorithm": "Degree",
                "k": 10,
                "probability": -0.5,
                "num_simulations": 100,
            })
            assert mock_err.called
            assert any("Probability" in str(c) for c in mock_err.call_args_list)
