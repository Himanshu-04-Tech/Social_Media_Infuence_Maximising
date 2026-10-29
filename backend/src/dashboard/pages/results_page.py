"""Page 3: Stored Results & Persistence Manager.

Uses Module 4 Storage Layer (StorageManager) to inspect, load, download,
and delete saved experiments, metrics summaries, and raw simulation outputs.
"""

from __future__ import annotations

import json
import streamlit as st

from storage.storage_manager import StorageManager


def render_results_page(params: dict[str, any]) -> None:
    """Render the Stored Results management interface."""
    st.title("💾 Stored Results & Persistence Manager")
    st.caption("Manage saved artifacts produced by the simulator (Module 4 Storage Layer).")

    storage = StorageManager()

    try:
        saved_files = storage.list_all_saved_results()
    except Exception as err:
        st.error(f"❌ Failed to query storage directories: {err}")
        return

    tab1, tab2, tab3 = st.tabs([
        "📁 Experiment DataFrames",
        "📊 Metrics Summaries",
        "🔬 Raw Simulations",
    ])

    # Tab 1: Experiment DataFrames
    with tab1:
        st.subheader("Saved Experiment CSVs")
        exp_files = saved_files.get("experiments", [])
        if not exp_files:
            st.info("No saved experiment files found in `results/experiments/`.")
        else:
            selected_exp = st.selectbox(
                "Select Experiment File",
                options=exp_files,
                format_func=lambda r: getattr(r, "file_name", str(r)),
                key="sel_exp",
            )
            if selected_exp:
                fname = getattr(selected_exp, "file_name", str(selected_exp))
                c1, c2 = st.columns([4, 1])
                try:
                    df = storage.load_experiment_results(selected_exp)
                    st.dataframe(df, use_container_width=True)

                    with c1:
                        st.download_button(
                            label=f"📥 Download {fname}",
                            data=df.to_csv(index=False),
                            file_name=fname,
                            mime="text/csv",
                            key=f"dl_{fname}",
                        )
                    with c2:
                        if st.button(f"🗑️ Delete", key=f"del_{fname}"):
                            storage.delete_saved_result(selected_exp, "experiments")
                            st.success(f"Deleted {fname}")
                            st.rerun()
                except Exception as err:
                    st.error(f"Error reading file: {err}")

    # Tab 2: Metrics Summaries
    with tab2:
        st.subheader("Saved Metrics CSVs")
        metrics_files = saved_files.get("metrics", [])
        if not metrics_files:
            st.info("No metrics summaries found in `results/metrics/`.")
        else:
            selected_m = st.selectbox(
                "Select Metrics File",
                options=metrics_files,
                format_func=lambda r: getattr(r, "file_name", str(r)),
                key="sel_m",
            )
            if selected_m:
                fname_m = getattr(selected_m, "file_name", str(selected_m))
                try:
                    df_m = storage.load_metrics_summary(selected_m)
                    st.dataframe(df_m, use_container_width=True)
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.download_button(
                            label=f"📥 Download {fname_m}",
                            data=df_m.to_csv(index=False),
                            file_name=fname_m,
                            mime="text/csv",
                            key=f"dl_{fname_m}",
                        )
                    with c2:
                        if st.button(f"🗑️ Delete", key=f"del_{fname_m}"):
                            storage.delete_saved_result(selected_m, "metrics")
                            st.success(f"Deleted {fname_m}")
                            st.rerun()
                except Exception as err:
                    st.error(f"Error reading file: {err}")

    # Tab 3: Raw Simulations
    with tab3:
        st.subheader("Raw Simulation JSON Records")
        raw_files = saved_files.get("raw", [])
        if not raw_files:
            st.info("No raw simulation JSON files found in `results/raw/`.")
        else:
            selected_raw = st.selectbox(
                "Select Raw JSON File",
                options=raw_files,
                format_func=lambda r: getattr(r, "file_name", str(r)),
                key="sel_raw",
            )
            if selected_raw:
                fname_raw = getattr(selected_raw, "file_name", str(selected_raw))
                try:
                    raw_data = storage.load_raw_simulation(selected_raw)
                    st.json(raw_data)
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.download_button(
                            label=f"📥 Download {fname_raw}",
                            data=json.dumps(raw_data, indent=2),
                            file_name=fname_raw,
                            mime="application/json",
                            key=f"dl_{fname_raw}",
                        )
                    with c2:
                        if st.button(f"🗑️ Delete", key=f"del_{fname_raw}"):
                            storage.delete_saved_result(selected_raw, "raw")
                            st.success(f"Deleted {fname_raw}")
                            st.rerun()
                except Exception as err:
                    st.error(f"Error reading file: {err}")
