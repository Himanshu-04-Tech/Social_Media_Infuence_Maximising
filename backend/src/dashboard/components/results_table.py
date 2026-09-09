"""Results table and export component for the Streamlit Dashboard.

Displays cumulative simulation runs in an interactive tabular grid (AgGrid or st.dataframe)
with sorting, filtering, column selection, and CSV export buttons.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_results_table(results_df: pd.DataFrame) -> None:
    """Render the interactive results table with sorting, pagination, and export tools."""
    st.subheader("📋 Cumulative Simulation Results")

    if results_df.empty:
        st.info("No simulation records logged yet. Run a simulation to populate the table.")
        return

    # Column Selection Filter
    with st.expander("🛠️ Table Column Settings", expanded=False):
        all_cols = list(results_df.columns)
        selected_cols = st.multiselect(
            "Visible Columns",
            options=all_cols,
            default=all_cols,
        )

    display_df = results_df[selected_cols] if selected_cols else results_df

    # Try AgGrid for advanced sorting/filtering/pagination
    rendered_aggrid = False
    try:
        from st_aggrid import AgGrid, GridOptionsBuilder  # type: ignore

        gb = GridOptionsBuilder.from_dataframe(display_df)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
        gb.configure_side_bar()
        gb.configure_default_column(sortable=True, filter=True, resizable=True)
        grid_options = gb.build()

        AgGrid(
            display_df,
            gridOptions=grid_options,
            height=300,
            width="100%",
            theme="streamlit",
        )
        rendered_aggrid = True
    except Exception:
        rendered_aggrid = False

    if not rendered_aggrid:
        st.dataframe(display_df, use_container_width=True, height=280)

    # Export Controls (SRS requirement)
    c1, c2 = st.columns([1, 4])
    with c1:
        csv_data = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Export Table as CSV",
            data=csv_data,
            file_name="influence_simulation_results.csv",
            mime="text/csv",
            help="Download the visible simulation records as a CSV file.",
        )
    with c2:
        json_data = display_df.to_json(orient="records", indent=2)
        st.download_button(
            label="📦 Export Table as JSON",
            data=json_data,
            file_name="influence_simulation_results.json",
            mime="application/json",
            help="Download the visible simulation records as formatted JSON.",
        )
