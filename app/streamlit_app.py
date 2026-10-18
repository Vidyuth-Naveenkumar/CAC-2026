from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

DATA_PATH = Path(__file__).parent / "data" / "risk_scores_app.parquet"

MAP_CENTER = (36.17, -115.14)  # Las Vegas valley
MAP_ZOOM = 10
MAX_MARKERS_DEFAULT = 5000

# risk_percentile bucket -> (label, color), checked high-to-low
PERCENTILE_COLORS = [
    (90, "Top 10% (90-100)", "red"),
    (75, "75-90", "orange"),
    (50, "50-75", "gold"),
    (0, "Below 50", "lightgray"),
]


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)


def marker_color(percentile: pd.Series) -> pd.Series:
    """Vectorized risk_percentile -> color, per the PERCENTILE_COLORS bins."""
    colors = pd.Series("lightgray", index=percentile.index)
    for threshold, _label, color in reversed(PERCENTILE_COLORS):
        colors[percentile >= threshold] = color
    return colors


def apply_filters(
    df: pd.DataFrame,
    jurisdictions: list[str],
    min_percentile: float,
    road_classes: list[str],
    crashes_only: bool,
) -> pd.DataFrame:
    mask = (
        df["jurisdiction"].isin(jurisdictions)
        & (df["risk_percentile"] >= min_percentile)
        & df["highway_class_max"].isin(road_classes)
    )
    if crashes_only:
        mask &= df["ped_crashes_10yr"] > 0
    return df[mask]


def render_sidebar(df: pd.DataFrame) -> tuple[list[str], float, list[str], bool]:
    st.sidebar.header("Filters")

    all_jurisdictions = sorted(df["jurisdiction"].unique())
    jurisdictions = st.sidebar.multiselect(
        "Jurisdiction", options=all_jurisdictions, default=all_jurisdictions
    )

    min_percentile = st.sidebar.slider("Minimum risk percentile", 0, 100, 0)

    all_road_classes = sorted(df["highway_class_max"].unique())
    road_classes = st.sidebar.multiselect(
        "Road class", options=all_road_classes, default=all_road_classes
    )

    crashes_only = st.sidebar.checkbox("Only intersections with a recorded crash")

    return jurisdictions, min_percentile, road_classes, crashes_only


def build_map(filtered: pd.DataFrame, max_markers: int = MAX_MARKERS_DEFAULT) -> folium.Map:
    st.caption(f"{len(filtered)} intersections match your filters.")

    show_all = True
    if len(filtered) > max_markers:
        show_all = st.checkbox(
            f"Render all {len(filtered)} matching intersections (slower)", value=False
        )

    if show_all or len(filtered) <= max_markers:
        plotted = filtered
    else:
        plotted = filtered.nlargest(max_markers, "risk_score")
        st.info(
            f"Showing the top {max_markers} by risk score out of {len(filtered)} matches. "
            "Check the box above to render everything."
        )

    fmap = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM)
    colors = marker_color(plotted["risk_percentile"])

    for row, color in zip(plotted.itertuples(), colors):
        popup = (
            f"<b>{row.int_id}</b><br>"
            f"Risk percentile: {row.risk_percentile:.1f}<br>"
            f"Lanes: {row.max_lanes if pd.notna(row.max_lanes) else 'unknown'}<br>"
            f"Speed limit: {row.max_speed_mph:.0f} mph<br>"
            f"Road class: {row.highway_class_max}<br>"
            f"Signalized: {'Yes' if row.is_signalized else 'No'}<br>"
            f"Recorded crashes (10yr): {row.ped_crashes_10yr}"
        )
        folium.CircleMarker(
            location=(row.lat, row.lon),
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(popup, max_width=250),
        ).add_to(fmap)

    return fmap


def render_legend() -> None:
    items = " &nbsp;&nbsp; ".join(
        f'<span style="color:{color}">●</span> {label}' for _, label, color in PERCENTILE_COLORS
    )
    st.markdown(f"**Risk percentile:** {items}", unsafe_allow_html=True)


def render_map_tab(filtered: pd.DataFrame) -> None:
    render_legend()
    fmap = build_map(filtered)
    st_folium(fmap, height=600, use_container_width=True, returned_objects=[])


PRIORITY_DISPLAY_COLUMNS = {
    "rank": "Rank",
    "int_id": "Intersection ID",
    "jurisdiction": "Jurisdiction",
    "risk_percentile": "Risk percentile",
    "max_speed_mph": "Speed (mph)",
    "max_lanes": "Lanes",
    "highway_class_max": "Road class",
    "ped_crashes_10yr": "Recorded crashes",
}


def build_priority_table(filtered: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Top top_n rows of `filtered` by risk_score, ranked and relabeled
    per the Tab 2 column list in phase6_spec.md."""
    top = filtered.nlargest(top_n, "risk_score").reset_index(drop=True)
    top.insert(0, "rank", range(1, len(top) + 1))
    return top[list(PRIORITY_DISPLAY_COLUMNS)].rename(columns=PRIORITY_DISPLAY_COLUMNS)


def render_priority_tab(filtered: pd.DataFrame) -> None:
    top_n = st.slider("Show top N by risk score", min_value=50, max_value=2000, value=250, step=50)
    table = build_priority_table(filtered, top_n)
    st.caption(f"Showing {len(table)} of {len(filtered)} filtered intersections.")

    st.dataframe(table, width="stretch", hide_index=True)

    csv = table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download this table as CSV",
        data=csv,
        file_name="priority_intersections.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="Pedestrian Infrastructure Gap Auditor", layout="wide")
    st.title("Pedestrian Infrastructure Gap Auditor")

    df = load_data()
    jurisdictions, min_percentile, road_classes, crashes_only = render_sidebar(df)
    filtered = apply_filters(df, jurisdictions, min_percentile, road_classes, crashes_only)

    map_tab, priority_tab, about_tab = st.tabs(["Map", "Priority list", "About"])

    with map_tab:
        render_map_tab(filtered)

    with priority_tab:
        render_priority_tab(filtered)


if __name__ == "__main__":
    main()
