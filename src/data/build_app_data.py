"""
Phase 6, Step 1: slim intersections.geojson + risk_scores.parquet down to
the single file the Streamlit app ships with.

data/ is gitignored and Streamlit Community Cloud only sees what's checked
into the public repo, so app/data/risk_scores_app.parquet is the sole
bridge between the pipeline and the deployed app. See docs/phase6_spec.md.
"""

import geopandas as gpd
import pandas as pd

from src.config import (
    APP_DATA_FALLBACK_N,
    APP_DATA_MAX_MB,
    APP_DATA_PATH,
    INTERIM_DIR,
    OUTPUTS_DIR,
)
from src.data.fetch_osm import get_study_area_polygon
from src.utils.geo import snap_to_nearest, to_analysis_crs

INTERSECTIONS_PATH = INTERIM_DIR / "intersections.geojson"
RISK_SCORES_PATH = OUTPUTS_DIR / "risk_scores.parquet"

APP_COLUMNS = [
    "int_id",
    "lat",
    "lon",
    "risk_score",
    "risk_percentile",
    "jurisdiction",
    "n_legs",
    "max_lanes",
    "max_speed_mph",
    "highway_class_max",
    "is_signalized",
    "has_marked_crossing",
    "dist_to_transit_m",
    "ped_crashes_10yr",
]


def load_joined() -> gpd.GeoDataFrame:
    """Load intersections.geojson + risk_scores.parquet, inner-join on int_id."""
    intersections = gpd.read_file(INTERSECTIONS_PATH)
    scores = pd.read_parquet(RISK_SCORES_PATH)
    joined = intersections.merge(scores, on="int_id", how="inner")
    if len(joined) != len(intersections):
        print(
            f"WARNING: join dropped rows -- {len(intersections)} intersections, "
            f"{len(joined)} after merging risk scores"
        )
    return joined


def assign_jurisdiction(gdf: gpd.GeoDataFrame) -> pd.Series:
    """One jurisdiction name per intersection (e.g. "Las Vegas", "Henderson"),
    matched by point-in-polygon against get_study_area_polygon()'s
    one-row-per-place boundaries (a "name" column per STUDY_AREA_PLACES entry).

    The 8 boundaries are adjacent cities/CDPs, not a perfect tiling, so two
    edge cases are handled explicitly rather than left to silently produce
    NaN or duplicate rows:
      - a point intersecting 2+ polygons (sliver overlap) keeps the first match
      - a point intersecting 0 polygons (boundary gap) falls back to
        nearest-boundary in ANALYSIS_CRS, per CLAUDE.md's CRS rule

    Returns a Series of jurisdiction names aligned to gdf's row order.
    """
    boundaries = get_study_area_polygon()[["name", "geometry"]]
    points = gdf[["int_id", "geometry"]]

    matched = gpd.sjoin(points, boundaries, how="left", predicate="intersects")
    matched = matched.drop_duplicates(subset="int_id", keep="first").set_index("int_id")

    missing_ids = matched.index[matched["name"].isna()]
    if len(missing_ids):
        missing_points = points[points["int_id"].isin(missing_ids)]
        nearest = snap_to_nearest(to_analysis_crs(missing_points), to_analysis_crs(boundaries))
        nearest = nearest.drop_duplicates(subset="int_id", keep="first").set_index("int_id")
        matched.loc[missing_ids, "name"] = nearest.loc[missing_ids, "name"]
        print(f"jurisdiction: {len(missing_ids)}/{len(gdf)} filled via nearest-boundary fallback")

    still_missing = matched["name"].isna().sum()
    if still_missing:
        print(f"WARNING: {still_missing} intersections have no jurisdiction match at all")

    return matched.loc[gdf["int_id"], "name"].reset_index(drop=True)


def build_app_table(gdf: gpd.GeoDataFrame, jurisdiction: pd.Series) -> pd.DataFrame:
    """Select the Step-1 column list, pull lat/lon off the point geometry,
    and round coordinates to 5 decimals / scores to 4 (per phase6_spec.md)."""
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    df["lat"] = gdf.geometry.y.round(5)
    df["lon"] = gdf.geometry.x.round(5)
    df["jurisdiction"] = jurisdiction.to_numpy()
    df["risk_score"] = df["risk_score"].round(4)
    df["risk_percentile"] = df["risk_percentile"].round(4)
    return df[APP_COLUMNS]


def enforce_size_budget(
    df: pd.DataFrame,
    path=APP_DATA_PATH,
    max_mb: float = APP_DATA_MAX_MB,
    fallback_n: int = APP_DATA_FALLBACK_N,
) -> pd.DataFrame:
    """Write df to path and print its size. If over max_mb, trim to the
    top fallback_n rows by risk_score and rewrite -- the About page's
    cutoff note (if any) should read this function's printed row count,
    not assume the full 39,183.

    Returns whichever DataFrame actually ended up on disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Wrote {len(df)} rows to {path} ({size_mb:.2f} MB)")

    if size_mb <= max_mb:
        return df

    print(f"Over the {max_mb} MB budget -- trimming to the top {fallback_n} by risk_score.")
    trimmed = df.sort_values("risk_score", ascending=False).head(fallback_n).reset_index(drop=True)
    trimmed.to_parquet(path, index=False)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Wrote {len(trimmed)} rows to {path} ({size_mb:.2f} MB) after trim")
    return trimmed


def main() -> None:
    gdf = load_joined()
    jurisdiction = assign_jurisdiction(gdf)
    app_df = build_app_table(gdf, jurisdiction)
    enforce_size_budget(app_df)


if __name__ == "__main__":
    main()
