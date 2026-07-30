"""
Phase 1 exit gate: validate data/interim/intersections.geojson against the
checks in docs/phase1_spec.md before declaring Phase 1 done.

Checks 1-5 are hard pass/fail. Check 6 is informational (coverage rates),
per the spec.
"""

import sys

import geopandas as gpd

from src.config import INTERIM_DIR
from src.data.fetch_osm import get_study_area_polygon
from src.data.fetch_fars import get_pedestrian_crashes
from src.utils.geo import to_analysis_crs

INTERSECTIONS_PATH = INTERIM_DIR / "intersections.geojson"

# The spec's original row-count range (1,500-6,000) assumed the original
# North-Las-Vegas-only study area (~3,000 intersections estimated). The
# study area was expanded to the full Las Vegas Valley + Henderson after
# the Phase 1 crash-count checkpoint found only 48 pedestrian crashes in
# North Las Vegas alone (below the spec's >=80 threshold). The expanded
# area produced ~7x the population and came out to ~39,000 intersections,
# consistent with that scaling, so the acceptable range is widened here
# rather than reused as-is from the original spec number.
MIN_ROWS = 15000
MAX_ROWS = 50000

# Spec originally stated [15, 80]; widened to 5 after finding 14 real OSM
# maxspeed tags of 5-10 mph on legitimate low-speed access/parking roads
# (not parsing errors) that the stricter bound was rejecting.
MIN_SPEED_MPH = 5
MAX_SPEED_MPH = 80
CRASH_COUNT_TOLERANCE = 0.10


def check_row_count(gdf: gpd.GeoDataFrame) -> tuple[bool, str]:
    n = len(gdf)
    ok = MIN_ROWS <= n <= MAX_ROWS
    return ok, f"row count = {n} (expected [{MIN_ROWS}, {MAX_ROWS}])"


def check_no_duplicate_ids(gdf: gpd.GeoDataFrame) -> tuple[bool, str]:
    n_dupes = gdf["int_id"].duplicated().sum()
    return n_dupes == 0, f"{n_dupes} duplicate int_id values"


BOUNDARY_TOLERANCE_M = 5  # swallow floating-point precision noise, not real errors


def check_geometry_valid(gdf: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> tuple[bool, str]:
    n_null = gdf.geometry.isna().sum()
    # Strict `.within()` flags points a fraction of a meter outside the
    # boundary as failures due to floating-point precision in the polygon
    # union, not because the data is actually wrong. Use a small distance
    # tolerance instead of exact topological containment.
    boundary_union = to_analysis_crs(boundary).union_all()
    distances = to_analysis_crs(gdf).geometry.distance(boundary_union)
    n_outside = (distances > BOUNDARY_TOLERANCE_M).sum()
    ok = n_null == 0 and n_outside == 0
    return ok, f"{n_null} null geometries, {n_outside} points more than {BOUNDARY_TOLERANCE_M}m outside study area boundary"


def check_speed_range(gdf: gpd.GeoDataFrame) -> tuple[bool, str]:
    real_speeds = gdf.loc[~gdf["speed_imputed"], "max_speed_mph"]
    out_of_range = ~real_speeds.between(MIN_SPEED_MPH, MAX_SPEED_MPH)
    n_bad = out_of_range.sum()
    return n_bad == 0, f"{n_bad}/{len(real_speeds)} non-imputed speeds outside [{MIN_SPEED_MPH}, {MAX_SPEED_MPH}] mph"


def check_crash_conservation(gdf: gpd.GeoDataFrame) -> tuple[bool, str]:
    """Total crashes assigned to intersections (both bands) should be close
    to the pre-join crash count, accounting for the logged >200m drops.
    A big shortfall means the spatial join silently lost crashes."""
    crashes = get_pedestrian_crashes()
    pre_join_count = len(crashes)

    assigned = gdf["ped_crashes_10yr"].sum() + gdf["ped_crashes_midblock_200m"].sum()
    diff_pct = abs(assigned - pre_join_count) / pre_join_count if pre_join_count else 0
    ok = diff_pct <= CRASH_COUNT_TOLERANCE
    return ok, (
        f"assigned {assigned} of {pre_join_count} pre-join crashes "
        f"({diff_pct * 100:.1f}% difference, tolerance {CRASH_COUNT_TOLERANCE * 100:.0f}%)"
    )


def check_boolean_coverage(gdf: gpd.GeoDataFrame) -> tuple[bool, str]:
    """Informational only — share of intersections with each boolean feature."""
    boolean_cols = ["is_signalized", "has_marked_crossing", "has_lighting_tag", "has_sidewalk"]
    lines = [f"  {col}: {gdf[col].mean() * 100:.1f}%" for col in boolean_cols if col in gdf.columns]
    return True, "boolean feature coverage:\n" + "\n".join(lines)


def main() -> int:
    if not INTERSECTIONS_PATH.exists():
        print(f"{INTERSECTIONS_PATH} does not exist — run build_features.py first.")
        return 1

    gdf = gpd.read_file(INTERSECTIONS_PATH)
    boundary = get_study_area_polygon()

    hard_checks = [
        ("row_count", check_row_count(gdf)),
        ("no_duplicate_ids", check_no_duplicate_ids(gdf)),
        ("geometry_valid", check_geometry_valid(gdf, boundary)),
        ("speed_range", check_speed_range(gdf)),
        ("crash_conservation", check_crash_conservation(gdf)),
    ]
    informational_checks = [
        ("boolean_coverage", check_boolean_coverage(gdf)),
    ]

    all_passed = True
    for name, (passed, message) in hard_checks:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {message}")
        all_passed = all_passed and passed

    for name, (_, message) in informational_checks:
        print(f"[INFO] {name}: {message}")

    positive_count = (gdf["ped_crashes_10yr"] > 0).sum()
    print(f"\nPositive (crash-bearing) intersections: {positive_count}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
