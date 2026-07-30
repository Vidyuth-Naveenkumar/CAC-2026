"""
Phase 1, Step 4: join OSM road features and FARS pedestrian crashes onto
each intersection, and write data/interim/intersections.geojson.

This is the training table for the Phase 2 crash-risk model — one row per
intersection, per the schema in docs/phase1_spec.md.
"""

import argparse

import geopandas as gpd
import pandas as pd

from src.config import (
    INTERIM_DIR,
    CRASH_SNAP_RADIUS_M,
    CRASH_MIDBLOCK_RADIUS_M,
    CROSSING_SEARCH_RADIUS_M,
)
from src.data.fetch_osm import get_graph, consolidate_and_filter, get_crossings, get_bus_stops
from src.data.fetch_fars import get_pedestrian_crashes
from src.utils.geo import to_analysis_crs, to_storage_crs, snap_to_nearest

OUTPUT_PATH = INTERIM_DIR / "intersections.geojson"

# Highest-order class among incident edges wins. Unranked classes
# (unclassified, living_street, etc.) default to rank 0 and lose ties.
HIGHWAY_RANK = {
    "motorway": 6,
    "trunk": 5,
    "primary": 4,
    "secondary": 3,
    "tertiary": 2,
    "residential": 1,
}

# maxspeed imputation when NO incident edge has a real maxspeed tag.
# Assumption, easy to revisit — one dict, not scattered through the code.
DEFAULT_MPH_BY_CLASS = {
    "motorway": 65,
    "trunk": 55,
    "primary": 45,
    "secondary": 40,
    "tertiary": 35,
    "residential": 25,
}

# A crossing tagged "marked" or "traffic_signals" within this radius of an
# intersection counts as that intersection having a marked crossing.
# Assumption — a crossing can sit a bit off the exact node.
MARKED_CROSSING_VALUES = {"marked", "traffic_signals"}


def _as_list(tag) -> list:
    """OSM tags are sometimes a single value, sometimes a list of values
    (when incident edges disagree, e.g. one has 2 lanes and another has 3)."""
    if tag is None or (isinstance(tag, float) and pd.isna(tag)):
        return []
    return tag if isinstance(tag, list) else [tag]


def parse_lanes(tag) -> list[float]:
    """Parse a 'lanes' tag into numeric values, skipping anything non-numeric."""
    values = []
    for v in _as_list(tag):
        try:
            values.append(float(v))
        except (TypeError, ValueError):
            continue
    return values


def parse_maxspeed_mph(tag) -> list[float]:
    """Parse a 'maxspeed' tag into mph values.

    US OSM extracts use "35 mph"-style strings. Defensively handle a bare
    "km/h" value too, in case a stray non-US-tagged way slipped in.
    """
    values = []
    for v in _as_list(tag):
        if not isinstance(v, str):
            continue
        v = v.strip().lower()
        try:
            if "km/h" in v:
                values.append(float(v.replace("km/h", "").strip()) * 0.621371)
            else:
                values.append(float(v.replace("mph", "").strip()))
        except ValueError:
            continue
    return values


def highway_class_max(tag) -> str:
    """Highest-ranked highway class among possibly-multiple tag values."""
    classes = _as_list(tag)
    if not classes:
        return "unknown"
    return max(classes, key=lambda c: HIGHWAY_RANK.get(c, 0))


def _tag_present(tag) -> bool:
    """True if a tag has any non-null, non-'no' value (handles list tags too)."""
    return any(v not in (None, "no") and not (isinstance(v, float) and pd.isna(v)) for v in _as_list(tag))


def build_edge_lookup(edges: gpd.GeoDataFrame) -> pd.core.groupby.DataFrameGroupBy:
    """Map each node id to all edges incident to it (as either endpoint).

    Edges are directed (u -> v), but a physical intersection is incident to
    an edge regardless of direction, so every edge is attached to both its
    u and v node.
    """
    edges_flat = edges.reset_index()
    by_u = edges_flat.rename(columns={"u": "node_id"}).drop(columns=["v"])
    by_v = edges_flat.rename(columns={"v": "node_id"}).drop(columns=["u"])
    incident = pd.concat([by_u, by_v], ignore_index=True)
    return incident.groupby("node_id")


def compute_road_features(intersections: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> pd.DataFrame:
    """Aggregate incident-edge tags into one feature row per intersection."""
    edge_groups = build_edge_lookup(edges)

    lit_coverage = 0
    sidewalk_coverage = 0
    total_edges = len(edges)

    rows = []
    for osmid, node in intersections.iterrows():
        incident = edge_groups.get_group(osmid) if osmid in edge_groups.groups else pd.DataFrame()

        lane_values = [v for tag in incident.get("lanes", []) for v in parse_lanes(tag)]
        real_speeds = [v for tag in incident.get("maxspeed", []) for v in parse_maxspeed_mph(tag)]
        highway_classes = [highway_class_max(tag) for tag in incident.get("highway", [])]

        if real_speeds:
            max_speed_mph = max(real_speeds)
            speed_imputed = False
        else:
            candidate_classes = highway_classes or ["residential"]
            max_speed_mph = max(DEFAULT_MPH_BY_CLASS.get(c, 25) for c in candidate_classes)
            speed_imputed = True

        has_lighting = any(_tag_present(tag) for tag in incident.get("lit", []))
        has_sidewalk = any(_tag_present(tag) for tag in incident.get("sidewalk", []))
        lit_coverage += incident.get("lit", pd.Series(dtype=object)).notna().sum()
        sidewalk_coverage += incident.get("sidewalk", pd.Series(dtype=object)).notna().sum()

        node_highway = _as_list(node.get("highway"))

        rows.append(
            {
                "osmid": osmid,
                "geometry": node.geometry,
                "n_legs": node.get("street_count"),
                "max_lanes": max(lane_values) if lane_values else float("nan"),
                "mean_lanes": (sum(lane_values) / len(lane_values)) if lane_values else float("nan"),
                "max_speed_mph": max_speed_mph,
                "speed_imputed": speed_imputed,
                "highway_class_max": max(highway_classes, key=lambda c: HIGHWAY_RANK.get(c, 0))
                if highway_classes
                else "unknown",
                "is_signalized": "traffic_signals" in node_highway,
                "has_lighting_tag": has_lighting,
                "has_sidewalk": has_sidewalk,
            }
        )

    print(
        f"Tag coverage: lit={lit_coverage}/{total_edges} edges, "
        f"sidewalk={sidewalk_coverage}/{total_edges} edges"
    )

    return pd.DataFrame(rows)


def add_marked_crossings(features: pd.DataFrame, crossings: gpd.GeoDataFrame) -> pd.DataFrame:
    """has_marked_crossing = a marked/signalized crossing within CROSSING_SEARCH_RADIUS_M."""
    crossings = crossings[crossings["crossing"].notna()].copy()
    crossings["is_marked"] = crossings["crossing"].apply(
        lambda tag: any(v in MARKED_CROSSING_VALUES for v in str(tag).split(";"))
    )
    marked = to_analysis_crs(crossings[crossings["is_marked"]])

    points = gpd.GeoDataFrame(
        features[["osmid", "geometry"]], geometry="geometry", crs="EPSG:4326"
    )
    points = to_analysis_crs(points)
    matched = snap_to_nearest(points, marked[["geometry"]], max_distance_m=CROSSING_SEARCH_RADIUS_M)
    has_crossing = matched.set_index("osmid")["dist_m"].notna()

    features = features.set_index("osmid")
    features["has_marked_crossing"] = has_crossing.reindex(features.index, fill_value=False)
    return features.reset_index()


def add_transit_distance(features: pd.DataFrame, bus_stops: gpd.GeoDataFrame) -> pd.DataFrame:
    """dist_to_transit_m = distance to the nearest bus stop, no cap."""
    points = gpd.GeoDataFrame(
        features[["osmid", "geometry"]], geometry="geometry", crs="EPSG:4326"
    )
    points = to_analysis_crs(points)
    stops = to_analysis_crs(bus_stops[["geometry"]])
    matched = snap_to_nearest(points, stops, max_distance_m=None)

    features = features.set_index("osmid")
    features["dist_to_transit_m"] = matched.set_index("osmid")["dist_m"].reindex(features.index)
    return features.reset_index()


def add_crash_counts(features: pd.DataFrame, crashes: gpd.GeoDataFrame) -> pd.DataFrame:
    """ped_crashes_10yr (<=50m) and ped_crashes_midblock_200m (50-200m)."""
    points = gpd.GeoDataFrame(
        features[["osmid", "geometry"]], geometry="geometry", crs="EPSG:4326"
    )
    points = to_analysis_crs(points)
    crashes_proj = to_analysis_crs(crashes)

    matched = snap_to_nearest(crashes_proj, points[["osmid", "geometry"]], max_distance_m=CRASH_MIDBLOCK_RADIUS_M)

    dropped = matched["osmid"].isna().sum()
    print(f"Crashes beyond {CRASH_MIDBLOCK_RADIUS_M}m of any intersection (dropped): {dropped}")

    at_intersection = matched[matched["dist_m"] <= CRASH_SNAP_RADIUS_M]
    midblock = matched[
        (matched["dist_m"] > CRASH_SNAP_RADIUS_M) & (matched["dist_m"] <= CRASH_MIDBLOCK_RADIUS_M)
    ]

    ped_10yr = at_intersection.groupby("osmid").size()
    ped_midblock = midblock.groupby("osmid").size()

    features = features.set_index("osmid")
    features["ped_crashes_10yr"] = ped_10yr.reindex(features.index, fill_value=0).astype(int)
    features["ped_crashes_midblock_200m"] = ped_midblock.reindex(features.index, fill_value=0).astype(int)
    return features.reset_index()


def build_intersections(force: bool = False) -> gpd.GeoDataFrame:
    if OUTPUT_PATH.exists() and not force:
        print(f"{OUTPUT_PATH} already exists, skipping (pass --force to rebuild).")
        return gpd.read_file(OUTPUT_PATH)

    graph = get_graph()
    intersections, edges = consolidate_and_filter(graph)
    crossings = get_crossings()
    bus_stops = get_bus_stops()
    crashes = get_pedestrian_crashes()

    features = compute_road_features(intersections, edges)
    features = add_marked_crossings(features, crossings)
    features = add_transit_distance(features, bus_stops)
    features = add_crash_counts(features, crashes)

    # Stable deterministic ordering before assigning IDs.
    features = features.sort_values("osmid").reset_index(drop=True)
    features["int_id"] = [f"nlv_{i}" for i in range(len(features))]

    columns = [
        "int_id",
        "osmid",
        "n_legs",
        "max_lanes",
        "mean_lanes",
        "max_speed_mph",
        "speed_imputed",
        "highway_class_max",
        "is_signalized",
        "has_marked_crossing",
        "has_lighting_tag",
        "has_sidewalk",
        "dist_to_transit_m",
        "ped_crashes_10yr",
        "ped_crashes_midblock_200m",
        "geometry",
    ]
    result = gpd.GeoDataFrame(features, geometry="geometry", crs="EPSG:4326")
    result = to_storage_crs(result)
    result = result.reindex(columns=[c for c in columns if c in result.columns])

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    result.to_file(OUTPUT_PATH, driver="GeoJSON")
    print(f"Wrote {len(result)} intersections to {OUTPUT_PATH}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rebuild even if output exists")
    args = parser.parse_args()
    build_intersections(force=args.force)
