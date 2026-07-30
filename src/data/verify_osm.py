"""
Phase 0 check: can we pull North Las Vegas road data from OpenStreetMap
and get roughly the ~3,000 intersections the project expects?

This is verification only — it doesn't save anything to data/processed.
The real OSM feature-extraction pipeline comes in a later phase.
"""

import networkx as nx
import geopandas as gpd
import osmnx as ox

from src.config import STUDY_AREA_NAME, OSM_RAW_DIR

# How far off from ~3,000 we tolerate before treating the count as a red
# flag (usually means the place name geocoded to the wrong boundary).
EXPECTED_INTERSECTIONS = 3000
TOLERANCE = 0.5  # +/- 50%


def get_study_area_graph(place_name: str = STUDY_AREA_NAME) -> nx.MultiDiGraph:
    """Download the drivable street network for the study area from OSM."""
    return ox.graph_from_place(place_name, network_type="drive")


def extract_intersections(graph: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    """Return nodes where 3+ streets meet — i.e. actual intersections.

    osmnx annotates every node with `street_count` when the graph is built,
    so we just filter on that instead of computing degree ourselves.
    """
    nodes, _ = ox.graph_to_gdfs(graph)
    return nodes[nodes["street_count"] >= 3]


def verify_osm(intersections: gpd.GeoDataFrame) -> dict:
    """Sanity-check the intersection count against the ~3,000 estimate."""
    count = len(intersections)
    low = EXPECTED_INTERSECTIONS * (1 - TOLERANCE)
    high = EXPECTED_INTERSECTIONS * (1 + TOLERANCE)
    in_range = low <= count <= high
    return {
        "source": "osm",
        "passed": count > 0,
        "intersection_count": count,
        "in_expected_range": in_range,
        "expected_range": (low, high),
    }


if __name__ == "__main__":
    OSM_RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading street network for: {STUDY_AREA_NAME!r}")
    graph = get_study_area_graph()

    intersections = extract_intersections(graph)
    result = verify_osm(intersections)

    print(result)
    if not result["in_expected_range"]:
        print(
            "WARNING: intersection count is far from the ~3,000 estimate. "
            "Double-check STUDY_AREA_NAME actually geocodes to North Las Vegas "
            "and not, e.g., all of Las Vegas or a single street."
        )
