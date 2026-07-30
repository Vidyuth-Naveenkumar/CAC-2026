"""
Phase 1, Step 1: download and cache everything we need from OpenStreetMap
for the study area — the drivable road network, the study area boundary
polygon, marked crossings, and bus stops.

`network_type="drive"` only returns the road graph; crossings and bus stops
are separate point features that don't live on drivable ways, so we fetch
them here too rather than splitting "OSM data" across two modules.

Every get_* function follows the same pattern: if the cached file exists,
load it; otherwise download it, cache it, and return it. Pass force=True
to re-download.
"""

import argparse

import geopandas as gpd
import networkx as nx
import osmnx as ox

from src.config import STUDY_AREA_PLACES, OSM_RAW_DIR, CONSOLIDATION_TOLERANCE_M, STORAGE_CRS

GRAPH_CACHE = OSM_RAW_DIR / "network.graphml"
BOUNDARY_CACHE = OSM_RAW_DIR / "boundary.geojson"
CROSSINGS_CACHE = OSM_RAW_DIR / "crossings.geojson"
BUS_STOPS_CACHE = OSM_RAW_DIR / "bus_stops.geojson"


def get_graph(place_names: list[str] = STUDY_AREA_PLACES, force: bool = False) -> nx.MultiDiGraph:
    """Drivable road network for the study area, cached as GraphML."""
    if GRAPH_CACHE.exists() and not force:
        return ox.load_graphml(GRAPH_CACHE)

    OSM_RAW_DIR.mkdir(parents=True, exist_ok=True)
    graph = ox.graph_from_place(place_names, network_type="drive")
    ox.save_graphml(graph, GRAPH_CACHE)
    return graph


def get_study_area_polygon(place_names: list[str] = STUDY_AREA_PLACES, force: bool = False) -> gpd.GeoDataFrame:
    """Study area boundary polygon, cached as GeoJSON.

    fetch_fars.py imports this too, so both modules clip to the exact same
    boundary instead of each geocoding the place name independently.
    """
    if BOUNDARY_CACHE.exists() and not force:
        return gpd.read_file(BOUNDARY_CACHE)

    OSM_RAW_DIR.mkdir(parents=True, exist_ok=True)
    boundary = ox.geocode_to_gdf(place_names)
    boundary.to_file(BOUNDARY_CACHE, driver="GeoJSON")
    return boundary


def consolidate_and_filter(
    graph: nx.MultiDiGraph, tolerance: float = CONSOLIDATION_TOLERANCE_M
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Merge nearby nodes into single intersections and keep real junctions.

    Divided roads produce 2+ raw OSM nodes for one real-world intersection,
    and plenty of nodes are just geometry vertices where a road bends.
    consolidate_intersections merges nodes within `tolerance` meters of each
    other (hence projecting first — tolerance is in meters, not degrees).
    Nodes with street_count < 3 are a bend, not a junction, and get dropped.

    Returns (intersections, edges) as GeoDataFrames in EPSG:4326. `edges` is
    the full incident-edge table (not filtered) — build_features.py uses it
    to aggregate road tags (lanes, maxspeed, ...) onto each intersection.
    """
    graph_proj = ox.project_graph(graph)
    consolidated = ox.consolidate_intersections(
        graph_proj, tolerance=tolerance, rebuild_graph=True
    )
    nodes, edges = ox.graph_to_gdfs(consolidated)

    print(f"consolidate_and_filter: node columns = {list(nodes.columns)}")
    if "osmid_original" not in nodes.columns:
        print("WARNING: 'osmid_original' missing — osmid traceability is broken.")

    intersections = nodes[nodes["street_count"] >= 3].copy()
    return intersections.to_crs(STORAGE_CRS), edges.to_crs(STORAGE_CRS)


def get_crossings(place_names: list[str] = STUDY_AREA_PLACES, force: bool = False) -> gpd.GeoDataFrame:
    """Marked/signalized pedestrian crossing nodes, cached as GeoJSON."""
    if CROSSINGS_CACHE.exists() and not force:
        return gpd.read_file(CROSSINGS_CACHE)

    OSM_RAW_DIR.mkdir(parents=True, exist_ok=True)
    crossings = ox.features_from_place(place_names, tags={"highway": "crossing"})
    crossings.to_file(CROSSINGS_CACHE, driver="GeoJSON")
    return crossings


def get_bus_stops(place_names: list[str] = STUDY_AREA_PLACES, force: bool = False) -> gpd.GeoDataFrame:
    """Bus stop nodes, cached as GeoJSON."""
    if BUS_STOPS_CACHE.exists() and not force:
        return gpd.read_file(BUS_STOPS_CACHE)

    OSM_RAW_DIR.mkdir(parents=True, exist_ok=True)
    bus_stops = ox.features_from_place(place_names, tags={"highway": "bus_stop"})
    bus_stops.to_file(BUS_STOPS_CACHE, driver="GeoJSON")
    return bus_stops


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()

    print("Fetching road network...")
    graph = get_graph(force=args.force)
    print(f"  {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    print("Fetching study area boundary...")
    boundary = get_study_area_polygon(force=args.force)
    print(f"  boundary rows: {len(boundary)}")

    print("Consolidating intersections...")
    intersections, edges = consolidate_and_filter(graph)
    print(f"  {len(intersections)} intersections (street_count >= 3), {len(edges)} edges")

    print("Fetching crossings...")
    crossings = get_crossings(force=args.force)
    print(f"  {len(crossings)} crossing nodes")

    print("Fetching bus stops...")
    bus_stops = get_bus_stops(force=args.force)
    print(f"  {len(bus_stops)} bus stop nodes")
