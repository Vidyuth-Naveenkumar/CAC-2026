"""
Small CRS and spatial-join helpers shared by the data pipeline.

Every other module should import these instead of calling .to_crs() or
gpd.sjoin_nearest() directly, so the CRS choice lives in exactly one place.
"""

import geopandas as gpd

from src.config import ANALYSIS_CRS, STORAGE_CRS


def to_analysis_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject to ANALYSIS_CRS (EPSG:26911, meters) for distance/buffer math.

    NEVER compute distances or buffers in STORAGE_CRS (EPSG:4326) — degrees
    are not meters.
    """
    return gdf.to_crs(ANALYSIS_CRS)


def to_storage_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject to STORAGE_CRS (EPSG:4326) for saving/output."""
    return gdf.to_crs(STORAGE_CRS)


def snap_to_nearest(
    points: gpd.GeoDataFrame,
    targets: gpd.GeoDataFrame,
    max_distance_m: float | None = None,
    distance_col: str = "dist_m",
) -> gpd.GeoDataFrame:
    """Nearest-neighbor join of points onto targets.

    Both inputs must already be in ANALYSIS_CRS (meters) — this function
    doesn't reproject, since callers often need to control CRS timing
    around other steps (e.g. clipping) in the same pipeline.

    Returns one row per point in `points`, with the matched target's columns
    attached and a `distance_col` column. If max_distance_m is set, points
    with no target inside that radius get NaN in the target columns.
    """
    return gpd.sjoin_nearest(
        points, targets, how="left", max_distance=max_distance_m, distance_col=distance_col
    )
