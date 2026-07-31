"""
Small CRS and spatial-join helpers shared by the data pipeline.

Every other module should import these instead of calling .to_crs() or
gpd.sjoin_nearest() directly, so the CRS choice lives in exactly one place.
"""

import geopandas as gpd
import numpy as np
import pandas as pd

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


def spatial_kfold(
    gdf: gpd.GeoDataFrame,
    n_splits: int = 5,
    cell_size_m: float = 2000,
    seed: int = 42,
) -> pd.Series:
    """Assign each row to one of n_splits folds using whole grid cells.

    Adjacent intersections share features and crash history, so a random
    row split leaks information between train and validation. Instead,
    bucket points into cell_size_m x cell_size_m grid cells and assign
    whole cells to folds — never splitting a cell across folds.

    gdf must already be in ANALYSIS_CRS (meters), consistent with
    snap_to_nearest's convention. Returns fold labels (0..n_splits-1) as a
    pandas Series aligned to gdf's index.

    Label-agnostic on purpose (no crash-count awareness here) so it stays
    reusable for other spatially-blocked CV needs later — the caller
    decides what to do if a fold ends up with too few positives.
    """
    cell_x = (gdf.geometry.x // cell_size_m).astype(int)
    cell_y = (gdf.geometry.y // cell_size_m).astype(int)
    cell_id = pd.Series(list(zip(cell_x, cell_y)), index=gdf.index)

    cell_counts = cell_id.value_counts()
    cells = cell_counts.index.to_list()

    # Shuffle before the greedy pass so cells aren't processed in
    # coordinate order, which would bias which fold gets which region.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cells))
    cells = [cells[i] for i in order]

    # Greedy bin-packing: assign the biggest cells first, each to
    # whichever fold currently has the fewest rows, for balanced fold
    # sizes despite uneven intersection density across cells.
    fold_sizes = [0] * n_splits
    cell_to_fold = {}
    for cell in sorted(cells, key=lambda c: cell_counts[c], reverse=True):
        fold = min(range(n_splits), key=lambda f: fold_sizes[f])
        cell_to_fold[cell] = fold
        fold_sizes[fold] += cell_counts[cell]

    return cell_id.map(cell_to_fold)
