"""
Phase 3, Step 0: Mapillary coverage probe (see docs/phase3_spec.md).

This is the highest-risk unknown in the project — the curb ramp
classifier only works if enough intersections actually have street-level
photos near them. Samples 200 intersections stratified across risk
deciles, queries Mapillary for images within 25m of each, and reports
coverage broken out by highway_class_max and risk decile so the decision
gate in phase3_spec.md can be checked before the downloader is built.
"""

import argparse
import sys
import time

import geopandas as gpd
import pandas as pd
import requests

from src.config import (
    INTERIM_DIR,
    OUTPUTS_DIR,
    MAPILLARY_API_URL,
    MAPILLARY_TOKEN,
    MAPILLARY_PROBE_SAMPLE_N,
    MAPILLARY_PROBE_RADIUS_M,
    MAPILLARY_PROBE_IMAGE_LIMIT,
)
from src.utils.geo import to_analysis_crs, to_storage_crs

INTERSECTIONS_PATH = INTERIM_DIR / "intersections.geojson"
RISK_SCORES_PATH = OUTPUTS_DIR / "risk_scores.parquet"
PROBE_OUTPUT_PATH = OUTPUTS_DIR / "mapillary_coverage_probe.csv"

# Decision gate thresholds from phase3_spec.md Step 0.
COVERAGE_PROCEED_THRESHOLD = 0.70
COVERAGE_RESTRICT_THRESHOLD = 0.40

MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 1.0
REQUEST_SLEEP_SECONDS = 0.2


def load_scored_intersections() -> gpd.GeoDataFrame:
    """Join risk_scores.parquet onto intersections.geojson.

    Returns a GeoDataFrame in STORAGE_CRS (EPSG:4326) with int_id,
    geometry, highway_class_max, and risk_percentile.
    """
    intersections = gpd.read_file(INTERSECTIONS_PATH)[["int_id", "highway_class_max", "geometry"]]
    risk = pd.read_parquet(RISK_SCORES_PATH)[["int_id", "risk_percentile"]]
    return intersections.merge(risk, on="int_id", how="inner")


def stratified_risk_sample(
    gdf: gpd.GeoDataFrame,
    n: int = MAPILLARY_PROBE_SAMPLE_N,
    n_deciles: int = 10,
    seed: int = 42,
) -> gpd.GeoDataFrame:
    """Sample n rows evenly split across n_deciles risk_percentile bins.

    Adds a risk_decile column (1 = lowest risk, n_deciles = highest).
    """
    bins = pd.cut(
        gdf["risk_percentile"],
        bins=[i * 100 / n_deciles for i in range(n_deciles + 1)],
        labels=range(1, n_deciles + 1),
        include_lowest=True,
    )
    gdf = gdf.assign(risk_decile=bins)

    per_decile = n // n_deciles
    rng_seed = seed
    sampled = []
    for decile in range(1, n_deciles + 1):
        group = gdf[gdf["risk_decile"] == decile]
        take = min(per_decile, len(group))
        sampled.append(group.sample(n=take, random_state=rng_seed))
    return gpd.GeoDataFrame(pd.concat(sampled, ignore_index=True), crs=gdf.crs)


def bbox_around_point(lon: float, lat: float, radius_m: float) -> tuple[float, float, float, float]:
    """Bounding box around (lon, lat) at radius_m, as (west, south, east, north).

    Buffers in ANALYSIS_CRS (meters) per CLAUDE.md — never buffer directly
    in STORAGE_CRS degrees — then reprojects the bounds back to STORAGE_CRS
    for Mapillary's bbox query param.
    """
    point = gpd.GeoDataFrame(geometry=gpd.points_from_xy([lon], [lat]), crs="EPSG:4326")
    buffered = to_analysis_crs(point).buffer(radius_m)
    buffered_gdf = gpd.GeoDataFrame(geometry=buffered, crs=buffered.crs)
    west, south, east, north = to_storage_crs(buffered_gdf).total_bounds
    return west, south, east, north


def query_images_near(
    lon: float,
    lat: float,
    radius_m: float = MAPILLARY_PROBE_RADIUS_M,
    limit: int = MAPILLARY_PROBE_IMAGE_LIMIT,
) -> list[dict]:
    """Query the Mapillary Graph API v4 for images within radius_m of a point.

    Retries with exponential backoff on HTTP 429. Returns [] (rather than
    raising) on any other failure — this is a measurement script, and one
    bad site shouldn't kill a 200-site probe.
    """
    west, south, east, north = bbox_around_point(lon, lat, radius_m)

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                f"{MAPILLARY_API_URL}/images",
                params={
                    "access_token": MAPILLARY_TOKEN,
                    "fields": "id,captured_at",
                    "bbox": f"{west},{south},{east},{north}",
                    "limit": limit,
                },
                timeout=30,
            )
        except requests.RequestException as exc:
            print(f"  request error at ({lon:.5f}, {lat:.5f}): {exc}")
            return []

        if response.status_code == 429:
            sleep_for = BACKOFF_BASE_SECONDS * (2 ** attempt)
            time.sleep(sleep_for)
            continue

        if not response.ok:
            print(f"  HTTP {response.status_code} at ({lon:.5f}, {lat:.5f})")
            return []

        return response.json().get("data", [])

    print(f"  gave up after {MAX_RETRIES} retries at ({lon:.5f}, {lat:.5f})")
    return []


def run_probe(sample: gpd.GeoDataFrame) -> pd.DataFrame:
    """Query Mapillary for every site in sample, one row of results each."""
    rows = []
    total = len(sample)
    for i, site in enumerate(sample.itertuples(), start=1):
        images = query_images_near(site.geometry.x, site.geometry.y)
        years = [img["captured_at"] // 1000 for img in images if img.get("captured_at")]
        capture_years = [pd.Timestamp(y, unit="s").year for y in years]
        rows.append(
            {
                "int_id": site.int_id,
                "highway_class_max": site.highway_class_max,
                "risk_decile": site.risk_decile,
                "n_images": len(images),
                "median_capture_year": pd.Series(capture_years).median() if capture_years else None,
            }
        )
        if i % 20 == 0 or i == total:
            print(f"  probed {i}/{total} sites")
        time.sleep(REQUEST_SLEEP_SECONDS)
    return pd.DataFrame(rows)


def summarize(probe_df: pd.DataFrame) -> dict:
    """Coverage fractions, median capture year, and breakdowns for the
    decision gate in phase3_spec.md Step 0."""
    n = len(probe_df)
    frac_at_least_1 = (probe_df["n_images"] >= 1).mean()
    frac_at_least_3 = (probe_df["n_images"] >= 3).mean()
    median_year = probe_df["median_capture_year"].median()

    by_highway_class = probe_df.groupby("highway_class_max")["n_images"].apply(lambda s: (s >= 1).mean())
    by_risk_decile = probe_df.groupby("risk_decile")["n_images"].apply(lambda s: (s >= 1).mean())

    if frac_at_least_1 >= COVERAGE_PROCEED_THRESHOLD:
        gate = "PROCEED — coverage >= 70%, proceed as specced"
    elif frac_at_least_1 >= COVERAGE_RESTRICT_THRESHOLD:
        gate = "PROCEED WITH RESTRICTION — 40-70% coverage: restrict study set to covered intersections, document coverage bias"
    else:
        gate = "STOP — coverage < 40%. Vision phase not viable as specced; fall back to risk model alone."

    return {
        "n_sampled": n,
        "frac_at_least_1_image": frac_at_least_1,
        "frac_at_least_3_images": frac_at_least_3,
        "median_capture_year": median_year,
        "coverage_by_highway_class": by_highway_class.to_dict(),
        "coverage_by_risk_decile": by_risk_decile.to_dict(),
        "decision_gate": gate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-run even if outputs/mapillary_coverage_probe.csv exists")
    args = parser.parse_args()

    if PROBE_OUTPUT_PATH.exists() and not args.force:
        print(f"{PROBE_OUTPUT_PATH} already exists — pass --force to re-run.")
        probe_df = pd.read_csv(PROBE_OUTPUT_PATH)
    else:
        if not MAPILLARY_TOKEN:
            print("MAPILLARY_TOKEN is not set in .env — see src/data/verify_mapillary.py.")
            return 1

        gdf = load_scored_intersections()
        sample = stratified_risk_sample(gdf)
        print(f"Sampled {len(sample)} intersections across risk deciles. Querying Mapillary...")
        probe_df = run_probe(sample)
        probe_df.to_csv(PROBE_OUTPUT_PATH, index=False)
        print(f"Wrote {PROBE_OUTPUT_PATH}")

    summary = summarize(probe_df)

    print("\n--- Mapillary coverage probe summary ---")
    print(f"n sampled: {summary['n_sampled']}")
    print(f">=1 image: {summary['frac_at_least_1_image'] * 100:.1f}%")
    print(f">=3 images: {summary['frac_at_least_3_images'] * 100:.1f}%")
    print(f"median capture year: {summary['median_capture_year']}")
    print("coverage by highway_class_max:")
    for cls, frac in sorted(summary["coverage_by_highway_class"].items(), key=lambda kv: -kv[1]):
        print(f"  {cls}: {frac * 100:.1f}%")
    print("coverage by risk decile (1=lowest risk, 10=highest):")
    for decile, frac in sorted(summary["coverage_by_risk_decile"].items()):
        print(f"  decile {decile}: {frac * 100:.1f}%")
    print(f"\nDECISION GATE: {summary['decision_gate']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
