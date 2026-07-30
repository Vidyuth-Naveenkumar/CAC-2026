"""
Phase 1, Step 3: load NHTSA FARS pedestrian crash data for the study area.

FARS ships one ZIP per year containing many CSVs; we need two:
- accident.csv — one row per crash, has STATE/COUNTY/LATITUDE/LONGITUD
- pbtype.csv    — one row per pedestrian/cyclist involved, joined on ST_CASE

We filter to Nevada + Clark County, keep only pedestrian-involved crashes,
drop FARS's sentinel "unknown location" coordinates, and clip to the study
area boundary. Every filter step is logged; per phase1_spec.md, a step that
drops more than half the remaining rows stops the script rather than
silently continuing, since that usually means a filter is wrong, not that
the data is actually that sparse.
"""

import argparse
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from src.config import FARS_RAW_DIR, FARS_YEARS
from src.data.fetch_osm import get_study_area_polygon

NEVADA_FIPS = 32
CLARK_COUNTY_FIPS = 3

# FARS encodes "location unknown" with these placeholder values instead of
# leaving the cell blank. Left in, they'd silently poison the spatial join
# with points in the middle of the ocean.
SENTINEL_COORDS = {77.7777, 88.8888, 99.9999}

FARS_URL_TEMPLATE = (
    "https://static.nhtsa.gov/nhtsa/downloads/FARS/{year}/National/"
    "FARS{year}NationalCSV.zip"
)


def download_fars_year(year: int, dest_dir: Path = FARS_RAW_DIR, force: bool = False) -> Path:
    """Download+extract one year's accident.csv and pbtype.csv into dest_dir/{year}/."""
    year_dir = dest_dir / str(year)
    accident_path = year_dir / "accident.csv"
    pbtype_path = year_dir / "pbtype.csv"

    if accident_path.exists() and pbtype_path.exists() and not force:
        return year_dir

    year_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"FARS{year}NationalCSV.zip"

    if not zip_path.exists() or force:
        url = FARS_URL_TEMPLATE.format(year=year)
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        zip_path.write_bytes(response.content)

    with zipfile.ZipFile(zip_path) as archive:
        # Case varies by year ("accident.csv" vs "ACCIDENT.CSV"), so match
        # case-insensitively and normalize the extracted filename.
        names = {name.lower(): name for name in archive.namelist()}
        for wanted, dest_name in [("accident.csv", "accident.csv"), ("pbtype.csv", "pbtype.csv")]:
            matches = [orig for lower, orig in names.items() if wanted in lower]
            member = next(iter(matches))
            with archive.open(member) as src, open(year_dir / dest_name, "wb") as dst:
                dst.write(src.read())

    return year_dir


def read_fars_csv(path: Path) -> pd.DataFrame:
    """Read a FARS CSV, handling inconsistent per-year encodings.

    Some years' files are UTF-8 with a BOM (utf-8-sig strips it — found in
    Phase 0, latin-1 mangled it onto the STATE column name). Other years
    contain Windows-1252 characters (e.g. a smart quote in a free-text
    field, found in 2016's pbtype.csv) that aren't valid UTF-8 at all.
    """
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", low_memory=False)


def load_pedestrian_crashes(year_dir: Path) -> pd.DataFrame:
    """Load one year's crashes, filtered to Nevada/Clark County pedestrian crashes."""
    accidents = read_fars_csv(year_dir / "accident.csv")
    pbtype = read_fars_csv(year_dir / "pbtype.csv")

    local = accidents[
        (accidents["STATE"] == NEVADA_FIPS) & (accidents["COUNTY"] == CLARK_COUNTY_FIPS)
    ]

    # Match on the PBPTYPENAME string, not the numeric PBPTYPE code — FARS
    # has changed the numeric coding across editions (per phase1_spec.md),
    # but the descriptive name is stable.
    pedestrians = pbtype[pbtype["PBPTYPENAME"] == "Pedestrian"]

    return local.merge(pedestrians[["ST_CASE"]].drop_duplicates(), on="ST_CASE", how="inner")


def drop_sentinel_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where LATITUDE or LONGITUD is one of FARS's sentinel values."""
    bad_lat = df["LATITUDE"].isin(SENTINEL_COORDS)
    bad_lon = df["LONGITUD"].isin(SENTINEL_COORDS)
    return df[~(bad_lat | bad_lon)]


def log_filter_step(name: str, before: int, after: int) -> None:
    """Print before/after counts; stop the pipeline if a step drops over half the rows."""
    dropped = before - after
    pct = (dropped / before * 100) if before else 0
    print(f"  [{name}] {before} -> {after} ({dropped} dropped, {pct:.1f}%)")
    if before > 0 and dropped / before > 0.5:
        raise RuntimeError(
            f"Filter step '{name}' dropped more than half the rows "
            f"({pct:.1f}%) — stopping per phase1_spec.md instead of "
            f"silently continuing. Check the filter logic before re-running."
        )


def get_pedestrian_crashes(years=FARS_YEARS, force: bool = False) -> gpd.GeoDataFrame:
    """Build the full multi-year pedestrian crash dataset, clipped to the study area."""
    yearly = []
    for year in years:
        print(f"  {year}...")
        year_dir = download_fars_year(year, force=force)
        yearly.append(load_pedestrian_crashes(year_dir))
    combined = pd.concat(yearly, ignore_index=True)
    print(f"Combined {len(years)} years: {len(combined)} pedestrian crash rows in Clark County")

    before = len(combined)
    combined = drop_sentinel_coords(combined)
    log_filter_step("drop_sentinel_coords", before, len(combined))

    crashes = gpd.GeoDataFrame(
        combined,
        geometry=gpd.points_from_xy(combined["LONGITUD"], combined["LATITUDE"]),
        crs="EPSG:4326",
    )

    boundary = get_study_area_polygon()
    before = len(crashes)
    clipped = gpd.clip(crashes, boundary)
    log_filter_step("clip_to_study_area", before, len(clipped))

    return clipped


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()

    print(f"Fetching FARS pedestrian crashes for {FARS_YEARS.start}-{FARS_YEARS.stop - 1}...")
    crashes = get_pedestrian_crashes(force=args.force)
    print(f"\nFinal pedestrian crash count in study area: {len(crashes)}")
