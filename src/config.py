"""
Central place for every filesystem path and project-wide constant.
Nothing outside this file should hardcode a path — see CLAUDE.md.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# --- Project root ---
# config.py lives in src/, so the repo root is one level up.
ROOT_DIR = Path(__file__).resolve().parent.parent

# --- Data directories ---
DATA_DIR = ROOT_DIR / "data"

RAW_DIR = DATA_DIR / "raw"          # immutable source data — never write here after download
FARS_RAW_DIR = RAW_DIR / "fars"
OSM_RAW_DIR = RAW_DIR / "osm"

INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

IMAGES_DIR = DATA_DIR / "images"
LABELED_RAMP_DIR = IMAGES_DIR / "labeled" / "ramp"
LABELED_NO_RAMP_DIR = IMAGES_DIR / "labeled" / "no_ramp"
UNLABELED_DIR = IMAGES_DIR / "unlabeled"

SAMPLE_DIR = DATA_DIR / "sample"

# --- Other top-level directories ---
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# --- Study area ---
# The full urbanized Las Vegas Valley: incorporated cities + unincorporated
# Clark County CDPs. North-Las-Vegas-only gave just 48 pedestrian crashes
# over 2015-2023 (below the Phase 1 spec's >=80 threshold), so the study
# area was expanded per the spec's own suggested fallback. Passed as a list
# to osmnx (ox.graph_from_place / ox.geocode_to_gdf both accept list[str]
# and union the results into one boundary).
STUDY_AREA_PLACES = [
    "Las Vegas, Nevada, USA",
    "North Las Vegas, Nevada, USA",
    "Henderson, Nevada, USA",
    "Enterprise, Nevada, USA",
    "Paradise, Nevada, USA",
    "Spring Valley, Nevada, USA",
    "Winchester, Nevada, USA",
    "Whitney, Nevada, USA",
    # Sunrise Manor excluded: OSM/Nominatim has no boundary polygon for it,
    # only a point node (verified directly against the Nominatim API) — it's
    # surrounded by Las Vegas/Paradise/Winchester, so coverage loss is small.
]

# --- Mapillary API ---
# Get a token from https://www.mapillary.com/dashboard/developers and put it
# in .env as MAPILLARY_TOKEN=... — it is not committed to git.
MAPILLARY_TOKEN = os.getenv("MAPILLARY_TOKEN")
MAPILLARY_API_URL = "https://graph.mapillary.com"

# --- Coordinate reference systems ---
# Storage/output CRS: lat/lon, used for saving GeoJSON and display.
STORAGE_CRS = "EPSG:4326"
# Analysis CRS: NAD83 UTM zone 11N (meters) — Las Vegas is in UTM 11N.
# NEVER compute distances or buffers in STORAGE_CRS; degrees are not meters.
ANALYSIS_CRS = "EPSG:26911"

# --- Phase 1: data pipeline parameters ---
FARS_YEARS = range(2015, 2024)  # 2015-2023 inclusive, per phase1_spec.md
CONSOLIDATION_TOLERANCE_M = 15  # osmnx consolidate_intersections tolerance
CRASH_SNAP_RADIUS_M = 50        # crash counts as "at" an intersection within this
CRASH_MIDBLOCK_RADIUS_M = 200   # crash counts as "near" an intersection within this
CROSSING_SEARCH_RADIUS_M = 30   # search radius for a marked crossing near a node


def ensure_dirs() -> None:
    """Create every writable data/output directory if it doesn't exist yet.

    Deliberately excludes RAW_DIR and its children — raw data arrives via an
    explicit download step, not by silently creating an empty folder for it.
    """
    writable_dirs = [
        INTERIM_DIR,
        PROCESSED_DIR,
        LABELED_RAMP_DIR,
        LABELED_NO_RAMP_DIR,
        UNLABELED_DIR,
        SAMPLE_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
    ]
    for directory in writable_dirs:
        directory.mkdir(parents=True, exist_ok=True)
