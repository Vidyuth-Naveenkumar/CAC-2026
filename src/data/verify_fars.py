"""
Phase 0 check: can we get NHTSA FARS (Fatality Analysis Reporting System)
crash data for Nevada, with usable coordinates?

FARS publishes one ZIP per year at a predictable URL containing several
CSVs (accident, person, vehicle, ...). We only need the accident-level
file, filtered to Nevada (STATE == 32, Nevada's FIPS code).

This is verification only — no spatial filtering to North Las Vegas yet,
and nothing gets written outside data/raw/fars (which becomes immutable
once downloaded, per CLAUDE.md).
"""

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.config import FARS_RAW_DIR

NEVADA_FIPS = 32

FARS_URL_TEMPLATE = (
    "https://static.nhtsa.gov/nhtsa/downloads/FARS/{year}/National/"
    "FARS{year}NationalCSV.zip"
)

REQUIRED_COLUMNS = ["LATITUDE", "LONGITUD"]


def download_fars_year(year: int, dest_dir: Path = FARS_RAW_DIR) -> Path:
    """Download and unzip one year of national FARS data into dest_dir.

    Returns the path to the accident-level CSV for that year.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"FARS{year}NationalCSV.zip"

    if not zip_path.exists():
        url = FARS_URL_TEMPLATE.format(year=year)
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        zip_path.write_bytes(response.content)

    with zipfile.ZipFile(zip_path) as archive:
        # The accident-level file's exact case/name varies by year
        # (e.g. "accident.csv" vs "ACCIDENT.CSV"), so match case-insensitively.
        accident_name = next(
            name for name in archive.namelist() if "accident" in name.lower()
        )
        archive.extract(accident_name, dest_dir)
        return dest_dir / accident_name


def load_fars_accidents(csv_path: Path) -> pd.DataFrame:
    """Load the accident CSV and filter to Nevada crashes."""
    # utf-8-sig strips the UTF-8 byte-order-mark FARS puts at the start of
    # the file; using latin-1 (or plain utf-8) leaves it mangled onto STATE.
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    return df[df["STATE"] == NEVADA_FIPS]


def verify_fars(df: pd.DataFrame) -> dict:
    """Sanity-check that Nevada rows exist and have usable coordinates."""
    has_columns = all(col in df.columns for col in REQUIRED_COLUMNS)
    valid_coords = 0
    if has_columns:
        valid_coords = df[REQUIRED_COLUMNS].notna().all(axis=1).sum()

    return {
        "source": "fars",
        "passed": len(df) > 0 and has_columns and valid_coords > 0,
        "nevada_row_count": len(df),
        "has_required_columns": has_columns,
        "rows_with_valid_coords": int(valid_coords),
    }


if __name__ == "__main__":
    year = 2022  # most recent year with a finalized FARS file at time of writing
    print(f"Downloading FARS {year} national data...")
    csv_path = download_fars_year(year)

    df = load_fars_accidents(csv_path)
    result = verify_fars(df)

    print(result)
