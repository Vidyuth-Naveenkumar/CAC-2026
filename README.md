# SightLine

Congressional App Challenge submission. Ranks street intersections in the
Las Vegas Valley, NV by pedestrian safety priority, by fusing:

1. A crash risk model (LightGBM trained on FARS crash data + OSM road features)
2. A curb ramp presence classifier (ResNet-18 on Mapillary street-level photos)

Study area: Las Vegas, North Las Vegas, Henderson, Enterprise, Paradise,
Spring Valley, Winchester, and Whitney, NV (~39,000 intersections).
Expanded from North Las Vegas alone during Phase 1 — see `LOG.md`.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install -r requirements.txt
```

Never `pip install` directly — always `uv pip install` so `.venv` and
`requirements.txt` stay in sync.

### Mapillary API token

Curb ramp photos come from Mapillary, which requires a free access token:

1. Create an account and register an app at
   https://www.mapillary.com/dashboard/developers
2. Copy the app's client token into `.env`:
   ```
   MAPILLARY_TOKEN=your_token_here
   ```

FARS (crash data) and OSM (road network) don't require any signup.

## Project layout

- `src/config.py` — every filesystem path and project constant lives here.
  Nothing else should hardcode a path.
- `src/data/` — data acquisition, pipeline, and verification scripts.
- `src/models/` — model training/inference code (crash risk, curb ramp).
- `src/utils/geo.py` — CRS reprojection and nearest-neighbor snapping helpers.
- `notebooks/` — exploration only; production code lives in `src/`.
- `docs/` — phase specs (e.g. `phase1_spec.md`).
- `data/raw/` — immutable source data once downloaded. Never modified in place.
- `data/interim/intersections.geojson` — Phase 1 output: one row per
  intersection with road features and pedestrian crash counts.
- `data/interim/`, `data/processed/` — derived/cleaned data.
- `data/images/` — Mapillary photos, split into `labeled/{ramp,no_ramp}` and `unlabeled/`.
- `models/` — trained model artifacts.
- `outputs/` — final rankings, reports, figures.

## Current phase

**Phase 1 complete: data pipeline.** `data/interim/intersections.geojson`
exists, passes all validation checks, and has 295 positive (crash-bearing)
intersections against a ≥80 gate. Phase 2 (crash risk model) is not yet
scoped. See `LOG.md` for the full history, including a study-area expansion
partway through Phase 1.

### Phase 1 pipeline scripts

```bash
uv run python -m src.data.fetch_osm          # road network, boundary, crossings, bus stops
uv run python -m src.data.fetch_fars         # NHTSA FARS pedestrian crashes, 2015-2023
uv run python -m src.data.build_features     # joins everything -> data/interim/intersections.geojson
uv run python -m src.data.validate_phase1    # Phase 1 exit-gate checks
```

Pass `--force` to any `fetch_*`/`build_features` script to rebuild instead
of using the cached/existing output.

### Phase 0 verification scripts

```bash
uv run python -m src.data.verify_osm         # OSM road network + intersection count
uv run python -m src.data.verify_fars        # NHTSA FARS Nevada crash data
uv run python -m src.data.verify_mapillary   # requires MAPILLARY_TOKEN in .env
```
Link to Deployment: https://cac-2026-dtheehpmaixjdafwrvdhwk.streamlit.app/
