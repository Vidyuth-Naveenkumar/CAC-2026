# Pedestrian Infrastructure Gap Auditor

Congressional App Challenge submission. Ranks street intersections in
North Las Vegas, NV by pedestrian safety priority, by fusing:

1. A crash risk model (LightGBM trained on FARS crash data + OSM road features)
2. A curb ramp presence classifier (ResNet-18 on Mapillary street-level photos)

Study area: North Las Vegas, NV (~3,000 intersections).

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
- `src/data/` — data acquisition and verification scripts.
- `src/models/` — model training/inference code (crash risk, curb ramp).
- `src/utils/` — shared helpers.
- `notebooks/` — exploration only; production code lives in `src/`.
- `data/raw/` — immutable source data once downloaded. Never modified in place.
- `data/interim/`, `data/processed/` — derived/cleaned data.
- `data/images/` — Mapillary photos, split into `labeled/{ramp,no_ramp}` and `unlabeled/`.
- `models/` — trained model artifacts.
- `outputs/` — final rankings, reports, figures.

## Current phase

**Phase 0: environment setup and data source verification.** Confirming
FARS, OSM, and Mapillary are all reachable before building any pipeline
or model code. See `LOG.md` for progress notes.

### Verification scripts

```bash
uv run python -m src.data.verify_osm         # OSM road network + intersection count
uv run python -m src.data.verify_fars        # NHTSA FARS Nevada crash data
uv run python -m src.data.verify_mapillary   # requires MAPILLARY_TOKEN in .env
```
