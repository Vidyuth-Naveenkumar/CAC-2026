# Pedestrian Infrastructure Gap Auditor

Congressional App Challenge submission. Deadline: October 24, 2026.
Author is a 10th-grade student learning Python — explain reasoning,
don't just produce code.

## Goal
Rank street intersections by pedestrian safety priority, fusing:
1. Crash risk model (LightGBM on FARS + OSM road features)
2. Curb ramp presence classifier (ResNet-18 on Mapillary photos)

Study area: North Las Vegas, NV. ~3,000 intersections.

## Environment
- macOS, Apple Silicon M4, 48GB. Use `mps` device for PyTorch, never `cuda`.
- Python 3.11 in `.venv`, managed with `uv`. Install via `uv pip install`.
- Never `pip install` directly.

## Geospatial conventions
- Storage and output CRS: EPSG:4326 (WGS84 lat/lon).
- Analysis CRS: EPSG:26911 (NAD83 / UTM zone 11N) — Las Vegas is in UTM 11N.
- NEVER compute distances or buffers in EPSG:4326. Degrees are not meters.
  Reproject to 26911 first, do the math, reproject back.
- Always state the CRS in docstrings when a function takes or returns geometry.

## Rules
- All paths come from `src/config.py`. Never hardcode a path.
- Never write to `data/raw/` — it is immutable source data.
- Every script in `src/data/` must be re-runnable without side effects.
  Check if the output exists and skip unless `--force` is passed.
- Notebooks in `notebooks/` are for exploration only. Production code
  lives in `src/`.
- Prefer clear code over clever code. Comment the non-obvious parts.
- Before writing a new module, propose the function signatures first.
- Large data files are gitignored, so you cannot read them. Small samples
  live in `data/sample/` — ask me to create one if you need to see a schema.

## Current phase
**Phase 1: Data pipeline** (Aug 3 – Aug 16)
Full spec: `docs/phase1_spec.md` — read it before working on Phase 1 tasks.

Gate to exit Phase 1: `data/interim/intersections.geojson` exists,
passes all validation checks in the spec, and has ≥80 positive
(crash-bearing) intersections.

## Phase history
- Phase 0 (complete): environment setup, data source verification.