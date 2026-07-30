# Pedestrian Infrastructure Gap Auditor

Congressional App Challenge submission. Deadline: October 24, 2026.
Author is a 10th-grade student learning Python — explain reasoning,
don't just produce code.

## Goal
Rank street intersections by pedestrian safety priority, fusing:
1. Crash risk model (LightGBM on FARS + OSM road features)
2. Curb ramp presence classifier (ResNet-18 on Mapillary photos)

Study area: Las Vegas Valley, NV — Las Vegas, North Las Vegas, Henderson,
Enterprise, Paradise, Spring Valley, Winchester, and Whitney. ~39,000
intersections. Expanded from North Las Vegas alone during Phase 1 after a
decision checkpoint found too few pedestrian crashes in that smaller area
— see Phase history.

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
**Phase 1 complete.** Phase 2 is not yet scoped — add a `docs/phase2_spec.md`
and update this section before starting Phase 2 work.

## Known issues carried into Phase 2
- `lit` and `sidewalk` OSM tags are completely absent from this extract
  (0/105,582 edges, not just sparse) — `has_lighting_tag` and `has_sidewalk`
  carry zero signal and are candidates to drop before model training.

## Phase history
- Phase 0 (complete): environment setup, data source verification.
- Phase 1 (complete): data pipeline. Full spec: `docs/phase1_spec.md`.
  Study area expanded from North Las Vegas alone to the Las Vegas Valley +
  Henderson after a decision checkpoint found only 48 pedestrian crashes in
  North Las Vegas alone (below the spec's ≥80 gate) and NV DOT's supplemental
  injury-crash service turned out to be offline. Output:
  `data/interim/intersections.geojson` — 39,183 intersections, 295 positive
  (crash-bearing). All validation checks pass (see `LOG.md` for the two
  validation bugs/assumptions found and fixed along the way).