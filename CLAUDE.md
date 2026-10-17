# Pedestrian Infrastructure Gap Auditor

Congressional App Challenge submission. **Deadline: October 24, 2026.**
Author is a 10th-grade student learning Python — explain reasoning,
don't just produce code.

## SCHEDULE OVERRIDE — read first
It is late in the project. The app is the only required deliverable
and it does not exist yet. Priority order is absolute:

1. A working, deployed Streamlit app on the Phase 2 risk scores
2. Video + writeup
3. The curb-ramp vision phase — OPTIONAL, attempted only if the
   Sunday Oct 18 checkpoint passes, hard cutoff Tuesday Oct 21

Never suggest vision work before the app is deployed and reachable
at a public URL. If asked to, say the app comes first.

## Goal
Rank street intersections by pedestrian safety priority.
- SHIPPING: crash risk model (LightGBM Poisson) scoring all
  39,183 intersections, delivered as an interactive map
- STRETCH: curb ramp classifier on the covered subset

Study area: Las Vegas, North Las Vegas, Henderson, Enterprise,
Paradise, Spring Valley, Winchester, Whitney.
39,183 intersections · 459 pedestrian crashes (2015–2023) ·
295 crash-bearing intersections (0.75% positive rate).

## Environment
- macOS, Apple Silicon M4, 48GB. PyTorch device `mps`, never `cuda`.
- Python 3.11 in `.venv` via `uv`. Install with `uv pip install`.
- LightGBM needs system libomp (`brew install libomp`) — already done.

## Geospatial conventions
- Storage/output CRS EPSG:4326. Analysis CRS EPSG:26911 (UTM 11N).
- NEVER compute distances or buffers in EPSG:4326. Reproject first.

## Data conventions
- FARS CSVs: `utf-8-sig` with `cp1252` fallback (2016 pbtype.csv).
- Match pedestrians on `PBPTYPENAME == "Pedestrian"`, never the
  numeric code — it drifts across FARS editions.
- `lit` / `sidewalk` way tags are absent from this OSM extract.

## Modeling rules
- Target is `ped_crashes_10yr` (50m band).
- `ped_crashes_midblock_200m` is NEVER a feature. Validation only.
- Positive rate 0.75%. Never report accuracy. PR-AUC beside base rate.
- Splits spatially blocked, never random.
- The risk model predicts crash FREQUENCY, not rate. Top features are
  exposure proxies. Never describe it as measuring design quality.

## Rules
- All paths and constants come from `src/config.py`. Never hardcode.
- Never write to `data/raw/` — immutable source data.
- Prefer clear code over clever code.
- Before writing a new module, propose function signatures first.
- Large data files are gitignored and unreadable to you. Samples
  live in `data/sample/`.

## Current phase
**Phase 6: Streamlit app** (Oct 17–20)
Full spec: `docs/phase6_spec.md`

Exit gate: app deployed at a public URL, loads in under 10 seconds,
map renders all intersections, filters work, CSV download works,
About page present. Verified from a browser that is not yours.

## Phase history
- Phase 0 (complete, Jul 29): environment, data source verification.
- Phase 1 (complete, Jul 30): pipeline. Study area expanded from
  North Las Vegas (48 crashes, below threshold) to 8 jurisdictions.
  39,183 intersections.
- Phase 2 (complete, Aug 6): LightGBM Poisson, 10 features.
  Spatial CV ROC-AUC 0.919, PR-AUC 0.080 (10.6x base rate),
  recall@top-5% 51%.
- Phase 3 Step 0 (complete, Oct 17): Mapillary coverage probe.
  **Finding:** 25% of intersections have any imagery at 25m, but
  coverage tracks road class almost exactly — 5.7% residential vs
  100% motorway/trunk. Top-2,000 risk set sits at 70%. This is a
  measured limitation of crowdsourced imagery for infrastructure
  auditing, and it is a reportable result in its own right.
- Phases 3–5 (deferred, Oct 17): vision work moved behind the app
  for schedule reasons. See `docs/phase3_revised_spec.md`.