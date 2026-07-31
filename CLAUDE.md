# Pedestrian Infrastructure Gap Auditor

Congressional App Challenge submission. Deadline: October 24, 2026.
Author is a 10th-grade student learning Python — explain reasoning,
don't just produce code.

## Goal
Rank street intersections by pedestrian safety priority using a
two-stage screening architecture:
1. Crash risk model (LightGBM, Poisson) scores ALL intersections
2. Top ~2,000 by risk go to a curb ramp classifier (ResNet-18, Mapillary)
3. Fusion produces the ranked priority list

Study area: Las Vegas + North Las Vegas + Henderson + Enterprise +
Paradise + Spring Valley + Winchester + Whitney.
39,183 intersections, 459 pedestrian crashes (2015-2023),
295 crash-bearing intersections (0.75% positive rate).

## Environment
- macOS, Apple Silicon M4, 48GB. Use `mps` device for PyTorch, never `cuda`.
- Python 3.11 in `.venv`, managed with `uv`. Install via `uv pip install`.
- Never `pip install` directly.

## Geospatial conventions
- Storage/output CRS: EPSG:4326. Analysis CRS: EPSG:26911 (UTM 11N).
- NEVER compute distances or buffers in EPSG:4326. Reproject first.
- State the CRS in docstrings for any function taking/returning geometry.

## Data conventions
- FARS CSVs need `utf-8-sig` with a `cp1252` fallback (2016 pbtype.csv).
- Match pedestrians on `PBPTYPENAME == "Pedestrian"`, never the numeric
  code — it drifts across FARS editions.
- `lit` and `sidewalk` way tags are absent from this OSM extract.
  Any code touching them must degrade gracefully, not crash.
- Recovered via separate footway/street-lamp query: `sidewalk_len_100m`
  (50.7% coverage, kept) — `n_street_lamps_100m` only had 4.5% coverage
  (< 20% threshold) and was dropped as noise.

## Rules
- All paths and constants come from `src/config.py`. Never hardcode.
- Never write to `data/raw/` — immutable source data.
- Scripts in `src/data/` must be re-runnable: check for existing output
  and skip unless `--force` is passed.
- Notebooks are exploration only. Production code lives in `src/`.
- Prefer clear code over clever code.
- Before writing a new module, propose the function signatures first.
- Large data files are gitignored and unreadable to you. Samples live in
  `data/sample/` — ask me to create one if you need to see a schema.

## Modeling rules
- The target is `ped_crashes_10yr` (50m band).
- `ped_crashes_midblock_200m` is NEVER a feature. Using nearby crash
  counts to predict crashes defeats the purpose of finding dangerous
  locations that have no crash history yet. Validation column only.
- Positive rate is 0.75%. Accuracy is meaningless. Report PR-AUC
  alongside the base rate every time.
- Splits must be spatially blocked, never random.

## Current phase
**Phase 2 complete.** Phase 3 (curb ramp classifier, ResNet-18 on
Mapillary photos for the top ~2,000 by risk) is not yet scoped — add a
`docs/phase3_spec.md` and update this section before starting Phase 3 work.

## Phase history
- Phase 0 (complete, Jul 29): environment, data source verification.
- Phase 1 (complete, Jul 30): pipeline built. Study area expanded from
  North Las Vegas only (48 crashes, below the 80 threshold) to 8
  jurisdictions. `data/interim/intersections.geojson`, 39,183 rows,
  all validation checks passing.
- Phase 2 (complete, Jul 31): LightGBM Poisson crash risk model.
  Recovered `sidewalk_len_100m` from a footway query (kept, 50.7%
  coverage); `n_street_lamps_100m` dropped (4.5% coverage, noise).
  10 features, native `highway_class_max` categorical, `LGBMRegressor`
  chosen over the raw Booster API for `PartialDependenceDisplay`
  compatibility. 5 spatial folds (all >=30 positives), no Poisson
  misbehavior, no midblock-band fallback needed. Spatially-blocked CV:
  ROC-AUC 0.919, PR-AUC 0.080 (~10.6x the 0.0075 base rate), recall@top-5%
  = 51%. Exit gate met on both metrics. Speed sanity check passed
  (SHAP correlation +0.866). Outputs: `outputs/risk_scores.parquet`
  (39,183 rows), `outputs/phase3_candidates.csv` (top 2,000), `models/risk_lgbm.txt`.