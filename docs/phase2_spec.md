# Phase 2 Spec — Crash Risk Model

Goal: predict expected pedestrian crashes per intersection from road
features alone, so that dangerous locations with no crash history are
surfaced. Output feeds Phase 3 site selection and the final fusion.

## Step 0 — Recover lost features (do first, timeboxed to 1 hour)

`lit` and `sidewalk` were absent because `network_type="drive"` excludes
footways, and street lamps are nodes rather than way tags. Test:

```python
tags = {"highway": ["footway", "street_lamp"], "footway": "sidewalk"}
gdf = ox.features_from_place(STUDY_AREA_NAMES, tags)
```

If footways/lamps return in volume, add to `build_features.py`:
- `sidewalk_len_100m` — metres of footway within 100m of the intersection
- `n_street_lamps_100m` — count of street lamp nodes within 100m

Then re-run Phase 1 and its validation. If coverage is under ~20% of
intersections, drop it and log the finding — sparse features are noise.

Do not spend more than an hour here. Move on either way.

## Step 1 — Feature prep (`src/models/risk_model.py`)

Load `data/interim/intersections.geojson`.

**Exclude from features:** `int_id`, `osmid`, `geometry`,
`ped_crashes_10yr` (target), `ped_crashes_midblock_200m` (leakage).

- One-hot or native-categorical `highway_class_max` (LightGBM handles
  categoricals natively — prefer that).
- Keep `speed_imputed` as a feature. Missingness is informative.
- Do not scale or normalize. Gradient boosting is scale-invariant.

Log the final feature list and count.

## Step 2 — Spatial cross-validation

Random splits leak: adjacent intersections share features and crash
history. Instead:

1. Reproject to EPSG:26911.
2. Assign each intersection to a 2km x 2km grid cell.
3. Group cells into 5 folds, whole cells only.
4. Check each fold has >= 30 positives. If not, use 3 folds.

Write this as `spatial_kfold(gdf, n_splits=5, cell_size_m=2000)` in
`src/utils/geo.py` so Phase 6 can reuse it.

**Also report the random-split score for comparison.** The gap between
random and spatial CV is a real finding worth putting in the writeup.

## Step 3 — Model

LightGBM with a Poisson objective — the outcome is a count, and Poisson
is standard for crash prediction models in the safety field.

```python
params = {
    "objective": "poisson",
    "metric": "poisson",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
}
```

`min_data_in_leaf` is high on purpose: with 295 positives, small leaves
memorize individual crashes.

Use early stopping on the validation fold. Record the best iteration
per fold and average.

If Poisson misbehaves, fall back to `objective="binary"` with
`is_unbalance=True` on a binarized target, and log why.

## Step 4 — Evaluation

Report for every fold and averaged:

| Metric | Note |
|---|---|
| PR-AUC | Primary. Always print base rate (0.0075) beside it |
| ROC-AUC | Secondary |
| Poisson deviance | Model-native fit |
| Recall @ top 5% | Of the 1,959 highest-scored, what share of crashes? |

**Recall @ top 5% is the metric that matters for the application** — it
directly answers "if an agency investigates the top 5%, how much of the
harm do they catch?" Put this one in the video.

Do NOT report accuracy anywhere.

## Step 5 — Interpretation (required, not optional)

1. Feature importance (gain) — plot to `outputs/figures/`.
2. SHAP summary plot. `uv pip install shap`.
3. Partial dependence for `max_speed_mph` and `max_lanes`.

**Sanity check:** importances must be physically sensible. More lanes
and higher speed should increase predicted risk. If speed comes out
protective, something is wrong — stop and investigate rather than
shipping it.

Write 3-5 sentences in `LOG.md` on what the model learned, in plain
English. This is writeup material and it is where you find bugs.

## Step 6 — Score and persist

1. Predict on all 39,183 intersections using a model refit on all data.
2. Write `outputs/risk_scores.parquet`: `int_id`, `risk_score`,
   `risk_percentile`.
3. Save the booster to `models/risk_lgbm.txt`.
4. Write the top 2,000 `int_id`s to `outputs/phase3_candidates.csv`
   for image collection.

## Exit gate

- Spatially-blocked CV: ROC-AUC >= 0.70 AND PR-AUC >= 0.023 (3x base rate)
- Feature importances physically sensible
- `outputs/risk_scores.parquet` covers all 39,183 rows
- `outputs/phase3_candidates.csv` has 2,000 rows

If the gate isn't met, that is a finding, not a failure — report the
numbers and we'll decide whether to adjust features or scope. Do not
tune against the test folds to force a passing number.

## KNOWN RISK

459 crashes across 39,183 intersections is a thin signal. The model may
land near baseline. If PR-AUC is below 2x base rate after honest
tuning, stop and tell me — the fallback is to widen the target to
include the 200m midblock band (raising positives to roughly 500+),
accepting that the unit of prediction becomes "intersection vicinity"
rather than "intersection."