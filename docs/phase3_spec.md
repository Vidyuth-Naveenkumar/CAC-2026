# Phase 3 Spec — Image Collection

Goal: assemble a labeled-ready image set for the curb ramp classifier,
and quantify Mapillary coverage before committing to the vision approach.

## Step 0 — COVERAGE PROBE (blocking, do before anything else)

This is the highest-risk unknown in the project. Do not build the
downloader until this passes.

Sample 200 intersections stratified across risk deciles. For each,
query Mapillary within 25m. Record:

- Fraction with >= 1 image
- Fraction with >= 3 images
- Median image capture year
- Coverage broken out by `highway_class_max` and by risk decile

Write results to `outputs/mapillary_coverage_probe.csv` and summarize
in LOG.md.

**Decision gate — report to me before proceeding:**
- >= 70% of intersections have >= 1 image → proceed as specced
- 40-70% → proceed, but restrict the study set to covered intersections
  and document the coverage bias explicitly (it will correlate with
  road class, which is a real limitation)
- < 40% → STOP. Vision phase is not viable. Fall back to submitting
  the risk model alone with a strong map and writeup.

Do not silently proceed on thin coverage.

## Step 1 — Site selection

Two sets, written to `outputs/phase3_sites.csv` with a `set` column:

| Set | n | Source | Purpose |
|---|---|---|---|
| `priority` | 2,000 | `outputs/phase3_candidates.csv` | The audit target |
| `random` | 1,000 | Stratified random, 100 per risk decile | Unbiased training + measuring gap-vs-risk |

Overlap is fine; dedupe on `int_id` and keep the `priority` label.

The `random` set is mandatory, not optional. Training the classifier
only on high-risk arterials would bias it toward arterial appearance.

## Step 2 — Downloader (`src/data/fetch_images.py`)

For each site:
1. Query Mapillary images within 25m of the intersection point
2. Keep up to 4, preferring recent capture date and, where available,
   varied compass headings so different corners are represented
3. Download at 1024px width, save to
   `data/images/unlabeled/{int_id}_{image_id}.jpg`
4. Append a row to `data/interim/image_manifest.csv`:
   `int_id`, `image_id`, `lat`, `lon`, `captured_at`, `compass_angle`,
   `set`, `filepath`

Requirements:
- Resumable. Skip any `image_id` already in the manifest. You will
  interrupt this run.
- Rate limit: sleep between requests, exponential backoff on 429.
- Log failures to `outputs/image_fetch_errors.csv`, never crash the run.
- Expect roughly 3,000 sites x up to 4 images = up to ~12,000 files.
  Check disk space before starting.

## Step 3 — Manifest validation

Write `src/data/validate_phase3.py`:
1. Every manifest row has a file that exists and opens in PIL
2. No duplicate `image_id`
3. Report images-per-intersection distribution
4. Report coverage by `set` and by risk decile
5. Report capture-year histogram — flag if median is older than 2018,
   since infrastructure may have changed since capture

## Exit gate

- Coverage probe complete, decision gate cleared
- >= 2,000 intersections with >= 1 usable image
- `data/interim/image_manifest.csv` passes all validation checks
- Coverage bias documented in LOG.md

## KNOWN RISK

Mapillary is crowdsourced. Coverage will be uneven and correlated with
road class — arterials get driven and photographed, residential streets
less so. This is a real limitation and must appear in the writeup, not
be hidden. Map the uncovered intersections as a distinct category
rather than treating them as compliant.