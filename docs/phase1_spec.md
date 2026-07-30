# Phase 1 Spec — Data Pipeline

Goal: produce `data/interim/intersections.geojson`, one row per road
intersection in the study area, with road features and pedestrian crash
counts attached. This is the training table for the Phase 2 risk model.

## Modules to build (in order)

| File | Responsibility |
|---|---|
| `src/data/fetch_osm.py` | Download + cache the road network, extract intersections |
| `src/data/fetch_fars.py` | Load FARS, filter to pedestrian crashes in study area |
| `src/utils/geo.py` | CRS helpers, nearest-neighbor snapping |
| `src/data/build_features.py` | Join everything, write the GeoJSON |

## Step 1 — Road network (`fetch_osm.py`)

1. `ox.graph_from_place("North Las Vegas, Nevada, USA", network_type="drive")`
2. Cache the raw graph to `data/raw/osm/` so we never re-download.
3. **Consolidate intersections.** Raw OSM nodes are not intersections —
   divided roads produce 2+ nodes for one real intersection, and many
   nodes are just geometry vertices. Use
   `ox.consolidate_intersections(G_proj, tolerance=15, rebuild_graph=True)`
   on the projected graph. Tolerance is in meters, so projection first
   is mandatory.
4. Keep only nodes with degree >= 3. Degree 2 is a road bending, not a
   junction.

## Step 2 — Road features

For each intersection, aggregate over its incident edges:

- `n_legs` — number of distinct approach legs
- `max_lanes`, `mean_lanes` — from the `lanes` tag
- `max_speed_mph` — from `maxspeed`. This tag is often missing.
  Impute by `highway` class and set `speed_imputed = True` so the
  model can learn from the missingness.
- `highway_class_max` — the highest-order class among incident edges
  (motorway > trunk > primary > secondary > tertiary > residential)
- `is_signalized` — node tag `highway=traffic_signals`
- `has_marked_crossing` — any nearby node tagged `highway=crossing`
  with `crossing=marked` or `crossing=traffic_signals`
- `has_lighting_tag` — the `lit` tag on incident edges. Sparse in OSM;
  record coverage rate in the log so we know how much to trust it.
- `has_sidewalk` — `sidewalk` tag on incident edges
- `dist_to_transit_m` — distance to nearest OSM `highway=bus_stop`

Note which of these are sparse. A feature present on 5% of edges is
noise, not signal, and should be dropped before Phase 2.

## Step 3 — FARS pedestrian crashes (`fetch_fars.py`)

Years 2015–2023. Two files per year in `data/raw/fars/{year}/`:

- `accident.csv` — has `ST_CASE`, `LATITUDE`, `LONGITUD`, `STATE`, `COUNTY`
- `pbtype.csv` — person-level; join on `ST_CASE`

Filtering:
1. `STATE == 32` (Nevada), `COUNTY == 3` (Clark)
2. Join to `pbtype.csv` and keep pedestrian person type only —
   confirm the exact code against the FARS analytical user's manual
   for the year, since codes have changed across editions.
3. **Drop sentinel coordinates.** FARS encodes unknown lat/lon as
   77.7777, 88.8888, and 99.9999. These are not real locations and
   will silently poison the spatial join if not removed.
4. Clip to the study area boundary.

Log the row count after each filter step. If a step drops more than
half the rows, stop and tell me before continuing.

## Step 4 — Spatial join (`build_features.py`)

1. Reproject both layers to EPSG:26911.
2. Snap each crash to the nearest intersection.
   - Within 50 m → `ped_crashes_10yr` for that intersection
   - 50–200 m → `ped_crashes_midblock_200m` (separate column;
     midblock crashes near a junction are a real signal about
     crossing spacing, not noise to discard)
   - Beyond 200 m → dropped, but log how many
3. Reproject to EPSG:4326 and write the GeoJSON.

## Output schema

| Field | Type | Notes |
|---|---|---|
| `int_id` | str | Stable ID, format `nlv_{n}` |
| `geometry` | Point | EPSG:4326 |
| `osmid` | int | For traceability back to OSM |
| `n_legs` | int | |
| `max_lanes` / `mean_lanes` | float | |
| `max_speed_mph` | float | |
| `speed_imputed` | bool | |
| `highway_class_max` | str | Categorical |
| `is_signalized` | bool | |
| `has_marked_crossing` | bool | |
| `has_lighting_tag` | bool | |
| `has_sidewalk` | bool | |
| `dist_to_transit_m` | float | |
| `ped_crashes_10yr` | int | **Model target** |
| `ped_crashes_midblock_200m` | int | |

## Validation checks (run before declaring Phase 1 done)

Write these as a script, `src/data/validate_phase1.py`:

1. Row count is between 1,500 and 6,000. Outside that range means
   consolidation went wrong.
2. No duplicate `int_id`.
3. No null geometries; all points fall inside the study boundary.
4. `max_speed_mph` between 15 and 80 where not imputed.
5. Total `ped_crashes_10yr` is within 10% of the pre-join crash count
   (accounting for logged drops). If not, the join lost crashes.
6. Print the share of intersections with each boolean feature.

## KNOWN RISK — read before starting

North Las Vegas is ~280k people out of ~2.3M in Clark County. If
pedestrian fatalities scale with population, the study area may only
have roughly 80–120 crashes over 10 years across ~3,000 intersections.
That is a very low positive rate, and Phase 2 may not train.

**Decision checkpoint:** after Step 3, count crashes inside the study
area. Report the number to me before continuing.

- ≥80 crashes → proceed as planned
- <80 crashes → stop and discuss. Options are to expand the study area
  to Las Vegas + North Las Vegas + unincorporated Clark County, or to
  source Nevada DOT injury-crash data (not fatal-only) as a supplement.

Do not silently proceed with too few positives.