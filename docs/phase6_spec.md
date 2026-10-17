# Phase 6 Spec — Streamlit App (REQUIRED DELIVERABLE)

Build this first. It is the only thing the Congressional App
Challenge actually requires. Target: deployed by Sunday Oct 18.

## Scope discipline
Build exactly the five features below. Do not add fusion layers,
authentication, databases, or a ramp overlay. If a feature is not
on this list, it is not in scope. Finishing early means making the
map faster, not bigger.

## Step 1 — Slim data file (DO THIS FIRST, ~20 min)

`data/` is gitignored, and Streamlit Community Cloud deploys from a
public GitHub repo. The app cannot read anything gitignored.

Write `src/data/build_app_data.py`:
1. Load `data/interim/intersections.geojson` + `outputs/risk_scores.parquet`
2. Join on `int_id`
3. Keep ONLY: `int_id`, `lat`, `lon`, `risk_score`, `risk_percentile`,
   `jurisdiction`, `n_legs`, `max_lanes`, `max_speed_mph`,
   `highway_class_max`, `is_signalized`, `has_marked_crossing`,
   `dist_to_transit_m`, `ped_crashes_10yr`
4. Round floats to 5 decimal places for coordinates, 4 for scores
5. Write `app/data/risk_scores_app.parquet`
6. Print the file size

If `jurisdiction` is not already a column, derive it with a spatial
join against the study-area boundaries from `fetch_osm.py`.

Target under 10 MB. If larger, drop to the top 15,000 by risk and
note the cutoff in the About page. Then un-ignore it:
### Tab 4: My neighborhood (resident view) — BUILD AFTER TABS 1-3 WORK

Purpose: give a non-expert a reason to open this app. Also the
primary demo in the submission video.

1. Text input: "Enter an address or intersection in the Las Vegas area"
2. Geocode with geopy Nominatim, custom user_agent, wrapped in
   @st.cache_data. Handle failures with a clear message, never a
   traceback.
3. Map zoomed to that point, 0.5 mile radius circle drawn
4. The 5 highest-risk intersections inside that radius, listed in
   plain language:
   "Lake Mead Blvd & Civic Center Dr — higher risk than 94% of
    intersections in the valley. 6 lanes, 45 mph, no marked crossing."
5. Fallback if geocoding fails or returns nothing nearby: a
   jurisdiction dropdown that does the same thing

Write every string on this tab for someone with no statistics
background. No "percentile," no "PR-AUC," no "Poisson." Say
"higher risk than X% of intersections."