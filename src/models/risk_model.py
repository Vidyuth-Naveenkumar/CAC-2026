"""
Phase 2: LightGBM Poisson crash risk model.

Predicts expected pedestrian crashes per intersection from road features
alone, so dangerous locations with no crash history yet still get
surfaced. See docs/phase2_spec.md for the full spec this implements.
"""

import geopandas as gpd
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import average_precision_score, mean_poisson_deviance, roc_auc_score
from sklearn.model_selection import KFold

from src.config import INTERIM_DIR, MODELS_DIR, OUTPUTS_DIR
from src.utils.geo import spatial_kfold, to_analysis_crs

INTERSECTIONS_PATH = INTERIM_DIR / "intersections.geojson"
FIGURES_DIR = OUTPUTS_DIR / "figures"
RISK_SCORES_PATH = OUTPUTS_DIR / "risk_scores.parquet"
CANDIDATES_PATH = OUTPUTS_DIR / "phase3_candidates.csv"
MODEL_PATH = MODELS_DIR / "risk_lgbm.txt"

TARGET_COL = "ped_crashes_10yr"
LEAKAGE_COL = "ped_crashes_midblock_200m"  # never a feature -- see phase2_spec.md
CATEGORICAL_COL = "highway_class_max"

# has_lighting_tag / has_sidewalk are always False -- 0% OSM tag coverage in
# this extract (see CLAUDE.md "Known issues"). A constant column carries no
# information, so it's dropped here rather than padding the feature list.
DEAD_COLS = ["has_lighting_tag", "has_sidewalk"]
EXCLUDE_COLS = ["int_id", "osmid", "geometry", TARGET_COL, LEAKAGE_COL] + DEAD_COLS

CELL_SIZE_M = 2000
MIN_POSITIVES_PER_FOLD = 30
TOP_N_CANDIDATES = 2000
RECALL_K_PCT = 0.05

LGBM_PARAMS = dict(
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,  # high on purpose: with 295 positives, small leaves memorize individual crashes
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    n_estimators=1000,  # upper bound; early stopping picks the real number
    verbose=-1,
)


def load_intersections() -> gpd.GeoDataFrame:
    return gpd.read_file(INTERSECTIONS_PATH)


def prepare_features(gdf: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split into (X, y). Casts highway_class_max to pandas 'category' dtype
    so LightGBM handles it natively instead of one-hot encoding it."""
    feature_cols = [c for c in gdf.columns if c not in EXCLUDE_COLS]
    X = pd.DataFrame(gdf[feature_cols]).copy()
    X[CATEGORICAL_COL] = X[CATEGORICAL_COL].astype("category")
    y = gdf[TARGET_COL].copy()
    print(f"Feature list ({len(feature_cols)}): {feature_cols}")
    return X, y


def choose_fold_labels(
    gdf_proj: gpd.GeoDataFrame, y: pd.Series, cell_size_m: float = CELL_SIZE_M, seed: int = 42
) -> tuple[int, pd.Series]:
    """Try 5 spatial folds; fall back to 3 if any fold has < 30 positives."""
    for n_splits in (5, 3):
        fold_labels = spatial_kfold(gdf_proj, n_splits=n_splits, cell_size_m=cell_size_m, seed=seed)
        positives_per_fold = y.groupby(fold_labels).apply(lambda s: (s > 0).sum())
        print(f"n_splits={n_splits} positives per fold: {positives_per_fold.to_dict()}")
        if (positives_per_fold >= MIN_POSITIVES_PER_FOLD).all():
            return n_splits, fold_labels
    print(
        f"WARNING: even 3 folds has a fold below {MIN_POSITIVES_PER_FOLD} positives -- using 3 anyway."
    )
    return 3, fold_labels


def train_one_fold(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, objective: str
) -> tuple[lgb.LGBMRegressor, int]:
    """Fit with early stopping on the validation fold. Returns (model, best_iteration)."""
    y_train_fit = y_train if objective == "poisson" else (y_train > 0).astype(int)
    y_val_fit = y_val if objective == "poisson" else (y_val > 0).astype(int)

    params = dict(LGBM_PARAMS)
    params["objective"] = objective
    if objective == "poisson":
        params["metric"] = "poisson"
    else:
        params["metric"] = "auc"
        params["is_unbalance"] = True

    model = lgb.LGBMRegressor(**params) if objective == "poisson" else lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train_fit,
        eval_set=[(X_val, y_val_fit)],
        categorical_feature=[CATEGORICAL_COL],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    return model, model.best_iteration_


def predict_score(model, X: pd.DataFrame, objective: str) -> np.ndarray:
    """Poisson regressor predicts an expected count directly; the binary
    fallback classifier predicts a probability -- both are usable as a
    ranking score for evaluation."""
    if objective == "poisson":
        return model.predict(X)
    return model.predict_proba(X)[:, 1]


def recall_at_top_k_pct(y_true: np.ndarray, y_score: np.ndarray, k_pct: float = RECALL_K_PCT) -> float:
    """Share of this fold's total crash COUNT captured in its top k_pct
    highest-scored rows -- "if an agency investigates the top 5%, how much
    of the harm do they catch?" """
    total = y_true.sum()
    if total == 0:
        return float("nan")
    k = max(1, int(np.ceil(k_pct * len(y_true))))
    top_idx = np.argsort(-y_score)[:k]
    return y_true[top_idx].sum() / total


def evaluate_fold(y_true: pd.Series, y_score: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_binary = (y_true > 0).astype(int)

    pr_auc = average_precision_score(y_binary, y_score)
    roc_auc = (
        roc_auc_score(y_binary, y_score) if 0 < y_binary.sum() < len(y_binary) else float("nan")
    )
    # mean_poisson_deviance needs strictly positive predictions.
    poisson_dev = mean_poisson_deviance(y_true, np.clip(y_score, 1e-6, None))
    recall_5pct = recall_at_top_k_pct(y_true, y_score)

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "poisson_deviance": poisson_dev,
        "recall_at_5pct": recall_5pct,
    }


def run_cv(X: pd.DataFrame, y: pd.Series, fold_labels: pd.Series, n_splits: int, objective: str) -> pd.DataFrame:
    fold_labels = fold_labels.reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    rows = []
    for fold in range(n_splits):
        train_mask = fold_labels != fold
        val_mask = fold_labels == fold

        model, best_iter = train_one_fold(
            X[train_mask], y[train_mask], X[val_mask], y[val_mask], objective
        )
        y_pred = predict_score(model, X[val_mask], objective)
        metrics = evaluate_fold(y[val_mask], y_pred)

        y_val = y[val_mask].to_numpy()
        std_collapsed = np.std(y_pred) < 1e-6
        roc_bad = np.isnan(metrics["roc_auc"]) or metrics["roc_auc"] <= 0.55
        stopped_immediately = best_iter == 0
        n_triggers = sum([std_collapsed, roc_bad, stopped_immediately])

        rows.append(
            {
                "fold": fold,
                "n_val": int(val_mask.sum()),
                "n_positive": int((y_val > 0).sum()),
                "best_iteration": best_iter,
                "misbehavior_triggers": n_triggers,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def run_cv_with_fallback(
    X: pd.DataFrame, y: pd.Series, fold_labels: pd.Series, n_splits: int
) -> tuple[pd.DataFrame, str]:
    """Poisson first; fall back to binary if >=2 of 3 misbehavior checks
    fire on any fold, per phase2_spec.md's open-ended 'if it misbehaves'."""
    results = run_cv(X, y, fold_labels, n_splits, objective="poisson")
    bad_folds = (results["misbehavior_triggers"] >= 2).sum()
    if bad_folds == 0:
        return results, "poisson"

    print(
        f"Poisson misbehaved on {bad_folds}/{n_splits} folds (>=2 of 3 triggers fired) "
        f"-- falling back to objective='binary', is_unbalance=True on a binarized target."
    )
    results = run_cv(X, y, fold_labels, n_splits, objective="binary")
    return results, "binary"


def summarize_cv(results: pd.DataFrame, base_rate: float, label: str) -> pd.DataFrame:
    mean_row = results.drop(columns=["fold"]).mean(numeric_only=True)
    mean_row["fold"] = "mean"
    summary = pd.concat([results, pd.DataFrame([mean_row])], ignore_index=True)
    print(f"\n=== {label} CV results (base rate = {base_rate:.4f}) ===")
    print(summary.to_string(index=False))
    return summary


def refit_and_score(
    X: pd.DataFrame, y: pd.Series, objective: str, best_iterations: np.ndarray
) -> tuple[object, np.ndarray]:
    """Refit on ALL data (no held-out fold) using the median best_iteration
    from CV as a stand-in for early stopping, then predict on everything."""
    num_boost_round = max(int(np.median(best_iterations)), 1)

    params = dict(LGBM_PARAMS)
    params["objective"] = objective
    params["n_estimators"] = num_boost_round
    if objective == "poisson":
        params["metric"] = "poisson"
    else:
        params["metric"] = "auc"
        params["is_unbalance"] = True

    y_fit = y if objective == "poisson" else (y > 0).astype(int)
    model = lgb.LGBMRegressor(**params) if objective == "poisson" else lgb.LGBMClassifier(**params)
    model.fit(X, y_fit, categorical_feature=[CATEGORICAL_COL])

    risk_score = predict_score(model, X, objective)
    return model, risk_score


def plot_feature_importance(model, feature_names: list[str], path) -> None:
    importance = model.booster_.feature_importance(importance_type="gain")
    order = np.argsort(importance)
    plt.figure(figsize=(8, 6))
    plt.barh(np.array(feature_names)[order], importance[order])
    plt.xlabel("Gain")
    plt.title("Feature importance (gain)")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
    print(f"Wrote {path}")


def plot_shap_summary(model, X: pd.DataFrame, path) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # binary classifier: positive-class SHAP values

    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")
    return shap_values


def plot_partial_dependence(model, X: pd.DataFrame, features: list[str], path) -> None:
    # sklearn's PDP grid-generation samples percentiles straight from the
    # column and can't handle NaN. max_lanes is ~71.5% missing (LightGBM
    # itself still trains on the full data with NaN as a native missing-
    # value split) so this one plot uses only the non-missing rows.
    X_ref = X.dropna(subset=["max_lanes"]) if "max_lanes" in features else X
    fig, ax = plt.subplots(1, len(features), figsize=(5 * len(features), 4))
    PartialDependenceDisplay.from_estimator(model, X_ref, features=features, ax=ax)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
    print(f"Wrote {path}")


def check_speed_direction(shap_values: np.ndarray, X: pd.DataFrame, feature: str = "max_speed_mph") -> float:
    """Sanity check per phase2_spec.md: higher speed should increase
    predicted risk. If it comes out protective, something is wrong."""
    idx = list(X.columns).index(feature)
    corr = np.corrcoef(X[feature], shap_values[:, idx])[0, 1]
    print(f"Correlation between {feature} and its SHAP value: {corr:.3f}")
    if corr < 0:
        print(
            f"WARNING: {feature} appears PROTECTIVE (higher speed -> lower predicted risk). "
            "Per phase2_spec.md this is a sign something is wrong -- investigate before shipping."
        )
    return corr


def write_outputs(
    gdf: gpd.GeoDataFrame, risk_score: np.ndarray, model, top_n: int = TOP_N_CANDIDATES
) -> pd.DataFrame:
    scores = pd.DataFrame({"int_id": gdf["int_id"].values, "risk_score": risk_score})
    scores["risk_percentile"] = scores["risk_score"].rank(pct=True) * 100

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(RISK_SCORES_PATH, index=False)
    print(f"Wrote {len(scores)} rows to {RISK_SCORES_PATH}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(MODEL_PATH))
    print(f"Saved model to {MODEL_PATH}")

    top = scores.sort_values("risk_score", ascending=False).head(top_n)
    top[["int_id"]].to_csv(CANDIDATES_PATH, index=False)
    print(f"Wrote top {len(top)} candidates to {CANDIDATES_PATH}")

    return scores


def make_random_fold_labels(n_rows: int, n_splits: int, seed: int = 42) -> pd.Series:
    """Baseline-only comparison: ordinary random KFold, no spatial blocking."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    labels = np.zeros(n_rows, dtype=int)
    for fold, (_, val_idx) in enumerate(kf.split(np.arange(n_rows))):
        labels[val_idx] = fold
    return pd.Series(labels)


def main() -> None:
    gdf = load_intersections()
    X, y = prepare_features(gdf)
    base_rate = (y > 0).mean()

    gdf_proj = to_analysis_crs(gdf)
    n_splits, fold_labels = choose_fold_labels(gdf_proj, y)

    spatial_results, objective = run_cv_with_fallback(X, y, fold_labels, n_splits)
    summarize_cv(spatial_results, base_rate, "Spatially-blocked")

    random_labels = make_random_fold_labels(len(X), n_splits)
    random_results, _ = run_cv_with_fallback(X, y, random_labels, n_splits)
    summarize_cv(random_results, base_rate, "Random-split (baseline comparison)")

    best_iterations = spatial_results["best_iteration"].to_numpy()
    model, risk_score = refit_and_score(X, y, objective, best_iterations)

    plot_feature_importance(model, list(X.columns), FIGURES_DIR / "feature_importance_gain.png")
    shap_values = plot_shap_summary(model, X, FIGURES_DIR / "shap_summary.png")
    plot_partial_dependence(model, X, ["max_speed_mph", "max_lanes"], FIGURES_DIR / "pdp_speed_lanes.png")
    check_speed_direction(shap_values, X)

    write_outputs(gdf, risk_score, model)

    mean_roc = spatial_results["roc_auc"].mean()
    mean_pr = spatial_results["pr_auc"].mean()
    gate_pr_threshold = 3 * base_rate
    print(f"\n=== Exit gate (objective={objective}) ===")
    print(f"ROC-AUC {mean_roc:.3f} >= 0.70: {mean_roc >= 0.70}")
    print(f"PR-AUC {mean_pr:.4f} >= {gate_pr_threshold:.4f} (3x base rate): {mean_pr >= gate_pr_threshold}")
    if mean_pr < 2 * base_rate:
        print(
            f"KNOWN RISK triggered: PR-AUC below 2x base rate ({2 * base_rate:.4f}). "
            "Per phase2_spec.md, stop and report rather than silently widening the target "
            "to the midblock band."
        )


if __name__ == "__main__":
    main()
