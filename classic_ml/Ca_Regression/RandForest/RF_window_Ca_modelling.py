#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Random Forest regression for Ca modelling from a prebuilt regression matrix.

This version:
- loads regression_matrix.npz
- does NOT rebuild raster extraction each run
- keeps predictor filtering similar to your RF classification script
- keeps spatial block CV
- saves variable importance
- saves simple prediction intervals from tree-wise predictions
"""


import os
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================

# USER SETTINGS
# =============================================================================


MATRIX_PATH = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\RandForest\Ca_conc_modelling\Regression_matrix_XRFtotal_lab\Regression_matrix_XRFtotal_lab_withLOI\regression_matrix.npz"
OUT_DIR = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\RandForest\Ca_conc_modelling\Model\Model_XRF_total_Lab\Test_noBioKlima"

# List of predictors to exclude from the regression (by name)
EXCLUDED_PREDICTORS = ["BioK_PCA1.tif","BioK_PCA2.tif"]


# === Target variables to run (each gets its own output subfolder and metrics) ===
TARGET_VARIABLES = ["KK"]  # Add or remove variables here
USE_LOG_TARGET = False                  # Apply log(target) transform to all target variables

# Per-target transforms applied AFTER loading (overrides USE_LOG_TARGET for that variable).
# Supported values: "exp" (np.exp), "log" (np.log), None (no transform).
TARGET_TRANSFORMS = {
    "CaMg_LOI_v": None,   # no transform (raw log-ratio values)
}

# === Cross-validation strategy ===
USE_SPATIAL_CV = True    # True = spatial block CV (recommended); False = random KFold
RUN_TOP15_RERUN = False  # Set True to also run a second CV with the top-15 predictors

# === Feature pre-selection using saved importance (set None to use all predictors) ===
# Loads variable_importance.csv from the previous run of the same target and keeps only
# the top N predictors. Useful for testing whether removing low-importance noise helps.
PRESELECT_TOP_N = 40    # used only when MULT_RUNS=False; set None to disable

# === Multi-run comparison mode ===
# MULT_RUNS=True : for each target, first run all predictors to build importances, then
#                  automatically rerun with each top-N subset. Outputs go to subfolders:
#                  OUT_DIR/<TARGET>/all/,  OUT_DIR/<TARGET>/top20/, ...
# MULT_RUNS=False: single run; if PRESELECT_TOP_N is set, filters using a prior
#                  importance CSV from OUT_DIR/<TARGET>/variable_importance.csv.
MULT_RUNS = True
MULT_RUNS_TOP_N = [20, 30, 40, 50, 60]  # top-N values to test when MULT_RUNS=True

USE_AREA_FILTER = False
AREA_POLYGON_PATH = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2025\kalk_prosjekt3.0\Mask\shape\Nordland_ed.shp"
AREA_POLYGON_LAYER = None
AREA_FILTER_MODE = "intersects"

SAMPLE_SIZE = None
N_SPLITS = 5
BLOCK_SIZE_M = 20000     # Spatial block size in metres (used when USE_SPATIAL_CV=True)
RANDOM_SEED = 42

N_ESTIMATORS = 700
MAX_DEPTH = 25
MIN_SAMPLES_LEAF = 2
MAX_FEATURES = "sqrt"
BOOTSTRAP = True
N_JOBS = -1

# Optional list of predictors to coerce as categorical (round, clamp negatives, fill NaN).
# Keep empty when quaternary/landuse are already represented as binary maps.
CATEGORICAL_PREDICTOR_NAMES = []

# === Load predictors from predictor_names.csv ===
PREDICTOR_NAMES_PATH = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\RandForest\Ca_conc_modelling\Regression_matrix_XRFtotal_lab\Regression_matrix_XRFtotal_lab_withLOI\predictor_names.csv"
_predictor_df = pd.read_csv(PREDICTOR_NAMES_PATH)
PREDICTOR_LIST = _predictor_df["predictor_name"].tolist()


## Interactive predictor exclusion removed: only EXCLUDED_PREDICTORS is used for exclusion.

SELECT_PREDICTORS = PREDICTOR_LIST
SELECT_PREFIXES = None
EXCLUDE_PREFIXES = None

IMPORTANCE_TOP_N = 30
SAVE_FULL_IMPORTANCE_CHART = True
MAX_VARS_IN_FULL_CHART = None
FULL_CHART_FIGHEIGHT_PER_VAR = 0.22
TOP_CHART_FIGHEIGHT_PER_VAR = 0.34

SAVE_PREDICTION_INTERVALS = True
LOWER_Q = 0.10
UPPER_Q = 0.90

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)


# =============================================================================
# HELPERS
# =============================================================================

def load_matrix(npz_path: str):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Regression matrix not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    required = ["X", "y", "rows", "cols", "var_names"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"Regression matrix missing required keys: {missing}")

    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.float32)  # target variable
    # Use x_coord and y_coord for coordinates
    if "x" in data.files and "y" in data.files:
        x_coord = np.asarray(data["x"], dtype=np.float64)
        y_coord = np.asarray(data["y"], dtype=np.float64)
    else:
        x_coord = np.asarray(data["rows"], dtype=np.float64)
        y_coord = np.asarray(data["cols"], dtype=np.float64)
    var_names = [str(v) for v in data["var_names"].tolist()]

    width = int(np.asarray(data["width"]).ravel()[0]) if "width" in data.files else None
    height = int(np.asarray(data["height"]).ravel()[0]) if "height" in data.files else None

    transform = None
    if "transform" in data.files:
        t = np.asarray(data["transform"], dtype=float).ravel()
        if t.size == 6:
            transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])

    crs_wkt = str(np.asarray(data["crs_wkt"]).ravel()[0]) if "crs_wkt" in data.files else None
    ref_mask = str(np.asarray(data["ref_mask"]).ravel()[0]) if "ref_mask" in data.files else None

    point_ids = None
    if "point_ids" in data.files:
        point_ids = np.asarray(data["point_ids"]).astype(str)

    return X, y, x_coord, y_coord, var_names, width, height, transform, crs_wkt, ref_mask, point_ids


def normalize_name(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_")


def apply_predictor_filters(
    X,
    var_names,
    excluded_names=None,
    selected_names=None,
    select_prefixes=None,
    exclude_prefixes=None,
):
    names_norm = [normalize_name(v) for v in var_names]
    keep_mask = np.ones(len(var_names), dtype=bool)

    if selected_names:
        selected_set = {normalize_name(n) for n in selected_names}
        keep_mask &= np.array([n in selected_set for n in names_norm], dtype=bool)
        missing = [n for n in selected_names if normalize_name(n) not in set(names_norm)]
        if missing:
            print(f"[WARNING] SELECT_PREDICTORS not found in var_names: {missing}")

    elif select_prefixes:
        prefixes = [normalize_name(p) for p in select_prefixes]
        keep_mask &= np.array([any(n.startswith(pref) for pref in prefixes) for n in names_norm], dtype=bool)
        missing_pref = [pref for pref in prefixes if not any(n.startswith(pref) for n in names_norm)]
        if missing_pref:
            print(f"[WARNING] SELECT_PREFIXES matched no predictors: {missing_pref}")

    if excluded_names:
        excluded_set = {normalize_name(n) for n in excluded_names}
        keep_mask &= np.array([n not in excluded_set for n in names_norm], dtype=bool)
        missing_excluded = [n for n in excluded_names if normalize_name(n) not in set(names_norm)]
        if missing_excluded:
            print(f"[WARNING] EXCLUDED_PREDICTORS not found in var_names: {missing_excluded}")

    if exclude_prefixes:
        ex_prefixes = [normalize_name(p) for p in exclude_prefixes]
        keep_mask &= np.array([not any(n.startswith(pref) for pref in ex_prefixes) for n in names_norm], dtype=bool)
        missing_ex_prefixes = [pref for pref in ex_prefixes if not any(n.startswith(pref) for n in names_norm)]
        if missing_ex_prefixes:
            print(f"[WARNING] EXCLUDE_PREFIXES matched no predictors: {missing_ex_prefixes}")

    keep_idx = [i for i, keep in enumerate(keep_mask) if keep]
    if len(keep_idx) == 0:
        raise RuntimeError("Predictor filtering removed all predictors.")

    print(f"[INFO] Keeping {len(keep_idx)} predictors")
    return X[:, keep_idx], [var_names[i] for i in keep_idx]


def find_categorical_indices(var_names, categorical_names=None):
    if not categorical_names:
        return {}
    lookup = {normalize_name(v): i for i, v in enumerate(var_names)}
    out = {}
    for name in categorical_names:
        idx = lookup.get(normalize_name(name), None)
        if idx is not None:
            out[name] = idx
    return out


def coerce_categorical_columns(X, cat_idx):
    X2 = X.copy()
    for idx in cat_idx:
        col = X2[:, idx]
        col = np.where(np.isfinite(col), col, 0)
        col = np.where(col < 0, 0, np.rint(col))
        X2[:, idx] = col.astype(np.float32)
    return X2


def filter_points_by_polygon(
    X, y, x, y_coord,
    crs_wkt, transform,
    polygon_path, polygon_layer=None, mode="intersects",
    point_ids=None
):
    if transform is None or crs_wkt is None:
        raise RuntimeError("Polygon filtering requires transform and crs_wkt")

    area_gdf = gpd.read_file(polygon_path, layer=polygon_layer)

    pts = gpd.GeoDataFrame(
        {"x": x, "y": y_coord},
        geometry=gpd.points_from_xy(x, y_coord),
        crs=crs_wkt,
    )

    if str(area_gdf.crs) != str(pts.crs):
        area_gdf = area_gdf.to_crs(pts.crs)

    union_geom = area_gdf.unary_union
    keep = pts.intersects(union_geom).to_numpy() if mode.lower() != "within" else pts.within(union_geom).to_numpy()

    X2 = X[keep]
    y2 = y[keep]
    x2 = x[keep]
    y2c = y_coord[keep]
    point_ids2 = point_ids[keep] if point_ids is not None else None
    return X2, y2, x2, y2c, point_ids2


def stratified_subsample_regression(X, y, rows, cols, sample_size=None, seed=42, point_ids=None):
    if sample_size is None or sample_size >= len(y):
        return X, y, rows, cols, point_ids

    rng = np.random.default_rng(seed)
    n_bins = min(10, max(3, len(y) // 200))
    q = np.unique(np.quantile(y, np.linspace(0, 1, n_bins + 1)))

    if len(q) <= 2:
        idx = rng.choice(len(y), size=sample_size, replace=False)
    else:
        bins = np.digitize(y, q[1:-1], right=True)
        selected = []
        for b in np.unique(bins):
            idx_b = np.where(bins == b)[0]
            take = max(1, int(round(sample_size * len(idx_b) / len(y))))
            take = min(take, len(idx_b))
            selected.extend(rng.choice(idx_b, size=take, replace=False).tolist())
        selected = np.array(sorted(set(selected)))
        if len(selected) > sample_size:
            selected = rng.choice(selected, size=sample_size, replace=False)
        elif len(selected) < sample_size:
            remaining = np.setdiff1d(np.arange(len(y)), selected)
            add = rng.choice(remaining, size=sample_size - len(selected), replace=False)
            selected = np.concatenate([selected, add])
        idx = np.asarray(selected)

    point_ids2 = point_ids[idx] if point_ids is not None else None
    return X[idx], y[idx], rows[idx], cols[idx], point_ids2



def make_spatial_block_folds(x_coords, y_coords, n_splits, block_size_m, random_seed=42):
    """Assign each point to a spatial block and return per-point fold labels (0..n_splits-1)."""
    if (x_coords.max() - x_coords.min()) < 1 or (y_coords.max() - y_coords.min()) < 1:
        warnings.warn("[SpatialCV] Coordinate range is near-zero; spatial folds will not be meaningful.")
    x_block = ((x_coords - x_coords.min()) / block_size_m).astype(int)
    y_block = ((y_coords - y_coords.min()) / block_size_m).astype(int)
    n_y = int((y_coords.max() - y_coords.min()) / block_size_m) + 2
    block_id = x_block * n_y + y_block
    unique_blocks = np.unique(block_id)
    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(len(unique_blocks))
    block_to_fold = {b: int(perm[i]) % n_splits for i, b in enumerate(unique_blocks)}
    fold_labels = np.array([block_to_fold[b] for b in block_id], dtype=int)
    sizes = [int((fold_labels == f).sum()) for f in range(n_splits)]
    print(f"[SpatialCV] {len(unique_blocks)} blocks → {n_splits} folds | sizes: {sizes}")
    return fold_labels


def fit_rf_regression(X_train, y_train, random_state=42):
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        bootstrap=BOOTSTRAP,
        random_state=random_state,
        n_jobs=N_JOBS,
    )
    model.fit(X_train, y_train)
    return model


def tree_prediction_intervals(model, X, q_low=0.10, q_high=0.90):
    all_tree_preds = np.vstack([tree.predict(X) for tree in model.estimators_]).T
    pred_mean = np.mean(all_tree_preds, axis=1)
    pred_low = np.quantile(all_tree_preds, q_low, axis=1)
    pred_high = np.quantile(all_tree_preds, q_high, axis=1)
    pred_width = pred_high - pred_low
    return pred_mean, pred_low, pred_high, pred_width


def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    return {"RMSE": float(rmse), "MAE": float(mae), "R2": float(r2), "Bias": float(bias)}


def save_importance_outputs(importance_df: pd.DataFrame, out_dir: str):
    csv_path = os.path.join(out_dir, "variable_importance.csv")
    try:
        importance_df.to_csv(csv_path, index=False)
    except PermissionError:
        fallback_csv_path = os.path.join(out_dir, f"variable_importance_{os.getpid()}.csv")
        importance_df.to_csv(fallback_csv_path, index=False)
        print(f"[WARNING] Could not overwrite variable_importance.csv; wrote {fallback_csv_path} instead.")

    top_df = importance_df.head(IMPORTANCE_TOP_N)
    fig_h = max(4, len(top_df) * TOP_CHART_FIGHEIGHT_PER_VAR)
    plt.figure(figsize=(10, fig_h))
    plt.barh(top_df["Variable"][::-1], top_df["Importance"][::-1])
    plt.xlabel("Importance")
    plt.ylabel("Variable")
    plt.title(f"Top {len(top_df)} Variable Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "variable_importance_top.png"), dpi=200)
    plt.close()

    if SAVE_FULL_IMPORTANCE_CHART:
        full_df = importance_df if MAX_VARS_IN_FULL_CHART is None else importance_df.head(MAX_VARS_IN_FULL_CHART)
        fig_h = max(6, len(full_df) * FULL_CHART_FIGHEIGHT_PER_VAR)
        plt.figure(figsize=(10, fig_h))
        plt.barh(full_df["Variable"][::-1], full_df["Importance"][::-1])
        plt.xlabel("Importance")
        plt.ylabel("Variable")
        plt.title("Full Variable Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "variable_importance_full.png"), dpi=200)
        plt.close()


def run_cv_and_save(X_samp, y_samp, x_samp, y_coord_samp, ids_samp,
                    var_names_this, target_label, TARGET_VARIABLE, out_dir):
    """Run spatial/random CV, save all outputs, fit final model. Returns importance_df or None."""
    os.makedirs(out_dir, exist_ok=True)
    n_preds = X_samp.shape[1]
    print(f"  Predictors: {n_preds}")

    if USE_SPATIAL_CV:
        cv_label = f"Spatial block CV ({BLOCK_SIZE_M} m)"
        print(f"\n=== {cv_label} | {N_SPLITS} folds ===")
        fold_labels = make_spatial_block_folds(
            x_samp, y_coord_samp, N_SPLITS, BLOCK_SIZE_M, RANDOM_SEED
        )
        fold_iter = [
            (np.where(fold_labels != f)[0], np.where(fold_labels == f)[0])
            for f in range(N_SPLITS)
        ]
    else:
        cv_label = f"Random KFold ({N_SPLITS} folds)"
        print(f"\n=== {cv_label} ===")
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        fold_iter = list(kf.split(X_samp, y_samp))

    fold_rows_list, importances_list, pred_rows_list = [], [], []
    for fold_id, (tr, te) in enumerate(fold_iter, start=1):
        print(f"\n--- Fold {fold_id}/{N_SPLITS} | train={len(tr)} test={len(te)} ---")
        model = fit_rf_regression(X_samp[tr], y_samp[tr], random_state=RANDOM_SEED + fold_id)
        y_pred = model.predict(X_samp[te])
        metrics = regression_metrics(y_samp[te], y_pred)
        metrics.update({"Fold": fold_id, "N_train": len(tr), "N_test": len(te)})
        fold_rows_list.append(metrics)
        if hasattr(model, "feature_importances_"):
            importances_list.append(model.feature_importances_.copy())
        pred_df = pd.DataFrame({
            "fold": fold_id, "y_true": y_samp[te], "y_pred": y_pred,
            "x": x_samp[te], "y": y_coord_samp[te],
        })
        if ids_samp is not None:
            pred_df["point_id"] = ids_samp[te]
        pred_rows_list.append(pred_df)
        print(f"RMSE={metrics['RMSE']:.4f} | MAE={metrics['MAE']:.4f} | "
              f"R2={metrics['R2']:.4f} | Bias={metrics['Bias']:.4f}")

    fold_df    = pd.DataFrame(fold_rows_list)
    pred_cv_df = pd.concat(pred_rows_list, ignore_index=True)
    fold_df.to_csv(os.path.join(out_dir, "cv_fold_metrics.csv"), index=False)
    pred_cv_df.to_csv(os.path.join(out_dir, "cv_predictions.csv"), index=False)

    mean_metrics = fold_df[["RMSE", "MAE", "R2", "Bias"]].mean().to_dict()
    std_metrics  = fold_df[["RMSE", "MAE", "R2", "Bias"]].std().to_dict()
    print(f"\nCV [{TARGET_VARIABLE} | {n_preds} preds]: "
          f"R2={mean_metrics['R2']:.4f}±{std_metrics['R2']:.4f} | "
          f"RMSE={mean_metrics['RMSE']:.4f}±{std_metrics['RMSE']:.4f}")

    with open(os.path.join(out_dir, "regression_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Random Forest regression — target: {TARGET_VARIABLE} ({target_label})\n")
        f.write("=" * 60 + "\n")
        f.write(f"CV strategy: {cv_label}\n")
        f.write(f"Predictors used: {n_preds}\n")
        f.write(f"MATRIX_PATH: {MATRIX_PATH}\n")
        f.write(f"SAMPLE_SIZE: {SAMPLE_SIZE}\n")
        f.write(f"N_SPLITS: {N_SPLITS}\n")
        f.write(f"N_ESTIMATORS: {N_ESTIMATORS}\n")
        f.write(f"MAX_DEPTH: {MAX_DEPTH}\n")
        f.write(f"MIN_SAMPLES_LEAF: {MIN_SAMPLES_LEAF}\n")
        f.write(f"MAX_FEATURES: {MAX_FEATURES}\n")
        f.write(f"BOOTSTRAP: {BOOTSTRAP}\n\n")
        f.write("Mean CV metrics\n")
        for k, v in mean_metrics.items():
            f.write(f"{k}: {v:.6f}\n")
        f.write("\nSD CV metrics\n")
        for k, v in std_metrics.items():
            f.write(f"{k}: {v:.6f}\n")
        f.write("\n\nPer-fold metrics\n")
        f.write(fold_df.to_string(index=False))

    importance_df = None
    if importances_list:
        mean_importances = np.mean(np.vstack(importances_list), axis=0)
        importance_df = pd.DataFrame({
            "Variable": var_names_this, "Importance": mean_importances,
        }).sort_values("Importance", ascending=False)
        save_importance_outputs(importance_df, out_dir)

    # Final model
    print(f"\n=== FINAL MODEL ({TARGET_VARIABLE} | {n_preds} preds) ===")
    final_model = fit_rf_regression(X_samp, y_samp, random_state=RANDOM_SEED)
    final_pred  = final_model.predict(X_samp)
    final_metrics_dict = regression_metrics(y_samp, final_pred)
    with open(os.path.join(out_dir, "final_model_train_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Training metrics — target: {TARGET_VARIABLE}\n{'='*50}\n")
        for k, v in final_metrics_dict.items():
            f.write(f"{k}: {v:.6f}\n")

    if SAVE_PREDICTION_INTERVALS:
        pred_mean_arr, pred_low, pred_high, pred_width = tree_prediction_intervals(
            final_model, X_samp, q_low=LOWER_Q, q_high=UPPER_Q
        )
        interval_df = pd.DataFrame({
            "x": x_samp, "y": y_coord_samp, "y_true": y_samp,
            "pred_mean": pred_mean_arr,
            f"pred_q{int(LOWER_Q*100)}": pred_low,
            f"pred_q{int(UPPER_Q*100)}": pred_high,
            "pred_interval_width": pred_width,
        })
        if ids_samp is not None:
            interval_df["point_id"] = ids_samp
        interval_df.to_csv(
            os.path.join(out_dir, "final_model_prediction_intervals.csv"), index=False
        )

    try:
        import joblib
        model_path = os.path.join(out_dir, f"rf_final_model_{TARGET_VARIABLE}.joblib")
        joblib.dump({
            "model": final_model,
            "var_names": var_names_this,
            "target_variable": TARGET_VARIABLE,
            "use_log_target": USE_LOG_TARGET,
            "target_transform": TARGET_TRANSFORMS.get(TARGET_VARIABLE, None),
            "categorical_predictors": find_categorical_indices(
                var_names_this, CATEGORICAL_PREDICTOR_NAMES
            ),
            "matrix_path": MATRIX_PATH,
            "rf_params": {
                "n_estimators": N_ESTIMATORS, "max_depth": MAX_DEPTH,
                "min_samples_leaf": MIN_SAMPLES_LEAF, "max_features": MAX_FEATURES,
                "bootstrap": BOOTSTRAP,
            },
            "prediction_interval_quantiles": [LOWER_Q, UPPER_Q],
        }, model_path)
        print(f"Saved model: {model_path}")
    except Exception as exc:
        print(f"[WARNING] Could not save final model: {exc}")

    try:
        plt.figure(figsize=(6, 6))
        plt.scatter(pred_cv_df["y_true"], pred_cv_df["y_pred"], s=10, alpha=0.5)
        mn = min(pred_cv_df["y_true"].min(), pred_cv_df["y_pred"].min())
        mx = max(pred_cv_df["y_true"].max(), pred_cv_df["y_pred"].max())
        plt.plot([mn, mx], [mn, mx], linestyle="--")
        plt.xlabel("Observed")
        plt.ylabel("Predicted")
        plt.title(f"CV: obs vs pred — {target_label} ({n_preds} preds)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "cv_observed_vs_predicted.png"), dpi=200)
        plt.close()
    except Exception as exc:
        print(f"[WARNING] Could not save scatter plot: {exc}")

    return importance_df


# =============================================================================
# MAIN
# =============================================================================

print("\n=== LOADING REGRESSION MATRIX ===")
X_all, _y_npz, _rows_npz, _cols_npz, var_names, width, height, transform, crs_wkt, ref_mask, point_ids = load_matrix(MATRIX_PATH)
print(f"Loaded matrix: {MATRIX_PATH} | X shape: {X_all.shape} | Predictors: {len(var_names)}")
print(f"X NaN: {np.isnan(X_all).sum()} | X inf: {np.isinf(X_all).sum()}")

# Load CSV for coordinates and all target columns (NPZ rows/cols are placeholder NaN)
_matrix_csv = MATRIX_PATH.replace(".npz", ".csv")
if not os.path.exists(_matrix_csv):
    raise FileNotFoundError(
        f"Regression matrix CSV not found (needed for coordinates and targets): {_matrix_csv}"
    )
_df_csv = pd.read_csv(_matrix_csv)
print(f"Loaded CSV: {_matrix_csv} | Rows: {len(_df_csv)}")

if "x" in _df_csv.columns and "y" in _df_csv.columns:
    _x_all = _df_csv["x"].to_numpy(dtype=np.float64)
    _y_all_coords = _df_csv["y"].to_numpy(dtype=np.float64)
    print(f"Coordinates: x [{_x_all.min():.0f}, {_x_all.max():.0f}]  "
          f"y [{_y_all_coords.min():.0f}, {_y_all_coords.max():.0f}]")
else:
    _x_all = np.zeros(len(X_all), dtype=np.float64)
    _y_all_coords = np.zeros(len(X_all), dtype=np.float64)
    print("[WARNING] x/y not found in CSV; spatial CV folds will not be spatially meaningful.")

# Apply predictor filters once — X is shared across all target runs
print(f"\nBefore predictor filter: X shape {X_all.shape}")
X_filtered, var_names_f = apply_predictor_filters(
    X_all, var_names,
    excluded_names=EXCLUDED_PREDICTORS,
    selected_names=SELECT_PREDICTORS,
    select_prefixes=SELECT_PREFIXES,
    exclude_prefixes=EXCLUDE_PREFIXES,
)
print(f"After predictor filter: X shape {X_filtered.shape}")

cat_lookup = find_categorical_indices(var_names_f, CATEGORICAL_PREDICTOR_NAMES)
cat_idx = sorted(cat_lookup.values())
print(f"Categorical predictors: {cat_lookup if cat_lookup else 'none'}")
X_filtered = coerce_categorical_columns(X_filtered, cat_idx)

_valid_x = ~np.all(~np.isfinite(X_filtered), axis=1)
X_filtered = np.nan_to_num(
    X_filtered[_valid_x], nan=-9999.0, posinf=-9999.0, neginf=-9999.0
).astype(np.float32)
_x_all = _x_all[_valid_x]
_y_all_coords = _y_all_coords[_valid_x]
if point_ids is not None:
    point_ids = point_ids[_valid_x]
_df_csv = _df_csv.iloc[_valid_x].reset_index(drop=True)
print(f"After finite-predictor filter: {X_filtered.shape[0]} rows")

# =============================================================================
# LOOP OVER TARGET VARIABLES
# =============================================================================

for TARGET_VARIABLE in TARGET_VARIABLES:
    print(f"\n{'='*60}")
    print(f"TARGET: {TARGET_VARIABLE}")
    print(f"{'='*60}")

    if TARGET_VARIABLE not in _df_csv.columns:
        print(f"[WARNING] '{TARGET_VARIABLE}' not found in CSV. Skipping.")
        continue

    # Extract and optionally transform the target
    y_all = _df_csv[TARGET_VARIABLE].to_numpy(dtype=np.float32)
    _per_target_transform = TARGET_TRANSFORMS.get(TARGET_VARIABLE, None)
    if _per_target_transform == "exp":
        print(f"Applying exp() to {TARGET_VARIABLE} (un-logging)")
        y_all = np.exp(y_all)
        target_label = f"exp({TARGET_VARIABLE})"
    elif _per_target_transform == "log":
        print(f"Applying log() to {TARGET_VARIABLE}")
        y_all = np.log(np.clip(y_all, a_min=1e-8, a_max=None))
        target_label = f"log({TARGET_VARIABLE})"
    elif USE_LOG_TARGET:
        print(f"Using log({TARGET_VARIABLE})")
        y_all = np.log(np.clip(y_all, a_min=1e-8, a_max=None))
        target_label = f"log({TARGET_VARIABLE})"
    else:
        target_label = TARGET_VARIABLE

    valid = np.isfinite(y_all)
    print(f"Finite y: {valid.sum()} / {len(y_all)}")
    X_cur       = X_filtered[valid]
    y_cur       = y_all[valid]
    x_cur       = _x_all[valid]
    y_coord_cur = _y_all_coords[valid]
    ids_cur     = point_ids[valid] if point_ids is not None else None

    if USE_AREA_FILTER:
        X_cur, y_cur, x_cur, y_coord_cur, ids_cur = filter_points_by_polygon(
            X_cur, y_cur, x_cur, y_coord_cur,
            crs_wkt=crs_wkt, transform=transform,
            polygon_path=AREA_POLYGON_PATH,
            polygon_layer=AREA_POLYGON_LAYER,
            mode=AREA_FILTER_MODE,
            point_ids=ids_cur,
        )
        print(f"After area filter: {X_cur.shape[0]} rows")

    # Subsample once with ALL predictors — column subsets are sliced per run
    X_sample_full, y_sample, x_sample, y_coord_sample, ids_sample = stratified_subsample_regression(
        X_cur, y_cur, x_cur, y_coord_cur,
        sample_size=SAMPLE_SIZE, seed=RANDOM_SEED, point_ids=ids_cur,
    )
    print(f"Sampled {len(y_sample)} points × {X_sample_full.shape[1]} predictors")

    if MULT_RUNS:
        # ── Run 1: all predictors ──────────────────────────────────────────
        full_out_dir = os.path.join(OUT_DIR, TARGET_VARIABLE, "All")
        print(f"\n{'─'*50}\nRun: all {len(var_names_f)} predictors  →  {full_out_dir}")
        imp_df = run_cv_and_save(
            X_sample_full, y_sample, x_sample, y_coord_sample, ids_sample,
            var_names_f, target_label, TARGET_VARIABLE, full_out_dir,
        )

        # ── Runs: top-N subsets from the full-run importances ──────────────
        if imp_df is not None:
            for top_n in MULT_RUNS_TOP_N:
                if top_n >= len(var_names_f):
                    print(f"[SKIP] top{top_n} >= total predictors ({len(var_names_f)})")
                    continue
                top_vars = [v for v in imp_df["Variable"].iloc[:top_n].tolist() if v in var_names_f]
                top_idx  = [var_names_f.index(v) for v in top_vars]
                topn_out = os.path.join(OUT_DIR, TARGET_VARIABLE, f"Top_{top_n}")
                print(f"\n{'─'*50}\nRun: top {len(top_vars)} predictors  →  {topn_out}")
                run_cv_and_save(
                    X_sample_full[:, top_idx], y_sample, x_sample, y_coord_sample, ids_sample,
                    top_vars, target_label, TARGET_VARIABLE, topn_out,
                )
        else:
            print("[WARNING] No importances from full run; skipping top-N runs.")

        print(f"\n✔ {TARGET_VARIABLE} multi-run outputs → {os.path.join(OUT_DIR, TARGET_VARIABLE)}/")

    else:
        # ── Single run with optional PRESELECT_TOP_N ──────────────────────
        target_out_dir = os.path.join(OUT_DIR, TARGET_VARIABLE)
        _imp_csv = os.path.join(target_out_dir, "variable_importance.csv")
        if PRESELECT_TOP_N is not None and os.path.exists(_imp_csv):
            _imp_df   = pd.read_csv(_imp_csv).head(PRESELECT_TOP_N)
            _keep     = [v for v in _imp_df["Variable"].tolist() if v in var_names_f]
            _keep_idx = [var_names_f.index(v) for v in _keep]
            X_run       = X_sample_full[:, _keep_idx]
            var_names_run = _keep
            print(f"[PRESELECT] top {len(_keep)} predictors from {_imp_csv}")
        else:
            X_run         = X_sample_full
            var_names_run = var_names_f
            if PRESELECT_TOP_N is not None:
                print(f"[PRESELECT] No importance CSV at {_imp_csv}; using all {len(var_names_f)} predictors.")

        run_cv_and_save(
            X_run, y_sample, x_sample, y_coord_sample, ids_sample,
            var_names_run, target_label, TARGET_VARIABLE, target_out_dir,
        )
        print(f"\n✔ {TARGET_VARIABLE} outputs → {target_out_dir}")

print("\n=== All target variables completed ===")

