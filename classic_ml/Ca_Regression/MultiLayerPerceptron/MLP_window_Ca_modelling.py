#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
XGBoost regression for Ca modelling from a prebuilt regression matrix.

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

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from affine import Affine

from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================

# USER SETTINGS
# =============================================================================


MATRIX_PATH = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\Regression_matrix\Regression_matrix_XRFtotal_lab\Regression_matrix_XRFtotal_lab_NoLOI\regression_matrix.npz"
OUT_DIR = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\XGBoost\Ca_conc_modelling\Model_XRF_total_Lab\No_LOI\Geo_topo_bedrock_modelling_tuned"

# List of predictors to exclude from the regression (by name)
EXCLUDED_PREDICTORS = ["Total_curvature_ch.tif","Hillshade_(MD)_ch.tif","Hillshade_(ZD)_ch.tif","Slope_ch.tif","Aspect(cos)_ch.tif","tri_ch.tif","VD1_ch.tif","slope.tif","Aspect(sin)_ch.tif",]  # Example: exclude specific predictors by nam


# === Target variables to run (each gets its own output subfolder and metrics) ===
TARGET_VARIABLES = ["KK","SS","KESP","CaO"]  # Add or remove variables here
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
PRESELECT_TOP_N = None    # baseline run uses all remaining predictors

# === Multi-run comparison mode ===
# MULT_RUNS=True : for each target, first run all predictors to build importances, then
#                  automatically rerun with each top-N subset. Outputs go to subfolders:
#                  OUT_DIR/<TARGET>/all/,  OUT_DIR/<TARGET>/top20/, ...
# MULT_RUNS=False: single run; if PRESELECT_TOP_N is set, filters using a prior
#                  importance CSV from OUT_DIR/<TARGET>/variable_importance.csv.
MULT_RUNS = False
MULT_RUNS_TOP_N = [20, 30, 40, 50, 60]  # top-N values to test when MULT_RUNS=True

USE_AREA_FILTER = False
AREA_POLYGON_PATH = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2025\kalk_prosjekt3.0\Mask\shape\Nordland_ed.shp"
AREA_POLYGON_LAYER = None
AREA_FILTER_MODE = "intersects"

SAMPLE_SIZE = None
N_SPLITS = 5
BLOCK_SIZE_M = 20000     # Spatial block size in metres (used when USE_SPATIAL_CV=True)
RANDOM_SEED = 42

N_ESTIMATORS = 1200
LEARNING_RATE = 0.03
MAX_DEPTH = 6
MIN_CHILD_WEIGHT = 5
SUBSAMPLE = 0.80
COLSAMPLE_BYTREE = 0.60
REG_ALPHA = 0.10
REG_LAMBDA = 2.0
GAMMA = 0.10
N_JOBS = -1

# === XGBoost tuning (step 1 before tail weighting) ===
ENABLE_XGB_TUNING = True
TUNING_MAX_TRIALS = 12
REUSE_BASELINE_TUNED_PARAMS = False

# === Tail weighting (deprecated for now) ===
# Weights are computed from the training-target rank so the extremes get more emphasis.
ENABLE_TAIL_WEIGHTING = False
TAIL_WEIGHT_STRENGTH = 1.0   # 1.0 = linear increase from center to tails
TAIL_WEIGHT_POWER = 1.0      # 1.0 keeps the effect linear; higher values sharpen the tails
TAIL_WEIGHT_CAP = 2.0        # Cap the maximum weight to avoid overfitting a few points

# === Post-hoc calibration (step 1 after tuned baseline) ===
ENABLE_POSTHOC_CALIBRATION = False
CALIBRATE_GLOBAL_LINEAR = True
CALIBRATE_TAIL_BIAS = True
CALIB_TAIL_Q_LOW = 0.10
CALIB_TAIL_Q_HIGH = 0.90

# === Two-stage residual correction (step 2) ===
# Stage 1: tuned XGBoost baseline (optionally calibrated)
# Stage 2: residual models fitted on low/high tails and applied only on tail-like predictions
ENABLE_TWO_STAGE_RESIDUAL = False
STAGE2_TAIL_Q_LOW = 0.10
STAGE2_TAIL_Q_HIGH = 0.90
STAGE2_MIN_TAIL_SAMPLES = 300
STAGE2_XGB_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "gamma": 0.1,
}

# === Quantile pilot (step 3) ===
ENABLE_QUANTILE_MODE = False
QUANTILE_LOW = 0.10
QUANTILE_HIGH = 0.90
QUANTILE_N_ESTIMATORS = 500
QUANTILE_LEARNING_RATE = 0.03
QUANTILE_MAX_DEPTH = 3
QUANTILE_MIN_SAMPLES_LEAF = 20

# Run only one selected subset per target for the tail-weighted pass.
RUN_SELECTED_SUBSETS_ONLY = False
SELECTED_SUBSET_BY_TARGET = {
    "KK": "Top_60",
    "SS": "Top_60",
    "KESP": "Top_50",
    "CaO": "Top_60",
    "CaMg_LOI_v": "All",
    "Comp_1": "Top_60",
}

# Optional list of predictors to coerce as categorical (round, clamp negatives, fill NaN).
# Keep empty when quaternary/landuse are already represented as binary maps.
CATEGORICAL_PREDICTOR_NAMES = []

# === Load predictors from predictor_names.csv ===
PREDICTOR_NAMES_PATH = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\Regression_matrix\Regression_matrix_XRFtotal_lab\Regression_matrix_XRFtotal_lab_NoLOI\predictor_names.csv"
_predictor_df = pd.read_csv(PREDICTOR_NAMES_PATH)
PREDICTOR_LIST = _predictor_df["predictor_name"].tolist()


## Interactive predictor exclusion removed: only EXCLUDED_PREDICTORS is used for exclusion.

SELECT_PREDICTORS = PREDICTOR_LIST
SELECT_PREFIXES = None
EXCLUDE_PREFIXES = ["quaternary_", "landuse_", "alphaearth_", "KalsiumElvInnsjo4_", "BioK_"]

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


def _env_float(name, default):
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default


def _env_bool(name, default):
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    val = str(raw).strip().lower()
    if val in {"1", "true", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "no", "n", "off"}:
        return False
    return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_targets(default_targets):
    raw = os.getenv("XGB_TARGETS")
    if raw is None or str(raw).strip() == "":
        return default_targets
    return [t.strip() for t in raw.split(",") if t.strip()]


TARGET_VARIABLES = _env_targets(TARGET_VARIABLES)
TAIL_WEIGHT_STRENGTH = _env_float("XGB_TAIL_STRENGTH", TAIL_WEIGHT_STRENGTH)
TAIL_WEIGHT_POWER = _env_float("XGB_TAIL_POWER", TAIL_WEIGHT_POWER)
TAIL_WEIGHT_CAP = _env_float("XGB_TAIL_CAP", TAIL_WEIGHT_CAP)
TAIL_TAG = os.getenv("XGB_TAIL_TAG", "").strip()
CALIB_TAG = os.getenv("XGB_CALIB_TAG", "").strip()
STAGE2_TAG = os.getenv("XGB_STAGE2_TAG", "").strip()
QUANTILE_TAG = os.getenv("XGB_QUANTILE_TAG", "").strip()
ENABLE_TWO_STAGE_RESIDUAL = _env_bool("XGB_ENABLE_TWO_STAGE", ENABLE_TWO_STAGE_RESIDUAL)
ENABLE_POSTHOC_CALIBRATION = _env_bool("XGB_ENABLE_POSTHOC", ENABLE_POSTHOC_CALIBRATION)
ENABLE_TAIL_WEIGHTING = _env_bool("XGB_ENABLE_TAIL_WEIGHT", ENABLE_TAIL_WEIGHTING)
ENABLE_QUANTILE_MODE = _env_bool("XGB_ENABLE_QUANTILE", ENABLE_QUANTILE_MODE)

if ENABLE_TAIL_WEIGHTING:
    ACTIVE_OUT_DIR = os.path.join(OUT_DIR, f"TailWeight_{TAIL_TAG}") if TAIL_TAG else os.path.join(OUT_DIR, "TailWeight")
elif ENABLE_TWO_STAGE_RESIDUAL:
    ACTIVE_OUT_DIR = os.path.join(OUT_DIR, f"TwoStageResidual_{STAGE2_TAG}") if STAGE2_TAG else os.path.join(OUT_DIR, "TwoStageResidual")
elif ENABLE_POSTHOC_CALIBRATION:
    ACTIVE_OUT_DIR = os.path.join(OUT_DIR, f"PostHocCalibrated_{CALIB_TAG}") if CALIB_TAG else os.path.join(OUT_DIR, "PostHocCalibrated")
elif ENABLE_QUANTILE_MODE:
    ACTIVE_OUT_DIR = os.path.join(OUT_DIR, f"QuantilePilot_{QUANTILE_TAG}") if QUANTILE_TAG else os.path.join(OUT_DIR, "QuantilePilot")
else:
    ACTIVE_OUT_DIR = OUT_DIR
os.makedirs(ACTIVE_OUT_DIR, exist_ok=True)

DEFAULT_XGB_PARAMS = {
    "n_estimators": N_ESTIMATORS,
    "learning_rate": LEARNING_RATE,
    "max_depth": MAX_DEPTH,
    "min_child_weight": MIN_CHILD_WEIGHT,
    "subsample": SUBSAMPLE,
    "colsample_bytree": COLSAMPLE_BYTREE,
    "reg_alpha": REG_ALPHA,
    "reg_lambda": REG_LAMBDA,
    "gamma": GAMMA,
}

TUNED_XGB_PARAMS_BY_TARGET: Dict[str, Dict[str, float]] = {}


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


def summarize_predictor_filters(
    var_names,
    excluded_names=None,
    selected_names=None,
    select_prefixes=None,
    exclude_prefixes=None,
):
    names_norm = [normalize_name(v) for v in var_names]
    norm_to_name = {normalize_name(v): v for v in var_names}

    selected_name_missing = []
    excluded_name_matches = []
    excluded_name_missing = []
    select_prefix_rows = []
    exclude_prefix_rows = []

    if selected_names:
        for name in selected_names:
            if normalize_name(name) not in norm_to_name:
                selected_name_missing.append(name)

    if select_prefixes:
        for prefix in select_prefixes:
            norm_prefix = normalize_name(prefix)
            matched = [name for name, norm_name in zip(var_names, names_norm) if norm_name.startswith(norm_prefix)]
            select_prefix_rows.append({
                "prefix": prefix,
                "match_count": len(matched),
                "matched_predictors": " | ".join(matched),
            })

    if excluded_names:
        for name in excluded_names:
            norm_name = normalize_name(name)
            if norm_name in norm_to_name:
                excluded_name_matches.append(norm_to_name[norm_name])
            else:
                excluded_name_missing.append(name)

    if exclude_prefixes:
        for prefix in exclude_prefixes:
            norm_prefix = normalize_name(prefix)
            matched = [name for name, norm_name in zip(var_names, names_norm) if norm_name.startswith(norm_prefix)]
            exclude_prefix_rows.append({
                "prefix": prefix,
                "match_count": len(matched),
                "matched_predictors": " | ".join(matched),
            })

    return {
        "selected_name_missing": selected_name_missing,
        "excluded_name_matches": excluded_name_matches,
        "excluded_name_missing": excluded_name_missing,
        "select_prefix_rows": select_prefix_rows,
        "exclude_prefix_rows": exclude_prefix_rows,
    }


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


def build_xgb_tuning_candidates():
    # Compact first-pass search around current baseline, informed by RF/XGB behavior.
    candidates = [
        {"n_estimators": 1000, "learning_rate": 0.04, "max_depth": 5, "min_child_weight": 4,
         "subsample": 0.8, "colsample_bytree": 0.6, "reg_alpha": 0.05, "reg_lambda": 2.0, "gamma": 0.0},
        {"n_estimators": 1200, "learning_rate": 0.03, "max_depth": 6, "min_child_weight": 5,
         "subsample": 0.8, "colsample_bytree": 0.6, "reg_alpha": 0.10, "reg_lambda": 2.0, "gamma": 0.10},
        {"n_estimators": 1400, "learning_rate": 0.025, "max_depth": 6, "min_child_weight": 6,
         "subsample": 0.85, "colsample_bytree": 0.65, "reg_alpha": 0.10, "reg_lambda": 2.5, "gamma": 0.15},
        {"n_estimators": 1600, "learning_rate": 0.02, "max_depth": 7, "min_child_weight": 8,
         "subsample": 0.8, "colsample_bytree": 0.7, "reg_alpha": 0.20, "reg_lambda": 3.0, "gamma": 0.20},
        {"n_estimators": 900, "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 3,
         "subsample": 0.75, "colsample_bytree": 0.55, "reg_alpha": 0.00, "reg_lambda": 1.5, "gamma": 0.0},
        {"n_estimators": 1100, "learning_rate": 0.035, "max_depth": 5, "min_child_weight": 5,
         "subsample": 0.85, "colsample_bytree": 0.7, "reg_alpha": 0.05, "reg_lambda": 2.0, "gamma": 0.05},
        {"n_estimators": 1300, "learning_rate": 0.03, "max_depth": 7, "min_child_weight": 10,
         "subsample": 0.8, "colsample_bytree": 0.6, "reg_alpha": 0.30, "reg_lambda": 4.0, "gamma": 0.20},
        {"n_estimators": 1000, "learning_rate": 0.04, "max_depth": 6, "min_child_weight": 4,
         "subsample": 0.9, "colsample_bytree": 0.8, "reg_alpha": 0.00, "reg_lambda": 1.0, "gamma": 0.0},
        {"n_estimators": 1400, "learning_rate": 0.025, "max_depth": 5, "min_child_weight": 6,
         "subsample": 0.9, "colsample_bytree": 0.75, "reg_alpha": 0.10, "reg_lambda": 2.0, "gamma": 0.10},
        {"n_estimators": 1200, "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 8,
         "subsample": 0.8, "colsample_bytree": 0.5, "reg_alpha": 0.20, "reg_lambda": 3.0, "gamma": 0.15},
        {"n_estimators": 1500, "learning_rate": 0.02, "max_depth": 6, "min_child_weight": 12,
         "subsample": 0.7, "colsample_bytree": 0.6, "reg_alpha": 0.30, "reg_lambda": 5.0, "gamma": 0.25},
        {"n_estimators": 1000, "learning_rate": 0.045, "max_depth": 5, "min_child_weight": 4,
         "subsample": 0.85, "colsample_bytree": 0.65, "reg_alpha": 0.05, "reg_lambda": 1.5, "gamma": 0.05},
    ]
    return candidates[:TUNING_MAX_TRIALS]


def get_xgb_params_for_target(target_variable: Optional[str] = None):
    params = dict(DEFAULT_XGB_PARAMS)
    if target_variable in TUNED_XGB_PARAMS_BY_TARGET:
        params.update(TUNED_XGB_PARAMS_BY_TARGET[target_variable])
    return params
def build_tail_sample_weights(y_train, strength=1.0, power=1.0, cap=2.0):
    if strength <= 0:
        return None
    y_arr = np.asarray(y_train, dtype=np.float64)
    ranks = pd.Series(y_arr).rank(method="average", pct=True).to_numpy(dtype=np.float64)
    tail_distance = np.abs(ranks - 0.5) / 0.5
    weights = 1.0 + strength * np.power(tail_distance, power)
    if cap is not None:
        weights = np.minimum(weights, float(cap))
    return weights.astype(np.float32)


def fit_xgb_regression(X_train, y_train, random_state=42, xgb_params=None, sample_weight=None):
    params = dict(DEFAULT_XGB_PARAMS)
    if xgb_params:
        params.update(xgb_params)
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=int(params["n_estimators"]),
        learning_rate=params["learning_rate"],
        max_depth=int(params["max_depth"]),
        min_child_weight=int(params["min_child_weight"]),
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        gamma=params["gamma"],
        random_state=random_state,
        n_jobs=N_JOBS,
        tree_method="hist",
        verbosity=0,
    )
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    return model


def empirical_prediction_intervals(model, X_train, y_train, X_eval, q_low=0.10, q_high=0.90):
    # XGBoost does not expose tree-wise predictions in sklearn API; use empirical residual quantiles.
    pred_train = model.predict(X_train)
    residuals = y_train - pred_train
    low_res = float(np.quantile(residuals, q_low))
    high_res = float(np.quantile(residuals, q_high))
    pred_mean = model.predict(X_eval)
    pred_low = pred_mean + low_res
    pred_high = pred_mean + high_res
    pred_width = pred_high - pred_low
    return pred_mean, pred_low, pred_high, pred_width


def tree_prediction_intervals(model, X, q_low=0.10, q_high=0.90):
    # Kept for compatibility. For XGBoost, prediction intervals are computed empirically.
    pred_mean = model.predict(X)
    pred_low = np.full_like(pred_mean, np.nan, dtype=np.float32)
    pred_high = np.full_like(pred_mean, np.nan, dtype=np.float32)
    pred_width = np.full_like(pred_mean, np.nan, dtype=np.float32)
    return pred_mean, pred_low, pred_high, pred_width


def fit_rf_regression(X_train, y_train, random_state=42, target_variable=None, sample_weight=None):
    # Backward-compatible wrapper used by the existing pipeline.
    xgb_params = get_xgb_params_for_target(target_variable)
    return fit_xgb_regression(
        X_train, y_train,
        random_state=random_state,
        xgb_params=xgb_params,
        sample_weight=sample_weight,
    )


def tune_xgb_params_for_target(X_samp, y_samp, x_samp, y_coord_samp, target_variable, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    candidates = build_xgb_tuning_candidates()
    if not candidates:
        return None

    if USE_SPATIAL_CV:
        fold_labels = make_spatial_block_folds(x_samp, y_coord_samp, N_SPLITS, BLOCK_SIZE_M, RANDOM_SEED)
        fold_iter = [
            (np.where(fold_labels != f)[0], np.where(fold_labels == f)[0])
            for f in range(N_SPLITS)
        ]
    else:
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        fold_iter = list(kf.split(X_samp, y_samp))

    trial_rows = []
    best = None
    best_key = (-np.inf, np.inf)

    print(f"\n=== TUNING XGBOOST ({target_variable}) | trials={len(candidates)} ===")
    for i, params in enumerate(candidates, start=1):
        fold_scores = []
        fold_rmse = []
        for fold_id, (tr, te) in enumerate(fold_iter, start=1):
            sample_weight = None
            if ENABLE_TAIL_WEIGHTING:
                sample_weight = build_tail_sample_weights(
                    y_samp[tr],
                    strength=TAIL_WEIGHT_STRENGTH,
                    power=TAIL_WEIGHT_POWER,
                    cap=TAIL_WEIGHT_CAP,
                )
            model = fit_xgb_regression(
                X_samp[tr], y_samp[tr],
                random_state=RANDOM_SEED + 1000 + i * 10 + fold_id,
                xgb_params=params,
                sample_weight=sample_weight,
            )
            pred = model.predict(X_samp[te])
            m = regression_metrics(y_samp[te], pred)
            fold_scores.append(m["R2"])
            fold_rmse.append(m["RMSE"])

        mean_r2 = float(np.mean(fold_scores))
        mean_rmse = float(np.mean(fold_rmse))
        row = {"trial": i, "mean_r2": mean_r2, "mean_rmse": mean_rmse, **params}
        trial_rows.append(row)
        print(f"[Tune {i:02d}] R2={mean_r2:.4f} | RMSE={mean_rmse:.4f}")

        key = (mean_r2, -mean_rmse)
        if key > best_key:
            best_key = key
            best = params

    trials_df = pd.DataFrame(trial_rows).sort_values(["mean_r2", "mean_rmse"], ascending=[False, True])
    trials_df.to_csv(os.path.join(out_dir, "xgb_tuning_trials.csv"), index=False)

    if best is not None:
        with open(os.path.join(out_dir, "xgb_best_params.txt"), "w", encoding="utf-8") as f:
            f.write(f"Target: {target_variable}\n")
            f.write(f"Best mean CV R2: {best_key[0]:.6f}\n")
            f.write(f"Best mean CV RMSE: {-best_key[1]:.6f}\n")
            for k, v in best.items():
                f.write(f"{k}: {v}\n")
        print(f"Best params ({target_variable}): R2={best_key[0]:.4f} | RMSE={-best_key[1]:.4f}")

    return best


def _deprecated_tree_prediction_intervals(model, X, q_low=0.10, q_high=0.90):
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


def fit_posthoc_calibrator(y_true_train, y_pred_train):
    y_true = np.asarray(y_true_train, dtype=np.float64)
    y_pred = np.asarray(y_pred_train, dtype=np.float64)

    slope = 1.0
    intercept = 0.0
    if CALIBRATE_GLOBAL_LINEAR and np.std(y_pred) > 1e-12:
        slope, intercept = np.polyfit(y_pred, y_true, 1)

    pred_cal = slope * y_pred + intercept
    low_thr = float(np.quantile(pred_cal, CALIB_TAIL_Q_LOW))
    high_thr = float(np.quantile(pred_cal, CALIB_TAIL_Q_HIGH))

    low_offset = 0.0
    high_offset = 0.0
    if CALIBRATE_TAIL_BIAS:
        low_mask = pred_cal <= low_thr
        high_mask = pred_cal >= high_thr
        if np.any(low_mask):
            low_offset = float(np.mean(y_true[low_mask] - pred_cal[low_mask]))
        if np.any(high_mask):
            high_offset = float(np.mean(y_true[high_mask] - pred_cal[high_mask]))

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "low_thr": low_thr,
        "high_thr": high_thr,
        "low_offset": low_offset,
        "high_offset": high_offset,
    }


def apply_posthoc_calibration(y_pred, calibrator):
    pred = np.asarray(y_pred, dtype=np.float64)
    out = calibrator["slope"] * pred + calibrator["intercept"]
    low_mask = out <= calibrator["low_thr"]
    high_mask = out >= calibrator["high_thr"]
    out[low_mask] += calibrator["low_offset"]
    out[high_mask] += calibrator["high_offset"]
    return out.astype(np.float32)


def compute_tail_metrics(y_true, y_pred, q_low=0.10, q_high=0.90):
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    lo = float(np.quantile(y_t, q_low))
    hi = float(np.quantile(y_t, q_high))
    low_mask = y_t <= lo
    high_mask = y_t >= hi

    def _safe_stats(mask):
        if not np.any(mask):
            return np.nan, np.nan, np.nan
        err = y_p[mask] - y_t[mask]
        rmse = float(np.sqrt(np.mean(np.square(err))))
        mae = float(np.mean(np.abs(err)))
        bias = float(np.mean(err))
        return rmse, mae, bias

    rmse_l, mae_l, bias_l = _safe_stats(low_mask)
    rmse_h, mae_h, bias_h = _safe_stats(high_mask)
    return {
        "TailLow_RMSE": rmse_l,
        "TailLow_MAE": mae_l,
        "TailLow_Bias": bias_l,
        "TailHigh_RMSE": rmse_h,
        "TailHigh_MAE": mae_h,
        "TailHigh_Bias": bias_h,
    }


def fit_quantile_regressor(X_train, y_train, quantile, random_state):
    model = GradientBoostingRegressor(
        loss="quantile",
        alpha=quantile,
        n_estimators=QUANTILE_N_ESTIMATORS,
        learning_rate=QUANTILE_LEARNING_RATE,
        max_depth=QUANTILE_MAX_DEPTH,
        min_samples_leaf=QUANTILE_MIN_SAMPLES_LEAF,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def interval_metrics(y_true, q_low_pred, q_high_pred, q_low=0.10, q_high=0.90):
    y_t = np.asarray(y_true, dtype=np.float64)
    ql = np.asarray(q_low_pred, dtype=np.float64)
    qh = np.asarray(q_high_pred, dtype=np.float64)
    lo = np.minimum(ql, qh)
    hi = np.maximum(ql, qh)
    inside = (y_t >= lo) & (y_t <= hi)
    width = hi - lo

    low_thr = float(np.quantile(y_t, q_low))
    high_thr = float(np.quantile(y_t, q_high))
    low_mask = y_t <= low_thr
    high_mask = y_t >= high_thr

    return {
        "PICP": float(np.mean(inside)),
        "MPIW": float(np.mean(width)),
        "PICP_LowTail": float(np.mean(inside[low_mask])) if np.any(low_mask) else np.nan,
        "PICP_HighTail": float(np.mean(inside[high_mask])) if np.any(high_mask) else np.nan,
    }


def fit_two_stage_residual_models(X_train, y_true_train, y_pred_stage1):
    y_true = np.asarray(y_true_train, dtype=np.float32)
    y_pred = np.asarray(y_pred_stage1, dtype=np.float32)
    residual = y_true - y_pred

    y_low_thr = float(np.quantile(y_true, STAGE2_TAIL_Q_LOW))
    y_high_thr = float(np.quantile(y_true, STAGE2_TAIL_Q_HIGH))
    low_train_mask = y_true <= y_low_thr
    high_train_mask = y_true >= y_high_thr

    pred_low_thr = float(np.quantile(y_pred, STAGE2_TAIL_Q_LOW))
    pred_high_thr = float(np.quantile(y_pred, STAGE2_TAIL_Q_HIGH))

    low_model = None
    high_model = None

    if int(np.sum(low_train_mask)) >= STAGE2_MIN_TAIL_SAMPLES:
        low_model = fit_xgb_regression(
            X_train[low_train_mask], residual[low_train_mask],
            random_state=RANDOM_SEED + 7001,
            xgb_params=STAGE2_XGB_PARAMS,
            sample_weight=None,
        )

    if int(np.sum(high_train_mask)) >= STAGE2_MIN_TAIL_SAMPLES:
        high_model = fit_xgb_regression(
            X_train[high_train_mask], residual[high_train_mask],
            random_state=RANDOM_SEED + 7002,
            xgb_params=STAGE2_XGB_PARAMS,
            sample_weight=None,
        )

    summary = {
        "y_low_thr": y_low_thr,
        "y_high_thr": y_high_thr,
        "pred_low_thr": pred_low_thr,
        "pred_high_thr": pred_high_thr,
        "n_low_train": int(np.sum(low_train_mask)),
        "n_high_train": int(np.sum(high_train_mask)),
        "low_model_fit": bool(low_model is not None),
        "high_model_fit": bool(high_model is not None),
    }
    return {"low_model": low_model, "high_model": high_model, "summary": summary}


def apply_two_stage_residual_correction(X_eval, y_pred_stage1, stage2_models):
    pred = np.asarray(y_pred_stage1, dtype=np.float32).copy()
    summary = stage2_models["summary"]
    low_model = stage2_models["low_model"]
    high_model = stage2_models["high_model"]

    low_apply_mask = pred <= summary["pred_low_thr"]
    high_apply_mask = pred >= summary["pred_high_thr"]

    if low_model is not None and np.any(low_apply_mask):
        pred[low_apply_mask] += low_model.predict(X_eval[low_apply_mask]).astype(np.float32)

    if high_model is not None and np.any(high_apply_mask):
        pred[high_apply_mask] += high_model.predict(X_eval[high_apply_mask]).astype(np.float32)

    return pred


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
    active_params = get_xgb_params_for_target(TARGET_VARIABLE)
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

    fold_rows_list, importances_list, pred_rows_list, calibrator_rows, stage2_rows = [], [], [], [], []
    for fold_id, (tr, te) in enumerate(fold_iter, start=1):
        print(f"\n--- Fold {fold_id}/{N_SPLITS} | train={len(tr)} test={len(te)} ---")
        sample_weight = None
        if ENABLE_TAIL_WEIGHTING:
            sample_weight = build_tail_sample_weights(
                y_samp[tr],
                strength=TAIL_WEIGHT_STRENGTH,
                power=TAIL_WEIGHT_POWER,
                cap=TAIL_WEIGHT_CAP,
            )
        model = fit_rf_regression(
            X_samp[tr], y_samp[tr],
            random_state=RANDOM_SEED + fold_id,
            target_variable=TARGET_VARIABLE,
            sample_weight=sample_weight,
        )
        y_pred_raw = model.predict(X_samp[te])
        y_pred = y_pred_raw.copy()
        y_train_pred_stage1 = model.predict(X_samp[tr])

        if ENABLE_POSTHOC_CALIBRATION:
            calibrator = fit_posthoc_calibrator(y_samp[tr], y_train_pred_stage1)
            y_train_pred_stage1 = apply_posthoc_calibration(y_train_pred_stage1, calibrator)
            y_pred = apply_posthoc_calibration(y_pred_raw, calibrator)
            calibrator_rows.append({"Fold": fold_id, **calibrator})

        if ENABLE_TWO_STAGE_RESIDUAL:
            stage2_models = fit_two_stage_residual_models(X_samp[tr], y_samp[tr], y_train_pred_stage1)
            y_pred = apply_two_stage_residual_correction(X_samp[te], y_pred, stage2_models)
            stage2_rows.append({"Fold": fold_id, **stage2_models["summary"]})

        q_low_pred = None
        q_high_pred = None
        if ENABLE_QUANTILE_MODE:
            q_low_model = fit_quantile_regressor(
                X_samp[tr], y_samp[tr], QUANTILE_LOW,
                random_state=RANDOM_SEED + 8000 + fold_id,
            )
            q_high_model = fit_quantile_regressor(
                X_samp[tr], y_samp[tr], QUANTILE_HIGH,
                random_state=RANDOM_SEED + 9000 + fold_id,
            )
            q_low_pred = q_low_model.predict(X_samp[te]).astype(np.float32)
            q_high_pred = q_high_model.predict(X_samp[te]).astype(np.float32)

        metrics = regression_metrics(y_samp[te], y_pred)
        metrics.update(compute_tail_metrics(y_samp[te], y_pred, CALIB_TAIL_Q_LOW, CALIB_TAIL_Q_HIGH))
        if ENABLE_QUANTILE_MODE and q_low_pred is not None and q_high_pred is not None:
            metrics.update(interval_metrics(y_samp[te], q_low_pred, q_high_pred, QUANTILE_LOW, QUANTILE_HIGH))
        metrics.update({"Fold": fold_id, "N_train": len(tr), "N_test": len(te)})
        fold_rows_list.append(metrics)
        if hasattr(model, "feature_importances_"):
            importances_list.append(model.feature_importances_.copy())
        pred_df = pd.DataFrame({
            "fold": fold_id, "y_true": y_samp[te], "y_pred_raw": y_pred_raw, "y_pred": y_pred,
            "x": x_samp[te], "y": y_coord_samp[te],
        })
        if ENABLE_QUANTILE_MODE and q_low_pred is not None and q_high_pred is not None:
            pred_df[f"y_pred_q{int(QUANTILE_LOW*100)}"] = q_low_pred
            pred_df[f"y_pred_q{int(QUANTILE_HIGH*100)}"] = q_high_pred
            pred_df["pred_interval_width"] = np.maximum(q_low_pred, q_high_pred) - np.minimum(q_low_pred, q_high_pred)
        if ids_samp is not None:
            pred_df["point_id"] = ids_samp[te]
        pred_rows_list.append(pred_df)
        print(f"RMSE={metrics['RMSE']:.4f} | MAE={metrics['MAE']:.4f} | "
              f"R2={metrics['R2']:.4f} | Bias={metrics['Bias']:.4f}")

    fold_df    = pd.DataFrame(fold_rows_list)
    pred_cv_df = pd.concat(pred_rows_list, ignore_index=True)
    fold_df.to_csv(os.path.join(out_dir, "cv_fold_metrics.csv"), index=False)
    pred_cv_df.to_csv(os.path.join(out_dir, "cv_predictions.csv"), index=False)
    if calibrator_rows:
        pd.DataFrame(calibrator_rows).to_csv(os.path.join(out_dir, "cv_posthoc_calibrators.csv"), index=False)
    if stage2_rows:
        pd.DataFrame(stage2_rows).to_csv(os.path.join(out_dir, "cv_two_stage_models.csv"), index=False)

    main_metric_cols = ["RMSE", "MAE", "R2", "Bias"]
    interval_metric_cols = ["PICP", "MPIW", "PICP_LowTail", "PICP_HighTail"]
    tail_metric_cols = [
        "TailLow_RMSE", "TailLow_MAE", "TailLow_Bias",
        "TailHigh_RMSE", "TailHigh_MAE", "TailHigh_Bias",
    ]
    mean_metrics = fold_df[main_metric_cols].mean().to_dict()
    std_metrics  = fold_df[main_metric_cols].std().to_dict()
    mean_tail_metrics = fold_df[tail_metric_cols].mean().to_dict() if all(c in fold_df.columns for c in tail_metric_cols) else {}
    mean_interval_metrics = fold_df[interval_metric_cols].mean().to_dict() if all(c in fold_df.columns for c in interval_metric_cols) else {}
    print(f"\nCV [{TARGET_VARIABLE} | {n_preds} preds]: "
          f"R2={mean_metrics['R2']:.4f}±{std_metrics['R2']:.4f} | "
          f"RMSE={mean_metrics['RMSE']:.4f}±{std_metrics['RMSE']:.4f}")

    with open(os.path.join(out_dir, "regression_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"XGBoost regression — target: {TARGET_VARIABLE} ({target_label})\n")
        f.write("=" * 60 + "\n")
        f.write(f"CV strategy: {cv_label}\n")
        f.write(f"Predictors used: {n_preds}\n")
        f.write(f"MATRIX_PATH: {MATRIX_PATH}\n")
        f.write(f"SAMPLE_SIZE: {SAMPLE_SIZE}\n")
        f.write(f"N_SPLITS: {N_SPLITS}\n")
        f.write(f"N_ESTIMATORS: {active_params['n_estimators']}\n")
        f.write(f"LEARNING_RATE: {active_params['learning_rate']}\n")
        f.write(f"MAX_DEPTH: {active_params['max_depth']}\n")
        f.write(f"MIN_CHILD_WEIGHT: {active_params['min_child_weight']}\n")
        f.write(f"SUBSAMPLE: {active_params['subsample']}\n")
        f.write(f"COLSAMPLE_BYTREE: {active_params['colsample_bytree']}\n")
        f.write(f"REG_ALPHA: {active_params['reg_alpha']}\n")
        f.write(f"REG_LAMBDA: {active_params['reg_lambda']}\n")
        f.write(f"GAMMA: {active_params['gamma']}\n\n")
        f.write("Mean CV metrics\n")
        for k, v in mean_metrics.items():
            f.write(f"{k}: {v:.6f}\n")
        f.write("\nSD CV metrics\n")
        for k, v in std_metrics.items():
            f.write(f"{k}: {v:.6f}\n")
        if mean_tail_metrics:
            f.write("\nMean tail metrics\n")
            for k, v in mean_tail_metrics.items():
                f.write(f"{k}: {v:.6f}\n")
        if mean_interval_metrics:
            f.write("\nMean interval metrics\n")
            for k, v in mean_interval_metrics.items():
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
    final_model = fit_rf_regression(
        X_samp, y_samp,
        random_state=RANDOM_SEED,
        target_variable=TARGET_VARIABLE,
        sample_weight=(build_tail_sample_weights(
            y_samp,
            strength=TAIL_WEIGHT_STRENGTH,
            power=TAIL_WEIGHT_POWER,
            cap=TAIL_WEIGHT_CAP,
        ) if ENABLE_TAIL_WEIGHTING else None),
    )
    final_pred_raw = final_model.predict(X_samp)
    final_pred = final_pred_raw.copy()
    final_pred_stage1 = final_pred_raw.copy()
    final_calibrator = None
    if ENABLE_POSTHOC_CALIBRATION:
        final_calibrator = fit_posthoc_calibrator(y_samp, final_pred_raw)
        final_pred = apply_posthoc_calibration(final_pred_raw, final_calibrator)
        final_pred_stage1 = final_pred.copy()

    final_stage2_models = None
    if ENABLE_TWO_STAGE_RESIDUAL:
        final_stage2_models = fit_two_stage_residual_models(X_samp, y_samp, final_pred_stage1)
        final_pred = apply_two_stage_residual_correction(X_samp, final_pred, final_stage2_models)

    final_q_low = None
    final_q_high = None
    if ENABLE_QUANTILE_MODE:
        final_q_low_model = fit_quantile_regressor(
            X_samp, y_samp, QUANTILE_LOW,
            random_state=RANDOM_SEED + 8100,
        )
        final_q_high_model = fit_quantile_regressor(
            X_samp, y_samp, QUANTILE_HIGH,
            random_state=RANDOM_SEED + 9100,
        )
        final_q_low = final_q_low_model.predict(X_samp).astype(np.float32)
        final_q_high = final_q_high_model.predict(X_samp).astype(np.float32)

    final_metrics_dict = regression_metrics(y_samp, final_pred)
    with open(os.path.join(out_dir, "final_model_train_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Training metrics — target: {TARGET_VARIABLE}\n{'='*50}\n")
        for k, v in final_metrics_dict.items():
            f.write(f"{k}: {v:.6f}\n")

    if SAVE_PREDICTION_INTERVALS:
        pred_mean_arr, pred_low, pred_high, pred_width = empirical_prediction_intervals(
            final_model, X_samp, y_samp, X_samp, q_low=LOWER_Q, q_high=UPPER_Q
        )
        if final_calibrator is not None:
            pred_mean_arr = apply_posthoc_calibration(pred_mean_arr, final_calibrator)
            pred_low = apply_posthoc_calibration(pred_low, final_calibrator)
            pred_high = apply_posthoc_calibration(pred_high, final_calibrator)
            pred_width = pred_high - pred_low
        if final_stage2_models is not None:
            pred_mean_arr = apply_two_stage_residual_correction(X_samp, pred_mean_arr, final_stage2_models)
            pred_low = apply_two_stage_residual_correction(X_samp, pred_low, final_stage2_models)
            pred_high = apply_two_stage_residual_correction(X_samp, pred_high, final_stage2_models)
            pred_width = pred_high - pred_low
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

    if ENABLE_QUANTILE_MODE and final_q_low is not None and final_q_high is not None:
        q_df = pd.DataFrame({
            "x": x_samp,
            "y": y_coord_samp,
            "y_true": y_samp,
            f"pred_q{int(QUANTILE_LOW*100)}": final_q_low,
            f"pred_q{int(QUANTILE_HIGH*100)}": final_q_high,
            "pred_quantile_width": np.maximum(final_q_low, final_q_high) - np.minimum(final_q_low, final_q_high),
        })
        if ids_samp is not None:
            q_df["point_id"] = ids_samp
        q_df.to_csv(os.path.join(out_dir, "final_model_quantile_predictions.csv"), index=False)

    try:
        import joblib
        model_path = os.path.join(out_dir, f"xgb_final_model_{TARGET_VARIABLE}.joblib")
        joblib.dump({
            "model": final_model,
            "model_type": "xgboost",
            "var_names": var_names_this,
            "target_variable": TARGET_VARIABLE,
            "use_log_target": USE_LOG_TARGET,
            "target_transform": TARGET_TRANSFORMS.get(TARGET_VARIABLE, None),
            "categorical_predictors": find_categorical_indices(
                var_names_this, CATEGORICAL_PREDICTOR_NAMES
            ),
            "matrix_path": MATRIX_PATH,
            "xgb_params": {
                "n_estimators": active_params["n_estimators"],
                "learning_rate": active_params["learning_rate"],
                "max_depth": active_params["max_depth"],
                "min_child_weight": active_params["min_child_weight"],
                "subsample": active_params["subsample"],
                "colsample_bytree": active_params["colsample_bytree"],
                "reg_alpha": active_params["reg_alpha"],
                "reg_lambda": active_params["reg_lambda"],
                "gamma": active_params["gamma"],
                "tree_method": "hist",
            },
            "prediction_interval_quantiles": [LOWER_Q, UPPER_Q],
            "posthoc_calibration": {
                "enabled": ENABLE_POSTHOC_CALIBRATION,
                "global_linear": CALIBRATE_GLOBAL_LINEAR,
                "tail_bias": CALIBRATE_TAIL_BIAS,
                "tail_q_low": CALIB_TAIL_Q_LOW,
                "tail_q_high": CALIB_TAIL_Q_HIGH,
                "params": final_calibrator,
            },
            "two_stage_residual": {
                "enabled": ENABLE_TWO_STAGE_RESIDUAL,
                "tail_q_low": STAGE2_TAIL_Q_LOW,
                "tail_q_high": STAGE2_TAIL_Q_HIGH,
                "min_tail_samples": STAGE2_MIN_TAIL_SAMPLES,
                "params": STAGE2_XGB_PARAMS,
                "summary": (final_stage2_models["summary"] if final_stage2_models is not None else None),
            },
            "quantile_mode": {
                "enabled": ENABLE_QUANTILE_MODE,
                "q_low": QUANTILE_LOW,
                "q_high": QUANTILE_HIGH,
                "model": "GradientBoostingRegressor",
            },
            "tail_weighting": {
                "enabled": ENABLE_TAIL_WEIGHTING,
                "strength": TAIL_WEIGHT_STRENGTH,
                "power": TAIL_WEIGHT_POWER,
                "cap": TAIL_WEIGHT_CAP,
            },
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


def parse_top_n_from_run_name(run_name: str):
    if run_name is None:
        return None
    if run_name.lower() == "all":
        return None
    if run_name.startswith("Top_"):
        try:
            return int(run_name.split("_", 1)[1])
        except ValueError:
            return None
    return None


def load_selected_predictors_for_tail_weighting(base_out_dir: str, target_variable: str, selected_run: str, all_var_names: List[str]):
    if selected_run is None or selected_run.lower() == "all":
        return all_var_names, list(range(len(all_var_names)))

    top_n = parse_top_n_from_run_name(selected_run)
    if top_n is None:
        raise ValueError(f"Unsupported selected run label for tail weighting: {selected_run}")

    imp_csv = os.path.join(base_out_dir, target_variable, selected_run, "variable_importance.csv")
    if not os.path.exists(imp_csv):
        raise FileNotFoundError(f"Missing baseline importance file for {target_variable}: {imp_csv}")

    imp_df = pd.read_csv(imp_csv).head(top_n)
    keep_vars = [v for v in imp_df["Variable"].tolist() if v in all_var_names]
    keep_idx = [all_var_names.index(v) for v in keep_vars]
    if not keep_idx:
        raise RuntimeError(f"No predictors found for {target_variable} from {imp_csv}")
    return keep_vars, keep_idx


def load_baseline_tuned_params(base_out_dir: str, target_variable: str):
    params_path = os.path.join(base_out_dir, target_variable, "_tuning", "xgb_best_params.txt")
    if not os.path.exists(params_path):
        return None

    params: Dict[str, float] = {}
    with open(params_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key not in {
                "n_estimators", "learning_rate", "max_depth", "min_child_weight",
                "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "gamma",
            }:
                continue
            if key in {"n_estimators", "max_depth", "min_child_weight"}:
                params[key] = float(int(float(value)))
            else:
                params[key] = float(value)

    return params if params else None


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
filter_summary = summarize_predictor_filters(
    var_names,
    excluded_names=EXCLUDED_PREDICTORS,
    selected_names=SELECT_PREDICTORS,
    select_prefixes=SELECT_PREFIXES,
    exclude_prefixes=EXCLUDE_PREFIXES,
)
print(f"Excluded predictor names matched: {len(filter_summary['excluded_name_matches'])}")
for name in filter_summary["excluded_name_matches"]:
    print(f"  name match: {name}")
if filter_summary["excluded_name_missing"]:
    print("Excluded predictor names not found:")
    for name in filter_summary["excluded_name_missing"]:
        print(f"  missing: {name}")
if filter_summary["exclude_prefix_rows"]:
    print("Excluded prefix matches:")
    for row in filter_summary["exclude_prefix_rows"]:
        print(f"  {row['prefix']}: {row['match_count']}")

summary_rows = []
for name in filter_summary["excluded_name_matches"]:
    summary_rows.append({"filter_type": "excluded_name", "filter_value": name, "match_count": 1, "matched_predictors": name})
for name in filter_summary["excluded_name_missing"]:
    summary_rows.append({"filter_type": "excluded_name_missing", "filter_value": name, "match_count": 0, "matched_predictors": ""})
for row in filter_summary["select_prefix_rows"]:
    summary_rows.append({"filter_type": "selected_prefix", "filter_value": row["prefix"], "match_count": row["match_count"], "matched_predictors": row["matched_predictors"]})
for row in filter_summary["exclude_prefix_rows"]:
    summary_rows.append({"filter_type": "excluded_prefix", "filter_value": row["prefix"], "match_count": row["match_count"], "matched_predictors": row["matched_predictors"]})
if summary_rows:
    pd.DataFrame(summary_rows).to_csv(os.path.join(ACTIVE_OUT_DIR, "predictor_filter_summary.csv"), index=False)

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

    tuned_params = None
    if REUSE_BASELINE_TUNED_PARAMS:
        tuned_params = load_baseline_tuned_params(OUT_DIR, TARGET_VARIABLE)
        if tuned_params is not None:
            TUNED_XGB_PARAMS_BY_TARGET[TARGET_VARIABLE] = tuned_params
            print(f"Loaded baseline tuned params for {TARGET_VARIABLE}: {tuned_params}")

    if tuned_params is None and ENABLE_XGB_TUNING:
        tuning_out_dir = os.path.join(ACTIVE_OUT_DIR, TARGET_VARIABLE, "_tuning")
        tuned_params = tune_xgb_params_for_target(
            X_sample_full, y_sample, x_sample, y_coord_sample,
            TARGET_VARIABLE, tuning_out_dir,
        )
        if tuned_params is not None:
            TUNED_XGB_PARAMS_BY_TARGET[TARGET_VARIABLE] = tuned_params
            print(f"Using tuned params for {TARGET_VARIABLE}: {tuned_params}")
        else:
            print(f"[WARNING] Tuning failed for {TARGET_VARIABLE}; using default params.")

    if RUN_SELECTED_SUBSETS_ONLY and (ENABLE_TAIL_WEIGHTING or ENABLE_POSTHOC_CALIBRATION or ENABLE_TWO_STAGE_RESIDUAL or ENABLE_QUANTILE_MODE):
        selected_run = SELECTED_SUBSET_BY_TARGET.get(TARGET_VARIABLE)
        if not selected_run:
            print(f"[WARNING] No selected subset configured for {TARGET_VARIABLE}; skipping.")
            continue

        if selected_run.lower() == "all":
            X_run = X_sample_full
            var_names_run = var_names_f
        else:
            var_names_run, keep_idx = load_selected_predictors_for_tail_weighting(
                OUT_DIR, TARGET_VARIABLE, selected_run, var_names_f
            )
            X_run = X_sample_full[:, keep_idx]

        target_out_dir = os.path.join(ACTIVE_OUT_DIR, TARGET_VARIABLE, selected_run)
        if ENABLE_TAIL_WEIGHTING:
            run_mode_label = "tail-weighted"
        elif ENABLE_TWO_STAGE_RESIDUAL:
            run_mode_label = "two-stage-residual"
        elif ENABLE_QUANTILE_MODE:
            run_mode_label = "quantile-pilot"
        else:
            run_mode_label = "posthoc-calibrated"
        print(f"\n{'─'*50}\n{run_mode_label.capitalize()} run: {selected_run} predictors  →  {target_out_dir}")
        run_cv_and_save(
            X_run, y_sample, x_sample, y_coord_sample, ids_sample,
            var_names_run, target_label, TARGET_VARIABLE, target_out_dir,
        )
        print(f"\n✔ {TARGET_VARIABLE} {run_mode_label} output → {os.path.join(ACTIVE_OUT_DIR, TARGET_VARIABLE)}/")

    elif MULT_RUNS:
        # ── Run 1: all predictors ──────────────────────────────────────────
        full_out_dir = os.path.join(ACTIVE_OUT_DIR, TARGET_VARIABLE, "All")
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
                topn_out = os.path.join(ACTIVE_OUT_DIR, TARGET_VARIABLE, f"Top_{top_n}")
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
        target_out_dir = os.path.join(ACTIVE_OUT_DIR, TARGET_VARIABLE)
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

