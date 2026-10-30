#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Patched point-based XGBoost model for the Kalk project using a prebuilt regression matrix.

This version:
- loads regression_matrix.npz directly
- keeps the workflow point/pixel-based (not tile-based)
- supports optional area-polygon filtering at the point level
- uses U-Net-style class weights as sample weights, with per-class multipliers
- handles categorical predictors without one-hot expansion
- tries GPU XGBoost first, with CPU fallback
- keeps spatial block cross-validation using rows/cols from the matrix
- saves fold metrics, confusion matrices, classification report
- saves variable importance tables and charts (top-N and full)

Important note on categorical predictors:
This script does NOT use neural embeddings like the U-Net. A tree model cannot
learn embeddings internally the same way. Instead, categorical predictors are kept
as compact integer-coded columns (default), with an optional frequency encoding mode.
"""

import os
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from affine import Affine
import geopandas as gpd

from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    recall_score,
)

warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# USER SETTINGS
# =====================================================================

MATRIX_PATH = r"/home/acosta_pedro/Classical_ML/Regression_matrix/Classic_ML/regression_matrix_2026_total_with_clean_ed.npz"
OUT_DIR = r"/home/acosta_pedro/Outputs/Models/XGBoost/Categorical/Tunned_shap"

# Target-status filtering based on regression matrix y_status values.
# - "all": keep clean + dirty
# - "clean": keep only clean
# - "dirty": keep only dirty
TRAIN_STATUS_FILTER = "clean"

# Optional area filter at point level
USE_AREA_FILTER = True
AREA_POLYGON_PATH = r"/home/acosta_pedro/Classical_ML/Regression_matrix/Norge_mask/Norge_mask.shp"
AREA_POLYGON_LAYER = None
AREA_FILTER_MODE = "intersects"   # "intersects" or "within"

# Sampling and CV
SAMPLE_SIZE = 500000               # set to None to use all filtered points
N_SPLITS = 5
BLOCK_SIZE_PX = 2000  # 2000 * 10m = 20km
RANDOM_SEED = 42

# Spatial CV toggle.
# Set False to skip fold-based validation and run only final training.
ENABLE_SPATIAL_CV = True

# XGBoost parameters
N_ESTIMATORS = 500
MAX_DEPTH = 25
MIN_SAMPLES_LEAF = 2
LEARNING_RATE = 0.05
SUBSAMPLE = 0.9
COLSAMPLE_BYTREE = 0.8
REG_LAMBDA = 1.0
REG_ALPHA = 0.0
GAMMA = 0.0

# Optional XGBoost tuning (same idea as XGBoost_window_Kalkklass_modelling.py).
# When enabled, a small candidate set is evaluated with spatial GroupKFold and
# best params (by mean macro-F1) are used for the main run.
ENABLE_XGB_TUNING = True
TUNING_MAX_TRIALS = 12
TUNING_SAMPLE_SIZE = 150000

# Ordered-XGB-style toggle:
# when True, train K-1 cumulative binary XGB models (P[y > c_k]) and
# reconstruct class probabilities for ordered classes.
USE_ORF_STYLE = False

# GPU
USE_GPU_IF_AVAILABLE = True

# Imbalance handling copied from U-Net logic, but now tunable per class
USE_UNET_STYLE_SAMPLE_WEIGHTS = True
CLASS_WEIGHT_MULTIPLIERS = {
    1: 1.00,   # low
    2: 2.20,   # medium
    3: 1.15,   # high
}

# Categorical handling mode.
# "native"      = use XGBoost native categorical splits (recommended here)
# "passthrough" = keep columns numeric as provided
# "frequency"   = replace category by training-fold frequency
CATEGORICAL_MODE = "native"
QUATERNARY_PREDICTOR_NAME = "QuaternaryClass_id"
LANDUSE_PREDICTOR_NAME = "landuseCode_18_cog"

# Optional predictor exclusions by exact var_names
USE_EXCLUDED_PREDICTORS = False
EXCLUDED_PREDICTORS = [
    "KlassCa_entropy_250K",
    "KlassCa_prob_1_Kalkfattig_250K",
    "KlassCa_prob_2_Intermediaer_250K",
    "KlassCa_prob_3_Kalkrik_250K",
]

# Optional predictor include-list by exact names from CSV.
# If the file has column "predictor_name" it is used, otherwise first column is used.
USE_INCLUDED_PREDICTORS_FILE = True
INCLUDED_PREDICTORS_FILE = r"/home/acosta_pedro/Classical_ML/Predictors_file/predictors_used_for_modelling.csv"

# Optional substring-based predictor exclusions.
USE_EXCLUDED_PREDICTOR_SUBSTRINGS = False
EXCLUDED_PREDICTOR_SUBSTRINGS = [
    "quaternary_",
    "landuse_",
    "Ca_conc_aveg_geology_Ca_ICP_Coe_cog",
    "Ca_conc_aveggeology_LogCa_ICP_cog",
]

# Predictors to force-include even if they are not present in the include-list file.
FORCE_INCLUDED_PREDICTORS = [
    "coord_X",
    "coord_Y",
    "QuaternaryClass_id",
    "landuseCode_18_cog",
]

# Force include by substring (e.g., bring in all geophysics predictors).
FORCE_INCLUDED_SUBSTRINGS = []

# Variable importance output
IMPORTANCE_TOP_N = 30
SAVE_FULL_IMPORTANCE_CHART = True
MAX_VARS_IN_FULL_CHART = None   # None = all variables
FULL_CHART_FIGHEIGHT_PER_VAR = 0.22
TOP_CHART_FIGHEIGHT_PER_VAR = 0.34

# Optional extra runs on predictor subsets from importance table.
# Set to False while tuning the main model to reduce runtime.
RUN_TOP_N_SUBSET_MODELS = False
TOP_N_SUBSET_LIST = [10, 20, 30, 40, 50, 60, 70, 80]

# Optional uncertainty diagnostics from CV predicted probabilities.
# This produces point-level diagnostics from held-out folds.
ENABLE_UNCERTAINTY_DIAGNOSTICS = True

# Optional SHAP-like export using XGBoost pred_contribs on final model.
ENABLE_SHAP_EXPORT = True
SHAP_MAX_SAMPLES = 50000

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)


# =====================================================================
# HELPERS
# =====================================================================

def load_matrix(npz_path: str):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Regression matrix not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    required = ["X", "y", "rows", "cols", "var_names"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"Regression matrix missing required keys: {missing}")

    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int32)
    if "y_status" in data.files:
        y_status = np.asarray(data["y_status"], dtype=np.int32)
    else:
        # Backward compatibility for older matrices without status labels.
        y_status = np.ones_like(y, dtype=np.int32)
    rows = np.asarray(data["rows"], dtype=np.int32)
    cols = np.asarray(data["cols"], dtype=np.int32)
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

    if (
        X.shape[0] != y.shape[0]
        or X.shape[0] != y_status.shape[0]
        or X.shape[0] != rows.shape[0]
        or X.shape[0] != cols.shape[0]
    ):
        raise RuntimeError("X, y, y_status, rows, and cols do not have the same number of samples")
    if X.shape[1] != len(var_names):
        raise RuntimeError("Number of columns in X does not match length of var_names")

    return X, y, y_status, rows, cols, var_names, width, height, transform, crs_wkt, ref_mask


def apply_status_filter(
    X: np.ndarray,
    y: np.ndarray,
    y_status: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    mode: str,
):
    mode_l = str(mode).strip().lower()
    if mode_l not in {"all", "clean", "dirty"}:
        raise ValueError("TRAIN_STATUS_FILTER must be one of: 'all', 'clean', 'dirty'")

    # 1=clean, 2=dirty in matrix builder output.
    if mode_l == "all":
        keep = np.isin(y_status, [1, 2])
    elif mode_l == "clean":
        keep = y_status == 1
    else:
        keep = y_status == 2

    kept = int(keep.sum())
    if kept == 0:
        raise RuntimeError(f"TRAIN_STATUS_FILTER='{mode_l}' removed all samples")

    print(f"[INFO] TRAIN_STATUS_FILTER='{mode_l}' kept {kept} / {len(keep)} samples")
    return X[keep], y[keep], y_status[keep], rows[keep], cols[keep], keep


def compute_unet_style_class_weights(y_train: np.ndarray, class_multipliers: Dict[int, float] = None) -> np.ndarray:
    """Mirror the U-Net class-weight logic for classes 1..3, with optional per-class multipliers."""
    if class_multipliers is None:
        class_multipliers = {}

    flat_y = y_train.astype(np.int32).ravel()
    counts = np.bincount(flat_y, minlength=4)
    _, c1, c2, c3 = counts[:4]
    non_bg = c1 + c2 + c3

    weights = np.zeros(4, dtype=np.float32)
    for cls, cnt in zip([1, 2, 3], [c1, c2, c3]):
        weights[cls] = non_bg / (3.0 * max(cnt, 1))
        weights[cls] *= float(class_multipliers.get(cls, 1.0))

    mean_nonzero = np.mean(weights[1:4]) if np.any(weights[1:4] > 0) else 1.0
    weights[1:4] /= mean_nonzero
    return weights


def xy_from_rows_cols(rows: np.ndarray, cols: np.ndarray, transform: Affine) -> Tuple[np.ndarray, np.ndarray]:
    xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    return xs.astype(np.float64), ys.astype(np.float64)


def filter_points_by_polygon(
    X: np.ndarray,
    y: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    crs_wkt: str,
    transform: Affine,
    polygon_path: str,
    polygon_layer: str = None,
    mode: str = "intersects",
):
    if mode not in {"intersects", "within"}:
        raise ValueError("AREA_FILTER_MODE must be 'intersects' or 'within'")
    if transform is None:
        raise RuntimeError("Cannot apply area filter because transform is missing in regression_matrix.npz")
    if not polygon_path or not os.path.exists(polygon_path):
        raise FileNotFoundError(f"Area polygon not found: {polygon_path}")

    area_gdf = gpd.read_file(polygon_path, layer=polygon_layer) if polygon_layer else gpd.read_file(polygon_path)
    if area_gdf.empty:
        raise RuntimeError(f"Area polygon file has no geometries: {polygon_path}")

    xs, ys = xy_from_rows_cols(rows, cols, transform)
    pts = gpd.GeoDataFrame(
        {"row": rows, "col": cols},
        geometry=gpd.points_from_xy(xs, ys),
        crs=crs_wkt if crs_wkt else area_gdf.crs,
    )

    if area_gdf.crs is not None and pts.crs is not None and area_gdf.crs != pts.crs:
        area_gdf = area_gdf.to_crs(pts.crs)

    if hasattr(area_gdf.geometry, "union_all"):
        area_union = area_gdf.geometry.union_all()
    else:
        area_union = area_gdf.geometry.unary_union

    keep = pts.within(area_union).to_numpy() if mode == "within" else pts.intersects(area_union).to_numpy()
    print(f"[INFO] Area filter kept {keep.sum()} / {len(keep)} points using mode='{mode}'")
    return X[keep], y[keep], rows[keep], cols[keep], keep


def apply_predictor_exclusions(X: np.ndarray, var_names: List[str], excluded_names: List[str]):
    if not excluded_names:
        return X, var_names
    excluded_set = {str(n).strip().lower() for n in excluded_names}
    keep_idx = [i for i, v in enumerate(var_names) if str(v).strip().lower() not in excluded_set]
    drop_names = [v for i, v in enumerate(var_names) if i not in keep_idx]
    print(f"[INFO] Excluding {len(drop_names)} predictors: {drop_names}")
    return X[:, keep_idx], [var_names[i] for i in keep_idx]


def apply_predictor_exclusion_substrings(X: np.ndarray, var_names: List[str], excluded_substrings: List[str]):
    substrings = [str(s).strip().lower() for s in excluded_substrings if str(s).strip()]
    if not substrings:
        return X, var_names, []

    keep_idx = []
    dropped_names = []
    for i, v in enumerate(var_names):
        name_lc = str(v).strip().lower()
        if any(sub in name_lc for sub in substrings):
            dropped_names.append(v)
        else:
            keep_idx.append(i)

    print(f"[INFO] Substring exclusion removed {len(dropped_names)} predictors")
    return X[:, keep_idx], [var_names[i] for i in keep_idx], dropped_names


def _resolve_csv_path(path_like: str) -> str:
    if os.path.exists(path_like):
        return path_like
    if os.path.exists(path_like + ".csv"):
        return path_like + ".csv"
    raise FileNotFoundError(f"Predictor include file not found: {path_like} (or .csv)")


def apply_predictor_inclusion_file(X: np.ndarray, var_names: List[str], include_file: str):
    csv_path = _resolve_csv_path(include_file)
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"Predictor include file is empty: {csv_path}")

    if "predictor_name" in df.columns:
        include_names_raw = df["predictor_name"].astype(str).tolist()
    else:
        include_names_raw = df.iloc[:, 0].astype(str).tolist()

    include_set = {str(n).strip().lower() for n in include_names_raw if str(n).strip()}
    if not include_set:
        raise RuntimeError(f"No valid predictor names found in include file: {csv_path}")

    # Exact matching first. If a requested name is not found exactly, allow
    # prefix expansion so base names can map to banded variables (e.g. *_b01).
    var_names_lc = [str(v).strip().lower() for v in var_names]
    var_set = set(var_names_lc)
    expanded_include_set = set()
    expanded_from_prefix = 0
    for name in include_set:
        if name in var_set:
            expanded_include_set.add(name)
            continue
        prefix = name + "_"
        pref_matches = [v for v in var_names_lc if v.startswith(prefix)]
        if pref_matches:
            expanded_include_set.update(pref_matches)
            expanded_from_prefix += len(pref_matches)

    keep_idx = [i for i, v in enumerate(var_names_lc) if v in expanded_include_set]
    kept_names = [var_names[i] for i in keep_idx]

    matched_lc = {str(v).strip().lower() for v in kept_names}
    missing = sorted(include_set - matched_lc)
    if missing:
        preview = missing[:15]
        print(f"[WARN] {len(missing)} include-list predictors were not found in matrix var_names. First entries: {preview}")

    if not keep_idx:
        raise RuntimeError("Inclusion file filtering removed all predictors. Check name matching.")

    if expanded_from_prefix > 0:
        print(f"[INFO] Expanded include-list by prefix matching: +{expanded_from_prefix} predictors")
    print(f"[INFO] Included {len(keep_idx)} predictors from {csv_path}")
    return X[:, keep_idx], kept_names


def apply_force_included_predictors(X: np.ndarray, var_names: List[str], X_ref: np.ndarray, var_names_ref: List[str], force_names: List[str]):
    if not force_names:
        return X, var_names

    current_lookup = {str(v).strip().lower(): i for i, v in enumerate(var_names)}
    ref_lookup = {str(v).strip().lower(): i for i, v in enumerate(var_names_ref)}

    add_indices = []
    add_names = []
    missing = []
    for name in force_names:
        name_lc = str(name).strip().lower()
        if not name_lc:
            continue
        if name_lc in current_lookup:
            continue
        ref_idx = ref_lookup.get(name_lc)
        if ref_idx is None:
            missing.append(name)
            continue
        add_indices.append(ref_idx)
        add_names.append(var_names_ref[ref_idx])

    if missing:
        print(f"[WARN] Force-included predictors not found in matrix var_names: {missing}")

    if not add_indices:
        return X, var_names

    X_added = X_ref[:, add_indices]
    X_out = np.concatenate([X, X_added], axis=1)
    var_names_out = list(var_names) + add_names
    print(f"[INFO] Force-included {len(add_names)} predictors: {add_names}")
    return X_out, var_names_out


def apply_force_included_substrings(
    X: np.ndarray,
    var_names: List[str],
    X_ref: np.ndarray,
    var_names_ref: List[str],
    force_substrings: List[str],
):
    substrings = [str(s).strip().lower() for s in force_substrings if str(s).strip()]
    if not substrings:
        return X, var_names

    current_lookup = {str(v).strip().lower() for v in var_names}
    add_indices = []
    add_names = []

    for i, name in enumerate(var_names_ref):
        name_lc = str(name).strip().lower()
        if name_lc in current_lookup:
            continue
        if any(sub in name_lc for sub in substrings):
            add_indices.append(i)
            add_names.append(name)

    if not add_indices:
        print(f"[INFO] No predictors matched force-include substrings: {substrings}")
        return X, var_names

    X_added = X_ref[:, add_indices]
    X_out = np.concatenate([X, X_added], axis=1)
    var_names_out = list(var_names) + add_names
    print(f"[INFO] Force-included {len(add_names)} predictors by substring: {substrings}")
    return X_out, var_names_out


def save_predictor_list(var_names: List[str], out_dir: str):
    pd.DataFrame({"predictor_name": var_names}).to_csv(
        os.path.join(out_dir, "predictors_used_for_modelling.csv"),
        index=False,
    )


def find_categorical_indices(var_names: List[str]) -> Dict[str, int]:
    lookup = {str(v).strip().lower(): i for i, v in enumerate(var_names)}
    out = {}
    for name in [QUATERNARY_PREDICTOR_NAME, LANDUSE_PREDICTOR_NAME]:
        idx = lookup.get(name.strip().lower(), None)
        if idx is not None:
            out[name] = idx
    return out


def coerce_categorical_columns(X: np.ndarray, cat_idx: List[int]) -> np.ndarray:
    X2 = X.copy()
    for idx in cat_idx:
        col = X2[:, idx]
        col = np.where(np.isfinite(col), col, 0)
        col = np.where(col < 0, 0, np.rint(col))
        X2[:, idx] = col.astype(np.float32)
    return X2


def frequency_encode_train_test(X_tr: np.ndarray, X_te: np.ndarray, cat_idx: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    Xtr = X_tr.copy()
    Xte = X_te.copy()
    for idx in cat_idx:
        tr_col = np.rint(np.where(np.isfinite(Xtr[:, idx]), Xtr[:, idx], 0)).astype(np.int64)
        te_col = np.rint(np.where(np.isfinite(Xte[:, idx]), Xte[:, idx], 0)).astype(np.int64)
        vals, counts = np.unique(tr_col, return_counts=True)
        freq = {v: c / len(tr_col) for v, c in zip(vals, counts)}
        Xtr[:, idx] = np.array([freq.get(v, 0.0) for v in tr_col], dtype=np.float32)
        Xte[:, idx] = np.array([freq.get(v, 0.0) for v in te_col], dtype=np.float32)
    return Xtr, Xte


def native_categorical_train_test(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    var_names: List[str],
    cat_idx: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = [str(v) for v in var_names]
    Xtr_df = pd.DataFrame(X_tr, columns=cols)
    Xte_df = pd.DataFrame(X_te, columns=cols)

    for idx in cat_idx:
        col = cols[idx]
        tr_vals = np.rint(np.where(np.isfinite(X_tr[:, idx]), X_tr[:, idx], -1)).astype(np.int64)
        te_vals = np.rint(np.where(np.isfinite(X_te[:, idx]), X_te[:, idx], -1)).astype(np.int64)

        # Build category vocabulary from training data only and reserve -1 for unknowns.
        train_cats = np.unique(tr_vals[tr_vals >= 0]).astype(np.int64)
        cats = np.concatenate((np.array([-1], dtype=np.int64), train_cats))

        # Keep split integrity: categories unseen in training fold become unknown (-1).
        tr_known = np.isin(tr_vals, train_cats)
        te_known = np.isin(te_vals, train_cats)
        tr_vals = np.where(tr_known, tr_vals, -1)
        te_vals = np.where(te_known, te_vals, -1)

        Xtr_df[col] = pd.Categorical(tr_vals, categories=cats)
        Xte_df[col] = pd.Categorical(te_vals, categories=cats)

    return Xtr_df, Xte_df


def make_groups(rows: np.ndarray, cols: np.ndarray, width: int, block_size_px: int) -> np.ndarray:
    if width is None:
        raise RuntimeError("Matrix width is required to build spatial block groups")
    n_block_cols = int(np.ceil(width / block_size_px))
    block_rows = rows // block_size_px
    block_cols = cols // block_size_px
    return (block_rows * n_block_cols + block_cols).astype(np.int64)


def stratified_subsample(X, y, rows, cols, sample_size, seed=42):
    if sample_size is None or len(y) <= sample_size:
        return X, y, rows, cols
    sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=seed)
    idx = next(sss.split(X, y))[0]
    return X[idx], y[idx], rows[idx], cols[idx]


DEFAULT_XGB_PARAMS: Dict[str, Any] = {
    "n_estimators": N_ESTIMATORS,
    "learning_rate": LEARNING_RATE,
    "max_depth": MAX_DEPTH,
    "min_child_weight": max(1.0, float(MIN_SAMPLES_LEAF)),
    "subsample": SUBSAMPLE,
    "colsample_bytree": COLSAMPLE_BYTREE,
    "reg_alpha": REG_ALPHA,
    "reg_lambda": REG_LAMBDA,
    "gamma": GAMMA,
}

TUNED_XGB_PARAMS: Dict[str, Any] = {
    "n_estimators": 1400,
    "learning_rate": 0.025,
    "max_depth": 10,
    "min_child_weight": 6.0,
    "subsample": 0.85,
    "colsample_bytree": 0.65,
    "reg_alpha": 0.1,
    "reg_lambda": 2.5,
    "gamma": 0.15,
}


def build_xgb_tuning_candidates():
    candidates = [
        {"n_estimators": 800, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 3.0, "subsample": 0.8, "colsample_bytree": 0.6, "reg_alpha": 0.05, "reg_lambda": 2.0, "gamma": 0.0},
        {"n_estimators": 1000, "learning_rate": 0.04, "max_depth": 8, "min_child_weight": 4.0, "subsample": 0.8, "colsample_bytree": 0.6, "reg_alpha": 0.10, "reg_lambda": 2.0, "gamma": 0.10},
        {"n_estimators": 1200, "learning_rate": 0.03, "max_depth": 10, "min_child_weight": 5.0, "subsample": 0.8, "colsample_bytree": 0.6, "reg_alpha": 0.10, "reg_lambda": 2.0, "gamma": 0.10},
        {"n_estimators": 1400, "learning_rate": 0.025, "max_depth": 10, "min_child_weight": 6.0, "subsample": 0.85, "colsample_bytree": 0.65, "reg_alpha": 0.10, "reg_lambda": 2.5, "gamma": 0.15},
        {"n_estimators": 1600, "learning_rate": 0.02, "max_depth": 12, "min_child_weight": 8.0, "subsample": 0.8, "colsample_bytree": 0.7, "reg_alpha": 0.20, "reg_lambda": 3.0, "gamma": 0.20},
        {"n_estimators": 900, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 3.0, "subsample": 0.75, "colsample_bytree": 0.55, "reg_alpha": 0.00, "reg_lambda": 1.5, "gamma": 0.0},
        {"n_estimators": 1100, "learning_rate": 0.035, "max_depth": 8, "min_child_weight": 5.0, "subsample": 0.85, "colsample_bytree": 0.7, "reg_alpha": 0.05, "reg_lambda": 2.0, "gamma": 0.05},
        {"n_estimators": 1300, "learning_rate": 0.03, "max_depth": 12, "min_child_weight": 10.0, "subsample": 0.8, "colsample_bytree": 0.6, "reg_alpha": 0.30, "reg_lambda": 4.0, "gamma": 0.20},
        {"n_estimators": 1000, "learning_rate": 0.04, "max_depth": 10, "min_child_weight": 4.0, "subsample": 0.9, "colsample_bytree": 0.8, "reg_alpha": 0.00, "reg_lambda": 1.0, "gamma": 0.0},
        {"n_estimators": 1400, "learning_rate": 0.025, "max_depth": 8, "min_child_weight": 6.0, "subsample": 0.9, "colsample_bytree": 0.75, "reg_alpha": 0.10, "reg_lambda": 2.0, "gamma": 0.10},
        {"n_estimators": 1200, "learning_rate": 0.03, "max_depth": 6, "min_child_weight": 8.0, "subsample": 0.8, "colsample_bytree": 0.5, "reg_alpha": 0.20, "reg_lambda": 3.0, "gamma": 0.15},
        {"n_estimators": 1500, "learning_rate": 0.02, "max_depth": 10, "min_child_weight": 12.0, "subsample": 0.7, "colsample_bytree": 0.6, "reg_alpha": 0.30, "reg_lambda": 5.0, "gamma": 0.25},
    ]
    return candidates[:TUNING_MAX_TRIALS]


def _new_xgb_classifier(
    use_gpu: bool,
    random_state: int,
    objective: str,
    num_class: int = None,
    xgb_params: Dict[str, Any] = None,
    use_native_categorical: bool = False,
):
    params: Dict[str, Any] = dict(DEFAULT_XGB_PARAMS)
    if xgb_params:
        params.update(xgb_params)

    model_params: Dict[str, Any] = {
        "n_estimators": int(params["n_estimators"]),
        "max_depth": int(params["max_depth"]),
        "learning_rate": float(params["learning_rate"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "min_child_weight": float(params["min_child_weight"]),
        "reg_alpha": float(params.get("reg_alpha", 0.0)),
        "reg_lambda": float(params["reg_lambda"]),
        "gamma": float(params.get("gamma", 0.0)),
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
        "objective": objective,
        "eval_metric": "mlogloss" if objective.startswith("multi") else "logloss",
    }
    if num_class is not None:
        model_params["num_class"] = int(num_class)
    if use_native_categorical:
        model_params["enable_categorical"] = True
    if use_gpu:
        model_params["device"] = "cuda"
    return XGBClassifier(**model_params)


def _fit_predict_orf_style_xgb(
    X_tr,
    y_tr,
    X_te,
    sample_weight=None,
    use_gpu=True,
    random_state=42,
    xgb_params: Dict[str, Any] = None,
    var_names: List[str] = None,
    cat_idx: List[int] = None,
    use_native_categorical: bool = False,
):
    classes = np.sort(np.unique(y_tr.astype(np.int32)))
    if classes.size < 2:
        raise RuntimeError("ORF-style requires at least 2 classes in training data.")

    thresholds = classes[:-1]
    cumulative_probs = []
    model_records = []
    importances = []

    X_tr_fit, X_te_fit = X_tr, X_te
    if use_native_categorical and cat_idx and var_names is not None:
        X_tr_fit, X_te_fit = native_categorical_train_test(X_tr, X_te, var_names, cat_idx)

    for i, thr in enumerate(thresholds):
        y_bin = (y_tr > thr).astype(np.int32)

        # Degenerate folds can occur in spatial CV; keep pipeline robust.
        if np.unique(y_bin).size < 2:
            const_p = float(y_bin[0])
            cumulative_probs.append(np.full(X_te.shape[0], const_p, dtype=np.float32))
            model_records.append({
                "threshold": int(thr),
                "model": None,
                "constant_prob": const_p,
            })
            continue

        model = _new_xgb_classifier(
            use_gpu=use_gpu,
            random_state=random_state + i,
            objective="binary:logistic",
            xgb_params=xgb_params,
            use_native_categorical=use_native_categorical,
        )
        if sample_weight is not None:
            model.fit(X_tr_fit, y_bin, sample_weight=sample_weight)
        else:
            model.fit(X_tr_fit, y_bin)

        prob = model.predict_proba(X_te_fit)
        cls_list = list(model.classes_)
        if 1 in cls_list:
            pos_idx = cls_list.index(1)
            p_gt = prob[:, pos_idx].astype(np.float32)
        else:
            p_gt = np.zeros(X_te.shape[0], dtype=np.float32)

        cumulative_probs.append(p_gt)
        model_records.append({
            "threshold": int(thr),
            "model": model,
            "constant_prob": None,
        })
        if hasattr(model, "feature_importances_"):
            importances.append(np.asarray(model.feature_importances_, dtype=np.float64))

    cum = np.column_stack(cumulative_probs).astype(np.float32)  # (n, K-1)
    cum = np.clip(cum, 0.0, 1.0)

    # Enforce monotone cumulative probabilities: P(y>c1) >= P(y>c2) >= ...
    for j in range(1, cum.shape[1]):
        cum[:, j] = np.minimum(cum[:, j], cum[:, j - 1])

    n = X_te.shape[0]
    k = classes.size
    probs = np.zeros((n, k), dtype=np.float32)
    probs[:, 0] = 1.0 - cum[:, 0]
    for cls_idx in range(1, k - 1):
        probs[:, cls_idx] = cum[:, cls_idx - 1] - cum[:, cls_idx]
    probs[:, -1] = cum[:, -1]
    probs = np.clip(probs, 0.0, 1.0)
    denom = probs.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    probs = probs / denom

    pred_idx = np.argmax(probs, axis=1)
    pred = classes[pred_idx].astype(np.int32)

    out_model: Dict[str, Any] = {
        "model_type": "xgb_orf_style",
        "classes": classes.astype(np.int32).tolist(),
        "thresholds": thresholds.astype(np.int32).tolist(),
        "models": model_records,
    }
    if importances:
        out_model["feature_importances_"] = np.mean(np.vstack(importances), axis=0)

    return pred, out_model, "xgboost_orf", probs


def extract_feature_importances(model: Any):
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=np.float64)
    if isinstance(model, dict) and model.get("feature_importances_") is not None:
        return np.asarray(model["feature_importances_"], dtype=np.float64)
    return None


def fit_predict_xgb(
    X_tr,
    y_tr,
    X_te,
    sample_weight=None,
    use_gpu=True,
    random_state=42,
    xgb_params: Dict[str, Any] = None,
    var_names: List[str] = None,
    cat_idx: List[int] = None,
    use_native_categorical: bool = False,
):
    if USE_ORF_STYLE:
        return _fit_predict_orf_style_xgb(
            X_tr,
            y_tr,
            X_te,
            sample_weight=sample_weight,
            use_gpu=use_gpu,
            random_state=random_state,
            xgb_params=xgb_params,
            var_names=var_names,
            cat_idx=cat_idx,
            use_native_categorical=use_native_categorical,
        )

    classes = np.sort(np.unique(y_tr.astype(np.int32)))
    num_class = int(classes.size)
    class_to_idx = {int(c): i for i, c in enumerate(classes.tolist())}
    idx_to_class = {i: int(c) for i, c in enumerate(classes.tolist())}

    y_tr_idx = np.asarray([class_to_idx[int(v)] for v in y_tr], dtype=np.int32)
    objective = "multi:softprob" if num_class > 2 else "binary:logistic"

    X_tr_fit, X_te_fit = X_tr, X_te
    if use_native_categorical and cat_idx and var_names is not None:
        X_tr_fit, X_te_fit = native_categorical_train_test(X_tr, X_te, var_names, cat_idx)

    try:
        model = _new_xgb_classifier(
            use_gpu=use_gpu,
            random_state=random_state,
            objective=objective,
            num_class=num_class if num_class > 2 else None,
            xgb_params=xgb_params,
            use_native_categorical=use_native_categorical,
        )
        if sample_weight is not None:
            model.fit(X_tr_fit, y_tr_idx, sample_weight=sample_weight)
        else:
            model.fit(X_tr_fit, y_tr_idx)
        pred_idx = model.predict(X_te_fit).astype(np.int32)
        pred = np.asarray([idx_to_class[int(v)] for v in pred_idx], dtype=np.int32)

        if hasattr(model, "predict_proba"):
            proba_idx = model.predict_proba(X_te_fit).astype(np.float32)
            proba = np.zeros((X_te_fit.shape[0], len(classes)), dtype=np.float32)
            cls_idx_model = np.asarray(model.classes_, dtype=np.int32)
            class_pos_lookup = {int(c): i for i, c in enumerate(classes.tolist())}
            for j, cls_idx in enumerate(cls_idx_model.tolist()):
                cls_label = idx_to_class[int(cls_idx)]
                tgt = class_pos_lookup[int(cls_label)]
                proba[:, tgt] = proba_idx[:, j]
        else:
            proba = np.full((X_te_fit.shape[0], len(classes)), np.nan, dtype=np.float32)

        backend = "xgboost_gpu" if use_gpu else "xgboost_cpu"
        return pred, model, backend, proba
    except Exception as exc:
        if use_gpu:
            print(f"[WARNING] GPU XGBoost failed, falling back to CPU. Reason: {exc}")
            return fit_predict_xgb(
                X_tr,
                y_tr,
                X_te,
                sample_weight=sample_weight,
                use_gpu=False,
                random_state=random_state,
                xgb_params=xgb_params,
                var_names=var_names,
                cat_idx=cat_idx,
                use_native_categorical=use_native_categorical,
            )
        raise


def tune_xgb_params(X_samp, y_samp, groups, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    candidates = build_xgb_tuning_candidates()
    gkf_tune = GroupKFold(n_splits=N_SPLITS)
    split_iter = list(gkf_tune.split(X_samp, y_samp, groups))

    trial_rows = []
    best_params = None
    best_f1 = -np.inf

    print(f"\n=== TUNING XGBoost (categorical maps) | trials={len(candidates)} ===")
    for i, params in enumerate(candidates, start=1):
        fold_f1s = []
        for train_idx, test_idx in split_iter:
            X_tr = X_samp[train_idx].copy()
            X_te = X_samp[test_idx].copy()
            y_tr = y_samp[train_idx]
            y_te = y_samp[test_idx]

            if CATEGORICAL_MODE == "frequency" and cat_idx:
                X_tr, X_te = frequency_encode_train_test(X_tr, X_te, cat_idx)
            elif CATEGORICAL_MODE not in ("passthrough", "native"):
                raise ValueError("CATEGORICAL_MODE must be 'passthrough', 'frequency', or 'native'")

            sample_weight = None
            if USE_UNET_STYLE_SAMPLE_WEIGHTS:
                class_weights = compute_unet_style_class_weights(
                    y_tr,
                    class_multipliers=CLASS_WEIGHT_MULTIPLIERS,
                )
                sample_weight = class_weights[y_tr].astype(np.float32)

            y_pred, _, _, _ = fit_predict_xgb(
                X_tr,
                y_tr,
                X_te,
                sample_weight=sample_weight,
                use_gpu=USE_GPU_IF_AVAILABLE,
                random_state=RANDOM_SEED + i * 10,
                xgb_params=params,
                var_names=var_names,
                cat_idx=cat_idx,
                use_native_categorical=(CATEGORICAL_MODE == "native"),
            )
            fold_f1s.append(f1_score(y_te, y_pred, average="macro", zero_division=0))

        mean_f1 = float(np.mean(fold_f1s))
        print(f"[Tune {i:02d}] macro-F1={mean_f1:.4f}")
        trial_rows.append({"trial": i, "mean_macro_f1": mean_f1, **params})

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_params = params

    pd.DataFrame(trial_rows).sort_values("mean_macro_f1", ascending=False).to_csv(
        os.path.join(out_dir, "xgb_tuning_trials.csv"),
        index=False,
    )
    if best_params:
        with open(os.path.join(out_dir, "xgb_best_params.txt"), "w", encoding="utf-8") as f:
            f.write(f"Best macro-F1: {best_f1:.6f}\n")
            for k, v in best_params.items():
                f.write(f"{k}: {v}\n")
        print(f"Best params: macro-F1={best_f1:.4f} | {best_params}")
        return best_params
    return {}


def save_importance_outputs(importance_df: pd.DataFrame, out_dir: str):
    importance_df = importance_df.copy()
    importance_df["Importance_pct"] = 100.0 * importance_df["Importance"] / importance_df["Importance"].sum()

    importance_df.to_csv(os.path.join(out_dir, "variable_importance.csv"), index=False)

    # Top-N chart
    top_n = min(IMPORTANCE_TOP_N, len(importance_df))
    plot_df = importance_df.head(top_n).iloc[::-1].copy()
    fig_h = max(6.0, TOP_CHART_FIGHEIGHT_PER_VAR * len(plot_df))
    plt.figure(figsize=(11, fig_h))
    plt.barh(plot_df["Variable"], plot_df["Importance"])
    plt.xlabel("Mean CV importance")
    plt.title(f"Top {top_n} Variable Importances (XGBoost from regression matrix)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"variable_importance_top{top_n}.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Full chart
    if SAVE_FULL_IMPORTANCE_CHART:
        full_df = importance_df.copy()
        if MAX_VARS_IN_FULL_CHART is not None:
            full_df = full_df.head(int(MAX_VARS_IN_FULL_CHART)).copy()
        full_df = full_df.iloc[::-1]
        fig_h = max(8.0, FULL_CHART_FIGHEIGHT_PER_VAR * len(full_df))
        plt.figure(figsize=(12, fig_h))
        plt.barh(full_df["Variable"], full_df["Importance"])
        plt.xlabel("Mean CV importance")
        plt.title("All Variable Importances (XGBoost from regression matrix)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "variable_importance_full.png"), dpi=220, bbox_inches="tight")
        plt.close()

    # Top table for quick reading
    importance_df.head(IMPORTANCE_TOP_N).to_csv(
        os.path.join(out_dir, f"variable_importance_top{IMPORTANCE_TOP_N}.csv"),
        index=False,
    )


def export_uncertainty_diagnostics(uncertainty_df: pd.DataFrame, out_dir: str):
    uncertainty_df.to_csv(os.path.join(out_dir, "cv_uncertainty_diagnostics.csv"), index=False)

    plt.figure(figsize=(8, 4.5))
    uncertainty_df["max_prob"].hist(bins=50)
    plt.xlabel("Max class probability")
    plt.ylabel("Count")
    plt.title("CV uncertainty diagnostic: max probability")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cv_uncertainty_maxprob_hist.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    uncertainty_df["entropy_norm"].hist(bins=50)
    plt.xlabel("Normalized entropy")
    plt.ylabel("Count")
    plt.title("CV uncertainty diagnostic: normalized entropy")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cv_uncertainty_entropy_hist.png"), dpi=180)
    plt.close()


def export_shap_contrib_summary(
    model: Any,
    X_ref: np.ndarray,
    var_names: List[str],
    out_dir: str,
    sample_size: int,
    random_seed: int,
    use_native_categorical: bool,
    cat_idx: List[int],
):
    if not hasattr(model, "get_booster"):
        print("[INFO] SHAP export skipped: final model does not expose Booster API.")
        return

    n = X_ref.shape[0]
    if n == 0:
        print("[INFO] SHAP export skipped: no samples available.")
        return

    if sample_size is not None and n > int(sample_size):
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(n, size=int(sample_size), replace=False)
        X_use = X_ref[idx].copy()
    else:
        X_use = X_ref.copy()

    try:
        if use_native_categorical and cat_idx:
            X_use_df, _ = native_categorical_train_test(X_use, X_use, var_names, cat_idx)
            dmat = xgb.DMatrix(X_use_df, feature_names=[str(v) for v in var_names], enable_categorical=True)
        else:
            dmat = xgb.DMatrix(X_use, feature_names=[str(v) for v in var_names])

        booster = model.get_booster()
        contrib = booster.predict(dmat, pred_contribs=True)
        n_feat = len(var_names)

        if contrib.ndim == 3:
            shap_abs = np.mean(np.abs(contrib[:, :, :-1]), axis=(0, 1))
        elif contrib.ndim == 2 and contrib.shape[1] == (n_feat + 1):
            shap_abs = np.mean(np.abs(contrib[:, :-1]), axis=0)
        elif contrib.ndim == 2 and contrib.shape[1] % (n_feat + 1) == 0:
            n_cls = contrib.shape[1] // (n_feat + 1)
            contrib3 = contrib.reshape(contrib.shape[0], n_cls, n_feat + 1)
            shap_abs = np.mean(np.abs(contrib3[:, :, :-1]), axis=(0, 1))
        else:
            print(f"[WARNING] SHAP export skipped: unexpected pred_contribs shape {contrib.shape}")
            return

        shap_df = pd.DataFrame(
            {
                "Variable": var_names,
                "MeanAbsSHAP": shap_abs,
            }
        ).sort_values("MeanAbsSHAP", ascending=False)
        shap_df.to_csv(os.path.join(out_dir, "shap_mean_abs.csv"), index=False)

        plot_df = shap_df.head(min(30, len(shap_df))).iloc[::-1]
        plt.figure(figsize=(11, max(6.0, 0.34 * len(plot_df))))
        plt.barh(plot_df["Variable"], plot_df["MeanAbsSHAP"])
        plt.xlabel("Mean absolute SHAP contribution")
        plt.title("Top SHAP contributions (XGBoost pred_contribs)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "shap_mean_abs_top30.png"), dpi=200)
        plt.close()
        print("[INFO] Saved SHAP contribution outputs.")
    except Exception as exc:
        print(f"[WARNING] SHAP export failed and was skipped: {exc}")


# =====================================================================
# MAIN
# =====================================================================

print("\n=== LOADING REGRESSION MATRIX ===")
X_all, y_all, y_status_all, rows_arr, cols_arr, var_names, width, height, transform, crs_wkt, ref_mask = load_matrix(MATRIX_PATH)
print(f"Loaded matrix: {MATRIX_PATH}")
print(f"X shape: {X_all.shape}")
print(f"Class distribution (raw): {np.bincount(y_all, minlength=4)}")
print(f"Status distribution (raw; 1=clean,2=dirty): {np.bincount(y_status_all, minlength=3)}")
print(f"Predictors: {len(var_names)}")

# Remove background if present
valid = y_all > 0
X_all = X_all[valid]
y_all = y_all[valid]
y_status_all = y_status_all[valid]
rows_arr = rows_arr[valid]
cols_arr = cols_arr[valid]
print(f"After removing background: {X_all.shape[0]} points")
print(f"Class distribution (1..3): {np.bincount(y_all, minlength=4)}")
print(f"Status distribution (1=clean,2=dirty): {np.bincount(y_status_all, minlength=3)}")

# Apply clean/dirty/all status filter before any spatial/predictor filtering.
X_all, y_all, y_status_all, rows_arr, cols_arr, _ = apply_status_filter(
    X_all,
    y_all,
    y_status_all,
    rows_arr,
    cols_arr,
    TRAIN_STATUS_FILTER,
)
print(f"Class distribution after status filter: {np.bincount(y_all, minlength=4)}")

# Area filter at point level
if USE_AREA_FILTER:
    X_all, y_all, rows_arr, cols_arr, _ = filter_points_by_polygon(
        X_all,
        y_all,
        rows_arr,
        cols_arr,
        crs_wkt=crs_wkt,
        transform=transform,
        polygon_path=AREA_POLYGON_PATH,
        polygon_layer=AREA_POLYGON_LAYER,
        mode=AREA_FILTER_MODE,
    )

X_all_ref = X_all.copy()
var_names_ref = list(var_names)

# Optional predictor include-list
if USE_INCLUDED_PREDICTORS_FILE:
    X_all, var_names = apply_predictor_inclusion_file(X_all, var_names, INCLUDED_PREDICTORS_FILE)

X_all, var_names = apply_force_included_predictors(
    X_all,
    var_names,
    X_all_ref,
    var_names_ref,
    FORCE_INCLUDED_PREDICTORS,
)

X_all, var_names = apply_force_included_substrings(
    X_all,
    var_names,
    X_all_ref,
    var_names_ref,
    FORCE_INCLUDED_SUBSTRINGS,
)

effective_excluded_predictors = []
if USE_EXCLUDED_PREDICTORS:
    effective_excluded_predictors = list(EXCLUDED_PREDICTORS)
    X_all, var_names = apply_predictor_exclusions(X_all, var_names, EXCLUDED_PREDICTORS)

effective_excluded_predictor_substrings = []
if USE_EXCLUDED_PREDICTOR_SUBSTRINGS:
    X_all, var_names, effective_excluded_predictor_substrings = apply_predictor_exclusion_substrings(
        X_all,
        var_names,
        EXCLUDED_PREDICTOR_SUBSTRINGS,
    )

save_predictor_list(var_names, OUT_DIR)
print(f"Predictors used for modelling: {len(var_names)}")

# Categorical predictors: compact handling, no one-hot explosion
cat_lookup = find_categorical_indices(var_names)
cat_idx = sorted(cat_lookup.values())
print(f"Categorical predictors found: {cat_lookup if cat_lookup else 'none'}")
X_all = coerce_categorical_columns(X_all, cat_idx)

# Optional stratified sample
X_sample, y_sample, rows_sample, cols_sample = stratified_subsample(
    X_all, y_all, rows_arr, cols_arr, SAMPLE_SIZE, seed=RANDOM_SEED
)
print(f"Using {len(y_sample)} sampled points")
print(f"Sample class distribution: {np.bincount(y_sample, minlength=4)}")

# Spatial groups
groups = make_groups(rows_sample, cols_sample, width=width, block_size_px=BLOCK_SIZE_PX)
print(f"Spatial blocks: {np.unique(groups).size} unique groups using block size = {BLOCK_SIZE_PX} px")

if ENABLE_XGB_TUNING:
    tune_size = min(TUNING_SAMPLE_SIZE, len(y_sample)) if TUNING_SAMPLE_SIZE is not None else len(y_sample)
    if tune_size < len(y_sample):
        X_tune, y_tune, rows_tune, cols_tune = stratified_subsample(
            X_sample,
            y_sample,
            rows_sample,
            cols_sample,
            tune_size,
            seed=RANDOM_SEED,
        )
    else:
        X_tune, y_tune = X_sample, y_sample
        rows_tune, cols_tune = rows_sample, cols_sample

    groups_tune = make_groups(rows_tune, cols_tune, width=width, block_size_px=BLOCK_SIZE_PX)
    TUNED_XGB_PARAMS = tune_xgb_params(X_tune, y_tune, groups_tune, OUT_DIR)
    if TUNED_XGB_PARAMS:
        print(f"[INFO] Using tuned XGBoost params for main CV/final fit: {TUNED_XGB_PARAMS}")
    else:
        print("[INFO] Tuning finished with no override params. Using baseline XGBoost params.")

# CV
print("\n=== SPATIAL BLOCK CROSS-VALIDATION ===")
gkf = GroupKFold(n_splits=N_SPLITS) if ENABLE_SPATIAL_CV else None

importances_list = []
y_true_all = []
y_pred_all = []
fold_rows = []
fold_confusion_matrices = []
per_fold_reports = []
uncertainty_records = []

if ENABLE_SPATIAL_CV:
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_sample, y_sample, groups), start=1):
        print(f"\nFold {fold}/{N_SPLITS}")
        print(f"  train groups: {np.unique(groups[train_idx]).size}")
        print(f"  test groups:  {np.unique(groups[test_idx]).size}")

        X_tr = X_sample[train_idx].copy()
        X_te = X_sample[test_idx].copy()
        y_tr = y_sample[train_idx]
        y_te = y_sample[test_idx]

        if CATEGORICAL_MODE == "frequency" and cat_idx:
            X_tr, X_te = frequency_encode_train_test(X_tr, X_te, cat_idx)
        elif CATEGORICAL_MODE not in ("passthrough", "native"):
            raise ValueError("CATEGORICAL_MODE must be 'passthrough', 'frequency', or 'native'")

        sample_weight = None
        if USE_UNET_STYLE_SAMPLE_WEIGHTS:
            class_weights = compute_unet_style_class_weights(
                y_tr,
                class_multipliers=CLASS_WEIGHT_MULTIPLIERS,
            )
            sample_weight = class_weights[y_tr].astype(np.float32)
            print(f"  class weights: {class_weights}")

        y_pred, model, backend, y_prob = fit_predict_xgb(
            X_tr,
            y_tr,
            X_te,
            sample_weight=sample_weight,
            use_gpu=USE_GPU_IF_AVAILABLE,
            random_state=RANDOM_SEED + fold,
            xgb_params=TUNED_XGB_PARAMS if TUNED_XGB_PARAMS else None,
            var_names=var_names,
            cat_idx=cat_idx,
            use_native_categorical=(CATEGORICAL_MODE == "native"),
        )
        print(f"  backend used: {backend}")

        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(y_pred.tolist())

        imp = extract_feature_importances(model)
        if imp is not None:
            importances_list.append(imp)

        fold_macro_f1 = f1_score(y_te, y_pred, average="macro")
        fold_weighted_f1 = f1_score(y_te, y_pred, average="weighted")
        fold_bal_acc = balanced_accuracy_score(y_te, y_pred)
        fold_cm = confusion_matrix(y_te, y_pred, labels=[1, 2, 3])
        fold_recalls = recall_score(y_te, y_pred, labels=[1, 2, 3], average=None, zero_division=0)

        fold_rows.append({
            "fold": fold,
            "backend": backend,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "macro_f1": float(fold_macro_f1),
            "weighted_f1": float(fold_weighted_f1),
            "balanced_accuracy": float(fold_bal_acc),
            "recall_low": float(fold_recalls[0]),
            "recall_medium": float(fold_recalls[1]),
            "recall_high": float(fold_recalls[2]),
        })
        fold_confusion_matrices.append({"fold": fold, "matrix": fold_cm.copy()})
        per_fold_reports.append(
            {
                "fold": fold,
                "balanced_accuracy": float(fold_bal_acc),
                "report": classification_report(
                    y_te,
                    y_pred,
                    labels=[1, 2, 3],
                    target_names=["low", "medium", "high"],
                    digits=4,
                    zero_division=0,
                ),
            }
        )

        print(f"  macro F1:          {fold_macro_f1:.4f}")
        print(f"  weighted F1:       {fold_weighted_f1:.4f}")
        print(f"  balanced accuracy: {fold_bal_acc:.4f}")
        print("  confusion matrix (rows=true, cols=pred):")
        print(fold_cm)

        if ENABLE_UNCERTAINTY_DIAGNOSTICS and y_prob is not None and y_prob.size > 0:
            p = np.clip(y_prob.astype(np.float64), 1e-12, 1.0)
            p = p / p.sum(axis=1, keepdims=True)
            max_prob = np.max(p, axis=1)
            if p.shape[1] >= 2:
                sorted_p = np.sort(p, axis=1)
                margin = sorted_p[:, -1] - sorted_p[:, -2]
            else:
                margin = np.zeros_like(max_prob)
            entropy = -np.sum(p * np.log(p), axis=1)
            entropy_norm = entropy / np.log(p.shape[1]) if p.shape[1] > 1 else np.zeros_like(entropy)

            fold_unc_df = pd.DataFrame(
                {
                    "fold": fold,
                    "row": rows_sample[test_idx],
                    "col": cols_sample[test_idx],
                    "y_true": y_te,
                    "y_pred": y_pred,
                    "max_prob": max_prob,
                    "margin_top2": margin,
                    "entropy_norm": entropy_norm,
                    "uncertainty_1_minus_maxprob": 1.0 - max_prob,
                }
            )
            uncertainty_records.append(fold_unc_df)
else:
    print("[INFO] ENABLE_SPATIAL_CV=False: skipping fold-based CV and running final model only.")

# Reports
if ENABLE_SPATIAL_CV and fold_rows:
    print("\n=== CROSS-VALIDATED REPORT ===")
    report_txt = classification_report(
        y_true_all,
        y_pred_all,
        labels=[1, 2, 3],
        target_names=["low", "medium", "high"],
        digits=4,
        zero_division=0,
    )
    print(report_txt)

    overall_bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)
    print(f"Overall balanced accuracy: {overall_bal_acc:.4f}")

    cm = confusion_matrix(y_true_all, y_pred_all, labels=[1, 2, 3])
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    fold_df = pd.DataFrame(fold_rows)
    print("\nPer-fold metric summary:")
    print(fold_df[["fold", "macro_f1", "weighted_f1", "balanced_accuracy"]].to_string(index=False))
    print("\nMean ± SD across folds:")
    print(
        f"  macro F1:          {fold_df['macro_f1'].mean():.4f} ± {fold_df['macro_f1'].std(ddof=1) if len(fold_df) > 1 else 0.0:.4f}\n"
        f"  weighted F1:       {fold_df['weighted_f1'].mean():.4f} ± {fold_df['weighted_f1'].std(ddof=1) if len(fold_df) > 1 else 0.0:.4f}\n"
        f"  balanced accuracy: {fold_df['balanced_accuracy'].mean():.4f} ± {fold_df['balanced_accuracy'].std(ddof=1) if len(fold_df) > 1 else 0.0:.4f}"
    )
    fold_df.to_csv(os.path.join(OUT_DIR, "cv_fold_metrics.csv"), index=False)

    fold_cm_records = []
    for item in fold_confusion_matrices:
        fold_id = item["fold"]
        mat = item["matrix"]
        for true_idx, true_name in enumerate(["low", "medium", "high"]):
            for pred_idx, pred_name in enumerate(["low", "medium", "high"]):
                fold_cm_records.append({
                    "fold": fold_id,
                    "true_class": true_name,
                    "pred_class": pred_name,
                    "count": int(mat[true_idx, pred_idx]),
                })
    pd.DataFrame(fold_cm_records).to_csv(os.path.join(OUT_DIR, "cv_fold_confusion_matrices.csv"), index=False)

    if ENABLE_UNCERTAINTY_DIAGNOSTICS and uncertainty_records:
        uncertainty_df = pd.concat(uncertainty_records, axis=0, ignore_index=True)
        export_uncertainty_diagnostics(uncertainty_df, OUT_DIR)
        print(f"Saved uncertainty diagnostics: {os.path.join(OUT_DIR, 'cv_uncertainty_diagnostics.csv')}")

    summary_metrics = {
        "macro_f1_mean": float(fold_df["macro_f1"].mean()),
        "macro_f1_std": float(fold_df["macro_f1"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        "weighted_f1_mean": float(fold_df["weighted_f1"].mean()),
        "weighted_f1_std": float(fold_df["weighted_f1"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        "balanced_accuracy_mean": float(fold_df["balanced_accuracy"].mean()),
        "balanced_accuracy_std": float(fold_df["balanced_accuracy"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
    }
    pd.DataFrame([summary_metrics]).to_csv(os.path.join(OUT_DIR, "cv_summary_metrics.csv"), index=False)

    with open(os.path.join(OUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write("XGBoost from regression matrix (patched)\n")
        f.write("=" * 60 + "\n")
        f.write(f"MATRIX_PATH: {MATRIX_PATH}\n")
        f.write(f"TRAIN_STATUS_FILTER: {TRAIN_STATUS_FILTER}\n")
        f.write(f"USE_AREA_FILTER: {USE_AREA_FILTER}\n")
        f.write(f"AREA_POLYGON_PATH: {AREA_POLYGON_PATH}\n")
        f.write(f"AREA_FILTER_MODE: {AREA_FILTER_MODE}\n")
        f.write(f"CATEGORICAL_MODE: {CATEGORICAL_MODE}\n")
        f.write(f"USE_UNET_STYLE_SAMPLE_WEIGHTS: {USE_UNET_STYLE_SAMPLE_WEIGHTS}\n")
        f.write(f"CLASS_WEIGHT_MULTIPLIERS: {CLASS_WEIGHT_MULTIPLIERS}\n")
        f.write(f"SAMPLE_SIZE: {SAMPLE_SIZE}\n")
        f.write(f"BLOCK_SIZE_PX: {BLOCK_SIZE_PX}\n")
        f.write(f"ENABLE_XGB_TUNING: {ENABLE_XGB_TUNING}\n")
        f.write(f"TUNING_MAX_TRIALS: {TUNING_MAX_TRIALS}\n")
        f.write(f"TUNED_XGB_PARAMS: {TUNED_XGB_PARAMS}\n")
        f.write(f"USE_EXCLUDED_PREDICTORS: {USE_EXCLUDED_PREDICTORS}\n")
        f.write(f"EXCLUDED_PREDICTORS_CONFIG: {EXCLUDED_PREDICTORS}\n")
        f.write(f"EXCLUDED_PREDICTORS_APPLIED: {effective_excluded_predictors}\n\n")
        f.write(f"USE_EXCLUDED_PREDICTOR_SUBSTRINGS: {USE_EXCLUDED_PREDICTOR_SUBSTRINGS}\n")
        f.write(f"EXCLUDED_PREDICTOR_SUBSTRINGS_CONFIG: {EXCLUDED_PREDICTOR_SUBSTRINGS}\n")
        f.write(f"EXCLUDED_PREDICTOR_SUBSTRINGS_APPLIED: {effective_excluded_predictor_substrings}\n\n")
        f.write(report_txt)
        f.write(f"\nOverall balanced accuracy: {overall_bal_acc:.4f}\n")
        f.write("\nConfusion matrix (rows=true, cols=pred):\n")
        f.write(np.array2string(cm))
        f.write("\n\nFold metrics:\n")
        f.write(fold_df.to_string(index=False))
        f.write("\n\nPer-fold reports:\n")
        for fold_item in per_fold_reports:
            f.write(f"\n--- Fold {fold_item['fold']} ---\n")
            f.write(f"Balanced accuracy: {fold_item['balanced_accuracy']:.4f}\n")
            f.write(fold_item["report"])
else:
    print("\n[INFO] CV reports skipped because ENABLE_SPATIAL_CV=False.")
    fold_df = pd.DataFrame([])
    with open(os.path.join(OUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write("XGBoost from regression matrix (patched)\n")
        f.write("=" * 60 + "\n")
        f.write("CV skipped because ENABLE_SPATIAL_CV=False.\n")

# Mean importances across folds when available
if importances_list:
    mean_importances = np.mean(np.vstack(importances_list), axis=0)
else:
    mean_importances = np.full(X_sample.shape[1], np.nan, dtype=float)

importance_df = pd.DataFrame({
    "Variable": var_names,
    "Importance": mean_importances,
}).sort_values("Importance", ascending=False, na_position="last")

if importance_df["Importance"].notna().any():
    save_importance_outputs(importance_df, OUT_DIR)

# Final model on all sampled points
print("\n=== FIT FINAL MODEL ON ALL SAMPLED POINTS ===")
X_final = X_sample.copy()
if CATEGORICAL_MODE == "frequency" and cat_idx:
    X_final_tmp, _ = frequency_encode_train_test(X_sample.copy(), X_sample.copy(), cat_idx)
    X_final = X_final_tmp

final_sw = None
if USE_UNET_STYLE_SAMPLE_WEIGHTS:
    final_cw = compute_unet_style_class_weights(y_sample, CLASS_WEIGHT_MULTIPLIERS)
    final_sw = final_cw[y_sample].astype(np.float32)
    print(f"Final class weights: {final_cw}")

_, final_model, final_backend, _ = fit_predict_xgb(
    X_final,
    y_sample,
    X_final[: min(100, len(X_final))],
    sample_weight=final_sw,
    use_gpu=USE_GPU_IF_AVAILABLE,
    random_state=RANDOM_SEED,
    xgb_params=TUNED_XGB_PARAMS if TUNED_XGB_PARAMS else None,
    var_names=var_names,
    cat_idx=cat_idx,
    use_native_categorical=(CATEGORICAL_MODE == "native"),
)
print(f"Final backend used: {final_backend}")

# Save final model with joblib
try:
    import joblib
    model_path = os.path.join(OUT_DIR, "xgb_final_model.joblib")
    joblib.dump({
        "model": final_model,
        "var_names": var_names,
        "train_status_filter": TRAIN_STATUS_FILTER,
        "categorical_mode": CATEGORICAL_MODE,
        "categorical_predictors": cat_lookup,
        "class_weight_multipliers": CLASS_WEIGHT_MULTIPLIERS,
        "enable_xgb_tuning": ENABLE_XGB_TUNING,
        "tuned_xgb_params": TUNED_XGB_PARAMS,
        "use_excluded_predictors": USE_EXCLUDED_PREDICTORS,
        "excluded_predictors_config": EXCLUDED_PREDICTORS,
        "excluded_predictors_applied": effective_excluded_predictors,
        "use_excluded_predictor_substrings": USE_EXCLUDED_PREDICTOR_SUBSTRINGS,
        "excluded_predictor_substrings_config": EXCLUDED_PREDICTOR_SUBSTRINGS,
        "excluded_predictor_substrings_applied": effective_excluded_predictor_substrings,
        "use_orf_style": USE_ORF_STYLE,
    }, model_path)
    print(f"Saved final model: {model_path}")
except Exception as exc:
    print(f"[WARNING] Could not save final model with joblib: {exc}")

if ENABLE_SHAP_EXPORT:
    export_shap_contrib_summary(
        model=final_model,
        X_ref=X_final,
        var_names=var_names,
        out_dir=OUT_DIR,
        sample_size=SHAP_MAX_SAMPLES,
        random_seed=RANDOM_SEED,
        use_native_categorical=(CATEGORICAL_MODE == "native"),
        cat_idx=cat_idx,
    )


# =====================================================================
# TOP-N VARIABLE MODELS
# =====================================================================
if RUN_TOP_N_SUBSET_MODELS:
    for top_n in TOP_N_SUBSET_LIST:

        print(f"\n=== TOP {top_n} VARIABLE MODEL ===")
        # Dynamically extract top N variables from variable_importance.csv
        importance_csv = os.path.join(OUT_DIR, "variable_importance.csv")
        if not os.path.exists(importance_csv):
            print(f"[WARNING] Importance file not found: {importance_csv}. Skipping Top {top_n}.")
            continue
        imp_df = pd.read_csv(importance_csv)
        top_vars = imp_df["Variable"].head(top_n).tolist()

        # Filter X_sample and var_names to top N
        var_idx = [i for i, v in enumerate(var_names) if v in top_vars]
        if len(var_idx) != top_n:
            print(f"[WARNING] Only found {len(var_idx)} of top {top_n} variables in var_names. Skipping.")
            continue
        X_top = X_sample[:, var_idx]
        var_names_top = [var_names[i] for i in var_idx]
        cat_lookup_top = find_categorical_indices(var_names_top)
        cat_idx_top = sorted(cat_lookup_top.values())

        # Output directory
        out_dir_top = os.path.join(OUT_DIR, f"Top{top_n}")
        os.makedirs(out_dir_top, exist_ok=True)

        # Spatial groups (same as before)
        groups_top = make_groups(rows_sample, cols_sample, width=width, block_size_px=BLOCK_SIZE_PX)

        # Cross-validation
        gkf = GroupKFold(n_splits=N_SPLITS)
        importances_list = []
        y_true_all = []
        y_pred_all = []
        fold_rows = []
        fold_confusion_matrices = []
        per_fold_reports = []

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X_top, y_sample, groups_top), start=1):
            print(f"\n[Top{top_n}] Fold {fold}/{N_SPLITS}")
            X_tr = X_top[train_idx].copy()
            X_te = X_top[test_idx].copy()
            y_tr = y_sample[train_idx]
            y_te = y_sample[test_idx]

            if CATEGORICAL_MODE == "frequency" and cat_idx_top:
                X_tr, X_te = frequency_encode_train_test(X_tr, X_te, cat_idx_top)
            elif CATEGORICAL_MODE not in ("passthrough", "native"):
                raise ValueError("CATEGORICAL_MODE must be 'passthrough', 'frequency', or 'native'")

            sample_weight = None
            if USE_UNET_STYLE_SAMPLE_WEIGHTS:
                class_weights = compute_unet_style_class_weights(
                    y_tr,
                    class_multipliers=CLASS_WEIGHT_MULTIPLIERS,
                )
                sample_weight = class_weights[y_tr].astype(np.float32)

            y_pred, model, backend, _ = fit_predict_xgb(
                X_tr,
                y_tr,
                X_te,
                sample_weight=sample_weight,
                use_gpu=USE_GPU_IF_AVAILABLE,
                random_state=RANDOM_SEED + fold + top_n,  # ensure different seed
                xgb_params=TUNED_XGB_PARAMS if TUNED_XGB_PARAMS else None,
                var_names=var_names_top,
                cat_idx=cat_idx_top,
                use_native_categorical=(CATEGORICAL_MODE == "native"),
            )

            y_true_all.extend(y_te.tolist())
            y_pred_all.extend(y_pred.tolist())

            imp = extract_feature_importances(model)
            if imp is not None:
                importances_list.append(imp)

            fold_macro_f1 = f1_score(y_te, y_pred, average="macro")
            fold_weighted_f1 = f1_score(y_te, y_pred, average="weighted")
            fold_bal_acc = balanced_accuracy_score(y_te, y_pred)
            fold_cm = confusion_matrix(y_te, y_pred, labels=[1, 2, 3])
            fold_recalls = recall_score(y_te, y_pred, labels=[1, 2, 3], average=None, zero_division=0)

            fold_rows.append({
                "fold": fold,
                "backend": backend,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "macro_f1": float(fold_macro_f1),
                "weighted_f1": float(fold_weighted_f1),
                "balanced_accuracy": float(fold_bal_acc),
                "recall_low": float(fold_recalls[0]),
                "recall_medium": float(fold_recalls[1]),
                "recall_high": float(fold_recalls[2]),
            })
            fold_confusion_matrices.append({"fold": fold, "matrix": fold_cm.copy()})
            per_fold_reports.append(
                {
                    "fold": fold,
                    "balanced_accuracy": float(fold_bal_acc),
                    "report": classification_report(
                        y_te,
                        y_pred,
                        labels=[1, 2, 3],
                        target_names=["low", "medium", "high"],
                        digits=4,
                        zero_division=0,
                    ),
                }
            )

        # Reports
        report_txt = classification_report(
            y_true_all,
            y_pred_all,
            labels=[1, 2, 3],
            target_names=["low", "medium", "high"],
            digits=4,
            zero_division=0,
        )
        overall_bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)
        cm = confusion_matrix(y_true_all, y_pred_all, labels=[1, 2, 3])
        fold_df = pd.DataFrame(fold_rows)
        fold_df.to_csv(os.path.join(out_dir_top, "cv_fold_metrics.csv"), index=False)

        fold_cm_records = []
        for item in fold_confusion_matrices:
            fold_id = item["fold"]
            mat = item["matrix"]
            for true_idx, true_name in enumerate(["low", "medium", "high"]):
                for pred_idx, pred_name in enumerate(["low", "medium", "high"]):
                    fold_cm_records.append({
                        "fold": fold_id,
                        "true_class": true_name,
                        "pred_class": pred_name,
                        "count": int(mat[true_idx, pred_idx]),
                    })
        fold_cm_df = pd.DataFrame(fold_cm_records)
        fold_cm_df.to_csv(os.path.join(out_dir_top, "cv_fold_confusion_matrices.csv"), index=False)

        summary_metrics = {
            "macro_f1_mean": float(fold_df["macro_f1"].mean()),
            "macro_f1_std": float(fold_df["macro_f1"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
            "weighted_f1_mean": float(fold_df["weighted_f1"].mean()),
            "weighted_f1_std": float(fold_df["weighted_f1"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
            "balanced_accuracy_mean": float(fold_df["balanced_accuracy"].mean()),
            "balanced_accuracy_std": float(fold_df["balanced_accuracy"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        }
        summary_df = pd.DataFrame([summary_metrics])
        summary_df.to_csv(os.path.join(out_dir_top, "cv_summary_metrics.csv"), index=False)

        with open(os.path.join(out_dir_top, "classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(f"XGBoost from regression matrix (Top {top_n} variables)\n")
            f.write("=" * 60 + "\n")
            f.write(f"MATRIX_PATH: {MATRIX_PATH}\n")
            f.write(f"USE_AREA_FILTER: {USE_AREA_FILTER}\n")
            f.write(f"AREA_POLYGON_PATH: {AREA_POLYGON_PATH}\n")
            f.write(f"AREA_FILTER_MODE: {AREA_FILTER_MODE}\n")
            f.write(f"CATEGORICAL_MODE: {CATEGORICAL_MODE}\n")
            f.write(f"USE_UNET_STYLE_SAMPLE_WEIGHTS: {USE_UNET_STYLE_SAMPLE_WEIGHTS}\n")
            f.write(f"CLASS_WEIGHT_MULTIPLIERS: {CLASS_WEIGHT_MULTIPLIERS}\n")
            f.write(f"SAMPLE_SIZE: {SAMPLE_SIZE}\n")
            f.write(f"BLOCK_SIZE_PX: {BLOCK_SIZE_PX}\n")
            f.write(f"TOP_VARIABLES: {top_vars}\n\n")
            f.write(report_txt)
            f.write(f"\nOverall balanced accuracy: {overall_bal_acc:.4f}\n")
            f.write("\nConfusion matrix (rows=true, cols=pred):\n")
            f.write(np.array2string(cm))
            f.write("\n\nFold metrics:\n")
            f.write(fold_df.to_string(index=False))
            f.write("\n\nPer-fold reports:\n")
            for fold_item in per_fold_reports:
                f.write(f"\n--- Fold {fold_item['fold']} ---\n")
                f.write(f"Balanced accuracy: {fold_item['balanced_accuracy']:.4f}\n")
                f.write(fold_item["report"])

        # Variable importance for this subset
        if importances_list:
            mean_importances = np.mean(np.vstack(importances_list), axis=0)
        else:
            mean_importances = np.full(X_top.shape[1], np.nan, dtype=float)
        importance_df = pd.DataFrame({
            "Variable": var_names_top,
            "Importance": mean_importances,
        }).sort_values("Importance", ascending=False, na_position="last")
        importance_df.to_csv(os.path.join(out_dir_top, "variable_importance.csv"), index=False)

        # Final model on all sampled points (top N)
        X_final = X_top.copy()
        final_sw = None
        if USE_UNET_STYLE_SAMPLE_WEIGHTS:
            final_cw = compute_unet_style_class_weights(y_sample, CLASS_WEIGHT_MULTIPLIERS)
            final_sw = final_cw[y_sample].astype(np.float32)
        _, final_model, final_backend, _ = fit_predict_xgb(
            X_final,
            y_sample,
            X_final[: min(100, len(X_final))],
            sample_weight=final_sw,
            use_gpu=USE_GPU_IF_AVAILABLE,
            random_state=RANDOM_SEED + 100 + top_n,
            xgb_params=TUNED_XGB_PARAMS if TUNED_XGB_PARAMS else None,
            var_names=var_names_top,
            cat_idx=cat_idx_top,
            use_native_categorical=(CATEGORICAL_MODE == "native"),
        )
        try:
            import joblib
            model_path = os.path.join(out_dir_top, "xgb_final_model.joblib")
            joblib.dump({
                "model": final_model,
                "var_names": var_names_top,
                "categorical_mode": CATEGORICAL_MODE,
                "categorical_predictors": cat_lookup,
                "class_weight_multipliers": CLASS_WEIGHT_MULTIPLIERS,
                "enable_xgb_tuning": ENABLE_XGB_TUNING,
                "tuned_xgb_params": TUNED_XGB_PARAMS,
                "top_variables": top_vars,
                "use_orf_style": USE_ORF_STYLE,
            }, model_path)
            print(f"[Top{top_n}] Saved final model: {model_path}")
        except Exception as exc:
            print(f"[Top{top_n}] [WARNING] Could not save final model with joblib: {exc}")
else:
    print(f"[INFO] RUN_TOP_N_SUBSET_MODELS=False: skipping Top-N subset model runs ({TOP_N_SUBSET_LIST}).")

print("\n✔ Outputs written to:")
print(f"  {OUT_DIR}")
print("\nDone.")
