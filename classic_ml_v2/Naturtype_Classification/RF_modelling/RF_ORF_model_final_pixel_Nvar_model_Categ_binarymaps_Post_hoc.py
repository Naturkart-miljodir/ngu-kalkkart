#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Patched point-based Random Forest for the Kalk project using a prebuilt regression matrix.

This version:
- loads regression_matrix.npz directly
- keeps the workflow point/pixel-based (not tile-based)
- supports optional area-polygon filtering at the point level
- uses U-Net-style class weights as sample weights, with per-class multipliers
- handles categorical predictors without one-hot expansion
- tries GPU Random Forest first (RAPIDS cuML), with CPU fallback
- keeps spatial block cross-validation using rows/cols from the matrix
- saves fold metrics, confusion matrices, classification report
- saves variable importance tables and charts (top-N and full)

Important note on categorical predictors:
This script does NOT use neural embeddings like the U-Net. A Random Forest cannot
learn embeddings internally the same way. Instead, categorical predictors are kept
as compact integer-coded columns (default), with an optional frequency encoding mode.
"""

import os
import inspect
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from affine import Affine
import geopandas as gpd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.isotonic import IsotonicRegression
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
OUT_DIR = r"/home/acosta_pedro/Outputs/Models/Random_forest/Models/ORF/Weighted_post_hoc/Experiment 3"

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

# Random Forest parameters
N_ESTIMATORS = 500
MAX_DEPTH = 25
MIN_SAMPLES_LEAF = 2
MAX_FEATURES = "sqrt"
BOOTSTRAP = True

# Ordered-RF-style toggle:
# when True, train K-1 cumulative binary RF models (P[y > c_k]) and
# reconstruct class probabilities for ordered classes.
USE_ORF_STYLE = True

# GPU
USE_GPU_IF_AVAILABLE = False
GPU_RESAMPLE_WHEN_NEEDED = True    # fallback if GPU fit() lacks sample_weight
GPU_RESAMPLE_MULTIPLIER = 1.0      # 1.0 => same size as original train fold

# Imbalance handling copied from U-Net logic, but now tunable per class
USE_UNET_STYLE_SAMPLE_WEIGHTS = False
CLASS_WEIGHT_MULTIPLIERS = {
    1: 1.00,   # low
    2: 1.00,   # medium
    3: 1.00,   # high
}

# Post-hoc probability calibration + mild class-specific decision policy.
# Calibrator and thresholds are tuned using spatial CV held-out predictions only.
ENABLE_POSTHOC_CALIBRATION = True
CALIBRATION_FRACTION_WITHIN_TRAIN = 0.20
MIN_CALIBRATION_SAMPLES = 2000

ENABLE_MILD_THRESHOLD_TUNING = True
THRESH_GRID_CLASS2 = [0.38, 0.39, 0.40, 0.41, 0.42]
THRESH_GRID_CLASS3 = [0.39, 0.41, 0.43, 0.45, 0.47]
MARGIN_GRID_CLASS2 = [0.00, 0.01, 0.02, 0.03]
MARGIN_GRID_CLASS3 = [0.00, 0.01, 0.02, 0.03]
MAX_CLASS2_SHARE_INCREASE = 0.006
MAX_CLASS3_SHARE_INCREASE = 0.01
MAX_LOW_TO_MEDIUM_INCREASE = 0.0025
TUNING_SCORE_MACRO_F1_WEIGHT = 0.35
TUNING_SCORE_MEDIUM_RECALL_WEIGHT = 0.12
LOW_TO_MEDIUM_PENALTY_WEIGHT = 0.45

# Categorical handling for RF-compatible treatment
# For this run, categorical maps are represented as binary predictors.
# "passthrough" = keep columns as provided
# "frequency"   = replace category by training-fold frequency
CATEGORICAL_MODE = "passthrough"
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
INCLUDED_PREDICTORS_FILE = r"/home/acosta_pedro/Classical_ML/Predictors_file/predictors_used_for_modelling_binary.csv"

# Optional substring-based predictor exclusions.
USE_EXCLUDED_PREDICTOR_SUBSTRINGS = False
EXCLUDED_PREDICTOR_SUBSTRINGS = [
    "quaternary_",
    "landuse_",
]

# Predictors to force-include even if they are not present in the include-list file.
FORCE_INCLUDED_PREDICTORS: List[str] = []

# Variable importance output
IMPORTANCE_TOP_N = 30
SAVE_FULL_IMPORTANCE_CHART = True
MAX_VARS_IN_FULL_CHART = None   # None = all variables
FULL_CHART_FIGHEIGHT_PER_VAR = 0.22
TOP_CHART_FIGHEIGHT_PER_VAR = 0.34

# Optional extra runs on predictor subsets (Top 20/30/40/50/60/70/80 from importance table).
# Set to False while tuning the main model to reduce runtime.
RUN_TOP_N_SUBSET_MODELS = False

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


def compute_unet_style_class_weights(y_train: np.ndarray, class_multipliers: Optional[Dict[int, float]] = None) -> np.ndarray:
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
    polygon_layer: Optional[str] = None,
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


def weighted_resample(X, y, sample_weight, multiplier=1.0, seed=42):
    rng = np.random.default_rng(seed)
    n = int(round(len(y) * float(multiplier)))
    probs = sample_weight / sample_weight.sum()
    idx = rng.choice(np.arange(len(y)), size=n, replace=True, p=probs)
    return X[idx], y[idx]


def try_get_gpu_rf():
    if not USE_GPU_IF_AVAILABLE:
        return None, None
    try:
        import cudf  # noqa
        from cuml.ensemble import RandomForestClassifier as cuRF  # noqa
        return cuRF, "cuml"
    except Exception as exc:
        print(f"[INFO] GPU RF not available, falling back to CPU sklearn. Reason: {exc}")
        return None, None


def _new_cpu_rf(random_state: int):
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        bootstrap=BOOTSTRAP,
        n_jobs=-1,
        random_state=random_state,
    )


def _fit_predict_orf_style_cpu(X_tr, y_tr, X_te, sample_weight=None, random_state=42):
    classes = np.sort(np.unique(y_tr.astype(np.int32)))
    if classes.size < 2:
        raise RuntimeError("ORF-style requires at least 2 classes in training data.")

    thresholds = classes[:-1]
    cumulative_probs = []
    model_records = []
    importances = []

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

        model = _new_cpu_rf(random_state=random_state + i)
        if sample_weight is not None:
            model.fit(X_tr, y_bin, sample_weight=sample_weight)
        else:
            model.fit(X_tr, y_bin)

        prob = model.predict_proba(X_te)
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
        "model_type": "orf_style",
        "classes": classes.astype(np.int32).tolist(),
        "thresholds": thresholds.astype(np.int32).tolist(),
        "models": model_records,
    }
    if importances:
        out_model["feature_importances_"] = np.mean(np.vstack(importances), axis=0)

    return pred, probs, classes.astype(np.int32), out_model, "sklearn_orf"


def extract_feature_importances(model: Any):
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=np.float64)
    if isinstance(model, dict) and model.get("feature_importances_") is not None:
        return np.asarray(model["feature_importances_"], dtype=np.float64)
    return None


def predict_proba_rf_or_orf(model: Any, X_data: np.ndarray):
    """Return class probabilities and class labels for sklearn RF or ORF-style dict bundle."""
    if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
        probs = model.predict_proba(X_data).astype(np.float32)
        classes = np.asarray(model.classes_, dtype=np.int32)
        if classes.size > 0 and np.array_equal(classes, np.arange(classes.size, dtype=np.int32)):
            classes = classes + 1
        return probs, classes

    if isinstance(model, dict) and model.get("model_type") in {"sklearn_orf", "orf_style", "xgb_orf_style"}:
        classes = np.asarray(model.get("classes", []), dtype=np.int32)
        model_records = model.get("models", [])

        if classes.size < 2 or len(model_records) != (classes.size - 1):
            raise RuntimeError("Invalid ORF model bundle: classes/models structure is inconsistent")

        cumulative_probs = []
        for rec in model_records:
            const_p = rec.get("constant_prob")
            if const_p is not None:
                p_gt = np.full(X_data.shape[0], float(const_p), dtype=np.float32)
            else:
                sub_model = rec.get("model")
                if sub_model is None or not hasattr(sub_model, "predict_proba"):
                    raise RuntimeError("Invalid ORF model bundle: missing binary sub-model")
                sub_prob = sub_model.predict_proba(X_data)
                sub_classes = list(sub_model.classes_)
                if 1 in sub_classes:
                    pos_idx = sub_classes.index(1)
                    p_gt = sub_prob[:, pos_idx].astype(np.float32)
                else:
                    p_gt = np.zeros(X_data.shape[0], dtype=np.float32)
            cumulative_probs.append(p_gt)

        cum = np.column_stack(cumulative_probs).astype(np.float32)
        cum = np.clip(cum, 0.0, 1.0)
        for j in range(1, cum.shape[1]):
            cum[:, j] = np.minimum(cum[:, j], cum[:, j - 1])

        n = X_data.shape[0]
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
        return probs.astype(np.float32), classes

    raise RuntimeError("Unsupported model type for probability prediction")


def reorder_probs_to_classes(probs: np.ndarray, classes: np.ndarray, target_classes: np.ndarray) -> np.ndarray:
    out = np.zeros((probs.shape[0], target_classes.size), dtype=np.float32)
    class_to_idx = {int(c): i for i, c in enumerate(classes.tolist())}
    for j, tc in enumerate(target_classes.tolist()):
        src_idx = class_to_idx.get(int(tc), None)
        if src_idx is not None:
            out[:, j] = probs[:, src_idx].astype(np.float32, copy=False)
    denom = out.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return (out / denom).astype(np.float32)


def fit_isotonic_calibrator_ovr(probs: np.ndarray, y_true: np.ndarray, class_labels: np.ndarray) -> Dict[str, Any]:
    calib_models = []
    for cls in class_labels.tolist():
        target = (y_true == cls).astype(np.int32)
        p = probs[:, int(np.where(class_labels == cls)[0][0])].astype(np.float64)

        if target.min() == target.max():
            calib_models.append({"identity": True})
            continue

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p, target)
        calib_models.append(
            {
                "identity": False,
                "x": np.asarray(iso.X_thresholds_, dtype=np.float64).tolist(),
                "y": np.asarray(iso.y_thresholds_, dtype=np.float64).tolist(),
            }
        )

    return {
        "method": "ovr_isotonic",
        "classes": class_labels.astype(np.int32).tolist(),
        "models": calib_models,
    }


def apply_isotonic_calibrator_ovr(probs: np.ndarray, calibrator: Dict[str, Any]) -> np.ndarray:
    out = probs.astype(np.float64, copy=True)
    models = calibrator.get("models", [])
    for j in range(out.shape[1]):
        if j >= len(models):
            continue
        spec = models[j]
        if spec.get("identity", False):
            continue
        x = np.asarray(spec.get("x", []), dtype=np.float64)
        y = np.asarray(spec.get("y", []), dtype=np.float64)
        if x.size < 2 or y.size < 2:
            continue
        out[:, j] = np.interp(out[:, j], x, y, left=y[0], right=y[-1])

    out = np.clip(out, 0.0, 1.0)
    denom = out.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return (out / denom).astype(np.float32)


def apply_mild_class_thresholds(
    probs: np.ndarray,
    class_labels: np.ndarray,
    thr2: float,
    thr3: float,
    margin2: float,
    margin3: float,
) -> np.ndarray:
    idx = {int(c): i for i, c in enumerate(class_labels.tolist())}
    if not all(c in idx for c in [1, 2, 3]):
        return class_labels[np.argmax(probs, axis=1)].astype(np.int32)

    p1 = probs[:, idx[1]]
    p2 = probs[:, idx[2]]
    p3 = probs[:, idx[3]]

    pred = class_labels[np.argmax(probs, axis=1)].astype(np.int32)

    m2 = pred == 2
    demote_2 = m2 & ((p2 < thr2) | ((p2 - p1) < margin2))
    pred[demote_2] = 1

    m3 = pred == 3
    demote_3 = m3 & ((p3 < thr3) | ((p3 - p2) < margin3))
    pred[demote_3] = np.where(p2[demote_3] >= p1[demote_3], 2, 1)

    return pred.astype(np.int32)


def tune_mild_thresholds_from_oof(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_labels: np.ndarray,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    base_pred = class_labels[np.argmax(probs, axis=1)].astype(np.int32)
    base_share2 = float(np.mean(base_pred == 2))
    base_share3 = float(np.mean(base_pred == 3))
    low_mask = y_true == 1
    base_low_to_medium = float(np.mean(base_pred[low_mask] == 2)) if np.any(low_mask) else 0.0

    rows = []
    for thr2 in THRESH_GRID_CLASS2:
        for thr3 in THRESH_GRID_CLASS3:
            for m2 in MARGIN_GRID_CLASS2:
                for m3 in MARGIN_GRID_CLASS3:
                    y_hat = apply_mild_class_thresholds(probs, class_labels, thr2, thr3, m2, m3)
                    share2 = float(np.mean(y_hat == 2))
                    share3 = float(np.mean(y_hat == 3))
                    bal = float(balanced_accuracy_score(y_true, y_hat))
                    mac = float(f1_score(y_true, y_hat, average="macro"))
                    med_rec = float(recall_score(y_true, y_hat, labels=[2], average="macro", zero_division=0))
                    low_to_medium = float(np.mean(y_hat[low_mask] == 2)) if np.any(low_mask) else 0.0
                    ok_share = (
                        share2 <= (base_share2 + MAX_CLASS2_SHARE_INCREASE)
                        and share3 <= (base_share3 + MAX_CLASS3_SHARE_INCREASE)
                    )
                    ok_low_to_medium = low_to_medium <= (base_low_to_medium + MAX_LOW_TO_MEDIUM_INCREASE)
                    score = (
                        bal
                        + TUNING_SCORE_MACRO_F1_WEIGHT * mac
                        + TUNING_SCORE_MEDIUM_RECALL_WEIGHT * med_rec
                        - LOW_TO_MEDIUM_PENALTY_WEIGHT * low_to_medium
                    )
                    rows.append(
                        {
                            "thr2": float(thr2),
                            "thr3": float(thr3),
                            "margin2": float(m2),
                            "margin3": float(m3),
                            "balanced_accuracy": bal,
                            "macro_f1": mac,
                            "recall_medium": med_rec,
                            "share_class2": share2,
                            "share_class3": share3,
                            "low_to_medium_rate": low_to_medium,
                            "share_ok": bool(ok_share),
                            "low_to_medium_ok": bool(ok_low_to_medium),
                            "score": score,
                        }
                    )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Threshold tuning produced no candidates")

    feasible = df[df["share_ok"] & df["low_to_medium_ok"]].copy()
    if feasible.empty:
        feasible = df.copy()
        print("[WARN] No threshold candidate satisfied share/low->medium constraints. Falling back to best unconstrained candidate.")

    feasible = feasible.sort_values(
        ["score", "balanced_accuracy", "macro_f1", "low_to_medium_rate", "share_class2", "share_class3"],
        ascending=[False, False, False, True, True, True],
    )
    best = feasible.iloc[0]
    policy = {
        "thr2": float(best["thr2"]),
        "thr3": float(best["thr3"]),
        "margin2": float(best["margin2"]),
        "margin3": float(best["margin3"]),
        "base_share2": base_share2,
        "base_share3": base_share3,
        "base_low_to_medium": base_low_to_medium,
    }
    return policy, df


def fit_predict_rf(X_tr, y_tr, X_te, sample_weight=None, use_gpu=True, random_state=42):
    if USE_ORF_STYLE:
        if use_gpu:
            print("[INFO] USE_ORF_STYLE=True: using sklearn CPU ORF-style path (GPU path not implemented).")
        return _fit_predict_orf_style_cpu(
            X_tr,
            y_tr,
            X_te,
            sample_weight=sample_weight,
            random_state=random_state,
        )

    cuRF, gpu_backend = try_get_gpu_rf() if use_gpu else (None, None)

    if cuRF is not None:
        try:
            import cudf
            model = cuRF(
                n_estimators=N_ESTIMATORS,
                max_depth=MAX_DEPTH,
                max_features=1.0 if MAX_FEATURES == "sqrt" else MAX_FEATURES,
                random_state=random_state,
                n_bins=128,
            )

            fit_params = inspect.signature(model.fit).parameters
            Xtr_gpu = cudf.DataFrame(X_tr)
            ytr_gpu = cudf.Series(y_tr)
            Xte_gpu = cudf.DataFrame(X_te)

            if sample_weight is not None and "sample_weight" in fit_params:
                sw_gpu = cudf.Series(sample_weight.astype(np.float32))
                model.fit(Xtr_gpu, ytr_gpu, sample_weight=sw_gpu)
            elif sample_weight is not None and GPU_RESAMPLE_WHEN_NEEDED:
                X_res, y_res = weighted_resample(
                    X_tr,
                    y_tr,
                    sample_weight=sample_weight,
                    multiplier=GPU_RESAMPLE_MULTIPLIER,
                    seed=random_state,
                )
                model.fit(cudf.DataFrame(X_res), cudf.Series(y_res))
            else:
                model.fit(Xtr_gpu, ytr_gpu)

            pred = model.predict(Xte_gpu)
            pred = pred.to_numpy() if hasattr(pred, "to_numpy") else np.asarray(pred)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(Xte_gpu)
                probs = probs.to_numpy() if hasattr(probs, "to_numpy") else np.asarray(probs)
                classes = np.asarray(model.classes_, dtype=np.int32)
            else:
                classes = np.sort(np.unique(y_tr.astype(np.int32)))
                probs = np.zeros((len(pred), classes.size), dtype=np.float32)
                for j, cls in enumerate(classes):
                    probs[:, j] = (pred.astype(np.int32) == int(cls)).astype(np.float32)
            return pred.astype(np.int32), probs.astype(np.float32), classes.astype(np.int32), model, gpu_backend
        except Exception as exc:
            print(f"[WARNING] GPU RF failed in practice, falling back to CPU. Reason: {exc}")

    model = _new_cpu_rf(random_state=random_state)
    if sample_weight is not None:
        model.fit(X_tr, y_tr, sample_weight=sample_weight)
    else:
        model.fit(X_tr, y_tr)
    probs, classes = predict_proba_rf_or_orf(model, X_te)
    probs = probs.astype(np.float32)
    pred = classes[np.argmax(probs, axis=1)].astype(np.int32)
    return pred.astype(np.int32), probs, classes.astype(np.int32), model, "sklearn"


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
    plt.title(f"Top {top_n} Variable Importances (RF from regression matrix)")
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
        plt.title("All Variable Importances (RF from regression matrix)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "variable_importance_full.png"), dpi=220, bbox_inches="tight")
        plt.close()

    # Top table for quick reading
    importance_df.head(IMPORTANCE_TOP_N).to_csv(
        os.path.join(out_dir, f"variable_importance_top{IMPORTANCE_TOP_N}.csv"),
        index=False,
    )


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

# CV
print("\n=== SPATIAL BLOCK CROSS-VALIDATION ===")
gkf = GroupKFold(n_splits=N_SPLITS)

importances_list = []
y_true_all = []
y_pred_all = []
oof_probs_raw_all = []
oof_probs_all = []
oof_fold_ids = []
fold_rows = []
fold_confusion_matrices = []
per_fold_reports = []
fixed_class_labels = np.array([1, 2, 3], dtype=np.int32)

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
    elif CATEGORICAL_MODE != "passthrough":
        raise ValueError("CATEGORICAL_MODE must be 'passthrough' or 'frequency'")

    # Split training fold into model-fit and calibration subsets.
    if ENABLE_POSTHOC_CALIBRATION and len(y_tr) >= max(MIN_CALIBRATION_SAMPLES, 10):
        sss_cal = StratifiedShuffleSplit(
            n_splits=1,
            test_size=CALIBRATION_FRACTION_WITHIN_TRAIN,
            random_state=RANDOM_SEED + 1000 + fold,
        )
        fit_sub_idx, cal_sub_idx = next(sss_cal.split(X_tr, y_tr))
    else:
        fit_sub_idx = np.arange(len(y_tr))
        cal_sub_idx = np.array([], dtype=np.int64)

    X_fit, y_fit = X_tr[fit_sub_idx], y_tr[fit_sub_idx]
    X_cal, y_cal = X_tr[cal_sub_idx], y_tr[cal_sub_idx]

    sample_weight_fit = None
    if USE_UNET_STYLE_SAMPLE_WEIGHTS:
        class_weights = compute_unet_style_class_weights(
            y_fit,
            class_multipliers=CLASS_WEIGHT_MULTIPLIERS,
        )
        sample_weight_fit = class_weights[y_fit].astype(np.float32)
        print(f"  class weights (fit split): {class_weights}")

    _, probs_te_raw, classes_te, model, backend = fit_predict_rf(
        X_fit,
        y_fit,
        X_te,
        sample_weight=sample_weight_fit,
        use_gpu=USE_GPU_IF_AVAILABLE,
        random_state=RANDOM_SEED + fold,
    )
    print(f"  backend used: {backend}")

    probs_te = reorder_probs_to_classes(probs_te_raw, classes_te, fixed_class_labels)
    oof_probs_raw_all.append(probs_te.copy())

    if ENABLE_POSTHOC_CALIBRATION and len(cal_sub_idx) > 0:
        probs_cal_raw, classes_cal = predict_proba_rf_or_orf(model, X_cal)
        probs_cal_raw = reorder_probs_to_classes(probs_cal_raw, classes_cal, fixed_class_labels)
        calibrator_fold = fit_isotonic_calibrator_ovr(probs_cal_raw, y_cal, fixed_class_labels)
        probs_te = apply_isotonic_calibrator_ovr(probs_te, calibrator_fold)

    y_pred = fixed_class_labels[np.argmax(probs_te, axis=1)].astype(np.int32)

    y_true_all.extend(y_te.tolist())
    y_pred_all.extend(y_pred.tolist())
    oof_probs_all.append(probs_te.astype(np.float32))
    oof_fold_ids.extend([fold] * len(y_te))

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

oof_probs_all = np.vstack(oof_probs_all).astype(np.float32)
oof_probs_raw_all = np.vstack(oof_probs_raw_all).astype(np.float32)
y_true_oof = np.asarray(y_true_all, dtype=np.int32)
fold_ids_oof = np.asarray(oof_fold_ids, dtype=np.int32)

decision_policy = {
    "type": "argmax",
    "classes": fixed_class_labels.tolist(),
}

if ENABLE_MILD_THRESHOLD_TUNING:
    decision_policy_vals, tuning_df = tune_mild_thresholds_from_oof(
        y_true_oof,
        oof_probs_all,
        fixed_class_labels,
    )
    decision_policy = {
        "type": "mild_thresholds",
        "classes": fixed_class_labels.tolist(),
        **decision_policy_vals,
    }
    tuning_df.to_csv(os.path.join(OUT_DIR, "threshold_tuning_candidates.csv"), index=False)
    print("\nSelected decision policy:")
    print(decision_policy)

if decision_policy["type"] == "mild_thresholds":
    y_pred_final = apply_mild_class_thresholds(
        oof_probs_all,
        fixed_class_labels,
        thr2=decision_policy["thr2"],
        thr3=decision_policy["thr3"],
        margin2=decision_policy["margin2"],
        margin3=decision_policy["margin3"],
    )
else:
    y_pred_final = fixed_class_labels[np.argmax(oof_probs_all, axis=1)].astype(np.int32)

# Reports
print("\n=== CROSS-VALIDATED REPORT ===")
report_txt = classification_report(
    y_true_oof,
    y_pred_final,
    labels=[1, 2, 3],
    target_names=["low", "medium", "high"],
    digits=4,
    zero_division=0,
)
print(report_txt)

overall_bal_acc = balanced_accuracy_score(y_true_oof, y_pred_final)
print(f"Overall balanced accuracy: {overall_bal_acc:.4f}")

cm = confusion_matrix(y_true_oof, y_pred_final, labels=[1, 2, 3])
print("Confusion matrix (rows=true, cols=pred):")
print(cm)

fold_df = pd.DataFrame(fold_rows)
if not fold_df.empty:
    print("\nPer-fold metric summary:")
    print(fold_df[["fold", "macro_f1", "weighted_f1", "balanced_accuracy"]].to_string(index=False))
    print("\nMean ± SD across folds:")
    print(
        f"  macro F1:          {fold_df['macro_f1'].mean():.4f} ± {fold_df['macro_f1'].std(ddof=1) if len(fold_df) > 1 else 0.0:.4f}\n"
        f"  weighted F1:       {fold_df['weighted_f1'].mean():.4f} ± {fold_df['weighted_f1'].std(ddof=1) if len(fold_df) > 1 else 0.0:.4f}\n"
        f"  balanced accuracy: {fold_df['balanced_accuracy'].mean():.4f} ± {fold_df['balanced_accuracy'].std(ddof=1) if len(fold_df) > 1 else 0.0:.4f}"
    )
fold_df.to_csv(os.path.join(OUT_DIR, "cv_fold_metrics.csv"), index=False)

# Save fold-level metrics after applying selected final decision policy on OOF probabilities.
fold_policy_rows = []
for fold in sorted(np.unique(fold_ids_oof).tolist()):
    m = fold_ids_oof == fold
    y_f = y_true_oof[m]
    p_f = oof_probs_all[m]
    if decision_policy["type"] == "mild_thresholds":
        y_hat_f = apply_mild_class_thresholds(
            p_f,
            fixed_class_labels,
            thr2=decision_policy["thr2"],
            thr3=decision_policy["thr3"],
            margin2=decision_policy["margin2"],
            margin3=decision_policy["margin3"],
        )
    else:
        y_hat_f = fixed_class_labels[np.argmax(p_f, axis=1)].astype(np.int32)

    fold_policy_rows.append(
        {
            "fold": int(fold),
            "macro_f1": float(f1_score(y_f, y_hat_f, average="macro")),
            "weighted_f1": float(f1_score(y_f, y_hat_f, average="weighted")),
            "balanced_accuracy": float(balanced_accuracy_score(y_f, y_hat_f)),
            "share_class1": float(np.mean(y_hat_f == 1)),
            "share_class2": float(np.mean(y_hat_f == 2)),
            "share_class3": float(np.mean(y_hat_f == 3)),
        }
    )
pd.DataFrame(fold_policy_rows).to_csv(os.path.join(OUT_DIR, "cv_fold_metrics_decision_policy.csv"), index=False)

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
fold_cm_df.to_csv(os.path.join(OUT_DIR, "cv_fold_confusion_matrices.csv"), index=False)

summary_metrics = {
    "macro_f1_mean": float(fold_df["macro_f1"].mean()),
    "macro_f1_std": float(fold_df["macro_f1"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
    "weighted_f1_mean": float(fold_df["weighted_f1"].mean()),
    "weighted_f1_std": float(fold_df["weighted_f1"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
    "balanced_accuracy_mean": float(fold_df["balanced_accuracy"].mean()),
    "balanced_accuracy_std": float(fold_df["balanced_accuracy"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
}
summary_df = pd.DataFrame([summary_metrics])
summary_df.to_csv(os.path.join(OUT_DIR, "cv_summary_metrics.csv"), index=False)

with open(os.path.join(OUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write("Random Forest from regression matrix (patched)\n")
    f.write("=" * 60 + "\n")
    f.write(f"MATRIX_PATH: {MATRIX_PATH}\n")
    f.write(f"TRAIN_STATUS_FILTER: {TRAIN_STATUS_FILTER}\n")
    f.write(f"USE_AREA_FILTER: {USE_AREA_FILTER}\n")
    f.write(f"AREA_POLYGON_PATH: {AREA_POLYGON_PATH}\n")
    f.write(f"AREA_FILTER_MODE: {AREA_FILTER_MODE}\n")
    f.write(f"CATEGORICAL_MODE: {CATEGORICAL_MODE}\n")
    f.write(f"USE_UNET_STYLE_SAMPLE_WEIGHTS: {USE_UNET_STYLE_SAMPLE_WEIGHTS}\n")
    f.write(f"CLASS_WEIGHT_MULTIPLIERS: {CLASS_WEIGHT_MULTIPLIERS}\n")
    f.write(f"ENABLE_POSTHOC_CALIBRATION: {ENABLE_POSTHOC_CALIBRATION}\n")
    f.write(f"CALIBRATION_FRACTION_WITHIN_TRAIN: {CALIBRATION_FRACTION_WITHIN_TRAIN}\n")
    f.write(f"ENABLE_MILD_THRESHOLD_TUNING: {ENABLE_MILD_THRESHOLD_TUNING}\n")
    f.write(f"DECISION_POLICY: {decision_policy}\n")
    f.write(f"SAMPLE_SIZE: {SAMPLE_SIZE}\n")
    f.write(f"BLOCK_SIZE_PX: {BLOCK_SIZE_PX}\n")
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

_, _, _, final_model, final_backend = fit_predict_rf(
    X_final,
    y_sample,
    X_final[: min(100, len(X_final))],
    sample_weight=final_sw,
    use_gpu=USE_GPU_IF_AVAILABLE,
    random_state=RANDOM_SEED,
)
print(f"Final backend used: {final_backend}")

# Fit a final global calibrator on OOF predictions only to avoid production-time fitting.
final_calibrator = None
if ENABLE_POSTHOC_CALIBRATION:
    final_calibrator = fit_isotonic_calibrator_ovr(oof_probs_raw_all, y_true_oof, fixed_class_labels)

# Save final model with joblib if sklearn; otherwise skip explicit serialization
try:
    import joblib
    if final_backend in {"sklearn", "sklearn_orf"}:
        model_path = os.path.join(OUT_DIR, "rf_final_model.joblib")
        joblib.dump({
            "model": final_model,
            "var_names": var_names,
            "train_status_filter": TRAIN_STATUS_FILTER,
            "categorical_mode": CATEGORICAL_MODE,
            "categorical_predictors": cat_lookup,
            "class_weight_multipliers": CLASS_WEIGHT_MULTIPLIERS,
            "use_excluded_predictors": USE_EXCLUDED_PREDICTORS,
            "excluded_predictors_config": EXCLUDED_PREDICTORS,
            "excluded_predictors_applied": effective_excluded_predictors,
            "use_excluded_predictor_substrings": USE_EXCLUDED_PREDICTOR_SUBSTRINGS,
            "excluded_predictor_substrings_config": EXCLUDED_PREDICTOR_SUBSTRINGS,
            "excluded_predictor_substrings_applied": effective_excluded_predictor_substrings,
            "use_orf_style": USE_ORF_STYLE,
            "posthoc_calibrator": final_calibrator,
            "decision_policy": decision_policy,
        }, model_path)
        print(f"Saved final model: {model_path}")
    else:
        print("[INFO] Final model trained with GPU backend; model serialization skipped in this script.")
except Exception as exc:
    print(f"[WARNING] Could not save final model with joblib: {exc}")


# =====================================================================
# TOP-N VARIABLE MODELS (Top 20, 30, 40, 50, 60, 70, 80)
# =====================================================================
if RUN_TOP_N_SUBSET_MODELS:
    for top_n in [20, 30, 40, 50, 60, 70, 80]:

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

            sample_weight = None
            if USE_UNET_STYLE_SAMPLE_WEIGHTS:
                class_weights = compute_unet_style_class_weights(
                    y_tr,
                    class_multipliers=CLASS_WEIGHT_MULTIPLIERS,
                )
                sample_weight = class_weights[y_tr].astype(np.float32)

            y_pred, _, _, model, backend = fit_predict_rf(
                X_tr,
                y_tr,
                X_te,
                sample_weight=sample_weight,
                use_gpu=USE_GPU_IF_AVAILABLE,
                random_state=RANDOM_SEED + fold + top_n,  # ensure different seed
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
            f.write(f"Random Forest from regression matrix (Top {top_n} variables)\n")
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
        _, _, _, final_model, final_backend = fit_predict_rf(
            X_final,
            y_sample,
            X_final[: min(100, len(X_final))],
            sample_weight=final_sw,
            use_gpu=USE_GPU_IF_AVAILABLE,
            random_state=RANDOM_SEED + 100 + top_n,
        )
        try:
            import joblib
            if final_backend in {"sklearn", "sklearn_orf"}:
                model_path = os.path.join(out_dir_top, "rf_final_model.joblib")
                joblib.dump({
                    "model": final_model,
                    "var_names": var_names_top,
                    "categorical_mode": CATEGORICAL_MODE,
                    "categorical_predictors": cat_lookup,
                    "class_weight_multipliers": CLASS_WEIGHT_MULTIPLIERS,
                    "top_variables": top_vars,
                    "use_orf_style": USE_ORF_STYLE,
                }, model_path)
                print(f"[Top{top_n}] Saved final model: {model_path}")
            else:
                print(f"[Top{top_n}] Final model trained with GPU backend; model serialization skipped.")
        except Exception as exc:
            print(f"[Top{top_n}] [WARNING] Could not save final model with joblib: {exc}")
else:
    print("[INFO] RUN_TOP_N_SUBSET_MODELS=False: skipping Top 20/30/40/50/60/70/80 subset model runs.")

print("\n✔ Outputs written to:")
print(f"  {OUT_DIR}")
print("\nDone.")
