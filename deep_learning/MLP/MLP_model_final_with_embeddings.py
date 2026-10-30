#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Point-based MLP model for the Kalk project using a prebuilt regression matrix.

This version:
- loads regression_matrix.npz directly
- keeps the workflow point/pixel-based (not tile-based)
- supports optional area-polygon filtering at the point level
- uses U-Net-style class weights as sample weights, with per-class multipliers
- uses categorical embeddings for selected categorical predictor maps
- keeps spatial block cross-validation using rows/cols from the matrix
- supports optional MLP tuning with GroupKFold
- saves fold metrics, confusion matrices, classification report
- saves variable ranking tables/charts for Top-N reruns
"""

import os
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from affine import Affine
import geopandas as gpd

from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    recall_score,
)
from sklearn.feature_selection import mutual_info_classif
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Embedding, Flatten, Input
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# USER SETTINGS
# =====================================================================

MATRIX_PATH = r"D:\Classical_ML\regression_matrix_2026_total_with_clean_ed.npz"
OUT_DIR = r"D:\MultiLayerPerceptron_models"

# Target-status filtering based on regression matrix y_status values.
# - "all": keep clean + dirty
# - "clean": keep only clean
# - "dirty": keep only dirty
TRAIN_STATUS_FILTER = "clean"

# Optional area filter at point level
USE_AREA_FILTER = True
AREA_POLYGON_PATH = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\MPM_MidNorge\Mask\Shape\TrondHed_pol_mask.shp"
AREA_POLYGON_LAYER = None
AREA_FILTER_MODE = "intersects"   # "intersects" or "within"

# Sampling and CV
SAMPLE_SIZE = 500000               # set to None to use all filtered points
N_SPLITS = 5
BLOCK_SIZE_PX = 2000  # 2000 * 10m = 20km
RANDOM_SEED = 42

# MLP parameters
MLP_HIDDEN_UNITS = [256, 128]
MLP_DROPOUT = 0.25
MLP_L2 = 1e-5
MLP_LEARNING_RATE = 1e-3
MLP_BATCH_SIZE = 4096
MLP_EPOCHS = 50
MLP_VAL_SPLIT = 0.1
MLP_PATIENCE = 6

# Optional MLP tuning with spatial GroupKFold.
ENABLE_MLP_TUNING = False
TUNING_MAX_TRIALS = 12
TUNING_SAMPLE_SIZE = 150000

# Categorical embeddings settings (aligned with U-Net conventions).
USE_CATEGORICAL_EMBEDDINGS = True
# Set to None to auto-detect class counts from filtered training data each run.
# Set explicit integers to force fixed caps.
QUATERNARY_NUM_CLASSES = None
LANDUSE_NUM_CLASSES = None
QUATERNARY_EMBED_DIM = 5
LANDUSE_EMBED_DIM = 6
REMOVE_EMBED_CHANNELS_FROM_CONTINUOUS = True

# GPU/CPU selection for TensorFlow
USE_GPU_IF_AVAILABLE = True

# Imbalance handling copied from U-Net logic, but now tunable per class
USE_UNET_STYLE_SAMPLE_WEIGHTS = True
CLASS_WEIGHT_MULTIPLIERS = {
    1: 1.00,   # low
    2: 1.80,   # medium (aligned with U-Net CLASS2_WEIGHT_MULTIPLIER)
    3: 1.00,   # high
}

# Continuous-feature preprocessing (critical for MLP stability).
# This is applied per fold using train-only statistics.
ENABLE_CONTINUOUS_STANDARDIZATION = True
CONTINUOUS_ZSCORE_CLIP = 8.0
LOG1P_CONTINUOUS_PREDICTORS = [
]
NO_SCALE_CONTINUOUS_PREDICTORS = [
    "KalsiumElvInnsjo4_10m_masked_cog",
]

# Categorical handling before embeddings
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
INCLUDED_PREDICTORS_FILE = r"D:\Classical_ML_models\Test\XGBoost\Categorical_maps\Ca_prob\Non_ORF_style\clean\Tunned\All_tunned\predictors_used_for_modelling.csv"

# Optional substring-based predictor exclusions.
USE_EXCLUDED_PREDICTOR_SUBSTRINGS = True
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

# Optional extra runs on predictor subsets (Top 10/20/30 from importance table).
# Set to False while tuning the main model to reduce runtime.
RUN_TOP_N_SUBSET_MODELS = False
IMPORTANCE_SAMPLE_SIZE = 120000

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)
tf.keras.utils.set_random_seed(RANDOM_SEED)


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


def summarize_effective_class_weights(y_train: np.ndarray, sample_weight: np.ndarray) -> Dict[int, float]:
    """Return normalized effective weight mass per class for quick diagnostics."""
    out: Dict[int, float] = {}
    if sample_weight is None or len(y_train) == 0:
        return out

    total = float(np.sum(sample_weight))
    if total <= 0:
        return out

    for cls in [1, 2, 3]:
        cls_mass = float(np.sum(sample_weight[y_train == cls]))
        out[cls] = cls_mass / total
    return out


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


DEFAULT_MLP_PARAMS: Dict[str, Any] = {
    "hidden_units": list(MLP_HIDDEN_UNITS),
    "dropout": float(MLP_DROPOUT),
    "l2": float(MLP_L2),
    "learning_rate": float(MLP_LEARNING_RATE),
    "batch_size": int(MLP_BATCH_SIZE),
    "epochs": int(MLP_EPOCHS),
}

TUNED_MLP_PARAMS: Dict[str, Any] = dict(DEFAULT_MLP_PARAMS)

# Runtime-resolved embedding class caps (detected from current run data unless fixed above).
ACTIVE_QUATERNARY_NUM_CLASSES = 1
ACTIVE_LANDUSE_NUM_CLASSES = 1


def build_mlp_tuning_candidates():
    candidates = [
        {"hidden_units": [256, 128], "dropout": 0.25, "l2": 1e-5, "learning_rate": 1e-3, "batch_size": 4096, "epochs": 35},
        {"hidden_units": [256, 128], "dropout": 0.35, "l2": 1e-5, "learning_rate": 7e-4, "batch_size": 4096, "epochs": 35},
        {"hidden_units": [384, 192], "dropout": 0.25, "l2": 1e-5, "learning_rate": 1e-3, "batch_size": 4096, "epochs": 35},
        {"hidden_units": [384, 192], "dropout": 0.30, "l2": 1e-4, "learning_rate": 8e-4, "batch_size": 4096, "epochs": 35},
        {"hidden_units": [512, 256], "dropout": 0.30, "l2": 1e-5, "learning_rate": 8e-4, "batch_size": 4096, "epochs": 35},
        {"hidden_units": [512, 256], "dropout": 0.35, "l2": 1e-4, "learning_rate": 6e-4, "batch_size": 4096, "epochs": 35},
        {"hidden_units": [320, 160], "dropout": 0.20, "l2": 1e-5, "learning_rate": 1e-3, "batch_size": 2048, "epochs": 35},
        {"hidden_units": [320, 160], "dropout": 0.30, "l2": 1e-4, "learning_rate": 7e-4, "batch_size": 2048, "epochs": 35},
        {"hidden_units": [256, 128, 64], "dropout": 0.30, "l2": 1e-5, "learning_rate": 8e-4, "batch_size": 4096, "epochs": 35},
        {"hidden_units": [384, 192, 96], "dropout": 0.30, "l2": 1e-4, "learning_rate": 7e-4, "batch_size": 4096, "epochs": 35},
        {"hidden_units": [192, 96], "dropout": 0.20, "l2": 1e-5, "learning_rate": 1.2e-3, "batch_size": 4096, "epochs": 35},
        {"hidden_units": [448, 224], "dropout": 0.35, "l2": 1e-4, "learning_rate": 6e-4, "batch_size": 4096, "epochs": 35},
    ]
    return candidates[:TUNING_MAX_TRIALS]


def _sanitize_embedding_ids(arr: np.ndarray, max_id: int) -> np.ndarray:
    out = np.rint(np.where(np.isfinite(arr), arr, 0)).astype(np.int32)
    out = np.where((out < 0) | (out > max_id), 0, out)
    return out


def resolve_embedding_class_counts(X: np.ndarray, var_names: List[str]) -> Tuple[int, int]:
    cat_lookup_local = find_categorical_indices(var_names)

    def _resolve_one(name: str, configured: Any) -> int:
        configured_int = int(configured) if configured is not None else None
        if configured_int is not None:
            return max(1, configured_int)

        idx = cat_lookup_local.get(name, None)
        if idx is None:
            return 1

        col = np.rint(np.where(np.isfinite(X[:, idx]), X[:, idx], 0)).astype(np.int32)
        col = np.where(col < 0, 0, col)
        max_id = int(col.max()) if col.size else 0
        return max(1, max_id)

    qua_n = _resolve_one(QUATERNARY_PREDICTOR_NAME, QUATERNARY_NUM_CLASSES)
    land_n = _resolve_one(LANDUSE_PREDICTOR_NAME, LANDUSE_NUM_CLASSES)
    return qua_n, land_n


def _prepare_mlp_inputs(X: np.ndarray, var_names: List[str], embedding_enabled: bool):
    cat_lookup_local = find_categorical_indices(var_names)
    qua_idx = cat_lookup_local.get(QUATERNARY_PREDICTOR_NAME, None)
    land_idx = cat_lookup_local.get(LANDUSE_PREDICTOR_NAME, None)

    if embedding_enabled:
        if qua_idx is None:
            X_qua = np.zeros(X.shape[0], dtype=np.int32)
        else:
            X_qua = _sanitize_embedding_ids(X[:, qua_idx], ACTIVE_QUATERNARY_NUM_CLASSES)

        if land_idx is None:
            X_land = np.zeros(X.shape[0], dtype=np.int32)
        else:
            X_land = _sanitize_embedding_ids(X[:, land_idx], ACTIVE_LANDUSE_NUM_CLASSES)

        if REMOVE_EMBED_CHANNELS_FROM_CONTINUOUS:
            remove_idx = {i for i in [qua_idx, land_idx] if i is not None}
            cont_idx = [i for i in range(X.shape[1]) if i not in remove_idx]
        else:
            cont_idx = list(range(X.shape[1]))

        if not cont_idx:
            X_cont = np.zeros((X.shape[0], 1), dtype=np.float32)
        else:
            X_cont = X[:, cont_idx].astype(np.float32)

        return [X_cont, X_qua, X_land], cont_idx

    return [X.astype(np.float32)], list(range(X.shape[1]))


def _fit_continuous_preprocessor(X_tr_cont: np.ndarray, cont_var_names: List[str]) -> Dict[str, Any]:
    """Fit train-only preprocessing stats for continuous MLP inputs."""
    Xtr = np.asarray(X_tr_cont, dtype=np.float32).copy()
    Xtr = np.where(np.isfinite(Xtr), Xtr, np.nan)

    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)

    tr_nan = np.isnan(Xtr)
    if np.any(tr_nan):
        Xtr[tr_nan] = med[np.where(tr_nan)[1]]

    log_targets = {str(v).strip().lower() for v in LOG1P_CONTINUOUS_PREDICTORS}
    no_scale_targets = {str(v).strip().lower() for v in NO_SCALE_CONTINUOUS_PREDICTORS}
    log1p_mask = np.zeros(Xtr.shape[1], dtype=bool)
    no_scale_mask = np.zeros(Xtr.shape[1], dtype=bool)

    for i, name in enumerate(cont_var_names):
        name_lc = str(name).strip().lower()
        no_scale_mask[i] = name_lc in no_scale_targets
        log1p_mask[i] = name_lc in log_targets and not no_scale_mask[i]
        if log1p_mask[i]:
            Xtr[:, i] = np.log1p(np.clip(Xtr[:, i], 0.0, None))

    mean = np.zeros(Xtr.shape[1], dtype=np.float32)
    std = np.ones(Xtr.shape[1], dtype=np.float32)

    if ENABLE_CONTINUOUS_STANDARDIZATION:
        scale_idx = np.where(~no_scale_mask)[0].tolist()
        if scale_idx:
            Xtr_s = Xtr[:, scale_idx]
            mu = Xtr_s.mean(axis=0, dtype=np.float64).astype(np.float32)
            sd = Xtr_s.std(axis=0, dtype=np.float64).astype(np.float32)
            sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
            mean[scale_idx] = mu
            std[scale_idx] = sd

    return {
        "feature_names": list(cont_var_names),
        "median_raw": med.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "log1p_mask": log1p_mask.astype(np.int8).tolist(),
        "no_scale_mask": no_scale_mask.astype(np.int8).tolist(),
        "enable_standardization": bool(ENABLE_CONTINUOUS_STANDARDIZATION),
        "zscore_clip": CONTINUOUS_ZSCORE_CLIP,
    }


def _apply_continuous_preprocessor(
    X_cont: np.ndarray,
    preprocessor: Dict[str, Any],
) -> np.ndarray:
    """Apply saved continuous preprocessing stats to a feature matrix."""
    out = np.asarray(X_cont, dtype=np.float32).copy()
    out = np.where(np.isfinite(out), out, np.nan)

    med = np.asarray(preprocessor["median_raw"], dtype=np.float32)
    mean = np.asarray(preprocessor["mean"], dtype=np.float32)
    std = np.asarray(preprocessor["std"], dtype=np.float32)
    log1p_mask = np.asarray(preprocessor["log1p_mask"], dtype=bool)
    no_scale_mask = np.asarray(preprocessor["no_scale_mask"], dtype=bool)

    nan_mask = np.isnan(out)
    if np.any(nan_mask):
        out[nan_mask] = med[np.where(nan_mask)[1]]

    if np.any(log1p_mask):
        for idx in np.where(log1p_mask)[0].tolist():
            out[:, idx] = np.log1p(np.clip(out[:, idx], 0.0, None))

    if bool(preprocessor.get("enable_standardization", False)):
        scale_idx = np.where(~no_scale_mask)[0].tolist()
        if scale_idx:
            out_s = out[:, scale_idx]
            mu = mean[scale_idx]
            sd = std[scale_idx]
            sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
            out_s = (out_s - mu) / sd
            clip_v = preprocessor.get("zscore_clip", None)
            if clip_v is not None:
                out_s = np.clip(out_s, -float(clip_v), float(clip_v))
            out[:, scale_idx] = out_s

    return out.astype(np.float32)


def _preprocess_continuous_train_test(
    X_tr_cont: np.ndarray,
    X_te_cont: np.ndarray,
    cont_var_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply train-fold-only preprocessing for MLP continuous inputs."""
    preprocessor = _fit_continuous_preprocessor(X_tr_cont, cont_var_names)
    Xtr = _apply_continuous_preprocessor(X_tr_cont, preprocessor)
    Xte = _apply_continuous_preprocessor(X_te_cont, preprocessor)
    return Xtr, Xte


def _build_mlp_model(n_cont_features: int, num_classes: int, params: Dict[str, Any], embedding_enabled: bool):
    hidden_units = list(params["hidden_units"])
    dropout = float(params["dropout"])
    l2_value = float(params["l2"])

    cont_input = Input(shape=(n_cont_features,), dtype="float32", name="continuous_input")
    to_concat = [cont_input]
    model_inputs = [cont_input]

    if embedding_enabled:
        qua_input = Input(shape=(1,), dtype="int32", name="quaternary_input")
        land_input = Input(shape=(1,), dtype="int32", name="landuse_input")

        qua_emb = Embedding(
            input_dim=ACTIVE_QUATERNARY_NUM_CLASSES + 1,
            output_dim=QUATERNARY_EMBED_DIM,
            name="quaternary_embedding",
        )(qua_input)
        land_emb = Embedding(
            input_dim=ACTIVE_LANDUSE_NUM_CLASSES + 1,
            output_dim=LANDUSE_EMBED_DIM,
            name="landuse_embedding",
        )(land_input)

        to_concat.append(Flatten(name="quaternary_flatten")(qua_emb))
        to_concat.append(Flatten(name="landuse_flatten")(land_emb))
        model_inputs.extend([qua_input, land_input])

    if len(to_concat) > 1:
        x = Concatenate(name="feature_concat")(to_concat)
    else:
        x = to_concat[0]

    for i, units in enumerate(hidden_units):
        x = Dense(
            int(units),
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(l2_value),
            name=f"dense_{i}",
        )(x)
        x = Dropout(dropout, name=f"dropout_{i}")(x)

    output = Dense(num_classes, activation="softmax", name="class_output")(x)
    model = Model(inputs=model_inputs, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=float(params["learning_rate"])),
        loss="sparse_categorical_crossentropy",
        weighted_metrics=[],
    )
    return model


def fit_predict_mlp(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    var_names_local: List[str],
    sample_weight: np.ndarray = None,
    random_state: int = 42,
    mlp_params: Dict[str, Any] = None,
):
    params = dict(DEFAULT_MLP_PARAMS)
    if mlp_params:
        params.update(mlp_params)

    classes = np.sort(np.unique(y_tr.astype(np.int32)))
    class_to_idx = {int(c): i for i, c in enumerate(classes.tolist())}
    idx_to_class = {i: int(c) for i, c in enumerate(classes.tolist())}

    y_tr_idx = np.asarray([class_to_idx[int(v)] for v in y_tr], dtype=np.int32)

    use_embeddings = bool(USE_CATEGORICAL_EMBEDDINGS)
    Xtr_inputs, cont_idx = _prepare_mlp_inputs(X_tr, var_names_local, embedding_enabled=use_embeddings)
    Xte_inputs, _ = _prepare_mlp_inputs(X_te, var_names_local, embedding_enabled=use_embeddings)

    cont_var_names = [var_names_local[i] for i in cont_idx]
    Xtr_inputs[0], Xte_inputs[0] = _preprocess_continuous_train_test(
        Xtr_inputs[0],
        Xte_inputs[0],
        cont_var_names,
    )

    n_cont_features = int(Xtr_inputs[0].shape[1])
    model = _build_mlp_model(n_cont_features, len(classes), params, embedding_enabled=use_embeddings)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=MLP_PATIENCE,
            restore_best_weights=True,
            verbose=0,
        )
    ]

    fit_kwargs: Dict[str, Any] = {
        "x": Xtr_inputs,
        "y": y_tr_idx,
        "epochs": int(params["epochs"]),
        "batch_size": int(params["batch_size"]),
        "verbose": 0,
        "callbacks": callbacks,
        "shuffle": True,
    }
    if len(y_tr_idx) > 1000:
        fit_kwargs["validation_split"] = float(MLP_VAL_SPLIT)
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    tf.keras.utils.set_random_seed(random_state)
    model.fit(**fit_kwargs)

    probs = model.predict(Xte_inputs, batch_size=int(params["batch_size"]), verbose=0)
    pred_idx = np.argmax(probs, axis=1).astype(np.int32)
    pred = np.asarray([idx_to_class[int(v)] for v in pred_idx], dtype=np.int32)

    bundle = {
        "model": model,
        "classes": classes.tolist(),
        "embedding_enabled": use_embeddings,
    }
    return pred, bundle, "mlp_tf"


def tune_mlp_params(X_samp, y_samp, groups, var_names_local: List[str], out_dir):
    os.makedirs(out_dir, exist_ok=True)
    candidates = build_mlp_tuning_candidates()
    gkf_tune = GroupKFold(n_splits=N_SPLITS)
    split_iter = list(gkf_tune.split(X_samp, y_samp, groups))

    trial_rows = []
    best_params = None
    best_f1 = -np.inf

    print(f"\n=== TUNING MLP (embeddings) | trials={len(candidates)} ===")
    for i, params in enumerate(candidates, start=1):
        fold_f1s = []
        for train_idx, test_idx in split_iter:
            X_tr = X_samp[train_idx].copy()
            X_te = X_samp[test_idx].copy()
            y_tr = y_samp[train_idx]
            y_te = y_samp[test_idx]

            if CATEGORICAL_MODE == "frequency" and cat_idx:
                X_tr, X_te = frequency_encode_train_test(X_tr, X_te, cat_idx)

            sample_weight = None
            if USE_UNET_STYLE_SAMPLE_WEIGHTS:
                class_weights = compute_unet_style_class_weights(
                    y_tr,
                    class_multipliers=CLASS_WEIGHT_MULTIPLIERS,
                )
                sample_weight = class_weights[y_tr].astype(np.float32)

            y_pred, _, _ = fit_predict_mlp(
                X_tr,
                y_tr,
                X_te,
                var_names_local=var_names_local,
                sample_weight=sample_weight,
                random_state=RANDOM_SEED + i * 10,
                mlp_params=params,
            )
            fold_f1s.append(f1_score(y_te, y_pred, average="macro", zero_division=0))

        mean_f1 = float(np.mean(fold_f1s))
        print(f"[Tune {i:02d}] macro-F1={mean_f1:.4f}")
        trial_rows.append({"trial": i, "mean_macro_f1": mean_f1, **params})

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_params = params

    pd.DataFrame(trial_rows).sort_values("mean_macro_f1", ascending=False).to_csv(
        os.path.join(out_dir, "mlp_tuning_trials.csv"),
        index=False,
    )
    if best_params:
        with open(os.path.join(out_dir, "mlp_best_params.txt"), "w", encoding="utf-8") as f:
            f.write(f"Best macro-F1: {best_f1:.6f}\n")
            for k, v in best_params.items():
                f.write(f"{k}: {v}\n")
        print(f"Best params: macro-F1={best_f1:.4f} | {best_params}")
        return best_params
    return {}


def compute_feature_ranking(X: np.ndarray, y: np.ndarray, var_names_local: List[str], max_samples: int):
    if len(y) > max_samples:
        X_sub, y_sub, _, _ = stratified_subsample(X, y, np.arange(len(y)), np.arange(len(y)), max_samples, seed=RANDOM_SEED)
    else:
        X_sub, y_sub = X, y

    # Mutual information provides a stable, model-agnostic ranking for Top-N reruns.
    scores = mutual_info_classif(X_sub, y_sub.astype(np.int32), discrete_features=False, random_state=RANDOM_SEED)
    scores = np.asarray(scores, dtype=np.float64)
    if np.all(~np.isfinite(scores)) or np.nansum(scores) <= 0:
        scores = np.nan_to_num(np.var(X_sub, axis=0), nan=0.0, posinf=0.0, neginf=0.0)

    return pd.DataFrame(
        {
            "Variable": var_names_local,
            "Importance": scores,
        }
    ).sort_values("Importance", ascending=False, na_position="last")


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
    plt.title(f"Top {top_n} Variable Ranking (MLP embeddings)")
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
        plt.title("All Variable Ranking (MLP embeddings)")
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

if USE_CATEGORICAL_EMBEDDINGS:
    ACTIVE_QUATERNARY_NUM_CLASSES, ACTIVE_LANDUSE_NUM_CLASSES = resolve_embedding_class_counts(X_all, var_names)
    print(
        "[INFO] Embedding class caps resolved "
        f"(quaternary={ACTIVE_QUATERNARY_NUM_CLASSES}, landuse={ACTIVE_LANDUSE_NUM_CLASSES})"
    )
else:
    ACTIVE_QUATERNARY_NUM_CLASSES = max(1, int(QUATERNARY_NUM_CLASSES) if QUATERNARY_NUM_CLASSES is not None else 1)
    ACTIVE_LANDUSE_NUM_CLASSES = max(1, int(LANDUSE_NUM_CLASSES) if LANDUSE_NUM_CLASSES is not None else 1)

# Optional stratified sample
X_sample, y_sample, rows_sample, cols_sample = stratified_subsample(
    X_all, y_all, rows_arr, cols_arr, SAMPLE_SIZE, seed=RANDOM_SEED
)
print(f"Using {len(y_sample)} sampled points")
print(f"Sample class distribution: {np.bincount(y_sample, minlength=4)}")

# Spatial groups
groups = make_groups(rows_sample, cols_sample, width=width, block_size_px=BLOCK_SIZE_PX)
print(f"Spatial blocks: {np.unique(groups).size} unique groups using block size = {BLOCK_SIZE_PX} px")

if ENABLE_MLP_TUNING:
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
    TUNED_MLP_PARAMS = tune_mlp_params(X_tune, y_tune, groups_tune, var_names, OUT_DIR)
    if TUNED_MLP_PARAMS:
        print(f"[INFO] Using tuned MLP params for main CV/final fit: {TUNED_MLP_PARAMS}")
    else:
        print("[INFO] Tuning finished with no override params. Using baseline MLP params.")

# CV
print("\n=== SPATIAL BLOCK CROSS-VALIDATION ===")
gkf = GroupKFold(n_splits=N_SPLITS)

y_true_all = []
y_pred_all = []
fold_rows = []
fold_confusion_matrices = []
per_fold_reports = []

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

    sample_weight = None
    if USE_UNET_STYLE_SAMPLE_WEIGHTS:
        class_weights = compute_unet_style_class_weights(
            y_tr,
            class_multipliers=CLASS_WEIGHT_MULTIPLIERS,
        )
        sample_weight = class_weights[y_tr].astype(np.float32)
        print(f"  class weights: {class_weights}")
        eff = summarize_effective_class_weights(y_tr, sample_weight)
        if eff:
            print(
                "  effective weight mass by class: "
                f"low={eff.get(1, 0.0):.3f}, medium={eff.get(2, 0.0):.3f}, high={eff.get(3, 0.0):.3f}"
            )

    y_pred, model_bundle, backend = fit_predict_mlp(
        X_tr,
        y_tr,
        X_te,
        var_names_local=var_names,
        sample_weight=sample_weight,
        random_state=RANDOM_SEED + fold,
        mlp_params=TUNED_MLP_PARAMS if TUNED_MLP_PARAMS else None,
    )
    print(f"  backend used: {backend}")

    y_true_all.extend(y_te.tolist())
    y_pred_all.extend(y_pred.tolist())

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

# Reports
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
    f.write("MLP with embeddings from regression matrix\n")
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
    f.write(f"ENABLE_MLP_TUNING: {ENABLE_MLP_TUNING}\n")
    f.write(f"TUNING_MAX_TRIALS: {TUNING_MAX_TRIALS}\n")
    f.write(f"USE_CATEGORICAL_EMBEDDINGS: {USE_CATEGORICAL_EMBEDDINGS}\n")
    f.write(f"QUATERNARY_NUM_CLASSES_CONFIG: {QUATERNARY_NUM_CLASSES}\n")
    f.write(f"LANDUSE_NUM_CLASSES_CONFIG: {LANDUSE_NUM_CLASSES}\n")
    f.write(f"QUATERNARY_NUM_CLASSES_ACTIVE: {ACTIVE_QUATERNARY_NUM_CLASSES}\n")
    f.write(f"LANDUSE_NUM_CLASSES_ACTIVE: {ACTIVE_LANDUSE_NUM_CLASSES}\n")
    f.write(f"QUATERNARY_EMBED_DIM: {QUATERNARY_EMBED_DIM}\n")
    f.write(f"LANDUSE_EMBED_DIM: {LANDUSE_EMBED_DIM}\n")
    f.write(f"TUNED_MLP_PARAMS: {TUNED_MLP_PARAMS}\n")
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

importance_df = compute_feature_ranking(
    X_sample,
    y_sample,
    var_names,
    max_samples=IMPORTANCE_SAMPLE_SIZE,
)

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
    eff = summarize_effective_class_weights(y_sample, final_sw)
    if eff:
        print(
            "Final effective weight mass by class: "
            f"low={eff.get(1, 0.0):.3f}, medium={eff.get(2, 0.0):.3f}, high={eff.get(3, 0.0):.3f}"
        )

final_inputs_for_stats, final_cont_idx = _prepare_mlp_inputs(
    X_final,
    var_names,
    embedding_enabled=bool(USE_CATEGORICAL_EMBEDDINGS),
)
final_cont_var_names = [var_names[i] for i in final_cont_idx]
final_continuous_preprocessor = _fit_continuous_preprocessor(
    final_inputs_for_stats[0],
    final_cont_var_names,
)

_, final_bundle, final_backend = fit_predict_mlp(
    X_final,
    y_sample,
    X_final[: min(100, len(X_final))],
    var_names_local=var_names,
    sample_weight=final_sw,
    random_state=RANDOM_SEED,
    mlp_params=TUNED_MLP_PARAMS if TUNED_MLP_PARAMS else None,
)
print(f"Final backend used: {final_backend}")

# Save final model (Keras) with metadata bundle
try:
    import joblib
    model_path = os.path.join(OUT_DIR, "mlp_final_model.keras")
    final_bundle["model"].save(model_path)
    meta_path = os.path.join(OUT_DIR, "mlp_final_model_meta.joblib")
    joblib.dump({
        "model_path": model_path,
        "model_type": "mlp_with_embeddings",
        "classes": final_bundle["classes"],
        "var_names": var_names,
        "train_status_filter": TRAIN_STATUS_FILTER,
        "categorical_mode": CATEGORICAL_MODE,
        "categorical_predictors": cat_lookup,
        "class_weight_multipliers": CLASS_WEIGHT_MULTIPLIERS,
        "enable_mlp_tuning": ENABLE_MLP_TUNING,
        "tuned_mlp_params": TUNED_MLP_PARAMS,
        "use_categorical_embeddings": USE_CATEGORICAL_EMBEDDINGS,
        "continuous_preprocessor": final_continuous_preprocessor,
        "quaternary_num_classes_config": QUATERNARY_NUM_CLASSES,
        "landuse_num_classes_config": LANDUSE_NUM_CLASSES,
        "quaternary_num_classes_active": ACTIVE_QUATERNARY_NUM_CLASSES,
        "landuse_num_classes_active": ACTIVE_LANDUSE_NUM_CLASSES,
        "quaternary_embed_dim": QUATERNARY_EMBED_DIM,
        "landuse_embed_dim": LANDUSE_EMBED_DIM,
        "use_excluded_predictors": USE_EXCLUDED_PREDICTORS,
        "excluded_predictors_config": EXCLUDED_PREDICTORS,
        "excluded_predictors_applied": effective_excluded_predictors,
        "use_excluded_predictor_substrings": USE_EXCLUDED_PREDICTOR_SUBSTRINGS,
        "excluded_predictor_substrings_config": EXCLUDED_PREDICTOR_SUBSTRINGS,
        "excluded_predictor_substrings_applied": effective_excluded_predictor_substrings,
    }, meta_path)
    print(f"Saved final model: {model_path}")
    print(f"Saved final metadata: {meta_path}")
except Exception as exc:
    print(f"[WARNING] Could not save final model: {exc}")


# =====================================================================
# TOP-N VARIABLE MODELS
# =====================================================================
if RUN_TOP_N_SUBSET_MODELS:
    for top_n in [10, 20, 30, 40, 50, 60, 70, 80]:

        print(f"\n=== TOP {top_n} VARIABLE MODEL ===")
        # Dynamically extract top N variables from variable_importance.csv
        importance_csv = os.path.join(OUT_DIR, "variable_importance.csv")
        if not os.path.exists(importance_csv):
            print(f"[WARNING] Importance file not found: {importance_csv}. Skipping Top {top_n}.")
            continue
        imp_df = pd.read_csv(importance_csv)
        top_vars = imp_df["Variable"].head(top_n).tolist()

        # Filter X_sample and var_names to top N
        if USE_CATEGORICAL_EMBEDDINGS:
            for forced_cat in [QUATERNARY_PREDICTOR_NAME, LANDUSE_PREDICTOR_NAME]:
                if forced_cat in var_names and forced_cat not in top_vars:
                    top_vars.append(forced_cat)

        var_idx = [i for i, v in enumerate(var_names) if v in top_vars]
        if len(var_idx) < top_n:
            print(f"[WARNING] Only found {len(var_idx)} variables for requested Top {top_n}. Skipping.")
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

            y_pred, model_bundle, backend = fit_predict_mlp(
                X_tr,
                y_tr,
                X_te,
                var_names_local=var_names_top,
                sample_weight=sample_weight,
                random_state=RANDOM_SEED + fold + top_n,  # ensure different seed
                mlp_params=TUNED_MLP_PARAMS if TUNED_MLP_PARAMS else None,
            )

            y_true_all.extend(y_te.tolist())
            y_pred_all.extend(y_pred.tolist())

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
            f.write(f"MLP with embeddings from regression matrix (Top {top_n} variables)\n")
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

        # Variable ranking for this subset
        importance_df = compute_feature_ranking(
            X_top,
            y_sample,
            var_names_top,
            max_samples=IMPORTANCE_SAMPLE_SIZE,
        )
        importance_df.to_csv(os.path.join(out_dir_top, "variable_importance.csv"), index=False)

        # Final model on all sampled points (top N)
        X_final = X_top.copy()
        final_sw = None
        if USE_UNET_STYLE_SAMPLE_WEIGHTS:
            final_cw = compute_unet_style_class_weights(y_sample, CLASS_WEIGHT_MULTIPLIERS)
            final_sw = final_cw[y_sample].astype(np.float32)

        final_inputs_top_for_stats, final_top_cont_idx = _prepare_mlp_inputs(
            X_final,
            var_names_top,
            embedding_enabled=bool(USE_CATEGORICAL_EMBEDDINGS),
        )
        final_top_cont_var_names = [var_names_top[i] for i in final_top_cont_idx]
        final_top_continuous_preprocessor = _fit_continuous_preprocessor(
            final_inputs_top_for_stats[0],
            final_top_cont_var_names,
        )

        _, final_bundle_top, final_backend = fit_predict_mlp(
            X_final,
            y_sample,
            X_final[: min(100, len(X_final))],
            var_names_local=var_names_top,
            sample_weight=final_sw,
            random_state=RANDOM_SEED + 100 + top_n,
            mlp_params=TUNED_MLP_PARAMS if TUNED_MLP_PARAMS else None,
        )
        try:
            import joblib
            model_path = os.path.join(out_dir_top, "mlp_final_model.keras")
            final_bundle_top["model"].save(model_path)
            meta_path = os.path.join(out_dir_top, "mlp_final_model_meta.joblib")
            joblib.dump({
                "model_path": model_path,
                "model_type": "mlp_with_embeddings",
                "classes": final_bundle_top["classes"],
                "var_names": var_names_top,
                "categorical_mode": CATEGORICAL_MODE,
                "categorical_predictors": cat_lookup,
                "class_weight_multipliers": CLASS_WEIGHT_MULTIPLIERS,
                "enable_mlp_tuning": ENABLE_MLP_TUNING,
                "tuned_mlp_params": TUNED_MLP_PARAMS,
                "use_categorical_embeddings": USE_CATEGORICAL_EMBEDDINGS,
                "continuous_preprocessor": final_top_continuous_preprocessor,
                "quaternary_num_classes_active": ACTIVE_QUATERNARY_NUM_CLASSES,
                "landuse_num_classes_active": ACTIVE_LANDUSE_NUM_CLASSES,
                "top_variables": top_vars,
            }, meta_path)
            print(f"[Top{top_n}] Saved final model: {model_path}")
            print(f"[Top{top_n}] Saved final metadata: {meta_path}")
        except Exception as exc:
            print(f"[Top{top_n}] [WARNING] Could not save final model: {exc}")
else:
    print("[INFO] RUN_TOP_N_SUBSET_MODELS=False: skipping Top-N subset model runs.")

print("\n✔ Outputs written to:")
print(f"  {OUT_DIR}")
print("\nDone.")
