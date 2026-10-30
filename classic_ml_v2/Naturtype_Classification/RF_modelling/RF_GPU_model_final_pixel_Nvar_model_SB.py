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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from affine import Affine
import geopandas as gpd

from sklearn.ensemble import RandomForestClassifier
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

MATRIX_PATH = "/home/acosta_pedro/Random_forest/Regression/regression_matrix_Update_covar_2026.npz"
OUT_DIR = "/home/acosta_pedro/Random_forest/output"

# Optional area filter at point level
USE_AREA_FILTER = None
AREA_POLYGON_PATH = None  # Update this if you have a new shapefile path
AREA_POLYGON_LAYER = None
AREA_FILTER_MODE = "intersects"   # "intersects" or "within"

# Additional directories (set variables only if needed)
COVARIATES_DIR = "/home/acosta_pedro/Random_forest/Covariates_to_model"
ALPHA_EARTH_DIR = "/home/acosta_pedro/Random_forest/Alpha_Earth"
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

# GPU
USE_GPU_IF_AVAILABLE = True
GPU_RESAMPLE_WHEN_NEEDED = True    # fallback if GPU fit() lacks sample_weight
GPU_RESAMPLE_MULTIPLIER = 1.0      # 1.0 => same size as original train fold

# Imbalance handling copied from U-Net logic, but now tunable per class
USE_UNET_STYLE_SAMPLE_WEIGHTS = True
CLASS_WEIGHT_MULTIPLIERS = {
    1: 1.00,   # low
    2: 2.20,   # medium
    3: 1.15,   # high
}

# Categorical handling for RF-compatible treatment
# "passthrough" = keep integer-coded categories as compact columns
# "frequency"   = replace category by training-fold frequency (avoids huge one-hot expansion)
CATEGORICAL_MODE = "passthrough"
QUATERNARY_PREDICTOR_NAME = "quaternary_forenkletk_cog"
LANDUSE_PREDICTOR_NAME = "landuse_code_18_cog"

# Optional predictor exclusions by exact var_names
EXCLUDED_PREDICTORS = [
    "kalk_class_1_cog",
    "kalk_class_2_cog",
    "kalk_class_3_cog",
    "kalk_class_4_cog",
    "kalk_class_5_cog",
    "kalk_raster_cog",
    "Lineament_total_length_clean_10m_cog",
    "BioK_PCA1",
    "BioK_PCA2",

]

# Variable importance output
IMPORTANCE_TOP_N = 30
SAVE_FULL_IMPORTANCE_CHART = True
MAX_VARS_IN_FULL_CHART = None   # None = all variables
FULL_CHART_FIGHEIGHT_PER_VAR = 0.22
TOP_CHART_FIGHEIGHT_PER_VAR = 0.34

if OUT_DIR and not os.path.exists(OUT_DIR):
    print(f"[WARNING] Output directory does not exist: {OUT_DIR}")
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

    if X.shape[0] != y.shape[0] or X.shape[0] != rows.shape[0] or X.shape[0] != cols.shape[0]:
        raise RuntimeError("X, y, rows, and cols do not have the same number of samples")
    if X.shape[1] != len(var_names):
        raise RuntimeError("Number of columns in X does not match length of var_names")

    return X, y, rows, cols, var_names, width, height, transform, crs_wkt, ref_mask


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


def fit_predict_rf(X_tr, y_tr, X_te, sample_weight=None, use_gpu=True, random_state=42):
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
            return pred.astype(np.int32), model, gpu_backend
        except Exception as exc:
            print(f"[WARNING] GPU RF failed in practice, falling back to CPU. Reason: {exc}")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        bootstrap=BOOTSTRAP,
        n_jobs=-1,
        random_state=random_state,
    )
    if sample_weight is not None:
        model.fit(X_tr, y_tr, sample_weight=sample_weight)
    else:
        model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return pred.astype(np.int32), model, "sklearn"


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
X_all, y_all, rows_arr, cols_arr, var_names, width, height, transform, crs_wkt, ref_mask = load_matrix(MATRIX_PATH)
print(f"Loaded matrix: {MATRIX_PATH}")
print(f"X shape: {X_all.shape}")
print(f"Class distribution (raw): {np.bincount(y_all, minlength=4)}")
print(f"Predictors: {len(var_names)}")

# Remove background if present
valid = y_all > 0
X_all = X_all[valid]
y_all = y_all[valid]
rows_arr = rows_arr[valid]
cols_arr = cols_arr[valid]
print(f"After removing background: {X_all.shape[0]} points")
print(f"Class distribution (1..3): {np.bincount(y_all, minlength=4)}")

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

# Optional predictor exclusion
X_all, var_names = apply_predictor_exclusions(X_all, var_names, EXCLUDED_PREDICTORS)

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

    y_pred, model, backend = fit_predict_rf(
        X_tr,
        y_tr,
        X_te,
        sample_weight=sample_weight,
        use_gpu=USE_GPU_IF_AVAILABLE,
        random_state=RANDOM_SEED + fold,
    )
    print(f"  backend used: {backend}")

    y_true_all.extend(y_te.tolist())
    y_pred_all.extend(y_pred.tolist())

    if hasattr(model, "feature_importances_"):
        importances_list.append(np.asarray(model.feature_importances_, dtype=np.float64))

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
    per_fold_reports.append((
        fold,
        classification_report(
            y_te,
            y_pred,
            labels=[1, 2, 3],
            target_names=["low", "medium", "high"],
            digits=4,
            zero_division=0,
        ),
    ))

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
    f.write("Random Forest from regression matrix (patched)\n")
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
    f.write(f"EXCLUDED_PREDICTORS: {EXCLUDED_PREDICTORS}\n\n")
    f.write(report_txt)
    f.write("\nConfusion matrix (rows=true, cols=pred):\n")
    f.write(np.array2string(cm))
    f.write("\n\nFold metrics:\n")
    f.write(fold_df.to_string(index=False))
    f.write("\n\nPer-fold reports:\n")
    for fold_id, fold_rep in per_fold_reports:
        f.write(f"\n--- Fold {fold_id} ---\n")
        f.write(fold_rep)

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

_, final_model, final_backend = fit_predict_rf(
    X_final,
    y_sample,
    X_final[: min(100, len(X_final))],
    sample_weight=final_sw,
    use_gpu=USE_GPU_IF_AVAILABLE,
    random_state=RANDOM_SEED,
)
print(f"Final backend used: {final_backend}")

# Save final model with joblib if sklearn; otherwise skip explicit serialization
try:
    import joblib
    if final_backend == "sklearn":
        model_path = os.path.join(OUT_DIR, "rf_final_model.joblib")
        joblib.dump({
            "model": final_model,
            "var_names": var_names,
            "categorical_mode": CATEGORICAL_MODE,
            "categorical_predictors": cat_lookup,
            "class_weight_multipliers": CLASS_WEIGHT_MULTIPLIERS,
            "excluded_predictors": EXCLUDED_PREDICTORS,
        }, model_path)
        print(f"Saved final sklearn model: {model_path}")
    else:
        print("[INFO] Final model trained with GPU backend; model serialization skipped in this script.")
except Exception as exc:
    print(f"[WARNING] Could not save final model with joblib: {exc}")


# =====================================================================
# TOP-N VARIABLE MODELS (Top 10, 20, 30)
# =====================================================================
for top_n in [10, 20, 30]:

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

        y_pred, model, backend = fit_predict_rf(
            X_tr,
            y_tr,
            X_te,
            sample_weight=sample_weight,
            use_gpu=USE_GPU_IF_AVAILABLE,
            random_state=RANDOM_SEED + fold + top_n,  # ensure different seed
        )

        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(y_pred.tolist())

        if hasattr(model, "feature_importances_"):
            importances_list.append(np.asarray(model.feature_importances_, dtype=np.float64))

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
        per_fold_reports.append((
            fold,
            classification_report(
                y_te,
                y_pred,
                labels=[1, 2, 3],
                target_names=["low", "medium", "high"],
                digits=4,
                zero_division=0,
            ),
        ))

    # Reports
    report_txt = classification_report(
        y_true_all,
        y_pred_all,
        labels=[1, 2, 3],
        target_names=["low", "medium", "high"],
        digits=4,
        zero_division=0,
    )
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
        f.write("\nConfusion matrix (rows=true, cols=pred):\n")
        f.write(np.array2string(cm))
        f.write("\n\nFold metrics:\n")
        f.write(fold_df.to_string(index=False))
        f.write("\n\nPer-fold reports:\n")
        for fold_id, fold_rep in per_fold_reports:
            f.write(f"\n--- Fold {fold_id} ---\n")
            f.write(fold_rep)

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
    _, final_model, final_backend = fit_predict_rf(
        X_final,
        y_sample,
        X_final[: min(100, len(X_final))],
        sample_weight=final_sw,
        use_gpu=USE_GPU_IF_AVAILABLE,
        random_state=RANDOM_SEED + 100 + top_n,
    )
    try:
        import joblib
        if final_backend == "sklearn":
            model_path = os.path.join(out_dir_top, "rf_final_model.joblib")
            joblib.dump({
                "model": final_model,
                "var_names": var_names_top,
                "categorical_mode": CATEGORICAL_MODE,
                "categorical_predictors": cat_lookup,
                "class_weight_multipliers": CLASS_WEIGHT_MULTIPLIERS,
                "top_variables": top_vars,
            }, model_path)
            print(f"[Top{top_n}] Saved final sklearn model: {model_path}")
        else:
            print(f"[Top{top_n}] Final model trained with GPU backend; model serialization skipped.")
    except Exception as exc:
        print(f"[Top{top_n}] [WARNING] Could not save final model with joblib: {exc}")

print("\n✔ Outputs written to:")
print(f"  {OUT_DIR}")
print("\nDone.")
