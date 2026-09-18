#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
XGBoost multi-class classification for Kalkklass from a prebuilt regression matrix.

Outputs per run:
  - cv_fold_metrics.csv        : per-fold accuracy, macro-F1, weighted-F1, per-class F1
  - cv_predictions.csv         : fold-level true labels, predicted class, per-class probabilities
  - cv_confusion_matrix.csv    : summed confusion matrix across all folds
  - variable_importance.csv/.png : mean gain importance from final model
  - final_model_train_metrics.txt
  - xgb_cls_final_model_Kalkklass.joblib
  - final_model_class_probabilities.csv : per-sample probs + entropy uncertainty

Class map (ordinal, 1-based):
  1 = Klasse_1_Karbonat
  2 = Klasse_2_Ultramafisk_basepreg
  3 = Klasse_3_Ca-rik_silikatisk_kile
  4 = Klasse_4_Mafisk_intermediÃ¦r_basepÃ¥virket
  5 = Klasse_5_Sursilikat
"""

import os
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from affine import Affine

from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# USER SETTINGS
# =============================================================================

MATRIX_PATH = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\Regression_matrix\Regression_matrix_XRFICP_sept2026\regression_matrix.npz"
OUT_DIR = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\XGBoost\KlassCa_modelling\Only_Geovariabler\250k\Balanced_SHAP"

# === Target column name in the regression matrix CSV ===
TARGET_COL = "Ca_klasse_int"   # integer labels 1-3

# Human-readable class labels (1-based)
CLASS_LABELS = {
    1: "Kalkfattig",
    2: "Intermediær",
    3: "Kalkrik",
}

# Optional ordinal model: class 1 < class 2 < class 3.
USE_ORDINAL_XGBOOST = True

# Optional SHAP explanations for the final fitted model.
ENABLE_SHAP = True
SHAP_SAMPLE_SIZE = 2000

# === Predictor filtering â€” same logic as regression script ===
EXCLUDED_PREDICTORS = ["KalsiumElvInnsjo4_10m_masked_cog","QuaternaryClass_id","Bedrock_class_id"]   # no exclusions — using full predictor set
EXCLUDE_PREFIXES = ["landuse","quaternary_","alphaearth_","BioK_","KalsiumElvInnsjo4_","Topo_","xgb_","KlassCa_","Losm_","Ca_conc","coord","Bedrock_class_id","QuaternaryClass_id"]      # no exclusions — using full predictor set
SELECT_PREFIXES = None   # If set, keep ONLY predictors matching these prefixes

# === Class imbalance handling ===
# "balanced"               â†’ per-sample weights = N / (n_classes * class_count); computed per fold
# "none"                   â†’ no weighting
# "balanced_damped_class3" â†’ balanced weights, then class 3 weight scaled by CLASS3_DAMPING_FACTOR
CLASS_WEIGHT_MODE = "balanced"

# Multiplier applied to class 3's balanced weight under "balanced_damped_class3" (<1.0 reduces its pull).
CLASS3_DAMPING_FACTOR = 0.70

# === Cross-validation ===
USE_SPATIAL_CV = True
N_SPLITS = 5
BLOCK_SIZE_M = 20000
RANDOM_SEED = 42
SAMPLE_SIZE = None   # set an integer to subsample before CV

# === XGBoost hyperparameters (baseline; used when ENABLE_XGB_TUNING=False) ===
N_ESTIMATORS = 1000
LEARNING_RATE = 0.05
MAX_DEPTH = 5
MIN_CHILD_WEIGHT = 5
SUBSAMPLE = 0.80
COLSAMPLE_BYTREE = 0.60
REG_ALPHA = 0.10
REG_LAMBDA = 2.0
GAMMA = 0.10
N_JOBS = -1

# === XGBoost tuning ===
ENABLE_XGB_TUNING = True
TUNING_MAX_TRIALS = 12

# === Multi-run mode: run all predictors first, then top-N subsets ===
MULT_RUNS = False
MULT_RUNS_TOP_N = [20, 30, 40, 50, 60]

# === Targeted experiment mode ===
# Used to compare a few changes aimed at improving classes 2 and 3.
RUN_EXPERIMENTS = False
EXPERIMENT_RUNS = [
    {"name": "minority_boost", "class_weight_mode": "minority_boost", "top_n": None},
    {"name": "top30_balanced", "class_weight_mode": "balanced", "top_n": 30},
    {"name": "top30_minority_boost", "class_weight_mode": "minority_boost", "top_n": 30},
]

# === Importance chart settings ===
IMPORTANCE_TOP_N = 30
SAVE_FULL_IMPORTANCE_CHART = True
FULL_CHART_FIGHEIGHT_PER_VAR = 0.22
TOP_CHART_FIGHEIGHT_PER_VAR = 0.34

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)

# =============================================================================
# HELPERS â€” predictor filtering
# =============================================================================

def normalize_name(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_")


def apply_predictor_filters(X, var_names, excluded_names=None, select_prefixes=None,
                             exclude_prefixes=None):
    names_norm = [normalize_name(v) for v in var_names]
    keep_mask = np.ones(len(var_names), dtype=bool)

    if excluded_names:
        excluded_set = {normalize_name(n) for n in excluded_names}
        keep_mask &= np.array([n not in excluded_set for n in names_norm], dtype=bool)
        missing = [n for n in excluded_names if normalize_name(n) not in set(names_norm)]
        if missing:
            print(f"[WARNING] EXCLUDED_PREDICTORS not found in var_names: {missing}")

    if select_prefixes:
        prefs = [normalize_name(p) for p in select_prefixes]
        keep_mask &= np.array([any(n.startswith(p) for p in prefs) for n in names_norm], dtype=bool)

    if exclude_prefixes:
        ex_prefs = [normalize_name(p) for p in exclude_prefixes]
        keep_mask &= np.array([not any(n.startswith(p) for p in ex_prefs) for n in names_norm], dtype=bool)
        matched_counts = {p: sum(1 for n in names_norm if n.startswith(normalize_name(p)))
                          for p in exclude_prefixes}
        print("Excluded prefix matches:")
        for p, c in matched_counts.items():
            print(f"  {p}: {c}")

    keep_idx = [i for i, k in enumerate(keep_mask) if k]
    if not keep_idx:
        raise RuntimeError("Predictor filtering removed all predictors.")
    print(f"[INFO] Keeping {len(keep_idx)} predictors after filtering")
    return X[:, keep_idx], [var_names[i] for i in keep_idx]


# =============================================================================
# HELPERS â€” spatial CV
# =============================================================================

def make_spatial_block_folds(x_coords, y_coords, n_splits, block_size_m, random_seed=42):
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
    print(f"[SpatialCV] {len(unique_blocks)} blocks â†’ {n_splits} folds | sizes: {sizes}")
    return fold_labels


# =============================================================================
# HELPERS â€” class weights
# =============================================================================

# XGBoost uses 0-based class indices; map labels 1-5 â†’ 0-4 internally.
LABEL_OFFSET = 1


def compute_sample_weights(y_train_0based: np.ndarray) -> np.ndarray:
    """Inverse-frequency sample weights so every class contributes equally."""
    classes, counts = np.unique(y_train_0based, return_counts=True)
    n_total = len(y_train_0based)
    n_classes = len(classes)
    class_weight = {c: n_total / (n_classes * cnt) for c, cnt in zip(classes, counts)}
    return np.array([class_weight[c] for c in y_train_0based], dtype=np.float32)


def build_sample_weights(y_train_0based: np.ndarray, class_weight_mode: str) -> Optional[np.ndarray]:
    if class_weight_mode == "none":
        return None

    weights = compute_sample_weights(y_train_0based)
    if class_weight_mode == "balanced":
        return weights

    if class_weight_mode == "minority_boost":
        # Extra lift for the two weakest classes from the baseline run:
        # class 2 (index 1) and class 3 (index 2).
        boosts = {
            1: 1.75,
            2: 2.25,
        }
        boosted = weights.copy()
        for class_idx, factor in boosts.items():
            boosted[y_train_0based == class_idx] *= factor
        return boosted.astype(np.float32)

    if class_weight_mode == "balanced_damped_class3":
        # Reduce class 3's (index 2) balanced pull so classes 1 and 2 are less overshadowed.
        damped = weights.copy()
        damped[y_train_0based == 2] *= CLASS3_DAMPING_FACTOR
        return damped.astype(np.float32)

    raise ValueError(f"Unsupported CLASS_WEIGHT_MODE: {class_weight_mode}")


def select_top_predictors_from_importance(
    X: np.ndarray,
    var_names: List[str],
    importance_csv_path: str,
    top_n: Optional[int],
):
    if top_n is None:
        return X, var_names

    if not os.path.exists(importance_csv_path):
        raise FileNotFoundError(
            f"Importance CSV not found for top-{top_n} selection: {importance_csv_path}"
        )

    importance_df = pd.read_csv(importance_csv_path).head(top_n)
    keep_vars = [v for v in importance_df["Variable"].tolist() if v in var_names]
    if not keep_vars:
        raise RuntimeError(f"No predictors found from importance CSV: {importance_csv_path}")

    keep_idx = [var_names.index(v) for v in keep_vars]
    return X[:, keep_idx], keep_vars


# =============================================================================
# HELPERS â€” XGBoost fit / tuning
# =============================================================================

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

TUNED_XGB_PARAMS: Dict[str, float] = {}


def fit_xgb_classifier(X_train, y_train_0based, random_state=42,
                        xgb_params=None, sample_weight=None):
    params = dict(DEFAULT_XGB_PARAMS)
    if xgb_params:
        params.update(xgb_params)
    n_cls = len(np.unique(y_train_0based))
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=n_cls,
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
        eval_metric="mlogloss",
    )
    if sample_weight is not None:
        model.fit(X_train, y_train_0based, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train_0based)
    return model


def fit_xgb_model(X_train, y_train_0based, random_state=42,
                  xgb_params=None, sample_weight=None, class_weight_mode=None):
    """Fit either the ordinary multiclass or cumulative ordinal model."""
    if not USE_ORDINAL_XGBOOST:
        return fit_xgb_classifier(
            X_train, y_train_0based, random_state, xgb_params, sample_weight
        )

    models = []
    for threshold in (0, 1):
        threshold_target = (y_train_0based > threshold).astype(np.int32)
        threshold_weights = build_sample_weights(
            threshold_target, class_weight_mode or CLASS_WEIGHT_MODE
        )
        models.append(
            fit_xgb_binary_classifier(
                X_train,
                threshold_target,
                random_state + threshold,
                xgb_params,
                threshold_weights,
            )
        )
    return models


def fit_xgb_binary_classifier(X_train, y_train, random_state=42,
                              xgb_params=None, sample_weight=None):
    params = dict(DEFAULT_XGB_PARAMS)
    if xgb_params:
        params.update(xgb_params)
    model = XGBClassifier(
        objective="binary:logistic",
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
        eval_metric="logloss",
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def predict_model_proba(model, X):
    if not USE_ORDINAL_XGBOOST:
        return model.predict_proba(X)
    q1 = model[0].predict_proba(X)[:, 1]
    q2 = model[1].predict_proba(X)[:, 1]
    q2 = np.minimum(q1, q2)
    return np.column_stack((1.0 - q1, q1 - q2, q2))


def predict_model_class(model, X):
    return np.argmax(predict_model_proba(model, X), axis=1)


def save_shap_outputs(model, X, var_names, out_dir):
    """Save SHAP values for the final model when the optional dependency is installed."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("ENABLE_SHAP=True requires the 'shap' package.") from exc

    rng = np.random.default_rng(RANDOM_SEED)
    sample_n = min(SHAP_SAMPLE_SIZE, len(X))
    sample_idx = rng.choice(len(X), size=sample_n, replace=False)
    X_sample = X[sample_idx]
    models = model if USE_ORDINAL_XGBOOST else [model]
    value_frames = []
    summary_frames = []
    for model_idx, fitted_model in enumerate(models, start=1):
        explainer = shap.TreeExplainer(fitted_model)
        values = explainer.shap_values(X_sample)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        prefix = f"threshold_{model_idx}_" if USE_ORDINAL_XGBOOST else ""
        value_frames.append(pd.DataFrame(
            values, columns=[prefix + str(name) for name in var_names]
        ))
        summary_frames.append(pd.DataFrame({
            "Variable": var_names,
            "MeanAbsSHAP": np.mean(np.abs(values), axis=0),
            "Model": prefix.rstrip("_") or "multiclass",
        }))

    shap_values_df = pd.concat(value_frames, axis=1)
    shap_values_df.insert(0, "sample_index", sample_idx)
    shap_values_df.to_csv(os.path.join(out_dir, "shap_values.csv"), index=False)
    summary_df = pd.concat(summary_frames, ignore_index=True)
    summary_df.sort_values("MeanAbsSHAP", ascending=False).to_csv(
        os.path.join(out_dir, "shap_summary.csv"), index=False
    )
    print(f"Saved SHAP outputs for {sample_n} samples to {out_dir}")


def build_xgb_tuning_candidates():
    candidates = [
        {"n_estimators": 800,  "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 3,  "subsample": 0.8,  "colsample_bytree": 0.6,  "reg_alpha": 0.05, "reg_lambda": 2.0, "gamma": 0.0},
        {"n_estimators": 1000, "learning_rate": 0.04, "max_depth": 5, "min_child_weight": 4,  "subsample": 0.8,  "colsample_bytree": 0.6,  "reg_alpha": 0.10, "reg_lambda": 2.0, "gamma": 0.10},
        {"n_estimators": 1200, "learning_rate": 0.03, "max_depth": 6, "min_child_weight": 5,  "subsample": 0.8,  "colsample_bytree": 0.6,  "reg_alpha": 0.10, "reg_lambda": 2.0, "gamma": 0.10},
        {"n_estimators": 1400, "learning_rate": 0.025,"max_depth": 6, "min_child_weight": 6,  "subsample": 0.85, "colsample_bytree": 0.65, "reg_alpha": 0.10, "reg_lambda": 2.5, "gamma": 0.15},
        {"n_estimators": 1600, "learning_rate": 0.02, "max_depth": 7, "min_child_weight": 8,  "subsample": 0.8,  "colsample_bytree": 0.7,  "reg_alpha": 0.20, "reg_lambda": 3.0, "gamma": 0.20},
        {"n_estimators": 900,  "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 3,  "subsample": 0.75, "colsample_bytree": 0.55, "reg_alpha": 0.00, "reg_lambda": 1.5, "gamma": 0.0},
        {"n_estimators": 1100, "learning_rate": 0.035,"max_depth": 5, "min_child_weight": 5,  "subsample": 0.85, "colsample_bytree": 0.7,  "reg_alpha": 0.05, "reg_lambda": 2.0, "gamma": 0.05},
        {"n_estimators": 1300, "learning_rate": 0.03, "max_depth": 7, "min_child_weight": 10, "subsample": 0.8,  "colsample_bytree": 0.6,  "reg_alpha": 0.30, "reg_lambda": 4.0, "gamma": 0.20},
        {"n_estimators": 1000, "learning_rate": 0.04, "max_depth": 6, "min_child_weight": 4,  "subsample": 0.9,  "colsample_bytree": 0.8,  "reg_alpha": 0.00, "reg_lambda": 1.0, "gamma": 0.0},
        {"n_estimators": 1400, "learning_rate": 0.025,"max_depth": 5, "min_child_weight": 6,  "subsample": 0.9,  "colsample_bytree": 0.75, "reg_alpha": 0.10, "reg_lambda": 2.0, "gamma": 0.10},
        {"n_estimators": 1200, "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 8,  "subsample": 0.8,  "colsample_bytree": 0.5,  "reg_alpha": 0.20, "reg_lambda": 3.0, "gamma": 0.15},
        {"n_estimators": 1500, "learning_rate": 0.02, "max_depth": 6, "min_child_weight": 12, "subsample": 0.7,  "colsample_bytree": 0.6,  "reg_alpha": 0.30, "reg_lambda": 5.0, "gamma": 0.25},
    ]
    return candidates[:TUNING_MAX_TRIALS]


def tune_xgb_params(X_samp, y_samp_0based, x_samp, y_coord_samp, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    candidates = build_xgb_tuning_candidates()

    if USE_SPATIAL_CV:
        fold_labels = make_spatial_block_folds(x_samp, y_coord_samp, N_SPLITS, BLOCK_SIZE_M, RANDOM_SEED)
        fold_iter = [(np.where(fold_labels != f)[0], np.where(fold_labels == f)[0]) for f in range(N_SPLITS)]
    else:
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        fold_iter = list(kf.split(X_samp))

    trial_rows = []
    best_params = None
    best_f1 = -np.inf

    print(f"\n=== TUNING XGBoost (Kalkklass) | trials={len(candidates)} ===")
    for i, params in enumerate(candidates, start=1):
        fold_f1s = []
        for tr, te in fold_iter:
            sw = compute_sample_weights(y_samp_0based[tr]) if CLASS_WEIGHT_MODE == "balanced" else None
            model = fit_xgb_model(X_samp[tr], y_samp_0based[tr],
                                  random_state=RANDOM_SEED + i * 10,
                                  xgb_params=params, sample_weight=sw)
            pred = predict_model_class(model, X_samp[te])
            fold_f1s.append(f1_score(y_samp_0based[te], pred, average="macro", zero_division=0))
        mean_f1 = float(np.mean(fold_f1s))
        print(f"[Tune {i:02d}] macro-F1={mean_f1:.4f}")
        trial_rows.append({"trial": i, "mean_macro_f1": mean_f1, **params})
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_params = params

    pd.DataFrame(trial_rows).sort_values("mean_macro_f1", ascending=False).to_csv(
        os.path.join(out_dir, "xgb_tuning_trials.csv"), index=False
    )
    if best_params:
        with open(os.path.join(out_dir, "xgb_best_params.txt"), "w", encoding="utf-8") as f:
            f.write(f"Best macro-F1: {best_f1:.6f}\n")
            for k, v in best_params.items():
                f.write(f"{k}: {v}\n")
        print(f"Best params: macro-F1={best_f1:.4f} | {best_params}")
    return best_params


# =============================================================================
# HELPERS â€” metrics
# =============================================================================

def classification_metrics(y_true_0based, y_pred_0based, class_labels: dict):
    sorted_ints = sorted(class_labels.keys())   # [1,2,3,4,5]
    zero_based = [k - LABEL_OFFSET for k in sorted_ints]

    acc = float(accuracy_score(y_true_0based, y_pred_0based))
    bal_acc = float(balanced_accuracy_score(y_true_0based, y_pred_0based))
    macro_f1 = float(f1_score(y_true_0based, y_pred_0based, average="macro",
                               labels=zero_based, zero_division=0))
    weighted_f1 = float(f1_score(y_true_0based, y_pred_0based, average="weighted",
                                  labels=zero_based, zero_division=0))
    per_class_f1 = f1_score(y_true_0based, y_pred_0based, average=None,
                             labels=zero_based, zero_division=0)

    metrics = {"accuracy": acc, "balanced_accuracy": bal_acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}
    for lb, f1 in zip(sorted_ints, per_class_f1):
        metrics[f"f1_class{lb}_{class_labels[lb][:12]}"] = float(f1)
    return metrics


def save_importance_outputs(importance_df: pd.DataFrame, out_dir: str):
    importance_df.to_csv(os.path.join(out_dir, "variable_importance.csv"), index=False)

    top_df = importance_df.head(IMPORTANCE_TOP_N)
    fig_h = max(4, len(top_df) * TOP_CHART_FIGHEIGHT_PER_VAR)
    plt.figure(figsize=(10, fig_h))
    plt.barh(top_df["Variable"][::-1], top_df["Importance"][::-1])
    plt.xlabel("Importance (mean gain)")
    plt.title(f"Top {len(top_df)} Variable Importances â€” Kalkklass")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "variable_importance_top.png"), dpi=200)
    plt.close()

    if SAVE_FULL_IMPORTANCE_CHART:
        fig_h = max(6, len(importance_df) * FULL_CHART_FIGHEIGHT_PER_VAR)
        plt.figure(figsize=(10, fig_h))
        plt.barh(importance_df["Variable"][::-1], importance_df["Importance"][::-1])
        plt.xlabel("Importance (mean gain)")
        plt.title("Full Variable Importance â€” Kalkklass")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "variable_importance_full.png"), dpi=200)
        plt.close()


# =============================================================================
# MAIN CV AND SAVE
# =============================================================================

def run_cv_and_save(X_samp, y_samp_1based, x_samp, y_coord_samp,
                    var_names_this, out_dir, xgb_params=None,
                    class_weight_mode: Optional[str] = None):
    """Run spatial/random CV, save all outputs, fit final model."""
    os.makedirs(out_dir, exist_ok=True)
    n_preds = X_samp.shape[1]
    n_classes = len(CLASS_LABELS)
    sorted_labels = sorted(CLASS_LABELS.keys())   # [1,2,3,4,5]

    y_samp_0based = y_samp_1based - LABEL_OFFSET
    active_class_weight_mode = class_weight_mode or CLASS_WEIGHT_MODE

    active_params = dict(DEFAULT_XGB_PARAMS)
    if xgb_params:
        active_params.update(xgb_params)

    print(f"\n  Predictors: {n_preds}")
    if USE_SPATIAL_CV:
        cv_label = f"Spatial block CV ({BLOCK_SIZE_M} m)"
        print(f"\n=== {cv_label} | {N_SPLITS} folds ===")
        fold_labels = make_spatial_block_folds(x_samp, y_coord_samp, N_SPLITS, BLOCK_SIZE_M, RANDOM_SEED)
        fold_iter = [(np.where(fold_labels != f)[0], np.where(fold_labels == f)[0]) for f in range(N_SPLITS)]
    else:
        cv_label = f"Random KFold ({N_SPLITS} folds)"
        print(f"\n=== {cv_label} ===")
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        fold_iter = list(kf.split(X_samp))

    fold_rows = []
    pred_rows = []
    importances_list = []
    all_conf = np.zeros((n_classes, n_classes), dtype=int)

    for fold_id, (tr, te) in enumerate(fold_iter, start=1):
        print(f"\n--- Fold {fold_id}/{N_SPLITS} | train={len(tr)} test={len(te)} ---")
        sw = build_sample_weights(y_samp_0based[tr], active_class_weight_mode)
        model = fit_xgb_model(X_samp[tr], y_samp_0based[tr],
                      random_state=RANDOM_SEED + fold_id,
                              xgb_params=active_params, sample_weight=sw,
                              class_weight_mode=active_class_weight_mode)

        proba = predict_model_proba(model, X_samp[te])
        pred_0based = np.argmax(proba, axis=1)
        pred_1based = pred_0based + LABEL_OFFSET

        metrics = classification_metrics(y_samp_0based[te], pred_0based, CLASS_LABELS)
        metrics["fold"] = fold_id
        metrics["n_train"] = len(tr)
        metrics["n_test"] = len(te)
        fold_rows.append(metrics)

        conf = confusion_matrix(y_samp_0based[te], pred_0based,
                                 labels=[lb - LABEL_OFFSET for lb in sorted_labels])
        all_conf += conf

        print(f"  Accuracy={metrics['accuracy']:.4f} | BalAcc={metrics['balanced_accuracy']:.4f} | "
              f"macro-F1={metrics['macro_f1']:.4f} | weighted-F1={metrics['weighted_f1']:.4f}")

        df_fold = pd.DataFrame({
            "fold": fold_id,
            "y_true": y_samp_1based[te],
            "y_pred": pred_1based,
            "x": x_samp[te],
            "y": y_coord_samp[te],
        })
        for i, lb in enumerate(sorted_labels):
            df_fold[f"prob_class{lb}"] = proba[:, i]
        pred_rows.append(df_fold)

        if hasattr(model, "feature_importances_"):
            importances_list.append(model.feature_importances_.copy())

    # â”€â”€ Aggregate CV metrics â”€â”€
    fold_df = pd.DataFrame(fold_rows)
    pred_cv_df = pd.concat(pred_rows, ignore_index=True)

    fold_df.to_csv(os.path.join(out_dir, "cv_fold_metrics.csv"), index=False)
    pred_cv_df.to_csv(os.path.join(out_dir, "cv_predictions.csv"), index=False)

    conf_df = pd.DataFrame(
        all_conf,
        index=[f"true_{CLASS_LABELS[lb][:18]}" for lb in sorted_labels],
        columns=[f"pred_{CLASS_LABELS[lb][:18]}" for lb in sorted_labels],
    )
    conf_df.to_csv(os.path.join(out_dir, "cv_confusion_matrix.csv"))

    # â”€â”€ Per-class probability uncertainty summary â”€â”€
    proba_stats_rows = []
    for lb in sorted_labels:
        col = f"prob_class{lb}"
        vals = pred_cv_df[col].to_numpy()
        proba_stats_rows.append({
            "class_int": lb,
            "class_label": CLASS_LABELS[lb],
            "prob_mean": float(vals.mean()),
            "prob_std": float(vals.std()),
            "prob_min": float(vals.min()),
            "prob_max": float(vals.max()),
        })
    pd.DataFrame(proba_stats_rows).to_csv(
        os.path.join(out_dir, "cv_class_proba_stats.csv"), index=False
    )

    # â”€â”€ Summary â”€â”€
    mean_acc = fold_df["accuracy"].mean()
    mean_bal_acc = fold_df["balanced_accuracy"].mean()
    mean_macro_f1 = fold_df["macro_f1"].mean()
    print(f"\nCV [Kalkklass | {n_preds} preds]: "
          f"Acc={mean_acc:.4f}\u00b1{fold_df['accuracy'].std():.4f} | "
          f"BalAcc={mean_bal_acc:.4f}\u00b1{fold_df['balanced_accuracy'].std():.4f} | "
          f"macro-F1={mean_macro_f1:.4f}Â±{fold_df['macro_f1'].std():.4f}")

    # â”€â”€ Text report â”€â”€
    with open(os.path.join(out_dir, "regression_report.txt"), "w", encoding="utf-8") as f:
        f.write("XGBoost classification â€” target: Kalkklass_int\n")
        f.write("=" * 60 + "\n")
        f.write(f"CV strategy: {cv_label}\n")
        f.write(f"Predictors used: {n_preds}\n")
        f.write(f"Class weight mode: {active_class_weight_mode}\n")
        f.write(f"MATRIX_PATH: {MATRIX_PATH}\n\n")
        f.write("Mean CV metrics\n")
        for k in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
            f.write(f"  {k}: {fold_df[k].mean():.6f} Â± {fold_df[k].std():.6f}\n")
        f.write("\nClass distribution in CV predictions:\n")
        f.write(pred_cv_df["y_true"].value_counts().sort_index().to_string())
        f.write("\n\nConfusion matrix (summed across folds):\n")
        f.write(conf_df.to_string())
        f.write("\n\nXGBoost params used:\n")
        for k, v in active_params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n\nPer-fold metrics:\n")
        f.write(fold_df.to_string(index=False))

    # â”€â”€ Variable importance â”€â”€
    importance_df = None
    if importances_list:
        mean_imp = np.mean(np.vstack(importances_list), axis=0)
        importance_df = pd.DataFrame({
            "Variable": var_names_this,
            "Importance": mean_imp,
        }).sort_values("Importance", ascending=False)
        save_importance_outputs(importance_df, out_dir)

    # â”€â”€ Class distribution plot â”€â”€
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        pred_cv_df["y_true"].value_counts().sort_index().plot(
            kind="bar", ax=axes[0], title="True class distribution")
        pred_cv_df["y_pred"].value_counts().sort_index().plot(
            kind="bar", ax=axes[1], title="Predicted class distribution")
        for ax in axes:
            ax.set_xlabel("Class (1-5)")
            ax.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "cv_class_distribution.png"), dpi=200)
        plt.close()
    except Exception as exc:
        print(f"[WARNING] Could not save distribution plot: {exc}")

    # â”€â”€ Confusion matrix heatmap â”€â”€
    try:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(all_conf, cmap="Blues")
        plt.colorbar(im, ax=ax)
        tick_labels = [CLASS_LABELS[lb][:18] for lb in sorted_labels]
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(tick_labels, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("CV Confusion Matrix (summed)")
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, str(all_conf[i, j]), ha="center", va="center", fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "cv_confusion_matrix.png"), dpi=200)
        plt.close()
    except Exception as exc:
        print(f"[WARNING] Could not save confusion matrix plot: {exc}")

    # ================================================================
    # FINAL MODEL (trained on full dataset)
    # ================================================================
    print(f"\n=== FINAL MODEL (Kalkklass | {n_preds} preds) ===")
    sw_final = build_sample_weights(y_samp_0based, active_class_weight_mode)
    final_model = fit_xgb_model(X_samp, y_samp_0based,
                                random_state=RANDOM_SEED,
                                xgb_params=active_params,
                                sample_weight=sw_final,
                                class_weight_mode=active_class_weight_mode)
    train_proba = predict_model_proba(final_model, X_samp)
    train_pred_0based = np.argmax(train_proba, axis=1)
    train_pred_1based = train_pred_0based + LABEL_OFFSET
    train_metrics = classification_metrics(y_samp_0based, train_pred_0based, CLASS_LABELS)

    with open(os.path.join(out_dir, "final_model_train_metrics.txt"), "w", encoding="utf-8") as f:
        f.write("Training metrics â€” Kalkklass\n")
        f.write("=" * 50 + "\n")
        for k, v in train_metrics.items():
            f.write(f"{k}: {v:.6f}\n")

    # Per-class probability maps + uncertainty (saved as CSV; production raster is separate)
    prob_df = pd.DataFrame({
        "x": x_samp, "y": y_coord_samp,
        "y_true": y_samp_1based, "y_pred": train_pred_1based,
    })
    for i, lb in enumerate(sorted_labels):
        prob_df[f"prob_class{lb}"] = train_proba[:, i]
    # Prediction entropy (high entropy = high uncertainty)
    eps = 1e-12
    entropy = -np.sum(train_proba * np.log(train_proba + eps), axis=1)
    prob_df["uncertainty_entropy"] = entropy
    # 1 - max_prob as a simple uncertainty proxy per sample
    prob_df["uncertainty_1minus_maxprob"] = 1.0 - train_proba.max(axis=1)
    prob_df.to_csv(os.path.join(out_dir, "final_model_class_probabilities.csv"), index=False)

    if ENABLE_SHAP:
        save_shap_outputs(final_model, X_samp, var_names_this, out_dir)

    try:
        import joblib
        model_path = os.path.join(out_dir, "xgb_cls_final_model_Kalkklass.joblib")
        joblib.dump({
            "model": final_model,
            "model_type": "xgboost_classifier",
            "ordinal": USE_ORDINAL_XGBOOST,
            "var_names": var_names_this,
            "target_variable": "Kalkklass_int",
            "label_offset": LABEL_OFFSET,
            "class_labels": CLASS_LABELS,
            "n_classes": n_classes,
            "class_weight_mode": active_class_weight_mode,
            "matrix_path": MATRIX_PATH,
            "xgb_params": active_params,
        }, model_path)
        print(f"Saved model: {model_path}")
    except Exception as exc:
        print(f"[WARNING] Could not save final model: {exc}")

    return importance_df


# =============================================================================
# LOAD MATRIX
# =============================================================================

def load_matrix(npz_path: str):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Regression matrix not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    var_names = [str(v) for v in data["var_names"].tolist()]
    transform = None
    if "transform" in data.files:
        t = np.asarray(data["transform"], dtype=float).ravel()
        if t.size == 6:
            transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
    crs_wkt = str(np.asarray(data["crs_wkt"]).ravel()[0]) if "crs_wkt" in data.files else None
    return X, var_names, transform, crs_wkt

def main():
    print("\n=== LOADING REGRESSION MATRIX ===")
    X_all, var_names, transform, crs_wkt = load_matrix(MATRIX_PATH)
    print(f"Loaded matrix: {MATRIX_PATH} | X shape: {X_all.shape} | Predictors: {len(var_names)}")
    print(f"X NaN: {np.isnan(X_all).sum()} | X inf: {np.isinf(X_all).sum()}")

    _matrix_csv = MATRIX_PATH.replace(".npz", ".csv")
    if not os.path.exists(_matrix_csv):
        raise FileNotFoundError(f"Regression matrix CSV not found: {_matrix_csv}")
    _df_csv = pd.read_csv(_matrix_csv)
    print(f"Loaded CSV: {_matrix_csv} | Rows: {len(_df_csv)}")

    if "x" in _df_csv.columns and "y" in _df_csv.columns:
        _x_all = _df_csv["x"].to_numpy(dtype=np.float64)
        _y_all = _df_csv["y"].to_numpy(dtype=np.float64)
        print(f"Coordinates: x [{_x_all.min():.0f}, {_x_all.max():.0f}]  "
              f"y [{_y_all.min():.0f}, {_y_all.max():.0f}]")
    else:
        _x_all = np.zeros(len(X_all), dtype=np.float64)
        _y_all = np.zeros(len(X_all), dtype=np.float64)
        print("[WARNING] x/y not found in CSV; spatial CV folds will not be spatially meaningful.")

    if TARGET_COL not in _df_csv.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in CSV. "
                       f"Available columns: {list(_df_csv.columns)}")
    _y_raw = _df_csv[TARGET_COL].to_numpy(dtype=float)

    print(f"\nBefore predictor filter: X shape {X_all.shape}")
    X_filtered, var_names_f = apply_predictor_filters(
        X_all, var_names,
        excluded_names=EXCLUDED_PREDICTORS,
        select_prefixes=SELECT_PREFIXES,
        exclude_prefixes=EXCLUDE_PREFIXES,
    )
    print(f"After predictor filter: X shape {X_filtered.shape}")

    _valid_x = ~np.all(~np.isfinite(X_filtered), axis=1)
    X_filtered = np.nan_to_num(
        X_filtered[_valid_x], nan=-9999.0, posinf=-9999.0, neginf=-9999.0
    ).astype(np.float32)
    _x_all = _x_all[_valid_x]
    _y_all = _y_all[_valid_x]
    _y_raw = _y_raw[_valid_x]
    _df_csv = _df_csv.iloc[_valid_x].reset_index(drop=True)
    print(f"After finite-predictor filter: {X_filtered.shape[0]} rows")

    _valid_y = np.isfinite(_y_raw) & np.isin(_y_raw, list(CLASS_LABELS.keys()))
    X_cls = X_filtered[_valid_y]
    y_cls = _y_raw[_valid_y].astype(int)
    x_cls = _x_all[_valid_y]
    y_coord_cls = _y_all[_valid_y]
    print(f"Valid classification samples: {X_cls.shape[0]}")

    print("\nClass distribution:")
    for lb in sorted(CLASS_LABELS.keys()):
        n = int((y_cls == lb).sum())
        print(f"  {lb} ({CLASS_LABELS[lb]}): {n} ({100.0*n/len(y_cls):.1f}%)")

    if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(y_cls):
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(len(y_cls), size=SAMPLE_SIZE, replace=False)
        X_cls, y_cls, x_cls, y_coord_cls = X_cls[idx], y_cls[idx], x_cls[idx], y_coord_cls[idx]
        print(f"Subsampled to {len(y_cls)} samples")

    print(f"\nSampled {len(y_cls)} points x {X_cls.shape[1]} predictors")

    tuned_params = None
    if ENABLE_XGB_TUNING:
        tuning_out_dir = os.path.join(OUT_DIR, "_tuning")
        tuned_params = tune_xgb_params(
            X_cls, y_cls - LABEL_OFFSET, x_cls, y_coord_cls, tuning_out_dir
        )
        if tuned_params is not None:
            print(f"Using tuned params: {tuned_params}")
        else:
            print("[WARNING] Tuning returned no params; using defaults.")

    if RUN_EXPERIMENTS:
        experiments_out_dir = os.path.join(OUT_DIR, "experiments")
        os.makedirs(experiments_out_dir, exist_ok=True)
        baseline_importance_csv = os.path.join(OUT_DIR, "variable_importance.csv")
        latest_importance_csv = baseline_importance_csv if os.path.exists(baseline_importance_csv) else None
        experiment_rows = []

        for experiment in EXPERIMENT_RUNS:
            exp_name = experiment["name"]
            exp_weight_mode = experiment.get("class_weight_mode", CLASS_WEIGHT_MODE)
            exp_top_n = experiment.get("top_n")
            print(f"\n{'=' * 70}\nEXPERIMENT: {exp_name}\n{'=' * 70}")

            if exp_top_n is None:
                X_run, var_names_run = X_cls, var_names_f
            else:
                importance_source = latest_importance_csv
                if not importance_source or not os.path.exists(importance_source):
                    raise FileNotFoundError(
                        "No importance CSV available for top-N experiment selection. "
                        "Run at least one experiment with top_n=None first."
                    )
                X_run, var_names_run = select_top_predictors_from_importance(
                    X_cls, var_names_f, importance_source, exp_top_n
                )
            exp_out_dir = os.path.join(experiments_out_dir, exp_name)
            run_cv_and_save(
                X_run, y_cls, x_cls, y_coord_cls,
                var_names_run, exp_out_dir,
                xgb_params=tuned_params,
                class_weight_mode=exp_weight_mode,
            )
            exp_importance_csv = os.path.join(exp_out_dir, "variable_importance.csv")
            if os.path.exists(exp_importance_csv):
                latest_importance_csv = exp_importance_csv

            fold_metrics_path = os.path.join(exp_out_dir, "cv_fold_metrics.csv")
            fold_df = pd.read_csv(fold_metrics_path)
            row = {
                "experiment": exp_name,
                "class_weight_mode": exp_weight_mode,
                "top_n": exp_top_n if exp_top_n is not None else len(var_names_run),
                "n_predictors": len(var_names_run),
                "accuracy_mean": float(fold_df["accuracy"].mean()),
                "macro_f1_mean": float(fold_df["macro_f1"].mean()),
                "weighted_f1_mean": float(fold_df["weighted_f1"].mean()),
            }
            for cls_id in sorted(CLASS_LABELS):
                pref = f"f1_class{cls_id}_"
                col = next((c for c in fold_df.columns if c.startswith(pref)), None)
                row[f"f1_class{cls_id}_mean"] = float(fold_df[col].mean()) if col else np.nan
            experiment_rows.append(row)

        experiment_df = pd.DataFrame(experiment_rows).sort_values("macro_f1_mean", ascending=False)
        experiment_df.to_csv(os.path.join(experiments_out_dir, "experiment_summary.csv"), index=False)
        print("\n=== Experiment summary written ===")
        print(experiment_df.to_string(index=False))

    elif MULT_RUNS:
        full_out_dir = os.path.join(OUT_DIR, "All")
        print(f"\nRun: all {len(var_names_f)} predictors -> {full_out_dir}")
        imp_df = run_cv_and_save(X_cls, y_cls, x_cls, y_coord_cls,
                                 var_names_f, full_out_dir, xgb_params=tuned_params)
        print(f"\nKalkklass All outputs -> {full_out_dir}")

        if imp_df is not None:
            for top_n in MULT_RUNS_TOP_N:
                if top_n >= len(var_names_f):
                    print(f"[SKIP] top{top_n} >= total predictors ({len(var_names_f)})")
                    continue
                top_vars = imp_df["Variable"].iloc[:top_n].tolist()
                top_idx = [var_names_f.index(v) for v in top_vars if v in var_names_f]
                topn_out = os.path.join(OUT_DIR, f"Top_{top_n}")
                print(f"\nRun: top {len(top_idx)} predictors -> {topn_out}")
                run_cv_and_save(X_cls[:, top_idx], y_cls, x_cls, y_coord_cls,
                                [var_names_f[i] for i in top_idx], topn_out,
                                xgb_params=tuned_params)
                print(f"\nKalkklass Top_{top_n} outputs -> {topn_out}")
    else:
        run_cv_and_save(X_cls, y_cls, x_cls, y_coord_cls,
                        var_names_f, OUT_DIR, xgb_params=tuned_params)
        print(f"\nKalkklass outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()

print("\n=== Kalkklass classification completed ===")

