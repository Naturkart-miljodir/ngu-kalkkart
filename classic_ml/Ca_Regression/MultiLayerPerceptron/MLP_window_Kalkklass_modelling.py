#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MLP multi-class classification for Kalkklass from a prebuilt regression matrix.

Replaces XGBoost with a PyTorch MLP. Key design choices vs. XGBoost version:

  Loss function (imbalance handling):
    - "focal"       : alpha-balanced focal loss (recommended).
                      FL = -alpha_c * (1-p_c)^gamma * log(p_c)
                      gamma=2 focuses learning on hard / minority examples.
                      alpha per class = N / (n_classes * count_c), normalised.
    - "weighted_ce" : standard class-weighted cross-entropy (simpler baseline).
    Why focal over focal-tversky?  Focal-tversky is designed for spatial
    segmentation (maximises Dice/IoU per pixel mask).  For tabular point
    classification the focal loss is the gold-standard choice — it naturally
    down-weights easy majority examples without requiring FP/FN geometry.

  Feature scaling:
    StandardScaler fit on training fold only — mandatory for MLP, never leaks.

  Variable importance:
    Permutation importance (macro-F1 drop) via sklearn — model-agnostic.

  Architecture:
    Linear -> BatchNorm -> ReLU -> Dropout  (repeated per hidden layer)
    Configurable width, depth, dropout.

  Early stopping:
    Monitors validation macro-F1 every 5 epochs; restores best weights.

Outputs (mirrors XGBoost_window_Kalkklass_modelling.py):
  - cv_fold_metrics.csv / cv_predictions.csv
  - cv_confusion_matrix.csv / .png
  - cv_class_distribution.png
  - cv_class_proba_stats.csv
  - variable_importance.csv / _top.png / _full.png
  - final_model_train_metrics.txt
  - mlp_cls_final_model_Kalkklass.joblib
  - final_model_class_probabilities.csv
  - regression_report.txt

Requires:  kalk_MLP conda environment (see kalk_MLP_env.yaml)
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from affine import Affine
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import RobustScaler, StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# USER SETTINGS
# =============================================================================

MATRIX_PATH = (
    r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse"
    r"\Geochemistry NGU_2026\Kalk_project\Modelling\Regression_matrix"
    r"\Regression_matrix_XRFICP_sept2026\regression_matrix.npz"
)
OUT_DIR = (
    r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse"
    r"\Geochemistry NGU_2026\Kalk_project\Modelling\MLP\KlassCa_modelling"
    r"\Only_Geovariabler\Balanced_SHAP_WCE_BalCW"
)

# === Target ===
TARGET_COL = "Ca_klasse_int"
CLASS_LABELS: Dict[int, str] = {
    1: "Kalkfattig",
    2: "Intermediær",
    3: "Kalkrik",
}
LABEL_OFFSET = 1   # class integers start at 1; internally XGBoost/MLP use 0-based

# === Predictor filtering (same logic as XGBoost version) ===
EXCLUDED_PREDICTORS: List[str] = []
EXCLUDE_PREFIXES: List[str] = [
    "landuse",
    "quaternary_",
    "alphaearth_",
    "BioK_",
    "KalsiumElvInnsjo4_",
    "Topo_",
    "xgb_",
    "KlassCa_",
    "Losm_",
    "Ca_conc",
    "coord",
    "Bedrock_class_id",
    "QuaternaryClass_id",
]
SELECT_PREFIXES: Optional[List[str]] = None

# === Cross-validation ===
USE_SPATIAL_CV = True
N_SPLITS = 5
BLOCK_SIZE_M = 20_000
RANDOM_SEED = 42
SAMPLE_SIZE: Optional[int] = None

# === MLP architecture ===
HIDDEN_LAYERS: List[int] = [256, 256, 128]
DROPOUT_RATE: float = 0.30
USE_BATCH_NORM: bool = True

# === Training ===
BATCH_SIZE: int = 512
MAX_EPOCHS: int = 500
PATIENCE: int = 50          # early-stopping patience in epochs (checked every epoch)
LEARNING_RATE: float = 5e-4
WEIGHT_DECAY: float = 1e-4
SCALER_TYPE: str = "robust"  # "robust" (median/IQR) | "standard" (mean/std)
USE_RESIDUAL: bool = True    # skip connections between consecutive same-width hidden layers

# === Loss function ===
# "focal"       recommended: handles minority class (Kalkrik 7%) better than CE
# "weighted_ce" simpler class-weighted CE (good baseline to compare against)
LOSS_FN: str = "weighted_ce"
FOCAL_GAMMA: float = 1.0    # lower gamma reduces minority over-amplification compared with gamma=2
FOCAL_ALPHA: Optional[torch.Tensor] = None  # None = auto-compute per fold from class freq.

# "balanced" : alpha_c = N / (n_classes * count_c)   (XGBoost-compatible)
# "sqrt_balanced" : softer minority upweighting using sqrt inverse frequency
# "minority_damped" : balanced weights, then damp class 2/3 to reduce map inflation
# "minority_plus" : balanced weights, then boost class 2/3 like XGBoost minority_boost
# "none"     : uniform alpha across classes
USE_BALANCED_CLASS_WEIGHTS: bool = True
USE_MINORITY_PLUS_WEIGHTS: bool = False
CLASS_WEIGHT_MODE: Optional[str] = None
MINORITY_DAMPING_1BASED: Dict[int, float] = {1: 1.00, 2: 0.82, 3: 0.74}
MINORITY_PLUS_FACTORS_1BASED: Dict[int, float] = {2: 1.75, 3: 2.25}
MAX_CLASS_ALPHA: Optional[float] = None

# Tuning objective: macro-F1 minus penalty for overpredicting minority classes.
TUNING_OVERPRED_PENALTY: float = 0.20
OVERPRED_CLASSES_1BASED: List[int] = [2, 3]

# === Device ===
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# === Permutation importance ===
ENABLE_PERMUTATION_IMPORTANCE: bool = True
IMPORTANCE_SAMPLE_N: Optional[int] = 3000  # subsample for speed; None = all test points
IMPORTANCE_TOP_N: int = 30
SAVE_FULL_IMPORTANCE_CHART: bool = True
FULL_CHART_FIGHEIGHT_PER_VAR: float = 0.22
TOP_CHART_FIGHEIGHT_PER_VAR: float = 0.34

# === SHAP explanations for the final model ===
ENABLE_SHAP: bool = True
SHAP_SAMPLE_SIZE: int = 2000
SHAP_BACKGROUND_SIZE: int = 100

# === MLP tuning ===
ENABLE_MLP_TUNING: bool = False
TUNING_MAX_TRIALS: int = 20

# === Multi-run mode ===
MULT_RUNS: bool = False
MULT_RUNS_TOP_N: List[int] = [20, 30, 40, 50, 60]

# === Experiment mode ===
RUN_EXPERIMENTS: bool = False
EXPERIMENT_RUNS: List[dict] = [
    {"name": "focal_g2_all",    "loss_fn": "focal",       "focal_gamma": 2.0, "top_n": None},
    {"name": "weighted_ce_all", "loss_fn": "weighted_ce", "focal_gamma": None, "top_n": None},
    {"name": "focal_g2_top30",  "loss_fn": "focal",       "focal_gamma": 2.0, "top_n": 30},
    {"name": "focal_g1_all",    "loss_fn": "focal",       "focal_gamma": 1.0, "top_n": None},
]

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
print(f"[INFO] Using device: {DEVICE}")


# =============================================================================
# HELPERS — predictor filtering
# =============================================================================

def normalize_name(s: str) -> str:
    raw = str(s).strip()
    low = raw.lower()
    no_normalize_prefixes = (
        "bedrock_",
        "metamorphic_",
        "tectonic_",
        "geochronology_",
    )
    if low.startswith(no_normalize_prefixes):
        # Keep geological-family predictor names untouched.
        return raw
    return low.replace(" ", "_")


def apply_predictor_filters(
    X: np.ndarray,
    var_names: List[str],
    excluded_names: Optional[List[str]] = None,
    select_prefixes: Optional[List[str]] = None,
    exclude_prefixes: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
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
        matched = {p: sum(1 for n in names_norm if n.startswith(normalize_name(p))) for p in exclude_prefixes}
        print("Excluded prefix matches:")
        for p, c in matched.items():
            print(f"  {p}: {c}")

    keep_idx = [i for i, k in enumerate(keep_mask) if k]
    if not keep_idx:
        raise RuntimeError("Predictor filtering removed all predictors.")
    print(f"[INFO] Keeping {len(keep_idx)} predictors after filtering")
    return X[:, keep_idx], [var_names[i] for i in keep_idx]


# =============================================================================
# HELPERS — spatial CV
# =============================================================================

def make_spatial_block_folds(
    x_coords: np.ndarray, y_coords: np.ndarray,
    n_splits: int, block_size_m: float, random_seed: int = 42,
) -> np.ndarray:
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


# =============================================================================
# CLASS WEIGHTING
# =============================================================================

def resolve_class_weight_mode(mode: Optional[str] = None) -> str:
    if mode is not None:
        return mode
    if USE_BALANCED_CLASS_WEIGHTS and USE_MINORITY_PLUS_WEIGHTS:
        raise ValueError(
            "USE_BALANCED_CLASS_WEIGHTS and USE_MINORITY_PLUS_WEIGHTS "
            "cannot both be True."
        )
    if USE_MINORITY_PLUS_WEIGHTS:
        return "minority_plus"
    if USE_BALANCED_CLASS_WEIGHTS:
        return "balanced"
    return "none"


def compute_class_alpha(
    y_0based: np.ndarray, n_classes: int, mode: str
) -> torch.Tensor:
    """Per-class focal/CE alpha weights from training label frequencies."""
    if mode == "none":
        alpha = np.ones(n_classes, dtype=np.float32)
    else:
        classes, counts = np.unique(y_0based, return_counts=True)
        n_total = len(y_0based)
        alpha = np.ones(n_classes, dtype=np.float32)
        for c, cnt in zip(classes, counts):
            if c >= n_classes:
                continue
            if mode in ("balanced", "minority_damped", "minority_plus"):
                alpha[int(c)] = n_total / (n_classes * cnt)
            elif mode == "sqrt_balanced":
                alpha[int(c)] = np.sqrt(n_total / (n_classes * cnt))
            else:
                raise ValueError(
                    f"Unknown CLASS_WEIGHT_MODE='{mode}'. "
                    "Use one of: none, balanced, sqrt_balanced, minority_damped, minority_plus"
                )

    if mode == "minority_damped":
        for class_int_1based, mult in MINORITY_DAMPING_1BASED.items():
            c0 = class_int_1based - LABEL_OFFSET
            if 0 <= c0 < n_classes:
                alpha[c0] *= float(mult)

    if mode == "minority_plus":
        for class_int_1based, mult in MINORITY_PLUS_FACTORS_1BASED.items():
            c0 = class_int_1based - LABEL_OFFSET
            if 0 <= c0 < n_classes:
                alpha[c0] *= float(mult)

    if MAX_CLASS_ALPHA is not None and MAX_CLASS_ALPHA > 0:
        alpha = np.clip(alpha, 0.0, float(MAX_CLASS_ALPHA))

    classes, counts = np.unique(y_0based, return_counts=True)
    if len(classes) < n_classes:
        # Keep stable magnitude even in sparse folds where one class may be absent.
        alpha = np.maximum(alpha, 1e-6)

    # Normalise so sum == n_classes (keeps loss magnitude stable across datasets)
    alpha = alpha / alpha.sum() * n_classes
    return torch.tensor(alpha, dtype=torch.float32)


def compute_overprediction_penalty(
    y_true_0based: np.ndarray,
    y_pred_0based: np.ndarray,
    classes_1based: List[int],
) -> float:
    """Sum of positive class-share excess for selected classes (0 = no overprediction)."""
    penalty = 0.0
    n = max(1, len(y_true_0based))
    for cls_1 in classes_1based:
        cls_0 = cls_1 - LABEL_OFFSET
        true_share = float((y_true_0based == cls_0).sum()) / n
        pred_share = float((y_pred_0based == cls_0).sum()) / n
        penalty += max(0.0, pred_share - true_share)
    return float(penalty)


# =============================================================================
# MLP — loss, architecture, wrapper
# =============================================================================

class ResBlock(nn.Module):
    """
    Pre-activation residual block for same-width MLP layers.
    Applied automatically in MLPNet when consecutive hidden layers share the same width
    and USE_RESIDUAL=True.  Improves gradient flow without adding hyperparameters.
    """

    def __init__(self, size: int, dropout: float, use_batch_norm: bool):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.bn1 = nn.BatchNorm1d(size) if use_batch_norm else nn.Identity()
        self.fc2 = nn.Linear(size, size)
        self.bn2 = nn.BatchNorm1d(size) if use_batch_norm else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(F.relu(self.bn1(self.fc1(x))))
        h = self.bn2(self.fc2(h))
        return F.relu(x + h)


class FocalLoss(nn.Module):
    """
    Alpha-balanced multi-class focal loss.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    gamma : float
        Focusing parameter.  0 = standard weighted cross-entropy.
        Typical range 1–3; 2.0 is the standard choice from the original paper.
    alpha : Tensor of shape (C,) or None
        Per-class weights. None = no per-class weighting.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)   # moves with .to(device)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha if self.alpha is not None else None
        ce = F.cross_entropy(logits, targets, weight=alpha, reduction="none")
        pt = torch.exp(-ce)                          # probability of correct class
        focal = (1.0 - pt) ** self.gamma * ce
        return focal.mean()


class MLPNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_layers: List[int],
        dropout: float,
        use_batch_norm: bool,
        use_residual: bool = False,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_size = n_features
        for idx, h in enumerate(hidden_layers):
            if use_residual and idx > 0 and h == hidden_layers[idx - 1]:
                # Same width as previous layer — use a residual block (skip connection)
                layers.append(ResBlock(h, dropout, use_batch_norm))
            else:
                # Different width or first layer — standard projection
                layers.append(nn.Linear(in_size, h))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
            in_size = h
        layers.append(nn.Linear(in_size, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPWrapper:
    """
    Thin sklearn-compatible wrapper around a trained MLPNet + StandardScaler.
    Exposes predict / predict_proba / score — needed for permutation importance.
    """

    def __init__(self, net: MLPNet, scaler: StandardScaler, device: str):
        self.net = net
        self.scaler = scaler
        self.device = device

    # Needed for sklearn's permutation_importance estimator validation.
    # The wrapped model is already fitted; this is intentionally a no-op.
    def fit(self, X: np.ndarray, y: np.ndarray):
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X).astype(np.float32)
        t = torch.from_numpy(Xs).to(self.device)
        self.net.eval()
        with torch.no_grad():
            proba = torch.softmax(self.net(t), dim=1).cpu().numpy()
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(f1_score(y, self.predict(X), average="macro", zero_division=0))


# =============================================================================
# TRAINING
# =============================================================================

def fit_mlp(
    X_train: np.ndarray,
    y_train_0based: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val_0based: Optional[np.ndarray] = None,
    n_classes: int = 3,
    hidden_layers: Optional[List[int]] = None,
    dropout: float = 0.30,
    use_batch_norm: bool = True,
    use_residual: bool = False,
    batch_size: int = 512,
    max_epochs: int = 200,
    patience: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    loss_fn: str = "focal",
    focal_gamma: float = 2.0,
    focal_alpha: Optional[torch.Tensor] = None,
    scaler_type: str = "standard",
    random_state: int = 42,
    device: str = "cpu",
) -> Tuple[MLPNet, StandardScaler]:
    """Fit an MLP classifier with early stopping on validation macro-F1."""
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    if hidden_layers is None:
        hidden_layers = HIDDEN_LAYERS

    scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
    X_scaled = scaler.fit_transform(X_train).astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(X_scaled),
        torch.from_numpy(y_train_0based.astype(np.int64)),
    )
    pin = device.startswith("cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin)

    if X_val is not None:
        X_val_t = torch.from_numpy(scaler.transform(X_val).astype(np.float32)).to(device)

    net = MLPNet(
        n_features=X_scaled.shape[1],
        n_classes=n_classes,
        hidden_layers=hidden_layers,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        use_residual=use_residual,
    ).to(device)

    # Build loss
    if loss_fn == "focal":
        criterion = FocalLoss(gamma=focal_gamma, alpha=focal_alpha).to(device)
    else:  # "weighted_ce"
        w = focal_alpha.to(device) if focal_alpha is not None else None
        criterion = nn.CrossEntropyLoss(weight=w)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_f1 = -np.inf
    patience_ctr = 0
    best_state: Optional[dict] = None

    for epoch in range(max_epochs):
        net.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(net(Xb), yb).backward()
            optimizer.step()
        scheduler.step()

        # Validate every epoch for accurate early stopping
        if X_val is not None:
            net.eval()
            with torch.no_grad():
                pred_val = net(X_val_t).argmax(dim=1).cpu().numpy()
            val_f1 = float(f1_score(y_val_0based, pred_val, average="macro", zero_division=0))
            if val_f1 > best_val_f1 + 1e-5:
                best_val_f1 = val_f1
                patience_ctr = 0
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            else:
                patience_ctr += 1
            if patience_ctr >= patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    net.eval()
    return net, scaler


# =============================================================================
# HELPERS — metrics, importance, outputs
# =============================================================================

def classification_metrics(
    y_true_0based: np.ndarray,
    y_pred_0based: np.ndarray,
    class_labels: Dict[int, str],
) -> dict:
    sorted_ints = sorted(class_labels.keys())
    zero_based = [k - LABEL_OFFSET for k in sorted_ints]
    acc = float(accuracy_score(y_true_0based, y_pred_0based))
    bal_acc = float(balanced_accuracy_score(y_true_0based, y_pred_0based))
    macro_f1 = float(f1_score(y_true_0based, y_pred_0based, average="macro",
                               labels=zero_based, zero_division=0))
    weighted_f1 = float(f1_score(y_true_0based, y_pred_0based, average="weighted",
                                  labels=zero_based, zero_division=0))
    per_class_f1 = f1_score(y_true_0based, y_pred_0based, average=None,
                             labels=zero_based, zero_division=0)
    d = {
        "accuracy": acc, "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1, "weighted_f1": weighted_f1,
    }
    for i, lb in enumerate(sorted_ints):
        safe = (class_labels[lb]
                .replace("æ", "ae").replace("Æ", "Ae")
                .replace("ø", "o").replace("Ø", "O").replace(" ", "_"))
        d[f"f1_class{lb}_{safe}"] = float(per_class_f1[i])
    return d


def compute_permutation_importance(
    wrapper: MLPWrapper,
    X_test: np.ndarray,
    y_test_0based: np.ndarray,
    var_names: List[str],
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance as sk_perm_imp
    result = sk_perm_imp(
        wrapper, X_test, y_test_0based,
        scoring="f1_macro",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    return pd.DataFrame({
        "Variable": var_names,
        "Importance": result.importances_mean,
    }).sort_values("Importance", ascending=False)


def save_importance_outputs(importance_df: pd.DataFrame, out_dir: str) -> None:
    importance_df.to_csv(os.path.join(out_dir, "variable_importance.csv"), index=False)
    top_df = importance_df.head(IMPORTANCE_TOP_N)
    fig_h = max(4, len(top_df) * TOP_CHART_FIGHEIGHT_PER_VAR)
    plt.figure(figsize=(10, fig_h))
    plt.barh(top_df["Variable"][::-1], top_df["Importance"][::-1])
    plt.xlabel("Permutation importance (macro-F1 drop)")
    plt.title(f"Top {len(top_df)} Variable Importances (MLP)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "variable_importance_top.png"), dpi=200)
    plt.close()
    if SAVE_FULL_IMPORTANCE_CHART:
        fig_h2 = max(6, len(importance_df) * FULL_CHART_FIGHEIGHT_PER_VAR)
        plt.figure(figsize=(10, fig_h2))
        plt.barh(importance_df["Variable"][::-1], importance_df["Importance"][::-1])
        plt.xlabel("Permutation importance")
        plt.title("Full Variable Importance (MLP)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "variable_importance_full.png"), dpi=200)
        plt.close()


def save_shap_outputs(
    net: MLPNet,
    scaler,
    X: np.ndarray,
    var_names: List[str],
    out_dir: str,
    device: str,
) -> None:
    """Save class-specific SHAP values for the final PyTorch MLP."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("ENABLE_SHAP=True requires the 'shap' package.") from exc

    rng = np.random.default_rng(RANDOM_SEED + 123)
    sample_n = min(SHAP_SAMPLE_SIZE, len(X))
    background_n = min(SHAP_BACKGROUND_SIZE, len(X))
    sample_idx = rng.choice(len(X), size=sample_n, replace=False)
    background_idx = rng.choice(len(X), size=background_n, replace=False)
    X_sample = scaler.transform(X[sample_idx]).astype(np.float32)
    X_background = scaler.transform(X[background_idx]).astype(np.float32)

    net.eval()
    background_tensor = torch.from_numpy(X_background).to(device)
    sample_tensor = torch.from_numpy(X_sample).to(device)
    explainer = shap.GradientExplainer(net, background_tensor)
    raw_values = explainer.shap_values(sample_tensor)

    if isinstance(raw_values, list):
        values_by_class = [np.asarray(values) for values in raw_values]
    else:
        values = np.asarray(raw_values)
        if values.ndim != 3:
            raise RuntimeError(f"Unexpected SHAP array shape: {values.shape}")
        if values.shape[0] == sample_n and values.shape[1] == len(var_names):
            values_by_class = [values[:, :, i] for i in range(values.shape[2])]
        elif values.shape[1] == sample_n and values.shape[2] == len(var_names):
            values_by_class = [values[i, :, :] for i in range(values.shape[0])]
        else:
            raise RuntimeError(f"Unexpected SHAP array shape: {values.shape}")

    shap_values_df = pd.DataFrame({"sample_index": sample_idx})
    summary_frames = []
    for class_index, class_values in enumerate(values_by_class):
        class_label = class_index + LABEL_OFFSET
        class_values = np.asarray(class_values)
        if class_values.shape != (sample_n, len(var_names)):
            raise RuntimeError(
                f"Unexpected SHAP class shape for class {class_label}: {class_values.shape}"
            )
        prefix = f"class_{class_label}_"
        for variable_index, variable_name in enumerate(var_names):
            shap_values_df[prefix + str(variable_name)] = class_values[:, variable_index]
        summary_frames.append(pd.DataFrame({
            "Variable": var_names,
            "MeanAbsSHAP": np.mean(np.abs(class_values), axis=0),
            "Class": class_label,
            "ClassLabel": CLASS_LABELS.get(class_label, str(class_label)),
        }))

    shap_values_df.to_csv(os.path.join(out_dir, "shap_values.csv"), index=False)
    pd.concat(summary_frames, ignore_index=True).sort_values(
        ["Class", "MeanAbsSHAP"], ascending=[True, False]
    ).to_csv(os.path.join(out_dir, "shap_summary.csv"), index=False)
    print(f"Saved SHAP outputs for {sample_n} samples to {out_dir}")


def select_top_predictors_from_importance(
    X: np.ndarray, var_names: List[str], importance_csv: str, top_n: int,
) -> Tuple[np.ndarray, List[str]]:
    imp_df = pd.read_csv(importance_csv).head(top_n)
    top_vars = [v for v in imp_df["Variable"].tolist() if v in var_names]
    top_idx = [var_names.index(v) for v in top_vars]
    return X[:, top_idx], top_vars


# =============================================================================
# TUNING
# =============================================================================

def build_mlp_tuning_candidates() -> List[dict]:
    # Architectures with consecutive same-width layers benefit from USE_RESIDUAL=True
    # e.g. [256,256,128] → the second 256 becomes a ResBlock with a skip connection
    candidates = [
        # --- Bias-controlled candidates (prefer less 2/3 inflation) ---
        {
            "hidden_layers": [256, 256, 128], "dropout": 0.30, "lr": 5e-4,
            "wd": 1e-4, "focal_gamma": 1.0, "class_weight_mode": "minority_damped",
            "loss_fn": "focal",
        },
        {
            "hidden_layers": [256, 128, 64], "dropout": 0.30, "lr": 1e-3,
            "wd": 1e-3, "focal_gamma": 1.0, "class_weight_mode": "minority_damped",
            "loss_fn": "focal",
        },
        {
            "hidden_layers": [256, 128, 64], "dropout": 0.30, "lr": 1e-3,
            "wd": 1e-4, "focal_gamma": 1.5, "class_weight_mode": "sqrt_balanced",
            "loss_fn": "focal",
        },
        {
            "hidden_layers": [256, 128, 64], "dropout": 0.25, "lr": 1e-3,
            "wd": 1e-4, "focal_gamma": 0.0, "class_weight_mode": "sqrt_balanced",
            "loss_fn": "weighted_ce",
        },
        # --- Proven baseline (previous best) ---
        {"hidden_layers": [256, 128, 64],   "dropout": 0.30, "lr": 1e-3, "wd": 1e-3, "focal_gamma": 1.5},
        # --- Residual-friendly architectures (consecutive same-width layers) ---
        {"hidden_layers": [256, 256, 128],  "dropout": 0.30, "lr": 5e-4, "wd": 1e-4, "focal_gamma": 1.5},
        {"hidden_layers": [512, 512, 256],  "dropout": 0.35, "lr": 5e-4, "wd": 1e-4, "focal_gamma": 1.5},
        {"hidden_layers": [512, 256, 256],  "dropout": 0.30, "lr": 5e-4, "wd": 1e-4, "focal_gamma": 2.0},
        {"hidden_layers": [256, 256, 128, 64], "dropout": 0.30, "lr": 5e-4, "wd": 1e-4, "focal_gamma": 1.5},
        {"hidden_layers": [512, 512, 256, 128], "dropout": 0.35, "lr": 3e-4, "wd": 1e-4, "focal_gamma": 1.5},
        # --- Wider single-step architectures ---
        {"hidden_layers": [512, 256, 128],  "dropout": 0.35, "lr": 5e-4, "wd": 1e-4, "focal_gamma": 2.0},
        {"hidden_layers": [1024, 512, 256], "dropout": 0.40, "lr": 3e-4, "wd": 1e-4, "focal_gamma": 1.5},
        {"hidden_layers": [512, 256, 128],  "dropout": 0.40, "lr": 1e-3, "wd": 1e-3, "focal_gamma": 1.5},
        # --- Shallower / faster baselines ---
        {"hidden_layers": [256, 128],       "dropout": 0.30, "lr": 1e-3, "wd": 1e-4, "focal_gamma": 2.0},
        {"hidden_layers": [512, 256],       "dropout": 0.35, "lr": 5e-4, "wd": 1e-4, "focal_gamma": 2.0},
        {"hidden_layers": [128, 128, 64],   "dropout": 0.20, "lr": 1e-3, "wd": 5e-4, "focal_gamma": 2.0},
        # --- LR variants on best architecture ---
        {"hidden_layers": [256, 128, 64],   "dropout": 0.25, "lr": 5e-4, "wd": 1e-3, "focal_gamma": 1.5},
        {"hidden_layers": [256, 128, 64],   "dropout": 0.30, "lr": 2e-3, "wd": 1e-4, "focal_gamma": 2.0},
        # --- Gamma sweep ---
        {"hidden_layers": [256, 256, 128],  "dropout": 0.30, "lr": 5e-4, "wd": 1e-3, "focal_gamma": 1.0},
        {"hidden_layers": [256, 256, 128],  "dropout": 0.30, "lr": 5e-4, "wd": 1e-4, "focal_gamma": 2.5},
        {"hidden_layers": [512, 512, 256],  "dropout": 0.40, "lr": 3e-4, "wd": 1e-4, "focal_gamma": 2.0},
        {"hidden_layers": [512, 256, 256, 128], "dropout": 0.30, "lr": 3e-4, "wd": 1e-4, "focal_gamma": 1.5},
        {"hidden_layers": [256, 128, 64],   "dropout": 0.30, "lr": 1e-3, "wd": 0.0,  "focal_gamma": 2.0},
        {"hidden_layers": [512, 512, 256, 128], "dropout": 0.25, "lr": 5e-4, "wd": 1e-4, "focal_gamma": 1.5},
    ]
    return candidates[:TUNING_MAX_TRIALS]


def tune_mlp_params(
    X_samp: np.ndarray,
    y_samp_0based: np.ndarray,
    x_samp: np.ndarray,
    y_coord_samp: np.ndarray,
    n_classes: int,
    out_dir: str,
) -> Optional[dict]:
    os.makedirs(out_dir, exist_ok=True)
    candidates = build_mlp_tuning_candidates()

    if USE_SPATIAL_CV:
        fold_labels = make_spatial_block_folds(x_samp, y_coord_samp, N_SPLITS, BLOCK_SIZE_M, RANDOM_SEED)
        fold_iter = [
            (np.where(fold_labels != f)[0], np.where(fold_labels == f)[0])
            for f in range(N_SPLITS)
        ]
    else:
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        fold_iter = list(kf.split(X_samp))

    trial_rows = []
    best_params: Optional[dict] = None
    best_score = -np.inf

    print(f"\n=== TUNING MLP (Kalkklass) | trials={len(candidates)} | device={DEVICE} ===")
    for i, cand in enumerate(candidates, start=1):
        fold_f1s = []
        fold_penalties = []
        cand_weight_mode = cand.get("class_weight_mode", resolve_class_weight_mode())
        cand_loss = cand.get("loss_fn", LOSS_FN)
        cand_gamma = cand.get("focal_gamma", FOCAL_GAMMA)
        for tr, te in fold_iter:
            alpha = compute_class_alpha(y_samp_0based[tr], n_classes, cand_weight_mode).to(DEVICE)
            net, scaler = fit_mlp(
                X_samp[tr], y_samp_0based[tr],
                X_val=X_samp[te], y_val_0based=y_samp_0based[te],
                n_classes=n_classes,
                hidden_layers=cand["hidden_layers"],
                dropout=cand["dropout"],
                use_batch_norm=USE_BATCH_NORM,
                use_residual=USE_RESIDUAL,
                batch_size=BATCH_SIZE,
                max_epochs=MAX_EPOCHS,
                patience=PATIENCE,
                lr=cand["lr"],
                weight_decay=cand["wd"],
                loss_fn=cand_loss,
                focal_gamma=cand_gamma,
                focal_alpha=alpha,
                scaler_type=SCALER_TYPE,
                random_state=RANDOM_SEED + i * 10,
                device=DEVICE,
            )
            wrapper = MLPWrapper(net, scaler, DEVICE)
            y_pred = wrapper.predict(X_samp[te])
            fold_f1s.append(float(f1_score(y_samp_0based[te], y_pred, average="macro", zero_division=0)))
            fold_penalties.append(
                compute_overprediction_penalty(y_samp_0based[te], y_pred, OVERPRED_CLASSES_1BASED)
            )
        mean_f1 = float(np.mean(fold_f1s))
        mean_pen = float(np.mean(fold_penalties))
        tune_score = mean_f1 - TUNING_OVERPRED_PENALTY * mean_pen
        print(
            f"[Tune {i:02d}] score={tune_score:.4f} macro-F1={mean_f1:.4f} "
            f"pen={mean_pen:.4f} | "
            f"layers={cand['hidden_layers']} drop={cand['dropout']} "
            f"lr={cand['lr']:.0e} gamma={cand_gamma} loss={cand_loss} wmode={cand_weight_mode}"
        )
        trial_rows.append({"trial": i, "tune_score": tune_score, "mean_macro_f1": mean_f1,
                           "mean_overpred_penalty": mean_pen,
                           "hidden_layers": str(cand["hidden_layers"]),
                           "dropout": cand["dropout"], "lr": cand["lr"],
                           "wd": cand["wd"], "focal_gamma": cand_gamma,
                           "loss_fn": cand_loss, "class_weight_mode": cand_weight_mode})
        if tune_score > best_score:
            best_score = tune_score
            best_params = cand

    pd.DataFrame(trial_rows).sort_values("tune_score", ascending=False).to_csv(
        os.path.join(out_dir, "mlp_tuning_trials.csv"), index=False
    )
    if best_params:
        with open(os.path.join(out_dir, "mlp_best_params.txt"), "w", encoding="utf-8") as f:
            f.write(f"Best tuning score: {best_score:.6f}\n")
            for k, v in best_params.items():
                f.write(f"{k}: {v}\n")
        print(f"Best params: score={best_score:.4f} | {best_params}")
    return best_params


# =============================================================================
# MAIN CV + FINAL MODEL
# =============================================================================

def run_cv_and_save(
    X_samp: np.ndarray,
    y_samp_1based: np.ndarray,
    x_samp: np.ndarray,
    y_coord_samp: np.ndarray,
    var_names_this: List[str],
    out_dir: str,
    mlp_params: Optional[dict] = None,
    loss_fn: Optional[str] = None,
    focal_gamma: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)
    n_preds = X_samp.shape[1]
    n_classes = len(CLASS_LABELS)
    sorted_labels = sorted(CLASS_LABELS.keys())
    y_samp_0based = y_samp_1based - LABEL_OFFSET

    # Resolve active hyperparams
    active_loss = loss_fn if loss_fn is not None else (
        mlp_params.get("loss_fn", LOSS_FN) if mlp_params else LOSS_FN
    )
    active_gamma = focal_gamma if focal_gamma is not None else FOCAL_GAMMA
    active_class_weight_mode = resolve_class_weight_mode(
        mlp_params.get("class_weight_mode") if mlp_params else None
    )
    active_hidden = mlp_params.get("hidden_layers", HIDDEN_LAYERS) if mlp_params else HIDDEN_LAYERS
    active_drop = mlp_params.get("dropout", DROPOUT_RATE) if mlp_params else DROPOUT_RATE
    active_lr = mlp_params.get("lr", LEARNING_RATE) if mlp_params else LEARNING_RATE
    active_wd = mlp_params.get("wd", WEIGHT_DECAY) if mlp_params else WEIGHT_DECAY
    if mlp_params and "focal_gamma" in mlp_params:
        active_gamma = mlp_params["focal_gamma"]

    print(f"\n  Predictors: {n_preds}")
    if USE_SPATIAL_CV:
        cv_label = f"Spatial block CV ({BLOCK_SIZE_M} m)"
        print(f"\n=== {cv_label} | {N_SPLITS} folds ===")
        fold_labels = make_spatial_block_folds(x_samp, y_coord_samp, N_SPLITS, BLOCK_SIZE_M, RANDOM_SEED)
        fold_iter = [
            (np.where(fold_labels != f)[0], np.where(fold_labels == f)[0])
            for f in range(N_SPLITS)
        ]
    else:
        cv_label = f"Random KFold ({N_SPLITS} folds)"
        print(f"\n=== {cv_label} ===")
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        fold_iter = list(kf.split(X_samp))

    fold_rows: List[dict] = []
    pred_rows: List[pd.DataFrame] = []
    all_conf = np.zeros((n_classes, n_classes), dtype=int)
    # Accumulate test data for permutation importance (last fold's wrapper used)
    imp_X: List[np.ndarray] = []
    imp_y: List[np.ndarray] = []
    last_wrapper: Optional[MLPWrapper] = None

    for fold_id, (tr, te) in enumerate(fold_iter, start=1):
        print(f"\n--- Fold {fold_id}/{N_SPLITS} | train={len(tr)} test={len(te)} ---")
        alpha = compute_class_alpha(y_samp_0based[tr], n_classes, active_class_weight_mode).to(DEVICE)
        net, scaler = fit_mlp(
            X_samp[tr], y_samp_0based[tr],
            X_val=X_samp[te], y_val_0based=y_samp_0based[te],
            n_classes=n_classes,
            hidden_layers=active_hidden,
            dropout=active_drop,
            use_batch_norm=USE_BATCH_NORM,
            use_residual=USE_RESIDUAL,
            batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            lr=active_lr,
            weight_decay=active_wd,
            loss_fn=active_loss,
            focal_gamma=active_gamma,
            focal_alpha=alpha,
            scaler_type=SCALER_TYPE,
            random_state=RANDOM_SEED + fold_id,
            device=DEVICE,
        )
        wrapper = MLPWrapper(net, scaler, DEVICE)
        last_wrapper = wrapper

        proba = wrapper.predict_proba(X_samp[te])
        pred_0based = np.argmax(proba, axis=1)
        pred_1based = pred_0based + LABEL_OFFSET

        metrics = classification_metrics(y_samp_0based[te], pred_0based, CLASS_LABELS)
        metrics.update({"fold": fold_id, "n_train": len(tr), "n_test": len(te)})
        fold_rows.append(metrics)

        conf = confusion_matrix(
            y_samp_0based[te], pred_0based,
            labels=[lb - LABEL_OFFSET for lb in sorted_labels],
        )
        all_conf += conf

        print(
            f"  Accuracy={metrics['accuracy']:.4f} | "
            f"BalAcc={metrics['balanced_accuracy']:.4f} | "
            f"macro-F1={metrics['macro_f1']:.4f} | "
            f"weighted-F1={metrics['weighted_f1']:.4f}"
        )

        df_fold = pd.DataFrame({
            "fold": fold_id, "y_true": y_samp_1based[te], "y_pred": pred_1based,
            "x": x_samp[te], "y": y_coord_samp[te],
        })
        for i, lb in enumerate(sorted_labels):
            df_fold[f"prob_class{lb}"] = proba[:, i]
        pred_rows.append(df_fold)
        imp_X.append(X_samp[te])
        imp_y.append(y_samp_0based[te])

    # ── Aggregate and save ──────────────────────────────────────────────────
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

    proba_stats = []
    for lb in sorted_labels:
        vals = pred_cv_df[f"prob_class{lb}"].to_numpy()
        proba_stats.append({
            "class_int": lb, "class_label": CLASS_LABELS[lb],
            "prob_mean": float(vals.mean()), "prob_std": float(vals.std()),
            "prob_min": float(vals.min()), "prob_max": float(vals.max()),
        })
    pd.DataFrame(proba_stats).to_csv(os.path.join(out_dir, "cv_class_proba_stats.csv"), index=False)

    mean_acc = fold_df["accuracy"].mean()
    mean_bal = fold_df["balanced_accuracy"].mean()
    mean_mf1 = fold_df["macro_f1"].mean()
    print(
        f"\nCV [Kalkklass MLP | {n_preds} preds]: "
        f"Acc={mean_acc:.4f}±{fold_df['accuracy'].std():.4f} | "
        f"BalAcc={mean_bal:.4f}±{fold_df['balanced_accuracy'].std():.4f} | "
        f"macro-F1={mean_mf1:.4f}±{fold_df['macro_f1'].std():.4f}"
    )

    with open(os.path.join(out_dir, "regression_report.txt"), "w", encoding="utf-8") as f:
        f.write("MLP classification — target: Kalkklass_int\n" + "=" * 60 + "\n")
        f.write(f"CV strategy:      {cv_label}\n")
        f.write(f"Predictors used:  {n_preds}\n")
        f.write(f"Loss function:    {active_loss} (gamma={active_gamma})\n")
        f.write(f"Class weights:    {active_class_weight_mode}\n")
        f.write(f"Architecture:     {active_hidden}  dropout={active_drop}  residual={USE_RESIDUAL}\n")
        f.write(f"Scaler:           {SCALER_TYPE}\n")
        f.write(f"Device:           {DEVICE}\n")
        f.write(f"MATRIX_PATH:      {MATRIX_PATH}\n\n")
        f.write("Mean CV metrics\n")
        for k in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
            f.write(f"  {k}: {fold_df[k].mean():.6f} ± {fold_df[k].std():.6f}\n")
        f.write("\nConfusion matrix (summed across folds):\n")
        f.write(conf_df.to_string())
        f.write("\n\nPer-fold metrics:\n")
        f.write(fold_df.to_string(index=False))

    # ── Permutation importance ──────────────────────────────────────────────
    importance_df: Optional[pd.DataFrame] = None
    if ENABLE_PERMUTATION_IMPORTANCE and last_wrapper is not None:
        try:
            print("\n[INFO] Computing permutation importance (may take a few minutes)...")
            X_imp = np.concatenate(imp_X, axis=0)
            y_imp = np.concatenate(imp_y, axis=0)
            if IMPORTANCE_SAMPLE_N is not None and len(y_imp) > IMPORTANCE_SAMPLE_N:
                rng = np.random.default_rng(RANDOM_SEED + 99)
                idx = rng.choice(len(y_imp), size=IMPORTANCE_SAMPLE_N, replace=False)
                X_imp, y_imp = X_imp[idx], y_imp[idx]
            importance_df = compute_permutation_importance(
                last_wrapper, X_imp, y_imp, var_names_this,
                n_repeats=5, random_state=RANDOM_SEED,
            )
            save_importance_outputs(importance_df, out_dir)
        except Exception as exc:
            print(f"[WARNING] Could not compute permutation importance: {exc}")

    # ── Plots ───────────────────────────────────────────────────────────────
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        pred_cv_df["y_true"].value_counts().sort_index().plot(
            kind="bar", ax=axes[0], title="True class distribution")
        pred_cv_df["y_pred"].value_counts().sort_index().plot(
            kind="bar", ax=axes[1], title="Predicted class distribution")
        for ax in axes:
            ax.set_xlabel("Class"); ax.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "cv_class_distribution.png"), dpi=200)
        plt.close()
    except Exception as exc:
        print(f"[WARNING] Could not save distribution plot: {exc}")

    try:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(all_conf, cmap="Blues")
        plt.colorbar(im, ax=ax)
        tick_labels = [CLASS_LABELS[lb][:18] for lb in sorted_labels]
        ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(tick_labels, fontsize=7)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("CV Confusion Matrix (summed) — MLP")
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, str(all_conf[i, j]), ha="center", va="center", fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "cv_confusion_matrix.png"), dpi=200)
        plt.close()
    except Exception as exc:
        print(f"[WARNING] Could not save confusion matrix plot: {exc}")

    # ── Final model (full dataset) ───────────────────────────────────────────
    print(f"\n=== FINAL MODEL (Kalkklass MLP | {n_preds} preds) ===")
    alpha_final = compute_class_alpha(y_samp_0based, n_classes, active_class_weight_mode).to(DEVICE)
    final_net, final_scaler = fit_mlp(
        X_samp, y_samp_0based,
        X_val=None, y_val_0based=None,
        n_classes=n_classes,
        hidden_layers=active_hidden,
        dropout=active_drop,
        use_batch_norm=USE_BATCH_NORM,
        use_residual=USE_RESIDUAL,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        lr=active_lr,
        weight_decay=active_wd,
        loss_fn=active_loss,
        focal_gamma=active_gamma,
        focal_alpha=alpha_final,
        scaler_type=SCALER_TYPE,
        random_state=RANDOM_SEED,
        device=DEVICE,
    )
    final_wrapper = MLPWrapper(final_net, final_scaler, DEVICE)
    train_proba = final_wrapper.predict_proba(X_samp)
    train_pred_0based = np.argmax(train_proba, axis=1)
    train_pred_1based = train_pred_0based + LABEL_OFFSET
    train_metrics = classification_metrics(y_samp_0based, train_pred_0based, CLASS_LABELS)

    with open(os.path.join(out_dir, "final_model_train_metrics.txt"), "w", encoding="utf-8") as f:
        f.write("Training metrics — Kalkklass MLP\n" + "=" * 50 + "\n")
        for k, v in train_metrics.items():
            f.write(f"{k}: {v:.6f}\n")

    prob_df = pd.DataFrame({
        "x": x_samp, "y": y_coord_samp,
        "y_true": y_samp_1based, "y_pred": train_pred_1based,
    })
    for i, lb in enumerate(sorted_labels):
        prob_df[f"prob_class{lb}"] = train_proba[:, i]
    eps = 1e-12
    prob_df["uncertainty_entropy"] = -np.sum(
        train_proba * np.log(train_proba + eps), axis=1
    )
    prob_df["uncertainty_1minus_maxprob"] = 1.0 - train_proba.max(axis=1)
    prob_df.to_csv(os.path.join(out_dir, "final_model_class_probabilities.csv"), index=False)

    if ENABLE_SHAP:
        try:
            save_shap_outputs(
                final_net,
                final_scaler,
                X_samp,
                var_names_this,
                out_dir,
                DEVICE,
            )
        except Exception as exc:
            print(f"[WARNING] Could not compute SHAP explanations: {exc}")

    # Save model bundle — compatible with production inference pipeline
    try:
        model_path = os.path.join(out_dir, "mlp_cls_final_model_Kalkklass.joblib")
        scaler_mean = getattr(final_scaler, "mean_", None)
        scaler_scale = getattr(final_scaler, "scale_", None)
        joblib.dump(
            {
                "model_type": "mlp_classifier",
                # State dict + architecture so the model can be reconstructed
                "state_dict": {k: v.cpu().numpy() for k, v in final_net.state_dict().items()},
                "architecture": {
                    "n_features": X_samp.shape[1],
                    "n_classes": n_classes,
                    "hidden_layers": active_hidden,
                    "dropout": active_drop,
                    "use_batch_norm": USE_BATCH_NORM,
                },
                # Store scaler object directly (works for StandardScaler and RobustScaler).
                "scaler": final_scaler,
                # Backward-compatible keys for old loaders expecting StandardScaler attrs.
                "scaler_mean": scaler_mean,
                "scaler_scale": scaler_scale,
                "var_names": var_names_this,
                "target_variable": TARGET_COL,
                "label_offset": LABEL_OFFSET,
                "class_labels": CLASS_LABELS,
                "n_classes": n_classes,
                "loss_fn": active_loss,
                "focal_gamma": active_gamma,
                "class_weight_mode": active_class_weight_mode,
                "scaler_type": SCALER_TYPE,
                "matrix_path": MATRIX_PATH,
            },
            model_path,
        )
        print(f"Saved model: {model_path}")
    except Exception as exc:
        print(f"[WARNING] Could not save final model: {exc}")

    return importance_df


# =============================================================================
# MATRIX LOADING
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


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
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
        raise KeyError(f"Target column '{TARGET_COL}' not found in CSV.")
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

    # ── Tuning ───────────────────────────────────────────────────────────────
    tuned_params: Optional[dict] = None
    if ENABLE_MLP_TUNING:
        tuning_out = os.path.join(OUT_DIR, "_tuning")
        tuned_params = tune_mlp_params(
            X_cls, y_cls - LABEL_OFFSET, x_cls, y_coord_cls,
            len(CLASS_LABELS), tuning_out,
        )
        if tuned_params:
            print(f"Using tuned params: {tuned_params}")

    # ── Experiment / multi-run / single run ──────────────────────────────────
    if RUN_EXPERIMENTS:
        exp_base = os.path.join(OUT_DIR, "experiments")
        os.makedirs(exp_base, exist_ok=True)
        latest_imp_csv = os.path.join(OUT_DIR, "variable_importance.csv")
        exp_rows: List[dict] = []

        for exp in EXPERIMENT_RUNS:
            exp_name = exp["name"]
            exp_loss = exp.get("loss_fn", LOSS_FN)
            exp_gamma = exp.get("focal_gamma", FOCAL_GAMMA)
            exp_top_n = exp.get("top_n")
            print(f"\n{'='*70}\nEXPERIMENT: {exp_name}\n{'='*70}")

            if exp_top_n is None:
                X_run, var_run = X_cls, var_names_f
            else:
                src = latest_imp_csv if os.path.exists(latest_imp_csv) else None
                if not src:
                    raise FileNotFoundError("No importance CSV available for top-N experiment.")
                X_run, var_run = select_top_predictors_from_importance(
                    X_cls, var_names_f, src, exp_top_n
                )
            exp_out = os.path.join(exp_base, exp_name)
            run_cv_and_save(
                X_run, y_cls, x_cls, y_coord_cls, var_run, exp_out,
                mlp_params=tuned_params, loss_fn=exp_loss, focal_gamma=exp_gamma,
            )
            exp_imp = os.path.join(exp_out, "variable_importance.csv")
            if os.path.exists(exp_imp):
                latest_imp_csv = exp_imp

            fold_df = pd.read_csv(os.path.join(exp_out, "cv_fold_metrics.csv"))
            row: dict = {
                "experiment": exp_name, "loss_fn": exp_loss, "focal_gamma": exp_gamma,
                "top_n": exp_top_n if exp_top_n is not None else len(var_run),
                "n_predictors": len(var_run),
                "accuracy_mean": float(fold_df["accuracy"].mean()),
                "macro_f1_mean": float(fold_df["macro_f1"].mean()),
                "weighted_f1_mean": float(fold_df["weighted_f1"].mean()),
            }
            for cls_id in sorted(CLASS_LABELS):
                pref = f"f1_class{cls_id}_"
                col = next((c for c in fold_df.columns if c.startswith(pref)), None)
                row[f"f1_class{cls_id}_mean"] = float(fold_df[col].mean()) if col else np.nan
            exp_rows.append(row)

        exp_df = pd.DataFrame(exp_rows).sort_values("macro_f1_mean", ascending=False)
        exp_df.to_csv(os.path.join(exp_base, "experiment_summary.csv"), index=False)
        print("\n=== Experiment summary ===")
        print(exp_df.to_string(index=False))

    elif MULT_RUNS:
        full_out = os.path.join(OUT_DIR, "All")
        imp_df = run_cv_and_save(
            X_cls, y_cls, x_cls, y_coord_cls, var_names_f, full_out, mlp_params=tuned_params
        )
        print(f"\nKalkklass MLP All outputs -> {full_out}")
        if imp_df is not None:
            for top_n in MULT_RUNS_TOP_N:
                if top_n >= len(var_names_f):
                    print(f"[SKIP] top{top_n} >= total predictors ({len(var_names_f)})")
                    continue
                top_vars = imp_df["Variable"].iloc[:top_n].tolist()
                top_idx = [var_names_f.index(v) for v in top_vars if v in var_names_f]
                topn_out = os.path.join(OUT_DIR, f"Top_{top_n}")
                run_cv_and_save(
                    X_cls[:, top_idx], y_cls, x_cls, y_coord_cls,
                    [var_names_f[i] for i in top_idx], topn_out,
                    mlp_params=tuned_params,
                )
                print(f"\nKalkklass MLP Top_{top_n} outputs -> {topn_out}")
    else:
        run_cv_and_save(
            X_cls, y_cls, x_cls, y_coord_cls, var_names_f, OUT_DIR, mlp_params=tuned_params
        )
        print(f"\nKalkklass MLP outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()

print("\n=== Kalkklass MLP classification completed ===")
