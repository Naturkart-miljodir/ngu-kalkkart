"""
Pixel-Based Random Forest Variable Selection for Kalk Project
- Loads rasters and polygons directly
- Rasterizes polygons to pixels
- Samples pixels for training
- Trains Random Forest classifier
- Outputs variable importance rankings
- Saves results to Modelling/RandForest
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GroupKFold
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def windows_path_to_wsl(path_text):
    normalized = path_text.replace("\\", "/")
    if len(normalized) >= 2 and normalized[1] == ":":
        drive_letter = normalized[0].lower()
        return f"/mnt/{drive_letter}{normalized[2:]}"
    return normalized

# ----------------------------------------------------
# COMPUTE BACKEND
# ----------------------------------------------------
# Options: "auto" (try cuML GPU, then XGBoost CUDA, fallback CPU), "gpu" (require GPU), "cpu"
RF_BACKEND = "auto"
SMOKE_TEST_ONLY = os.getenv("RF_SMOKE_TEST", "0") == "1"
DISABLE_CUPY = os.getenv("RF_DISABLE_CUPY", "0") == "1"

GPU_AVAILABLE = False
cp = None
cuRF = None
XGB_AVAILABLE = False
xgb = None
ACTIVE_BACKEND = "sklearn"
CUPY_AVAILABLE = False

if RF_BACKEND in ("auto", "gpu"):
    try:
        if not DISABLE_CUPY:
            import cupy as cp
        from cuml.ensemble import RandomForestClassifier as cuRF
        GPU_AVAILABLE = True
        ACTIVE_BACKEND = "cuml"
        print("GPU backend enabled: cuML RandomForestClassifier")
        CUPY_AVAILABLE = True
    except Exception as e:
        if cp is not None:
            CUPY_AVAILABLE = True
        print(f"cuML backend unavailable ({e}).")

if RF_BACKEND in ("auto", "gpu") and not GPU_AVAILABLE:
    try:
        import xgboost as xgb
        XGB_AVAILABLE = True
        ACTIVE_BACKEND = "xgboost_cuda"
        print("GPU backend enabled: XGBoost CUDA classifier")
    except Exception as e:
        if RF_BACKEND == "gpu":
            raise ImportError(
                "RF_BACKEND='gpu' but no supported GPU backend was found. "
                "Install RAPIDS (cuml, cupy) or install xgboost with CUDA support."
            ) from e
        print(f"XGBoost CUDA backend unavailable ({e}). Falling back to CPU sklearn RandomForestClassifier.")

if cp is None:
    try:
        if not DISABLE_CUPY:
            import cupy as cp
        CUPY_AVAILABLE = True
    except Exception:
        CUPY_AVAILABLE = False

if RF_BACKEND == "cpu":
    ACTIVE_BACKEND = "sklearn"

if SMOKE_TEST_ONLY:
    if ACTIVE_BACKEND == "cuml":
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            device_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
            print(f"SMOKE_TEST: GPU backend active (cuML/CuPy). Devices: {device_count}. Device[0]: {device_name}")
        except Exception as e:
            print(f"SMOKE_TEST: GPU backend active (cuML/CuPy), but device query failed: {e}")
    elif ACTIVE_BACKEND == "xgboost_cuda":
        if CUPY_AVAILABLE:
            print("SMOKE_TEST: GPU backend active (XGBoost CUDA + CuPy arrays).")
        else:
            print("SMOKE_TEST: GPU backend active (XGBoost CUDA). CuPy unavailable for GPU prediction arrays.")
    else:
        print("SMOKE_TEST: CPU fallback active (sklearn). cuML/CuPy not available.")
    sys.exit(0)


def _to_numpy(arr):
    if GPU_AVAILABLE and cp is not None:
        try:
            if isinstance(arr, cp.ndarray):
                return cp.asnumpy(arr)
        except Exception:
            pass
    return np.asarray(arr)


def _make_rf_model():
    if ACTIVE_BACKEND == "cuml":
        return cuRF(
            n_estimators=100,
            max_depth=20,
            random_state=42
        )
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )


def _fit_predict_and_importance(model, X_tr, y_tr, X_te):
    if ACTIVE_BACKEND == "cuml":
        X_tr_gpu = cp.asarray(X_tr, dtype=cp.float32)
        y_tr_gpu = cp.asarray(y_tr, dtype=cp.int32)
        X_te_gpu = cp.asarray(X_te, dtype=cp.float32)

        model.fit(X_tr_gpu, y_tr_gpu)
        y_pred = _to_numpy(model.predict(X_te_gpu)).astype(int)
        importances = _to_numpy(model.feature_importances_)
        return y_pred, importances

    if ACTIVE_BACKEND == "xgboost_cuda":
        n_classes = int(np.unique(y_tr).size)
        y_tr_zero = y_tr.astype(np.int32) - 1

        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softmax",
            num_class=n_classes,
            tree_method="hist",
            device="cuda",
            random_state=42,
            n_jobs=0,
            eval_metric="mlogloss"
        )
        xgb_model.fit(X_tr, y_tr_zero)

        if CUPY_AVAILABLE:
            try:
                booster = xgb_model.get_booster()
                booster.set_param({"device": "cuda:0"})
                X_te_gpu = cp.asarray(X_te, dtype=cp.float32)
                y_pred_zero = cp.asnumpy(booster.inplace_predict(X_te_gpu)).astype(np.int32)
            except Exception:
                xgb_model.set_params(device="cpu")
                y_pred_zero = xgb_model.predict(X_te).astype(np.int32)
        else:
            xgb_model.set_params(device="cpu")
            y_pred_zero = xgb_model.predict(X_te).astype(np.int32)

        y_pred = (y_pred_zero + 1).astype(int)
        importances = np.asarray(xgb_model.feature_importances_)
        return y_pred, importances

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    importances = np.asarray(model.feature_importances_)
    return y_pred, importances

# ----------------------------------------------------
# PATHS
# ----------------------------------------------------
PROJECT_DIR_WINDOWS = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project"

if os.name == "nt":
    PROJECT_DIR = PROJECT_DIR_WINDOWS
else:
    PROJECT_DIR = windows_path_to_wsl(PROJECT_DIR_WINDOWS)

OUT_DIR = os.path.join(PROJECT_DIR, "Modelling", "RandForest")
MATRIX_PATH = os.path.join(OUT_DIR, "regression_matrix.npz")

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------
# LOAD REGRESSION MATRIX
# ----------------------------------------------------
if not os.path.exists(MATRIX_PATH):
    raise FileNotFoundError(
        f"Regression matrix not found: {MATRIX_PATH}\n"
        "Run RF_regression_matrix.py first to create it."
    )

matrix = np.load(MATRIX_PATH, allow_pickle=True)
X_all = matrix["X"]
y_all = matrix["y"].astype(int)
rows_arr = matrix["rows"].astype(int)
cols_arr = matrix["cols"].astype(int)
var_names = matrix["var_names"].tolist()
width = int(matrix["width"][0]) if "width" in matrix.files else int(cols_arr.max() + 1)

print(f"Loaded regression matrix: {MATRIX_PATH}")
print(f"Samples: {len(y_all)}, Predictors: {X_all.shape[1]}")
print(f"Class distribution: {np.bincount(y_all)}")

# ----------------------------------------------------
# SAMPLE DATA (balanced if possible) + Spatial-block CV
# ----------------------------------------------------
SAMPLE_SIZE = 50000

# Define spatial block size in pixels (adjust if needed)
BLOCK_SIZE = 500

if len(y_all) > SAMPLE_SIZE:
    sss = StratifiedShuffleSplit(n_splits=1, train_size=SAMPLE_SIZE, random_state=42)
    idx = next(sss.split(X_all, y_all))[0]
    X_sample = X_all[idx]
    y_sample = y_all[idx]
    rows_sample = rows_arr[idx]
    cols_sample = cols_arr[idx]
else:
    X_sample, y_sample = X_all, y_all
    rows_sample, cols_sample = rows_arr, cols_arr

print(f"Sampled {len(y_sample)} pixels")

# Assign block IDs based on pixel indices
n_block_cols = int(np.ceil(width / BLOCK_SIZE))
block_rows = rows_sample // BLOCK_SIZE
block_cols = cols_sample // BLOCK_SIZE
groups = block_rows * n_block_cols + block_cols

print(f"Using spatial blocks of {BLOCK_SIZE} px; {np.unique(groups).size} blocks in sample")

# ----------------------------------------------------
# Spatial-block cross-validation (GroupKFold)
# ----------------------------------------------------
print("\nRunning spatial-block cross-validation...")

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

importances_list = []
y_true_all = []
y_pred_all = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_sample, y_sample, groups)):
    print(f"Fold {fold + 1}/{n_splits}: train groups={np.unique(groups[train_idx]).size}, test groups={np.unique(groups[test_idx]).size}")

    X_tr, X_te = X_sample[train_idx], X_sample[test_idx]
    y_tr, y_te = y_sample[train_idx], y_sample[test_idx]

    rf = _make_rf_model()

    y_pred, fold_importances = _fit_predict_and_importance(rf, X_tr, y_tr, X_te)
    y_true_all.extend(y_te.tolist())
    y_pred_all.extend(y_pred.tolist())
    importances_list.append(fold_importances)

# Aggregate results across folds
print("\nCross-validated Classification Report:")
print(classification_report(y_true_all, y_pred_all, target_names=["low", "medium", "high"]))

mean_importances = np.mean(importances_list, axis=0)

# Train final model on full sampled data to save a final importance ranking as well
final_rf = _make_rf_model()
if ACTIVE_BACKEND == "cuml":
    final_rf.fit(
        cp.asarray(X_sample, dtype=cp.float32),
        cp.asarray(y_sample, dtype=cp.int32)
    )
    final_importances = _to_numpy(final_rf.feature_importances_)
elif ACTIVE_BACKEND == "xgboost_cuda":
    n_classes = int(np.unique(y_sample).size)
    y_sample_zero = y_sample.astype(np.int32) - 1
    final_rf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=n_classes,
        tree_method="hist",
        device="cuda",
        random_state=42,
        n_jobs=0,
        eval_metric="mlogloss"
    )
    final_rf.fit(X_sample, y_sample_zero)
    final_importances = np.asarray(final_rf.feature_importances_)
else:
    final_rf.fit(X_sample, y_sample)
    final_importances = final_rf.feature_importances_

# Use mean_importances for cross-validated importance ranking
importances = mean_importances

# ----------------------------------------------------
# VARIABLE IMPORTANCE (cross-validated)
# ----------------------------------------------------
importances = mean_importances
importance_df = pd.DataFrame({
    'Variable': var_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nTop Variables by Importance:")
print(importance_df.head(10))

# ----------------------------------------------------
# SAVE RESULTS
# ----------------------------------------------------
importance_path = os.path.join(OUT_DIR, "variable_importance.csv")
importance_df.to_csv(importance_path, index=False)

report_path = os.path.join(OUT_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write("Random Forest Cross-validated Classification Report\n")
    f.write("=" * 40 + "\n")
    f.write(classification_report(y_true_all, y_pred_all, target_names=["low", "medium", "high"]))
    f.write(f"\nTotal samples (sampled): {len(y_sample)}\n")
    f.write(f"Total test predictions (CV): {len(y_pred_all)}\n")

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Variable'][:15], importance_df['Importance'][:15])
plt.xlabel('Importance')
plt.title('Top Variable Importances (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()

plot_path = os.path.join(OUT_DIR, "variable_importance.png")
plt.savefig(plot_path, dpi=150)
plt.close()

print("\n✔ Results saved to:")
print(f"  - {importance_path}")
print(f"  - {report_path}")
print(f"  - {plot_path}")

print("\nWorkflow complete! Use top variables for U-Net input.")