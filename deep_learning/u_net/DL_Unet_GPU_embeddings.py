# -*- coding: utf-8 -*-
"""
U-Net training for Kalk project tiles - GPU OPTIMIZED VERSION
- Masked loss (background ignored)
- Class-weighted loss for classes 1–3
- Optional ADVANCED LOSSES:
      • Focal-Tversky
      • Dice + CE hybrid
- Spatial CV (N folds) with KFold
- Per-epoch macro-F1 (background ignored)
- MC DROPOUT UNCERTAINTY QUANTIFICATION
- QC plots + metric logs
- GPU configuration with memory growth
"""

import os
import sys
import glob
import csv
import importlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

try:
    tqdm = importlib.import_module("tqdm.auto").tqdm
except Exception:
    # Fallback: no external dependency required
    def tqdm(x, **kwargs):
        return x


from sklearn.model_selection import train_test_split, KFold
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Input,
    Concatenate,
    Dropout,
    Embedding,
    Lambda,
)
from keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint, TerminateOnNaN
import matplotlib.pyplot as plt

# Add zlib path for Windows (fixes zlibwapi.dll error)
if sys.platform == "win32":
    conda_env = os.path.dirname(os.path.dirname(sys.executable))
    zlib_path = os.path.join(conda_env, "Library", "bin")
    if os.path.exists(zlib_path):
        # Add to PATH as fallback
        if zlib_path not in os.environ["PATH"]:
            os.environ["PATH"] = zlib_path + os.pathsep + os.environ["PATH"]
        # Python 3.8+: explicitly add DLL search path (more reliable)
        try:
            os.add_dll_directory(zlib_path)
        except AttributeError:
            pass  # Python < 3.8, PATH modification above is the fallback

# =====================================================================
# GPU CONFIGURATION
# =====================================================================
print("\n" + "=" * 70)
print("GPU CONFIGURATION")
print("=" * 70)

print(f"TensorFlow version: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

# List available GPUs
gpus = tf.config.list_physical_devices("GPU")
print(f"Number of GPUs available: {len(gpus)}")

if gpus:
    try:
        # Enable memory growth for all GPUs (prevents TensorFlow from allocating all GPU memory)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Enabled memory growth for: {gpu.name}")

        # Get logical devices
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")

        # Test GPU availability
        if tf.test.gpu_device_name():
            print(f"✓ Default GPU device: {tf.test.gpu_device_name()}")
        else:
            print("⚠ GPU detected but not set as default device")

    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(f"✗ GPU configuration error: {e}")
else:
    print("✗ WARNING: No GPU found!")
    print("  Training will run on CPU (this will be VERY slow for U-Net)")
    print("  Consider:")
    print("    1. Installing tensorflow-gpu")
    print("    2. Installing CUDA toolkit")
    print("    3. Installing cuDNN")

print("=" * 70)
print()

# =====================================================================
# DISTRIBUTED TRAINING STRATEGY
# =====================================================================
if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using MirroredStrategy across {strategy.num_replicas_in_sync} GPUs")
else:
    strategy = tf.distribute.get_strategy()
    print(f"Using default strategy with {strategy.num_replicas_in_sync} replica")

BATCH_SIZE_PER_REPLICA = 1
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
print(f"Batch size per replica: {BATCH_SIZE_PER_REPLICA}")
print(f"Global batch size: {GLOBAL_BATCH_SIZE}")
print()

# =====================================================================
# 0. USER SWITCHES & DEFAULT ADVANCED LOSS PARAMETERS
# =====================================================================

DO_SPATIAL_CV = True
N_FOLDS = 3
DO_MAIN_TRAIN = True
CV_EPOCHS = 5
MAIN_EPOCHS = 40
LEARNING_RATE = 5e-5
CLASS2_WEIGHT_MULTIPLIER = 1.8

# UNCERTAINTY QUANTIFICATION
USE_MC_DROPOUT = True  # Enable MC Dropout for uncertainty
MC_DROPOUT_RATE = 0.3  # Dropout rate during training (keep same at inference)
MC_ITERATIONS = 20  # Number of forward passes for uncertainty
MC_BATCH_SIZE = 8  # Batch size for MC inference (prevents GPU OOM)

# CATEGORICAL EMBEDDINGS SETTINGS
USE_CATEGORICAL_EMBEDDINGS = True
QUATERNARY_NUM_CLASSES = 22
LANDUSE_NUM_CLASSES = 33
QUATERNARY_EMBED_DIM = 5
LANDUSE_EMBED_DIM = 6
QUATERNARY_PREDICTOR_NAME = "quaternary_ForenkletK_cog"
LANDUSE_PREDICTOR_NAME = "landuse_Code_18_cog"
REMOVE_EMBED_CHANNELS_FROM_CONTINUOUS = True

# Channel exclusion toggle (name-based using channel_map.csv)
# False -> keep all channels
# ("A", "B") -> exclude predictors A and B from continuous branch
CHANNELS_EXCL = False

# Optional feature ablation (for diagnostics)
DROP_DERIVED_CHANNELS = False  # Toggle to False to re-enable normalized indices
DERIVED_CHANNEL_IDX = [
    11,
    12,
    13,
    14,
    15,
]  # NDVI, NDWI, NDMI, NDSI, NBR (now properly normalized)

# MAIN TOGGLE: Basic vs Advanced losses
USE_ADVANCED_LOSS = True  # ⬅️ Change this to True to activate advanced loss
ADVANCED_LOSS_TYPE = "dice_ce"  # options: "focal_tversky", "dice_ce"

# Default parameters for Focal-Tversky
FT_ALPHA = 0.7
FT_BETA = 0.3
FT_GAMMA = 1.33

print("\n=== LOSS SETTINGS ===")
print("USE_ADVANCED_LOSS:", USE_ADVANCED_LOSS)
print("ADVANCED_LOSS_TYPE:", ADVANCED_LOSS_TYPE)
print(f"Focal-Tversky: α={FT_ALPHA}, β={FT_BETA}, γ={FT_GAMMA}")

print("\n=== UNCERTAINTY SETTINGS ===")
print("USE_MC_DROPOUT:", USE_MC_DROPOUT)
print("MC_DROPOUT_RATE:", MC_DROPOUT_RATE)
print("MC_ITERATIONS:", MC_ITERATIONS)
print("MC_BATCH_SIZE:", MC_BATCH_SIZE)
print("USE_CATEGORICAL_EMBEDDINGS:", USE_CATEGORICAL_EMBEDDINGS)
print("QUATERNARY_NUM_CLASSES:", QUATERNARY_NUM_CLASSES)
print("LANDUSE_NUM_CLASSES:", LANDUSE_NUM_CLASSES)
print("QUATERNARY_EMBED_DIM:", QUATERNARY_EMBED_DIM)
print("LANDUSE_EMBED_DIM:", LANDUSE_EMBED_DIM)
print("QUATERNARY_PREDICTOR_NAME:", QUATERNARY_PREDICTOR_NAME)
print("LANDUSE_PREDICTOR_NAME:", LANDUSE_PREDICTOR_NAME)
print("REMOVE_EMBED_CHANNELS_FROM_CONTINUOUS:", REMOVE_EMBED_CHANNELS_FROM_CONTINUOUS)
print("CHANNELS_EXCL:", CHANNELS_EXCL)
print("DROP_DERIVED_CHANNELS:", DROP_DERIVED_CHANNELS)
print("DERIVED_CHANNEL_IDX:", DERIVED_CHANNEL_IDX)
print("LEARNING_RATE:", LEARNING_RATE)
print("CV_EPOCHS:", CV_EPOCHS)
print("CLASS2_WEIGHT_MULTIPLIER:", CLASS2_WEIGHT_MULTIPLIER)
print()

# =====================================================================
# 1. Paths
# =====================================================================

TILE_DIR = r"/home/acosta_pedro/DL_tiles"
MODEL_OUT = r"/home/acosta_pedro/outputs/Models"
QC_PLOTS = r"/home/acosta_pedro/outputs/QC_plots"
CHANNEL_MAP_PATH = r"/home/acosta_pedro/DL_chips_spatial_location/channel_map.csv"

X_dir = os.path.join(TILE_DIR, "X")
y_dir = os.path.join(TILE_DIR, "y")
# Legacy fallback (used only if channel map is unavailable)
qua_dir = os.path.join(TILE_DIR, "quaternary_soil")
land_dir = os.path.join(TILE_DIR, "landuse_code18")

os.makedirs(MODEL_OUT, exist_ok=True)
os.makedirs(QC_PLOTS, exist_ok=True)

print(f"\n=== DATA PATHS ===")
print(f"Input tiles: {TILE_DIR}")
print(f"Models output: {MODEL_OUT}")
print(f"QC plots: {QC_PLOTS}")
print(f"Channel map: {CHANNEL_MAP_PATH}")


def load_channel_map(channel_map_path):
    if not os.path.exists(channel_map_path):
        raise FileNotFoundError(f"Channel map not found: {channel_map_path}")

    with open(channel_map_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"Channel map is empty: {channel_map_path}")

    by_predictor = {}
    for row in rows:
        name = row["predictor_name"].strip().lower()
        idx = int(row["channel_idx"])
        by_predictor.setdefault(name, []).append(idx)

    for name in by_predictor:
        by_predictor[name] = sorted(by_predictor[name])

    return rows, by_predictor


def resolve_single_channel_idx(by_predictor, predictor_name):
    key = predictor_name.strip().lower()
    idxs = by_predictor.get(key, [])
    if len(idxs) == 0:
        raise RuntimeError(f"Predictor '{predictor_name}' not found in channel map")
    if len(idxs) > 1:
        raise RuntimeError(
            f"Predictor '{predictor_name}' has multiple channels {idxs}; expected a single categorical channel"
        )
    return idxs[0]


def parse_excluded_predictor_names(channels_excl):
    if channels_excl is False or channels_excl is None:
        return []

    if isinstance(channels_excl, str):
        raw_names = [channels_excl]
    elif isinstance(channels_excl, (list, tuple, set)):
        raw_names = list(channels_excl)
    else:
        raise ValueError(
            "CHANNELS_EXCL must be False, a string, or a list/tuple/set of strings"
        )

    parsed = []
    for name in raw_names:
        if not isinstance(name, str):
            raise ValueError("CHANNELS_EXCL entries must be strings")
        key = name.strip().lower()
        if key:
            parsed.append(key)

    return parsed


# =====================================================================
# 2. Load tiles
# =====================================================================

print("\n=== LOADING DATA ===")
X_paths = sorted(glob.glob(os.path.join(X_dir, "*.npy")))
y_paths = sorted(glob.glob(os.path.join(y_dir, "*.npy")))

print(f"Found {len(X_paths)} X tiles and {len(y_paths)} y tiles")

if len(X_paths) == 0 or len(y_paths) == 0:
    print(f"\n[ERROR] No tiles found!")
    print(f"  X dir: {X_dir}")
    print(f"  y dir: {y_dir}")
    exit(1)

if len(X_paths) != len(y_paths):
    print("\n[ERROR] Tile count mismatch across X/y!")
    print(f"  X: {len(X_paths)}, y: {len(y_paths)}")
    exit(1)

print("Loading X tiles...")
X = np.array([np.load(p) for p in tqdm(X_paths, desc="Loading X", unit="tile")])

print("Loading y tiles...")
cleaned = []
for p in tqdm(y_paths, desc="Loading y", unit="tile"):
    arr = np.load(p).astype(np.int16)
    arr = np.where((arr < 0) | (arr > 3), 0, arr)
    cleaned.append(arr)

y = np.array(cleaned, dtype=np.int16)

embeddings_from_channel_map = False
channel_map_rows = None
channel_map_by_predictor = None
quaternary_channel_idx_orig = -1
landuse_channel_idx_orig = -1
excluded_predictor_names = parse_excluded_predictor_names(CHANNELS_EXCL)
excluded_channel_idx_orig = set()

if USE_CATEGORICAL_EMBEDDINGS and os.path.exists(CHANNEL_MAP_PATH):
    print(f"Loading channel map: {CHANNEL_MAP_PATH}")
    channel_map_rows, channel_map_by_predictor = load_channel_map(CHANNEL_MAP_PATH)
    print(f"Loaded {len(channel_map_rows)} channel-map rows")

    quaternary_channel_idx_orig = resolve_single_channel_idx(
        channel_map_by_predictor, QUATERNARY_PREDICTOR_NAME
    )
    landuse_channel_idx_orig = resolve_single_channel_idx(
        channel_map_by_predictor, LANDUSE_PREDICTOR_NAME
    )

    print(
        f"Channel map selection -> {QUATERNARY_PREDICTOR_NAME}: idx {quaternary_channel_idx_orig}, "
        f"{LANDUSE_PREDICTOR_NAME}: idx {landuse_channel_idx_orig}"
    )
    embeddings_from_channel_map = True

if excluded_predictor_names:
    if channel_map_by_predictor is None:
        raise RuntimeError(
            "CHANNELS_EXCL requires CHANNEL_MAP_PATH to be available so names can be resolved."
        )

    unresolved = []
    for predictor_name in excluded_predictor_names:
        idxs = channel_map_by_predictor.get(predictor_name, [])
        if not idxs:
            unresolved.append(predictor_name)
            continue
        excluded_channel_idx_orig.update(idxs)

    if unresolved:
        raise RuntimeError(
            f"These CHANNELS_EXCL predictors were not found in channel_map.csv: {unresolved}"
        )

    print(
        f"[INFO] CHANNELS_EXCL resolved to original channel indices: {sorted(excluded_channel_idx_orig)}"
    )

quaternary = np.empty((0,), dtype=np.int32)
landuse = np.empty((0,), dtype=np.int32)

if USE_CATEGORICAL_EMBEDDINGS and not embeddings_from_channel_map:
    print(
        "[WARNING] Channel map not found. Falling back to legacy quaternary/landuse tile folders."
    )
    qua_paths = sorted(glob.glob(os.path.join(qua_dir, "*.npy")))
    land_paths = sorted(glob.glob(os.path.join(land_dir, "*.npy")))
    print(
        f"Found {len(qua_paths)} quaternary tiles and {len(land_paths)} landuse tiles"
    )

    if len(qua_paths) == 0 or len(land_paths) == 0:
        print(
            "\n[ERROR] Categorical embeddings enabled but neither channel_map nor legacy categorical tiles are available."
        )
        print(f"  Channel map: {CHANNEL_MAP_PATH}")
        print(f"  quaternary dir: {qua_dir}")
        print(f"  landuse dir: {land_dir}")
        exit(1)

    if not (len(X_paths) == len(y_paths) == len(qua_paths) == len(land_paths)):
        print("\n[ERROR] Tile count mismatch across inputs!")
        print(
            f"  X: {len(X_paths)}, y: {len(y_paths)}, quaternary: {len(qua_paths)}, landuse: {len(land_paths)}"
        )
        exit(1)

    print("Loading quaternary tiles...")
    qua_cleaned = []
    for p in tqdm(qua_paths, desc="Loading quaternary", unit="tile"):
        arr = np.load(p)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        arr = arr.astype(np.int32)
        arr = np.where((arr < 0) | (arr > QUATERNARY_NUM_CLASSES), 0, arr)
        qua_cleaned.append(arr)
    quaternary = np.array(qua_cleaned, dtype=np.int32)[..., np.newaxis]

    print("Loading landuse tiles...")
    land_cleaned = []
    for p in tqdm(land_paths, desc="Loading landuse", unit="tile"):
        arr = np.load(p)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        arr = arr.astype(np.int32)
        arr = np.where((arr < 0) | (arr > LANDUSE_NUM_CLASSES), 0, arr)
        land_cleaned.append(arr)
    landuse = np.array(land_cleaned, dtype=np.int32)[..., np.newaxis]

print(f"\nShapes -> X: {X.shape}, y: {y.shape}")
print(f"Unique labels: {np.unique(y)}")
print(f"Memory usage: X={X.nbytes / 1e9:.2f} GB, y={y.nbytes / 1e9:.2f} GB")

# =====================================================================
# 3. Convert X to channels-last (if needed)
# =====================================================================

# Check if data needs transposing from (N, C, H, W) to (N, H, W, C)
if X.ndim == 4 and X.shape[1] < X.shape[2]:
    print(
        f"[INFO] Transposing X from {X.shape} (channels_first) to channels_last format..."
    )
    X = np.transpose(X, (0, 2, 3, 1))
    print(f"[INFO] X shape after transpose: {X.shape}")
elif X.ndim == 4 and X.shape[-1] < X.shape[1]:
    print(f"[INFO] Data already in channels_last format: {X.shape}")
else:
    print(f"[WARNING] Unexpected X shape: {X.shape}")

dropped_channels: set[int] = set()
if DROP_DERIVED_CHANNELS:
    dropped_channels = set(i for i in DERIVED_CHANNEL_IDX if 0 <= i < X.shape[-1])
    keep_idx = [i for i in range(X.shape[-1]) if i not in DERIVED_CHANNEL_IDX]
    print(f"[INFO] Dropping derived channels: {DERIVED_CHANNEL_IDX}")
    print(f"[INFO] Keeping channels: {keep_idx}")
    X = X[..., keep_idx]
    print(f"[INFO] X shape after channel drop: {X.shape}")

if USE_CATEGORICAL_EMBEDDINGS and embeddings_from_channel_map:
    if quaternary_channel_idx_orig < 0 or landuse_channel_idx_orig < 0:
        raise RuntimeError(
            "Invalid channel-map categorical indices. Check predictor names and channel map."
        )

    if quaternary_channel_idx_orig in dropped_channels:
        raise RuntimeError(
            f"Configured quaternary predictor channel index {quaternary_channel_idx_orig} was dropped by DERIVED_CHANNEL_IDX"
        )
    if landuse_channel_idx_orig in dropped_channels:
        raise RuntimeError(
            f"Configured landuse predictor channel index {landuse_channel_idx_orig} was dropped by DERIVED_CHANNEL_IDX"
        )

    quaternary_channel_idx = quaternary_channel_idx_orig - len(
        [d for d in dropped_channels if d < quaternary_channel_idx_orig]
    )
    landuse_channel_idx = landuse_channel_idx_orig - len(
        [d for d in dropped_channels if d < landuse_channel_idx_orig]
    )

    print(
        f"[INFO] Extracting embedding channels from X -> "
        f"quaternary idx {quaternary_channel_idx}, landuse idx {landuse_channel_idx}"
    )

    if not (0 <= quaternary_channel_idx < X.shape[-1]):
        raise RuntimeError(
            f"Quaternary channel index {quaternary_channel_idx} out of bounds for X with {X.shape[-1]} channels"
        )
    if not (0 <= landuse_channel_idx < X.shape[-1]):
        raise RuntimeError(
            f"Landuse channel index {landuse_channel_idx} out of bounds for X with {X.shape[-1]} channels"
        )

    quaternary_arr = np.rint(
        np.nan_to_num(X[..., quaternary_channel_idx], nan=0.0)
    ).astype(np.int32)
    landuse_arr = np.rint(np.nan_to_num(X[..., landuse_channel_idx], nan=0.0)).astype(
        np.int32
    )

    quaternary_arr = np.where(
        (quaternary_arr < 0) | (quaternary_arr > QUATERNARY_NUM_CLASSES),
        0,
        quaternary_arr,
    )
    landuse_arr = np.where(
        (landuse_arr < 0) | (landuse_arr > LANDUSE_NUM_CLASSES), 0, landuse_arr
    )

    quaternary = quaternary_arr[..., np.newaxis]
    landuse = landuse_arr[..., np.newaxis]

continuous_remove_idx = set()

if excluded_channel_idx_orig:
    for orig_idx in sorted(excluded_channel_idx_orig):
        if orig_idx in dropped_channels:
            continue
        current_idx = orig_idx - len([d for d in dropped_channels if d < orig_idx])
        if 0 <= current_idx < X.shape[-1]:
            continuous_remove_idx.add(current_idx)

if (
    USE_CATEGORICAL_EMBEDDINGS
    and embeddings_from_channel_map
    and REMOVE_EMBED_CHANNELS_FROM_CONTINUOUS
):
    continuous_remove_idx.update({quaternary_channel_idx, landuse_channel_idx})

if continuous_remove_idx:
    keep_idx = [i for i in range(X.shape[-1]) if i not in continuous_remove_idx]
    print(
        f"[INFO] Removing channels from continuous branch: {sorted(continuous_remove_idx)}"
    )
    X = X[..., keep_idx]
    print(f"[INFO] X shape after channel removal: {X.shape}")

if USE_CATEGORICAL_EMBEDDINGS:
    if quaternary.size == 0 or landuse.size == 0:
        raise RuntimeError(
            "Categorical embeddings are enabled but categorical tensors were not prepared."
        )
    quaternary = np.asarray(quaternary, dtype=np.int32)
    landuse = np.asarray(landuse, dtype=np.int32)
    print(f"Shapes -> quaternary: {quaternary.shape}, landuse: {landuse.shape}")

num_classes = 3
n_channels = X.shape[-1]
print("X shape:", X.shape)
print(f"Number of channels: {n_channels}")
print(
    f"[DEBUG] X after loading - has NaN: {np.isnan(X).any()}, range: [{np.nanmin(X):.4f}, {np.nanmax(X):.4f}]"
)

# Store NaN mask before cleaning (True where data is VALID)
nan_mask_original = np.isnan(X)
# Pixel is valid only if ALL channels are valid (not NaN)
valid_data_mask = ~np.any(nan_mask_original, axis=-1)  # Shape: (N, H, W)

# Handle NaN values (tiles are already normalized)
has_nan = np.isnan(X).any()
if has_nan:
    n_nan = np.isnan(X).sum()
    pct_nan = (n_nan / X.size) * 100
    print(f"[INFO] Found {n_nan} NaN pixels ({pct_nan:.2f}% of data).")
    # Replace NaN with 0 (missing data sentinel)
    X = np.nan_to_num(X, nan=0.0)
    print(f"[INFO] Replaced NaN with 0 (missing data sentinel).")
else:
    print(f"[INFO] No NaN values found in data.")

print(f"[INFO] Data is already normalized. No additional standardization applied.")

# Stabilize extreme outliers to reduce NaN risk during distributed training
if valid_data_mask.any():
    valid_vals = X[valid_data_mask]
    clip_low = np.percentile(valid_vals, 0.1)
    clip_high = np.percentile(valid_vals, 99.9)
    X = np.clip(X, clip_low, clip_high)
    print(
        f"[INFO] Clipped X to robust range [{clip_low:.4f}, {clip_high:.4f}] for training stability."
    )
else:
    print("[WARNING] No valid pixels found for robust clipping; skipping clip step.")

print(
    f"[INFO] After preprocessing - has NaN: {np.isnan(X).any()}, range: [{np.min(X):.4f}, {np.max(X):.4f}]"
)
print(f"[INFO] Valid data mask will be applied to uncertainty outputs.")

# =====================================================================
# 4. CLASS WEIGHTS + LOSS FUNCTIONS
# =====================================================================

# ----- Class weights -----
flat_y = y.flatten()
counts = np.bincount(flat_y, minlength=4)
bg, c1, c2, c3 = counts

non_bg = c1 + c2 + c3

weights_np = np.zeros(4)
for c, cnt in zip([1, 2, 3], [c1, c2, c3]):
    weights_np[c] = non_bg / (3 * max(cnt, 1))

if CLASS2_WEIGHT_MULTIPLIER != 1.0:
    weights_np[2] *= CLASS2_WEIGHT_MULTIPLIER

weights_np[1:4] /= weights_np[1:4].mean()
class_weights_tf = tf.constant(weights_np, dtype=tf.float32)

print("\nClass weights:", weights_np)


# ----- BASE LOSS (weighted CE) -----
@tf.autograph.experimental.do_not_convert
def masked_sparse_ce_weighted(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.cast(y_true > 0, tf.float32)

    y_shift = tf.where(mask > 0, y_true - 1, 0)
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_shift, y_pred)

    w = tf.gather(class_weights_tf, tf.clip_by_value(y_true, 0, 3))
    ce = ce * mask * w
    return tf.reduce_sum(ce) / (tf.reduce_sum(mask * w) + 1e-7)


# ----- Focal-Tversky Loss -----
def tversky(y_true, y_pred, alpha=FT_ALPHA, beta=FT_BETA):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.cast(y_true > 0, tf.float32)

    y_shift = tf.where(mask > 0, y_true - 1, 0)
    onehot = tf.one_hot(y_shift, 3) * tf.expand_dims(mask, -1)

    p1 = onehot
    p0 = y_pred

    tp = tf.reduce_sum(p0 * p1, axis=[1, 2])
    fp = tf.reduce_sum(p0 * (1 - p1), axis=[1, 2])
    fn = tf.reduce_sum((1 - p0) * p1, axis=[1, 2])

    return (tp + 1e-7) / (tp + alpha * fp + beta * fn + 1e-7)


@tf.autograph.experimental.do_not_convert
def focal_tversky_loss(y_true, y_pred):
    TI = tversky(y_true, y_pred)
    return tf.reduce_mean(tf.pow((1 - TI), FT_GAMMA))


# ----- Dice + CE hybrid -----
def dice_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.cast(y_true > 0, tf.float32)

    y_shift = tf.where(mask > 0, y_true - 1, 0)
    onehot = tf.one_hot(y_shift, 3) * tf.expand_dims(mask, -1)

    inter = tf.reduce_sum(onehot * y_pred, axis=[1, 2])
    sums = tf.reduce_sum(onehot + y_pred, axis=[1, 2])
    return tf.reduce_mean((2 * inter + 1e-7) / (sums + 1e-7))


@tf.autograph.experimental.do_not_convert
def dice_ce_loss(y_true, y_pred):
    ce = masked_sparse_ce_weighted(y_true, y_pred)
    dice = dice_coef(y_true, y_pred)
    return 0.5 * ce + 0.5 * (1 - dice)


# ----- MASKED ACCURACY -----
@tf.autograph.experimental.do_not_convert
def masked_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.cast(y_true > 0, tf.float32)

    y_shift = tf.where(mask > 0, y_true - 1, 0)

    # FIX: force argmax to int32
    y_pred_label = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)

    correct = tf.cast(tf.equal(y_shift, y_pred_label), tf.float32)
    correct = correct * mask

    return tf.reduce_sum(correct) / (tf.reduce_sum(mask) + 1e-7)


# ----- FINAL LOSS SELECTOR -----
if not USE_ADVANCED_LOSS:
    LOSS_FUNCTION = masked_sparse_ce_weighted
    print("\n➡ Using BASIC weighted CE loss")
else:
    if ADVANCED_LOSS_TYPE == "focal_tversky":
        LOSS_FUNCTION = focal_tversky_loss
        print("\n➡ Using ADVANCED LOSS: Focal-Tversky")
    else:
        LOSS_FUNCTION = dice_ce_loss
        print("\n➡ Using ADVANCED LOSS: Dice + CE hybrid")

# =====================================================================
# 5. Build U-Net
# =====================================================================


def conv_block(x, f):
    x = Conv2D(f, 3, padding="same", activation="relu")(x)
    if USE_MC_DROPOUT:
        x = Dropout(MC_DROPOUT_RATE)(
            x
        )  # Dropout layer - training mode controlled at call time
    x = Conv2D(f, 3, padding="same", activation="relu")(x)
    if USE_MC_DROPOUT:
        x = Dropout(MC_DROPOUT_RATE)(
            x
        )  # Dropout layer - training mode controlled at call time
    return x


def encoder_block(x, f):
    c = conv_block(x, f)
    return c, MaxPooling2D(2)(c)


def decoder_block(x, skip, f):
    x = UpSampling2D(2)(x)
    return conv_block(Concatenate()([x, skip]), f)


def build_unet_with_embeddings(input_shape, num_classes):
    cont_in = Input(shape=input_shape, dtype=tf.float32, name="cont_in")
    qua_in = Input(shape=input_shape[:2] + (1,), dtype=tf.int32, name="quaternary_in")
    land_in = Input(shape=input_shape[:2] + (1,), dtype=tf.int32, name="landuse_in")

    qua_idx = Lambda(lambda t: tf.squeeze(t, axis=-1), name="quaternary_squeeze")(
        qua_in
    )
    land_idx = Lambda(lambda t: tf.squeeze(t, axis=-1), name="landuse_squeeze")(land_in)

    qua_feat = Embedding(
        input_dim=QUATERNARY_NUM_CLASSES + 1,
        output_dim=QUATERNARY_EMBED_DIM,
        mask_zero=False,
        name="quaternary_embedding",
    )(qua_idx)
    land_feat = Embedding(
        input_dim=LANDUSE_NUM_CLASSES + 1,
        output_dim=LANDUSE_EMBED_DIM,
        mask_zero=False,
        name="landuse_embedding",
    )(land_idx)

    model_input = Concatenate(axis=-1, name="concat_cont_embeddings")(
        [cont_in, qua_feat, land_feat]
    )

    c1, p1 = encoder_block(model_input, 32)
    c2, p2 = encoder_block(p1, 64)
    c3, p3 = encoder_block(p2, 128)
    c4, p4 = encoder_block(p3, 256)

    bn = conv_block(p4, 512)

    d1 = decoder_block(bn, c4, 256)
    d2 = decoder_block(d1, c3, 128)
    d3 = decoder_block(d2, c2, 64)
    d4 = decoder_block(d3, c1, 32)

    out = Conv2D(num_classes, 1, activation="softmax")(d4)
    return Model(inputs=[cont_in, qua_in, land_in], outputs=out)


# =====================================================================
# 6. F1 callback
# =====================================================================


class F1MetricsCallback(Callback):
    def __init__(self, Xv_cont, Xv_qua, Xv_land, yv, out_txt, prefix=""):
        super().__init__()
        self.Xv_cont = Xv_cont
        self.Xv_qua = Xv_qua
        self.Xv_land = Xv_land
        self.yv = yv
        self.out = out_txt
        self.prefix = prefix
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        y_prob = self.model.predict(
            {
                "cont_in": self.Xv_cont,
                "quaternary_in": self.Xv_qua,
                "landuse_in": self.Xv_land,
            },
            batch_size=2,
            verbose=0,
        )
        y_pred = np.argmax(y_prob, axis=-1).astype(np.int32) + 1

        mask = self.yv > 0
        y_true = self.yv[mask]
        y_pred = y_pred[mask]

        eps = 1e-7
        f1s = []

        ep_dict = {}
        for cls in [1, 2, 3]:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))

            prec = tp / (tp + fp + eps)
            rec = tp / (tp + fn + eps)
            f1 = 2 * prec * rec / (prec + rec + eps)
            f1s.append(f1)

            ep_dict[f"f1_{cls}"] = f1

        ep_dict["macro_f1"] = np.mean(f1s)
        logs["val_macro_f1"] = ep_dict["macro_f1"]

        pred_counts = np.array(
            [np.sum(y_pred == cls) for cls in [1, 2, 3]], dtype=np.int64
        )
        true_counts = np.array(
            [np.sum(y_true == cls) for cls in [1, 2, 3]], dtype=np.int64
        )
        pred_total = max(np.sum(pred_counts), 1)
        true_total = max(np.sum(true_counts), 1)
        pred_pct = (pred_counts / pred_total) * 100.0
        true_pct = (true_counts / true_total) * 100.0

        print(
            f"\n[{self.prefix} Epoch {epoch + 1}] "
            f"macro_F1={ep_dict['macro_f1']:.4f}  "
            f"F1_1={ep_dict['f1_1']:.3f}  "
            f"F1_2={ep_dict['f1_2']:.3f}  "
            f"F1_3={ep_dict['f1_3']:.3f} | "
            f"Pred%[1,2,3]=[{pred_pct[0]:.1f},{pred_pct[1]:.1f},{pred_pct[2]:.1f}] "
            f"True%[1,2,3]=[{true_pct[0]:.1f},{true_pct[1]:.1f},{true_pct[2]:.1f}]",
            flush=True,
        )

        self.history.append(ep_dict)

    def on_train_end(self, logs=None):
        with open(self.out, "w") as f:
            for i, h in enumerate(self.history):
                f.write(f"Epoch {i + 1}: {h}\n")
        print(f"Saved F1 log to {self.out}")


# =====================================================================
# 6. MC DROPOUT UNCERTAINTY FUNCTION
# =====================================================================


def compute_mc_uncertainty(
    model,
    X_cont,
    X_qua,
    X_land,
    valid_mask=None,
    mc_iterations=MC_ITERATIONS,
    mc_batch_size=MC_BATCH_SIZE,
):
    """
    Compute pixel-wise uncertainty using MC Dropout.
    Explicitly enables all Dropout layers for MC sampling.

    Args:
        model: Trained model with MC Dropout
        X_cont: Continuous input data (N, H, W, C)
        X_qua: Quaternary categorical IDs (N, H, W, 1)
        X_land: Landuse categorical IDs (N, H, W, 1)
        valid_mask: Boolean mask (N, H, W) - True where data is valid, False for NaN regions
        mc_iterations: Number of forward passes

    Returns:
        mean_pred: Mean prediction (N, H, W, num_classes)
        std_pred: Standard deviation (N, H, W, num_classes)
        entropy: Predictive entropy (N, H, W) - NaN in masked regions
    """
    print("  [DEBUG] Enabling MC Dropout for inference...")
    print(f"  [DEBUG] Input X_cont shape: {X_cont.shape}, dtype: {X_cont.dtype}")
    print(f"  [DEBUG] Input X_cont range: [{np.min(X_cont):.4f}, {np.max(X_cont):.4f}]")
    print(f"  [DEBUG] Input X_cont has NaN: {np.isnan(X_cont).any()}")

    predictions = []
    n_samples = X_cont.shape[0]
    print(f"  [DEBUG] MC batch size: {mc_batch_size}, samples: {n_samples}")

    for i in tqdm(range(mc_iterations), desc="MC Dropout passes", unit="pass"):
        # Use training=True to keep dropout active; run in mini-batches to avoid OOM
        pass_predictions = []
        for start_idx in range(0, n_samples, mc_batch_size):
            end_idx = min(start_idx + mc_batch_size, n_samples)
            X_cont_batch = tf.convert_to_tensor(
                X_cont[start_idx:end_idx], dtype=tf.float32
            )
            X_qua_batch = tf.convert_to_tensor(X_qua[start_idx:end_idx], dtype=tf.int32)
            X_land_batch = tf.convert_to_tensor(
                X_land[start_idx:end_idx], dtype=tf.int32
            )
            batch_pred = model(
                {
                    "cont_in": X_cont_batch,
                    "quaternary_in": X_qua_batch,
                    "landuse_in": X_land_batch,
                },
                training=True,
            ).numpy()
            pass_predictions.append(batch_pred)

        pred = np.concatenate(pass_predictions, axis=0)

        # Check first prediction
        if i == 0:
            print(
                f"  [DEBUG] First prediction shape: {pred.shape}, dtype: {pred.dtype}"
            )
            print(
                f"  [DEBUG] First prediction range: [{np.min(pred):.4f}, {np.max(pred):.4f}]"
            )
            print(f"  [DEBUG] First prediction has NaN: {np.isnan(pred).any()}")

        predictions.append(pred)

    predictions = np.array(predictions)  # (mc_iterations, N, H, W, num_classes)

    # Diagnostic: check if dropout is having an effect
    pred_variance = np.var(predictions, axis=0)
    mean_variance = np.mean(pred_variance)
    print(f"  MC prediction variance (mean): {mean_variance:.6f}")

    # Check first few predictions to see if they differ
    if mc_iterations >= 2:
        pred_diff = np.abs(predictions[0] - predictions[1])
        max_diff = np.max(pred_diff)
        print(f"  Max diff between first two MC passes: {max_diff:.6f}")

    if mean_variance < 1e-8:
        print(
            "  [WARNING] Very low variance across MC passes — dropout may not be active!"
        )

    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)

    # Entropy: -sum(p * log(p)) with numerical stability
    # Clamp probabilities to avoid log(0) issues
    mean_pred_clamped = np.clip(mean_pred, 1e-10, 1.0 - 1e-10)

    # Normalize to ensure sum=1 across classes (in case of numerical drift)
    mean_pred_norm = mean_pred_clamped / np.sum(
        mean_pred_clamped, axis=-1, keepdims=True
    )

    # Compute entropy with stable log
    entropy = -np.sum(mean_pred_norm * np.log(mean_pred_norm), axis=-1)

    # Apply valid data mask to entropy (set NaN regions to NaN)
    if valid_mask is not None:
        entropy_masked = np.where(valid_mask, entropy, np.nan)
        n_masked = (~valid_mask).sum()
        n_total = valid_mask.size
        print(
            f"  Applied mask: {n_masked}/{n_total} pixels ({n_masked / n_total * 100:.1f}%) marked as no-data (NaN)"
        )
    else:
        entropy_masked = entropy
        print("  [INFO] No mask applied - computing uncertainty for all pixels")

    # Diagnostic: check entropy range (only valid pixels)
    valid_entropy = entropy_masked[~np.isnan(entropy_masked)]
    if len(valid_entropy) > 0:
        print(
            f"  Entropy range (valid pixels): [{np.min(valid_entropy):.4f}, {np.max(valid_entropy):.4f}]"
        )
    else:
        print("  [WARNING] All entropy values are NaN!")

    return mean_pred, std_pred, entropy_masked

    return mean_pred, std_pred, entropy


def save_uncertainty_maps(mean_pred, std_pred, entropy, fold_idx, data_type="cv"):
    """Save uncertainty maps as numpy files."""
    out_dir = os.path.join(QC_PLOTS, f"uncertainty_{data_type}_fold{fold_idx}")
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "mean_pred.npy"), mean_pred)
    np.save(os.path.join(out_dir, "std_pred.npy"), std_pred)
    np.save(os.path.join(out_dir, "entropy.npy"), entropy)

    # Calculate statistics for valid (non-NaN) pixels
    total_pixels = entropy.size
    nan_pixels = np.isnan(entropy).sum()
    valid_pixels = total_pixels - nan_pixels
    pct_valid = (valid_pixels / total_pixels) * 100
    pct_masked = (nan_pixels / total_pixels) * 100

    # Plot mean entropy per sample
    mean_entropy_per_sample = np.mean(entropy, axis=(1, 2))

    # Remove NaN values for histogram
    valid_entropy = mean_entropy_per_sample[~np.isnan(mean_entropy_per_sample)]

    if len(valid_entropy) > 0:
        plt.figure(figsize=(10, 5))
        plt.hist(valid_entropy, bins=30, alpha=0.7, edgecolor="black")
        plt.xlabel("Mean Entropy per Sample")
        plt.ylabel("Frequency")
        plt.title(
            f"Predictive Uncertainty Distribution ({data_type.upper()} Fold {fold_idx})"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "entropy_histogram.png"), dpi=150)
        plt.close()

        print(f"Saved uncertainty maps to {out_dir}")
        print(f"  Valid pixels: {valid_pixels}/{total_pixels} ({pct_valid:.1f}%)")
        print(f"  Masked (no-data) pixels: {nan_pixels} ({pct_masked:.1f}%)")
        print(f"  Mean entropy (valid pixels): {np.mean(valid_entropy):.4f}")
        print(f"  Max entropy: {np.max(valid_entropy):.4f}")
        print(f"  Min entropy: {np.min(valid_entropy):.4f}")
    else:
        print(
            f"[WARNING] No valid entropy values found for {data_type} fold {fold_idx}"
        )
        print(f"  All {total_pixels} pixels are masked (no-data)")
        print(f"Saved uncertainty maps to {out_dir} (all NaN - check data mask)")


# =====================================================================
# CREATE tf.data.Dataset for efficient GPU data streaming
# =====================================================================


def create_dataset(
    X_cont,
    X_qua,
    X_land,
    y,
    batch_size=GLOBAL_BATCH_SIZE,
    shuffle=True,
    drop_remainder=True,
):
    """Create streaming tf.data.Dataset to avoid huge tensor copies to GPU."""
    x_cont_shape = tuple(X_cont.shape[1:])
    x_qua_shape = tuple(X_qua.shape[1:])
    x_land_shape = tuple(X_land.shape[1:])
    y_shape = tuple(y.shape[1:])

    def sample_generator():
        indices = np.arange(len(X_cont))
        if shuffle:
            np.random.shuffle(indices)
        for idx in indices:
            yield (
                {
                    "cont_in": X_cont[idx].astype(np.float32, copy=False),
                    "quaternary_in": X_qua[idx].astype(np.int32, copy=False),
                    "landuse_in": X_land[idx].astype(np.int32, copy=False),
                },
                y[idx].astype(np.float32, copy=False),
            )

    dataset = tf.data.Dataset.from_generator(
        sample_generator,
        output_signature=(
            {
                "cont_in": tf.TensorSpec(shape=x_cont_shape, dtype=tf.float32),
                "quaternary_in": tf.TensorSpec(shape=x_qua_shape, dtype=tf.int32),
                "landuse_in": tf.TensorSpec(shape=x_land_shape, dtype=tf.int32),
            },
            tf.TensorSpec(shape=y_shape, dtype=tf.float32),
        ),
    )

    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Auto-tune prefetching

    return dataset


# =====================================================================
# 7. SPATIAL CV  (with KFold shuffle)
# =====================================================================

if DO_SPATIAL_CV:
    print("\n=== Spatial CV ===")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        print(f"\n--- Fold {fold}/{N_FOLDS} ---")

        Xtr_cont, Xva_cont = X[tr_idx], X[va_idx]
        Xtr_qua, Xva_qua = quaternary[tr_idx], quaternary[va_idx]
        Xtr_land, Xva_land = landuse[tr_idx], landuse[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        # Make copies to prevent any in-place modification during training
        Xva_cont_clean = Xva_cont.copy()
        Xva_qua_clean = Xva_qua.copy()
        Xva_land_clean = Xva_land.copy()

        # Debug: check for NaN before training
        print(
            f"  [DEBUG CV FOLD {fold}] Xva after split - has NaN: {np.isnan(Xva_cont).any()}, range: [{np.nanmin(Xva_cont):.4f}, {np.nanmax(Xva_cont):.4f}]"
        )

        with strategy.scope():
            model = build_unet_with_embeddings(X.shape[1:], num_classes)
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=LEARNING_RATE, clipnorm=1.0
            )
            model.compile(
                optimizer=optimizer, loss=LOSS_FUNCTION, metrics=[masked_accuracy]
            )

        ckpt = ModelCheckpoint(
            os.path.join(MODEL_OUT, f"unet_cv_fold{fold}.keras"),
            monitor="val_loss",
            save_best_only=True,
        )

        f1_log = os.path.join(QC_PLOTS, f"cv_fold{fold}_f1.txt")
        f1cb = F1MetricsCallback(
            Xva_cont, Xva_qua, Xva_land, yva, f1_log, prefix=f"CV Fold {fold}"
        )

        # Create tf.data.Dataset for efficient GPU streaming
        train_dataset = create_dataset(
            Xtr_cont,
            Xtr_qua,
            Xtr_land,
            ytr,
            batch_size=GLOBAL_BATCH_SIZE,
            shuffle=True,
            drop_remainder=True,
        )
        val_dataset = create_dataset(
            Xva_cont,
            Xva_qua,
            Xva_land,
            yva,
            batch_size=GLOBAL_BATCH_SIZE,
            shuffle=False,
            drop_remainder=True,
        )
        train_steps = max(1, len(Xtr_cont) // GLOBAL_BATCH_SIZE)
        val_steps = max(1, len(Xva_cont) // GLOBAL_BATCH_SIZE)

        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=CV_EPOCHS,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            callbacks=[ckpt, f1cb, TerminateOnNaN()],
            verbose=1,
        )

        # Debug: check for NaN after training
        print(
            f"  [DEBUG CV FOLD {fold}] After fit - Xva has NaN: {np.isnan(Xva_cont).any()}, Xva_clean has NaN: {np.isnan(Xva_cont_clean).any()}"
        )

        # Compute uncertainty on validation set
        if USE_MC_DROPOUT:
            print(f"\nComputing MC Dropout uncertainty (Fold {fold})...")
            # Get mask for validation samples
            va_mask = valid_data_mask[va_idx]
            mean_pred, std_pred, entropy = compute_mc_uncertainty(
                model,
                Xva_cont_clean,
                Xva_qua_clean,
                Xva_land_clean,
                valid_mask=va_mask,
                mc_iterations=MC_ITERATIONS,
            )
            save_uncertainty_maps(
                mean_pred, std_pred, entropy, fold_idx=fold, data_type="cv"
            )

# =====================================================================
# 8. MAIN TRAINING  (Option A → uses selected loss)
# =====================================================================

if DO_MAIN_TRAIN:
    print("\n=== Main Training ===")

    Xtr_cont, Xva_cont, Xtr_qua, Xva_qua, Xtr_land, Xva_land, ytr, yva = (
        train_test_split(X, quaternary, landuse, y, test_size=0.2, random_state=42)
    )

    # Make copies to prevent any in-place modification during training
    Xva_cont_clean = Xva_cont.copy()
    Xva_qua_clean = Xva_qua.copy()
    Xva_land_clean = Xva_land.copy()
    print(
        f"  [DEBUG MAIN] Xva after split - has NaN: {np.isnan(Xva_cont).any()}, range: [{np.nanmin(Xva_cont):.4f}, {np.nanmax(Xva_cont):.4f}]"
    )

    with strategy.scope():
        model = build_unet_with_embeddings(X.shape[1:], num_classes)
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
        model.compile(
            optimizer=optimizer, loss=LOSS_FUNCTION, metrics=[masked_accuracy]
        )

    ckpt = ModelCheckpoint(
        os.path.join(MODEL_OUT, "unet_best_model.keras"),
        monitor="val_loss",
        save_best_only=True,
    )

    f1_main_log = os.path.join(QC_PLOTS, "main_train_f1.txt")
    f1cb = F1MetricsCallback(
        Xva_cont, Xva_qua, Xva_land, yva, f1_main_log, prefix="MAIN"
    )

    # Create tf.data.Dataset for efficient GPU streaming
    train_dataset = create_dataset(
        Xtr_cont,
        Xtr_qua,
        Xtr_land,
        ytr,
        batch_size=GLOBAL_BATCH_SIZE,
        shuffle=True,
        drop_remainder=True,
    )
    val_dataset = create_dataset(
        Xva_cont,
        Xva_qua,
        Xva_land,
        yva,
        batch_size=GLOBAL_BATCH_SIZE,
        shuffle=False,
        drop_remainder=True,
    )
    train_steps = max(1, len(Xtr_cont) // GLOBAL_BATCH_SIZE)
    val_steps = max(1, len(Xva_cont) // GLOBAL_BATCH_SIZE)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=MAIN_EPOCHS,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=[ckpt, f1cb, TerminateOnNaN()],
        verbose=1,
    )

    # Plot QC training curves
    hist = history.history
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(hist["loss"], label="loss")
    ax[0].plot(hist["val_loss"], label="val_loss")
    ax[0].legend()
    ax[0].set_title("Loss")

    ax[1].plot(hist["masked_accuracy"], label="masked_acc")
    ax[1].plot(hist["val_masked_accuracy"], label="val_masked_acc")
    ax[1].legend()
    ax[1].set_title("Masked Accuracy")

    out_png = os.path.join(QC_PLOTS, "training_curves.png")
    plt.savefig(out_png, dpi=150)
    plt.close()

    print("\n✔ Saved:", out_png)
    print("✔ F1 log:", f1_main_log)

    # Compute uncertainty on validation set (main training)
    if USE_MC_DROPOUT:
        print("\nComputing MC Dropout uncertainty (Main Training)...")
        print(
            f"  [DEBUG MAIN] Before MC - Xva has NaN: {np.isnan(Xva_cont).any()}, Xva_clean has NaN: {np.isnan(Xva_cont_clean).any()}"
        )
        # Get indices for validation samples
        _, va_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
        va_mask_main = valid_data_mask[va_idx]
        mean_pred, std_pred, entropy = compute_mc_uncertainty(
            model,
            Xva_cont_clean,
            Xva_qua_clean,
            Xva_land_clean,
            valid_mask=va_mask_main,
            mc_iterations=MC_ITERATIONS,
        )
        save_uncertainty_maps(
            mean_pred, std_pred, entropy, fold_idx=0, data_type="main"
        )

    print("\n✓ Training completed successfully!")
    print("✓ Check the saved plots and F1 logs for results.")
