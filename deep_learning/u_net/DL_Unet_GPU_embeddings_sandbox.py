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

TILE_DIR = r"/home/acosta_pedro/Pedro_test"
MODEL_OUT = r"/home/acosta_pedro/outputs/Models"
QC_PLOTS = r"/home/acosta_pedro/outputs/QC_plots"
CHANNEL_MAP_PATH = r"/home/acosta_pedro/Pedro_test/tile_meta_data/channel_map.csv"
TILE_METADATA_PATH = r"/home/acosta_pedro/Pedro_test/tile_metadata/tile_metadata.csv"

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

# ...existing code...
