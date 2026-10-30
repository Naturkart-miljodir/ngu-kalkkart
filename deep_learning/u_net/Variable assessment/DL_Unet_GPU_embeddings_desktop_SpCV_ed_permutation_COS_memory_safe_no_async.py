# -*- coding: utf-8 -*-
"""
U-Net training for Kalk project tiles - GPU OPTIMIZED VERSION
- Masked loss (background ignored)
- Class-weighted loss for classes 1–3
- Optional ADVANCED LOSSES:
      • Focal-Tversky
      • Dice + CE hybrid
- TRUE SPATIAL CV (block-based)
- Per-epoch macro-F1 (background ignored)
- MC DROPOUT UNCERTAINTY QUANTIFICATION
- QC plots + metric logs
- GPU configuration with memory growth
- CHANNEL OCCLUSION SENSITIVITY (COS) for fast predictor importance
"""

import os
# Disabled cuda_malloc_async for TF 2.6 / Windows stability
if "TF_GPU_ALLOCATOR" in os.environ:
    del os.environ["TF_GPU_ALLOCATOR"]
import sys
import glob
import csv
import importlib
import gc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

try:
    tqdm = importlib.import_module("tqdm.auto").tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

from sklearn.model_selection import train_test_split
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
from tensorflow.keras import backend as K

# Add zlib path for Windows (fixes zlibwapi.dll error)
if sys.platform == "win32":
    conda_env = os.path.dirname(os.path.dirname(sys.executable))
    zlib_path = os.path.join(conda_env, "Library", "bin")
    if os.path.exists(zlib_path):
        if zlib_path not in os.environ["PATH"]:
            os.environ["PATH"] = zlib_path + os.pathsep + os.environ["PATH"]
        try:
            os.add_dll_directory(zlib_path)
        except AttributeError:
            pass

#######################################################################
# === HOST-SPECIFIC CODE: GPU CONFIGURATION (MACHINE/ENVIRONMENT) === #
#######################################################################
print("\n" + "=" * 70)
print("GPU CONFIGURATION")
print("=" * 70)

print(f"TensorFlow version: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

gpus = tf.config.list_physical_devices("GPU")
print(f"Number of GPUs available: {len(gpus)}")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Enabled memory growth for: {gpu.name}")

        logical_gpus = tf.config.list_logical_devices("GPU")
        print(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")

        if tf.test.gpu_device_name():
            print(f"✓ Default GPU device: {tf.test.gpu_device_name()}")
        else:
            print("⚠ GPU detected but not set as default device")

    except RuntimeError as e:
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
# 0. USER SWITCHES & DEFAULT ADVANCED LOSS PARAMETERS
# =====================================================================

DO_SPATIAL_CV = True
N_FOLDS = 3
DO_MAIN_TRAIN = False

# === PATCHED HYPERPARAMETERS ===
CV_EPOCHS = 25
LEARNING_RATE = 1e-4
CLASS2_WEIGHT_MULTIPLIER = 1.8

# TRUE SPATIAL CV SETTINGS
SPATIAL_BLOCK_SIZE_KM = 20
SPATIAL_BUFFER_BLOCKS = 0
SPATIAL_RANDOM_SEED = 42

# UNCERTAINTY QUANTIFICATION

# === PATCHED HYPERPARAMETERS ===
USE_MC_DROPOUT = True
MC_DROPOUT_RATE = 0.3
MC_ITERATIONS = 20
MC_BATCH_SIZE = 1

# CATEGORICAL EMBEDDINGS SETTINGS
USE_CATEGORICAL_EMBEDDINGS = True
QUATERNARY_NUM_CLASSES = 22
LANDUSE_NUM_CLASSES = 33
QUATERNARY_EMBED_DIM = 5
LANDUSE_EMBED_DIM = 6
QUATERNARY_PREDICTOR_NAME = "quaternary_forenkletk_cog"
LANDUSE_PREDICTOR_NAME = "landuse_code_18_cog"
REMOVE_EMBED_CHANNELS_FROM_CONTINUOUS = True

# Channel exclusion toggle (name-based using channel_map.csv)
CHANNELS_EXCL = False

# Optional area filter (tile-level selection using tile_metadata extents)
USE_AREA_FILTER = True
AREA_POLYGON_PATH = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\MPM_MidNorge\Mask\Shape\NordTrond_pol_mask.shp"
AREA_POLYGON_LAYER = None
AREA_FILTER_MODE = "intersects"  # "intersects" or "within"
TILE_METADATA_CRS = "EPSG:25833"

# Optional feature ablation (for diagnostics)
DROP_DERIVED_CHANNELS = False
DERIVED_CHANNEL_IDX = [11, 12, 13, 14, 15]


# === PATCHED HYPERPARAMETERS ===
USE_ADVANCED_LOSS = True
ADVANCED_LOSS_TYPE = "dice_ce"   # options: "focal_tversky", "dice_ce"

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
print("USE_AREA_FILTER:", USE_AREA_FILTER)
print("AREA_POLYGON_PATH:", AREA_POLYGON_PATH)
print("AREA_POLYGON_LAYER:", AREA_POLYGON_LAYER)
print("AREA_FILTER_MODE:", AREA_FILTER_MODE)
print("TILE_METADATA_CRS:", TILE_METADATA_CRS)
print("DROP_DERIVED_CHANNELS:", DROP_DERIVED_CHANNELS)
print("DERIVED_CHANNEL_IDX:", DERIVED_CHANNEL_IDX)
print("LEARNING_RATE:", LEARNING_RATE)
print("CV_EPOCHS:", CV_EPOCHS)
print("CLASS2_WEIGHT_MULTIPLIER:", CLASS2_WEIGHT_MULTIPLIER)
print("SPATIAL_BLOCK_SIZE_KM:", SPATIAL_BLOCK_SIZE_KM)
print("SPATIAL_BUFFER_BLOCKS:", SPATIAL_BUFFER_BLOCKS)
print()


#######################################################################
# === INPUT/OUTPUT DIRECTORIES (DO NOT MODIFY UNLESS NECESSARY) === #
#######################################################################

TILE_DIR = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\DL_AE_chips"
MODEL_OUT = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\Models"
QC_PLOTS = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\QC_plots"
PERMUTATION_OUT = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\Permutation_test"

# PERMUTATION TEST SETTINGS
DO_PERMUTATION_TEST = True
PERMUTATION_N_REPEATS = 1          # Increase to 3 in round 2 for more stable estimates
PERMUTATION_RANDOM_SEED = 42
PERMUTATION_GROUPS_TO_RUN = None   # None -> run all groups found in Perm_test_group

# CHANNEL OCCLUSION SENSITIVITY (COS) SETTINGS
DO_CHANNEL_OCCLUSION_TEST = True
COS_MODE = "channel"               # "channel" or "group"
COS_GROUPS_TO_RUN = ["Alpha_earth"]  # e.g. ["Alpha_earth"], ["Topography"], or None for all groups
COS_FILL_VALUE = 0.0
COS_BATCH_SIZE = 1
COS_GROUP_SUBGROUP_SIZE = 4
F1_PRED_BATCH_SIZE = 1

SPATIAL_METADATA_DIR_CANDIDATES = [
    r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\DL_AE_chips\tile_metadata",
    r"E:\Test\National_test\DL_chips_spatial_location",
]


def resolve_spatial_metadata_dir(candidates):
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


SPATIAL_METADATA_DIR = resolve_spatial_metadata_dir(SPATIAL_METADATA_DIR_CANDIDATES)
CHANNEL_MAP_PATH = os.path.join(SPATIAL_METADATA_DIR, "channel_map.csv")
TILE_METADATA_PATH = os.path.join(SPATIAL_METADATA_DIR, "tile_metadata.csv")

X_dir = os.path.join(TILE_DIR, "X")
y_dir = os.path.join(TILE_DIR, "y")
qua_dir = os.path.join(TILE_DIR, "quaternary_soil")   # legacy fallback
land_dir = os.path.join(TILE_DIR, "landuse_code18")   # legacy fallback

os.makedirs(MODEL_OUT, exist_ok=True)
os.makedirs(QC_PLOTS, exist_ok=True)
os.makedirs(PERMUTATION_OUT, exist_ok=True)

print(f"\n=== DATA PATHS ===")
print(f"Input tiles: {TILE_DIR}")
print(f"Models output: {MODEL_OUT}")
print(f"QC plots: {QC_PLOTS}")
print(f"Permutation output: {PERMUTATION_OUT}")
print(f"DO_PERMUTATION_TEST: {DO_PERMUTATION_TEST}")
print(f"PERMUTATION_N_REPEATS: {PERMUTATION_N_REPEATS}")
print(f"PERMUTATION_GROUPS_TO_RUN: {PERMUTATION_GROUPS_TO_RUN}")
print(f"DO_CHANNEL_OCCLUSION_TEST: {DO_CHANNEL_OCCLUSION_TEST}")
print(f"COS_MODE: {COS_MODE}")
print(f"COS_GROUPS_TO_RUN: {COS_GROUPS_TO_RUN}")
print(f"COS_FILL_VALUE: {COS_FILL_VALUE}")
print(f"COS_BATCH_SIZE: {COS_BATCH_SIZE}")
print(f"COS_GROUP_SUBGROUP_SIZE: {COS_GROUP_SUBGROUP_SIZE}")
print(f"F1_PRED_BATCH_SIZE: {F1_PRED_BATCH_SIZE}")
print(f"Spatial metadata dir: {SPATIAL_METADATA_DIR}")
print(f"Channel map: {CHANNEL_MAP_PATH}")
print(f"Tile metadata: {TILE_METADATA_PATH}")


#######################################################################
# === HELPER FUNCTIONS === #
#######################################################################

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
        raise ValueError("CHANNELS_EXCL must be False, a string, or a list/tuple/set of strings")

    parsed = []
    for name in raw_names:
        if not isinstance(name, str):
            raise ValueError("CHANNELS_EXCL entries must be strings")
        key = name.strip().lower()
        if key:
            parsed.append(key)

    return parsed


def tile_id_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]


def filter_paths_by_tile_ids(paths, selected_tile_ids):
    selected = set(selected_tile_ids)
    return [p for p in paths if tile_id_from_path(p) in selected]


def build_tile_path_dict(paths):
    out = {}
    for p in paths:
        out[tile_id_from_path(p)] = p
    return out


def align_Xy_paths(X_paths, y_paths):
    """
    Align X and y paths by tile_id.
    """
    X_dict = build_tile_path_dict(X_paths)
    y_dict = build_tile_path_dict(y_paths)

    common_ids = sorted(set(X_dict).intersection(y_dict))

    if len(common_ids) == 0:
        raise RuntimeError("No matching tile_ids found between X and y.")

    missing_in_y = sorted(set(X_dict) - set(y_dict))
    missing_in_X = sorted(set(y_dict) - set(X_dict))

    if missing_in_y:
        print(f"[WARNING] {len(missing_in_y)} X tiles have no matching y tile.")
    if missing_in_X:
        print(f"[WARNING] {len(missing_in_X)} y tiles have no matching X tile.")

    X_aligned = [X_dict[tid] for tid in common_ids]
    y_aligned = [y_dict[tid] for tid in common_ids]

    for xp, yp, tid in zip(X_aligned, y_aligned, common_ids):
        if tile_id_from_path(xp) != tid or tile_id_from_path(yp) != tid:
            raise RuntimeError(f"Alignment error for tile_id={tid}")

    return X_aligned, y_aligned, common_ids


def load_tile_ids_from_polygon_filter(
    tile_metadata_path,
    polygon_path,
    polygon_layer=None,
    mode="intersects",
    tile_metadata_crs="EPSG:25833",
):
    if mode not in {"intersects", "within"}:
        raise ValueError("AREA_FILTER_MODE must be 'intersects' or 'within'")

    if not os.path.exists(tile_metadata_path):
        raise FileNotFoundError(f"Tile metadata not found: {tile_metadata_path}")

    if not polygon_path or not os.path.exists(polygon_path):
        raise FileNotFoundError(f"Area polygon not found: {polygon_path}")

    import geopandas as gpd
    from shapely.geometry import box

    with open(tile_metadata_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"Tile metadata is empty: {tile_metadata_path}")

    required_cols = {"tile_id", "xmin", "ymin", "xmax", "ymax"}
    missing_cols = required_cols.difference(rows[0].keys())
    if missing_cols:
        raise RuntimeError(f"Tile metadata missing required columns: {sorted(missing_cols)}")

    tile_ids = []
    geoms = []
    for row in rows:
        tile_ids.append(row["tile_id"])
        geoms.append(
            box(
                float(row["xmin"]),
                float(row["ymin"]),
                float(row["xmax"]),
                float(row["ymax"]),
            )
        )

    tiles_gdf = gpd.GeoDataFrame({"tile_id": tile_ids}, geometry=geoms, crs=tile_metadata_crs)

    if polygon_layer:
        area_gdf = gpd.read_file(polygon_path, layer=polygon_layer)
    else:
        area_gdf = gpd.read_file(polygon_path)

    if area_gdf.empty:
        raise RuntimeError(f"Area polygon file has no geometries: {polygon_path}")

    if hasattr(area_gdf.geometry, "union_all"):
        area_union = area_gdf.geometry.union_all()
    else:
        area_union = area_gdf.geometry.unary_union

    if area_gdf.crs and tiles_gdf.crs and area_gdf.crs != tiles_gdf.crs:
        area_union = gpd.GeoSeries([area_union], crs=area_gdf.crs).to_crs(tiles_gdf.crs).iloc[0]

    if mode == "within":
        keep_mask = tiles_gdf.within(area_union)
    else:
        keep_mask = tiles_gdf.intersects(area_union)

    selected_tile_ids = set(tiles_gdf.loc[keep_mask, "tile_id"].astype(str).tolist())

    print(
        f"[INFO] Area filter selected {len(selected_tile_ids)} / {len(tiles_gdf)} tiles using mode='{mode}'."
    )

    return selected_tile_ids


def load_tile_metadata_lookup(tile_metadata_path):
    with open(tile_metadata_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"Tile metadata is empty: {tile_metadata_path}")

    required_cols = {"tile_id", "xmin", "ymin", "xmax", "ymax"}
    missing_cols = required_cols.difference(rows[0].keys())
    if missing_cols:
        raise RuntimeError(f"Tile metadata missing required columns: {sorted(missing_cols)}")

    meta = {}
    for row in rows:
        tid = str(row["tile_id"])
        meta[tid] = {
            "xmin": float(row["xmin"]),
            "ymin": float(row["ymin"]),
            "xmax": float(row["xmax"]),
            "ymax": float(row["ymax"]),
        }
    return meta


def assign_spatial_blocks(tile_ids, tile_metadata_path, block_size_km=20):
    """
    Assign each tile to a spatial block using tile centroid.
    """
    meta = load_tile_metadata_lookup(tile_metadata_path)
    block_size_m = block_size_km * 1000.0

    tile_to_block = {}
    block_to_tiles = {}

    for tid in tile_ids:
        if tid not in meta:
            raise RuntimeError(f"Tile {tid} not found in tile_metadata.csv")

        xmin = meta[tid]["xmin"]
        ymin = meta[tid]["ymin"]
        xmax = meta[tid]["xmax"]
        ymax = meta[tid]["ymax"]

        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)

        bx = int(np.floor(cx / block_size_m))
        by = int(np.floor(cy / block_size_m))
        block = (bx, by)

        tile_to_block[tid] = block
        block_to_tiles.setdefault(block, []).append(tid)

    return tile_to_block, block_to_tiles


def assign_blocks_to_folds(block_to_tiles, n_folds=3, random_seed=42):
    """
    Greedy balancing: assign largest blocks first to the currently lightest fold.
    """
    rng = np.random.RandomState(random_seed)
    blocks = list(block_to_tiles.keys())
    block_sizes = {b: len(block_to_tiles[b]) for b in blocks}

    rng.shuffle(blocks)
    blocks = sorted(blocks, key=lambda b: block_sizes[b], reverse=True)

    fold_loads = [0] * n_folds
    block_to_fold = {}

    for b in blocks:
        fold_id = int(np.argmin(fold_loads))
        block_to_fold[b] = fold_id
        fold_loads[fold_id] += block_sizes[b]

    print(f"[INFO] Fold tile counts (before buffering): {fold_loads}")
    return block_to_fold


def neighbor_blocks(block, buffer_blocks=1):
    bx, by = block
    out = set()
    for dx in range(-buffer_blocks, buffer_blocks + 1):
        for dy in range(-buffer_blocks, buffer_blocks + 1):
            out.add((bx + dx, by + dy))
    return out


def make_spatial_fold_indices(tile_ids, tile_to_block, block_to_fold, fold_id, buffer_blocks=0):
    """
    Validation = all tiles in fold_id blocks.
    Training = all tiles not in validation blocks, excluding neighbor buffer if requested.
    """
    val_blocks = {b for b, f in block_to_fold.items() if f == fold_id}

    if buffer_blocks > 0:
        excluded_train_blocks = set()
        for vb in val_blocks:
            excluded_train_blocks.update(neighbor_blocks(vb, buffer_blocks))
    else:
        excluded_train_blocks = set(val_blocks)

    tr_idx = []
    va_idx = []

    for i, tid in enumerate(tile_ids):
        b = tile_to_block[tid]
        if b in val_blocks:
            va_idx.append(i)
        elif b not in excluded_train_blocks:
            tr_idx.append(i)

    return np.array(tr_idx, dtype=np.int32), np.array(va_idx, dtype=np.int32)


def compute_class_weights_from_y(y_train, class2_multiplier=1.0):
    flat_y = y_train.flatten()
    counts = np.bincount(flat_y, minlength=4)
    _, c1, c2, c3 = counts

    non_bg = c1 + c2 + c3

    weights_np = np.zeros(4, dtype=np.float32)
    for c, cnt in zip([1, 2, 3], [c1, c2, c3]):
        weights_np[c] = non_bg / (3.0 * max(cnt, 1))

    if class2_multiplier != 1.0:
        weights_np[2] *= class2_multiplier

    weights_np[1:4] /= np.mean(weights_np[1:4])
    return weights_np



#######################################################################
# === PERMUTATION TEST HELPERS === #
#######################################################################

def compute_numpy_metrics(y_true, y_pred):
    mask = y_true > 0
    y_true_m = y_true[mask]
    y_pred_m = y_pred[mask]

    eps = 1e-7
    f1s = []
    recalls = []
    for cls in [1, 2, 3]:
        tp = np.sum((y_true_m == cls) & (y_pred_m == cls))
        fp = np.sum((y_true_m != cls) & (y_pred_m == cls))
        fn = np.sum((y_true_m == cls) & (y_pred_m != cls))

        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        f1s.append(float(f1))
        recalls.append(float(rec))

    masked_acc = float(np.sum(y_true_m == y_pred_m) / (len(y_true_m) + eps))
    balanced_acc = float(np.mean(recalls))

    return {
        "f1_1": f1s[0],
        "f1_2": f1s[1],
        "f1_3": f1s[2],
        "macro_f1": float(np.mean(f1s)),
        "balanced_accuracy": balanced_acc,
        "masked_accuracy": masked_acc,
    }


def predict_label_map(model, X_cont, X_qua=None, X_land=None, batch_size=2):
    if X_qua is not None and X_land is not None:
        pred_input = {
            "cont_in": X_cont,
            "quaternary_in": X_qua,
            "landuse_in": X_land,
        }
    else:
        pred_input = X_cont

    y_prob = model.predict(pred_input, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(y_prob, axis=-1).astype(np.int32) + 1
    return y_pred


def sanitize_group_name(name):
    return str(name).strip().replace(" ", "_").replace("/", "_")


def build_current_channel_lookup(
    channel_map_rows,
    dropped_channels,
    excluded_channel_idx_orig,
    remove_embed_channels,
    quaternary_channel_idx_orig,
    landuse_channel_idx_orig,
):
    rows_out = []
    for row in channel_map_rows:
        row_copy = dict(row)
        orig_idx = int(row_copy["channel_idx"])
        row_copy["orig_channel_idx"] = orig_idx
        row_copy["current_channel_idx"] = None
        row_copy["in_continuous_branch"] = False

        if orig_idx in dropped_channels:
            rows_out.append(row_copy)
            continue

        current_idx = orig_idx - len([d for d in dropped_channels if d < orig_idx])
        remove_from_cont = False
        if orig_idx in excluded_channel_idx_orig:
            remove_from_cont = True
        if remove_embed_channels and orig_idx in {quaternary_channel_idx_orig, landuse_channel_idx_orig}:
            remove_from_cont = True

        if remove_from_cont:
            rows_out.append(row_copy)
            continue

        removed_before = len([idx for idx in excluded_channel_idx_orig if idx not in dropped_channels and idx < orig_idx])
        if remove_embed_channels:
            if quaternary_channel_idx_orig not in dropped_channels and quaternary_channel_idx_orig < orig_idx:
                removed_before += 1
            if landuse_channel_idx_orig not in dropped_channels and landuse_channel_idx_orig < orig_idx:
                removed_before += 1
            # If predictor is itself an embedding channel, it is excluded above.
            if quaternary_channel_idx_orig < orig_idx and landuse_channel_idx_orig < orig_idx and quaternary_channel_idx_orig == landuse_channel_idx_orig:
                pass

        row_copy["current_channel_idx"] = current_idx - removed_before
        row_copy["in_continuous_branch"] = True
        rows_out.append(row_copy)

    return rows_out


def build_permutation_plan(
    current_channel_rows,
    quaternary_predictor_name,
    landuse_predictor_name,
    groups_to_run=None,
):
    plan = []

    # Continuous-branch groups from channel map rows
    grouped = {}
    for row in current_channel_rows:
        group_name = row.get("Perm_test_group", "")
        if not group_name:
            continue
        if not row.get("in_continuous_branch", False):
            continue
        grouped.setdefault(group_name, []).append(int(row["current_channel_idx"]))

    for group_name, idxs in grouped.items():
        idxs_sorted = sorted(set(idxs))
        if not idxs_sorted:
            continue
        plan.append({
            "group_name": str(group_name),
            "perm_type": "continuous",
            "channel_indices": idxs_sorted,
            "predictor_names": [
                row["predictor_name"] for row in current_channel_rows
                if row.get("in_continuous_branch", False) and row.get("Perm_test_group", "") == group_name
            ],
        })

    # Explicit embedding groups
    plan.append({
        "group_name": "Quaternary",
        "perm_type": "quaternary",
        "channel_indices": [],
        "predictor_names": [quaternary_predictor_name],
    })
    plan.append({
        "group_name": "Land_use",
        "perm_type": "landuse",
        "channel_indices": [],
        "predictor_names": [landuse_predictor_name],
    })

    # Deduplicate by group_name, preferring first occurrence
    dedup = []
    seen = set()
    for item in plan:
        if item["group_name"] in seen:
            continue
        seen.add(item["group_name"])
        dedup.append(item)

    if groups_to_run is not None:
        allowed = {str(g) for g in groups_to_run}
        dedup = [item for item in dedup if item["group_name"] in allowed]

    dedup = sorted(dedup, key=lambda d: d["group_name"].lower())
    return dedup


def apply_tile_permutation(X_cont, X_qua, X_land, perm_indices, perm_item):
    X_cont_perm = X_cont
    X_qua_perm = X_qua
    X_land_perm = X_land

    if perm_item["perm_type"] == "continuous":
        X_cont_perm = X_cont.copy()
        idxs = perm_item["channel_indices"]
        X_cont_perm[..., idxs] = X_cont[perm_indices][..., idxs]
    elif perm_item["perm_type"] == "quaternary":
        if X_qua is None:
            raise RuntimeError("Quaternary permutation requested, but X_qua is None.")
        X_qua_perm = X_qua[perm_indices].copy()
    elif perm_item["perm_type"] == "landuse":
        if X_land is None:
            raise RuntimeError("Land_use permutation requested, but X_land is None.")
        X_land_perm = X_land[perm_indices].copy()
    else:
        raise RuntimeError(f"Unknown permutation type: {perm_item['perm_type']}")

    return X_cont_perm, X_qua_perm, X_land_perm


def run_permutation_test_for_fold(
    model,
    fold_idx,
    Xva,
    yva,
    permutation_plan,
    permutation_out_dir,
    n_repeats=1,
    random_seed=42,
    Xva_qua=None,
    Xva_land=None,
    batch_size=2,
):
    os.makedirs(permutation_out_dir, exist_ok=True)

    baseline_pred = predict_label_map(
        model,
        Xva,
        X_qua=Xva_qua,
        X_land=Xva_land,
        batch_size=batch_size,
    )
    baseline_metrics = compute_numpy_metrics(yva, baseline_pred)

    rows = []
    n_tiles = Xva.shape[0]
    rng = np.random.RandomState(random_seed + int(fold_idx))

    for perm_item in permutation_plan:
        print(f"\n[PERM] Fold {fold_idx} -> {perm_item['group_name']} ({perm_item['perm_type']})")
        repeat_macro_drops = []
        repeat_bal_drops = []
        repeat_masked_drops = []

        for rep in range(n_repeats):
            perm_indices = rng.permutation(n_tiles)
            Xc_perm, Xq_perm, Xl_perm = apply_tile_permutation(
                Xva,
                Xva_qua,
                Xva_land,
                perm_indices,
                perm_item,
            )

            perm_pred = predict_label_map(
                model,
                Xc_perm,
                X_qua=Xq_perm,
                X_land=Xl_perm,
                batch_size=batch_size,
            )
            perm_metrics = compute_numpy_metrics(yva, perm_pred)

            macro_drop = baseline_metrics["macro_f1"] - perm_metrics["macro_f1"]
            bal_drop = baseline_metrics["balanced_accuracy"] - perm_metrics["balanced_accuracy"]
            masked_drop = baseline_metrics["masked_accuracy"] - perm_metrics["masked_accuracy"]

            repeat_macro_drops.append(macro_drop)
            repeat_bal_drops.append(bal_drop)
            repeat_masked_drops.append(masked_drop)

            rows.append({
                "fold": fold_idx,
                "repeat": rep + 1,
                "group_name": perm_item["group_name"],
                "perm_type": perm_item["perm_type"],
                "n_channels": len(perm_item.get("channel_indices", [])),
                "predictor_names": " | ".join(perm_item.get("predictor_names", [])),
                "baseline_macro_f1": baseline_metrics["macro_f1"],
                "permuted_macro_f1": perm_metrics["macro_f1"],
                "macro_f1_drop": macro_drop,
                "baseline_balanced_accuracy": baseline_metrics["balanced_accuracy"],
                "permuted_balanced_accuracy": perm_metrics["balanced_accuracy"],
                "balanced_accuracy_drop": bal_drop,
                "baseline_masked_accuracy": baseline_metrics["masked_accuracy"],
                "permuted_masked_accuracy": perm_metrics["masked_accuracy"],
                "masked_accuracy_drop": masked_drop,
            })

        print(
            f"[PERM] {perm_item['group_name']} -> mean macro_F1_drop={np.mean(repeat_macro_drops):.4f}, "
            f"mean bal_acc_drop={np.mean(repeat_bal_drops):.4f}, "
            f"mean masked_acc_drop={np.mean(repeat_masked_drops):.4f}"
        )

    fold_csv = os.path.join(permutation_out_dir, f"fold{fold_idx}_permutation_results.csv")
    with open(fold_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "fold", "repeat", "group_name", "perm_type", "n_channels", "predictor_names",
            "baseline_macro_f1", "permuted_macro_f1", "macro_f1_drop",
            "baseline_balanced_accuracy", "permuted_balanced_accuracy", "balanced_accuracy_drop",
            "baseline_masked_accuracy", "permuted_masked_accuracy", "masked_accuracy_drop",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return baseline_metrics, rows


def write_permutation_summary(all_rows, summary_csv_path):
    grouped = {}
    for row in all_rows:
        grouped.setdefault(row["group_name"], []).append(row)

    out_rows = []
    for group_name, rows in sorted(grouped.items()):
        macro = np.array([float(r["macro_f1_drop"]) for r in rows], dtype=float)
        bal = np.array([float(r["balanced_accuracy_drop"]) for r in rows], dtype=float)
        masked = np.array([float(r["masked_accuracy_drop"]) for r in rows], dtype=float)
        out_rows.append({
            "group_name": group_name,
            "n_rows": len(rows),
            "mean_macro_f1_drop": float(np.mean(macro)),
            "std_macro_f1_drop": float(np.std(macro)),
            "mean_balanced_accuracy_drop": float(np.mean(bal)),
            "std_balanced_accuracy_drop": float(np.std(bal)),
            "mean_masked_accuracy_drop": float(np.mean(masked)),
            "std_masked_accuracy_drop": float(np.std(masked)),
        })

    os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else [
            "group_name", "n_rows",
            "mean_macro_f1_drop", "std_macro_f1_drop",
            "mean_balanced_accuracy_drop", "std_balanced_accuracy_drop",
            "mean_masked_accuracy_drop", "std_masked_accuracy_drop",
        ])
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)




#######################################################################
# === CHANNEL OCCLUSION SENSITIVITY (COS) HELPERS === #
#######################################################################

def build_channel_occlusion_plan(current_channel_rows, groups_to_run=None, mode="channel", subgroup_size=None):
    if mode not in {"channel", "group"}:
        raise ValueError("COS_MODE must be 'channel' or 'group'")

    allowed = None if groups_to_run is None else {str(g) for g in groups_to_run}
    plan = []

    grouped_rows = {}
    for row in current_channel_rows:
        group_name = row.get("Perm_test_group", "")
        if not group_name:
            continue
        if not row.get("in_continuous_branch", False):
            continue
        if allowed is not None and str(group_name) not in allowed:
            continue
        grouped_rows.setdefault(str(group_name), []).append(row)

    for group_name, rows in grouped_rows.items():
        rows = sorted(rows, key=lambda r: int(r["current_channel_idx"]))
        if mode == "group":
            if subgroup_size is not None and subgroup_size > 0 and len(rows) > subgroup_size:
                for i in range(0, len(rows), subgroup_size):
                    sub_rows = rows[i:i + subgroup_size]
                    first_idx = int(sub_rows[0]["current_channel_idx"])
                    last_idx = int(sub_rows[-1]["current_channel_idx"])
                    plan.append({
                        "test_name": f"{group_name}__subset_{i//subgroup_size + 1:02d}__ch{first_idx:03d}_to_ch{last_idx:03d}",
                        "group_name": group_name,
                        "channel_indices": [int(r["current_channel_idx"]) for r in sub_rows],
                        "predictor_names": [r["predictor_name"] for r in sub_rows],
                        "n_channels": len(sub_rows),
                    })
            else:
                plan.append({
                    "test_name": group_name,
                    "group_name": group_name,
                    "channel_indices": [int(r["current_channel_idx"]) for r in rows],
                    "predictor_names": [r["predictor_name"] for r in rows],
                    "n_channels": len(rows),
                })
        else:
            for row in rows:
                current_idx = int(row["current_channel_idx"])
                predictor_name = str(row["predictor_name"])
                plan.append({
                    "test_name": f"{group_name}__ch{current_idx:03d}__{sanitize_group_name(predictor_name)}",
                    "group_name": group_name,
                    "channel_indices": [current_idx],
                    "predictor_names": [predictor_name],
                    "n_channels": 1,
                })

    plan = sorted(plan, key=lambda d: (d["group_name"].lower(), d["channel_indices"][0]))
    return plan


def run_channel_occlusion_test_for_fold(
    model,
    fold_idx,
    Xva,
    yva,
    occlusion_plan,
    occlusion_out_dir,
    fill_value=0.0,
    batch_size=2,
    Xva_qua=None,
    Xva_land=None,
):
    os.makedirs(occlusion_out_dir, exist_ok=True)

    n_samples = Xva.shape[0]
    batch_size = max(1, int(batch_size))

    baseline_pred = np.empty_like(yva)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        baseline_pred[start:end] = predict_label_map(
            model,
            Xva[start:end],
            X_qua=Xva_qua[start:end] if Xva_qua is not None else None,
            X_land=Xva_land[start:end] if Xva_land is not None else None,
            batch_size=batch_size,
        )
    baseline_metrics = compute_numpy_metrics(yva, baseline_pred)
    del baseline_pred
    gc.collect()

    rows = []

    for item in occlusion_plan:
        idxs = list(item["channel_indices"])
        print(
            f"\n[COS] Fold {fold_idx} -> {item['test_name']} "
            f"(group={item['group_name']}, n_channels={len(idxs)})"
        )

        occ_pred = np.empty_like(yva)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = Xva[start:end].copy()
            X_batch[..., idxs] = fill_value
            occ_pred[start:end] = predict_label_map(
                model,
                X_batch,
                X_qua=Xva_qua[start:end] if Xva_qua is not None else None,
                X_land=Xva_land[start:end] if Xva_land is not None else None,
                batch_size=batch_size,
            )
            del X_batch
            gc.collect()

        occ_metrics = compute_numpy_metrics(yva, occ_pred)
        del occ_pred
        gc.collect()

        macro_drop = baseline_metrics["macro_f1"] - occ_metrics["macro_f1"]
        bal_drop = baseline_metrics["balanced_accuracy"] - occ_metrics["balanced_accuracy"]
        masked_drop = baseline_metrics["masked_accuracy"] - occ_metrics["masked_accuracy"]

        row = {
            "fold": fold_idx,
            "test_name": item["test_name"],
            "group_name": item["group_name"],
            "n_channels": item.get("n_channels", len(idxs)),
            "channel_indices": " | ".join(str(i) for i in idxs),
            "predictor_names": " | ".join(item.get("predictor_names", [])),
            "baseline_macro_f1": baseline_metrics["macro_f1"],
            "occluded_macro_f1": occ_metrics["macro_f1"],
            "macro_f1_drop": macro_drop,
            "baseline_balanced_accuracy": baseline_metrics["balanced_accuracy"],
            "occluded_balanced_accuracy": occ_metrics["balanced_accuracy"],
            "balanced_accuracy_drop": bal_drop,
            "baseline_masked_accuracy": baseline_metrics["masked_accuracy"],
            "occluded_masked_accuracy": occ_metrics["masked_accuracy"],
            "masked_accuracy_drop": masked_drop,
        }
        rows.append(row)

        print(
            f"[COS] {item['test_name']} -> macro_F1_drop={macro_drop:.4f}, "
            f"bal_acc_drop={bal_drop:.4f}, masked_acc_drop={masked_drop:.4f}"
        )

    fold_csv = os.path.join(occlusion_out_dir, f"fold{fold_idx}_channel_occlusion_results.csv")
    with open(fold_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "fold", "test_name", "group_name", "n_channels", "channel_indices", "predictor_names",
            "baseline_macro_f1", "occluded_macro_f1", "macro_f1_drop",
            "baseline_balanced_accuracy", "occluded_balanced_accuracy", "balanced_accuracy_drop",
            "baseline_masked_accuracy", "occluded_masked_accuracy", "masked_accuracy_drop",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return baseline_metrics, rows


def write_channel_occlusion_summary(all_rows, summary_csv_path):
    grouped = {}
    for row in all_rows:
        grouped.setdefault(row["test_name"], []).append(row)

    out_rows = []
    for test_name, rows in sorted(grouped.items()):
        macro = np.array([float(r["macro_f1_drop"]) for r in rows], dtype=float)
        bal = np.array([float(r["balanced_accuracy_drop"]) for r in rows], dtype=float)
        masked = np.array([float(r["masked_accuracy_drop"]) for r in rows], dtype=float)
        out_rows.append({
            "test_name": test_name,
            "group_name": rows[0]["group_name"],
            "n_folds": len(rows),
            "n_channels": rows[0]["n_channels"],
            "channel_indices": rows[0]["channel_indices"],
            "predictor_names": rows[0]["predictor_names"],
            "mean_macro_f1_drop": float(np.mean(macro)),
            "std_macro_f1_drop": float(np.std(macro)),
            "mean_balanced_accuracy_drop": float(np.mean(bal)),
            "std_balanced_accuracy_drop": float(np.std(bal)),
            "mean_masked_accuracy_drop": float(np.mean(masked)),
            "std_masked_accuracy_drop": float(np.std(masked)),
        })

    os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else [
            "test_name", "group_name", "n_folds", "n_channels", "channel_indices", "predictor_names",
            "mean_macro_f1_drop", "std_macro_f1_drop",
            "mean_balanced_accuracy_drop", "std_balanced_accuracy_drop",
            "mean_masked_accuracy_drop", "std_masked_accuracy_drop",
        ])
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

#######################################################################
# === DATA LOADING SECTION === #
#######################################################################

print("\n=== LOADING DATA ===")
X_paths = sorted(glob.glob(os.path.join(X_dir, "*.npy")))
y_paths = sorted(glob.glob(os.path.join(y_dir, "*.npy")))

if USE_AREA_FILTER:
    selected_tile_ids = load_tile_ids_from_polygon_filter(
        tile_metadata_path=TILE_METADATA_PATH,
        polygon_path=AREA_POLYGON_PATH,
        polygon_layer=AREA_POLYGON_LAYER,
        mode=AREA_FILTER_MODE,
        tile_metadata_crs=TILE_METADATA_CRS,
    )
    X_paths = filter_paths_by_tile_ids(X_paths, selected_tile_ids)
    y_paths = filter_paths_by_tile_ids(y_paths, selected_tile_ids)

# IMPORTANT: align by tile_id
X_paths, y_paths, tile_ids = align_Xy_paths(X_paths, y_paths)

print(f"[INFO] After alignment -> matched tiles: {len(tile_ids)}")
print(f"Found {len(X_paths)} X tiles and {len(y_paths)} y tiles")

if len(X_paths) == 0 or len(y_paths) == 0:
    print(f"\n[ERROR] No tiles found!")
    print(f"  X dir: {X_dir}")
    print(f"  y dir: {y_dir}")
    sys.exit(1)

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

    print(f"[INFO] CHANNELS_EXCL resolved to original channel indices: {sorted(excluded_channel_idx_orig)}")

quaternary = np.empty((0,), dtype=np.int32)
landuse = np.empty((0,), dtype=np.int32)

if USE_CATEGORICAL_EMBEDDINGS and not embeddings_from_channel_map:
    print("[WARNING] Channel map not found. Falling back to legacy quaternary/landuse tile folders.")
    qua_paths = sorted(glob.glob(os.path.join(qua_dir, "*.npy")))
    land_paths = sorted(glob.glob(os.path.join(land_dir, "*.npy")))

    if USE_AREA_FILTER:
        qua_paths = filter_paths_by_tile_ids(qua_paths, selected_tile_ids)
        land_paths = filter_paths_by_tile_ids(land_paths, selected_tile_ids)

    qua_dict = build_tile_path_dict(qua_paths)
    land_dict = build_tile_path_dict(land_paths)

    if not all(tid in qua_dict for tid in tile_ids):
        missing = [tid for tid in tile_ids if tid not in qua_dict]
        raise RuntimeError(f"Missing quaternary tiles for {len(missing)} matched tile_ids")
    if not all(tid in land_dict for tid in tile_ids):
        missing = [tid for tid in tile_ids if tid not in land_dict]
        raise RuntimeError(f"Missing landuse tiles for {len(missing)} matched tile_ids")

    qua_paths = [qua_dict[tid] for tid in tile_ids]
    land_paths = [land_dict[tid] for tid in tile_ids]

    print(f"Found {len(qua_paths)} quaternary tiles and {len(land_paths)} landuse tiles")

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


#######################################################################
# === EMBEDDINGS CREATION & CHANNEL PROCESSING === #
#######################################################################

if X.ndim == 4 and X.shape[1] < X.shape[2]:
    print(f"[INFO] Transposing X from {X.shape} (channels_first) to channels_last format...")
    X = np.transpose(X, (0, 2, 3, 1))
    print(f"[INFO] X shape after transpose: {X.shape}")
elif X.ndim == 4 and X.shape[-1] < X.shape[1]:
    print(f"[INFO] Data already in channels_last format: {X.shape}")
else:
    print(f"[WARNING] Unexpected X shape: {X.shape}")

dropped_channels = set()
if DROP_DERIVED_CHANNELS:
    dropped_channels = set(i for i in DERIVED_CHANNEL_IDX if 0 <= i < X.shape[-1])
    keep_idx = [i for i in range(X.shape[-1]) if i not in DERIVED_CHANNEL_IDX]
    print(f"[INFO] Dropping derived channels: {DERIVED_CHANNEL_IDX}")
    print(f"[INFO] Keeping channels: {keep_idx}")
    X = X[..., keep_idx]
    print(f"[INFO] X shape after channel drop: {X.shape}")

if USE_CATEGORICAL_EMBEDDINGS and embeddings_from_channel_map:
    if quaternary_channel_idx_orig < 0 or landuse_channel_idx_orig < 0:
        raise RuntimeError("Invalid channel-map categorical indices. Check predictor names and channel map.")

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
        f"[INFO] Extracting embedding channels from X -> quaternary idx {quaternary_channel_idx}, "
        f"landuse idx {landuse_channel_idx}"
    )

    if not (0 <= quaternary_channel_idx < X.shape[-1]):
        raise RuntimeError(
            f"Quaternary channel index {quaternary_channel_idx} out of bounds for X with {X.shape[-1]} channels"
        )
    if not (0 <= landuse_channel_idx < X.shape[-1]):
        raise RuntimeError(
            f"Landuse channel index {landuse_channel_idx} out of bounds for X with {X.shape[-1]} channels"
        )

    quaternary_arr = np.rint(np.nan_to_num(X[..., quaternary_channel_idx], nan=0.0)).astype(np.int32)
    landuse_arr = np.rint(np.nan_to_num(X[..., landuse_channel_idx], nan=0.0)).astype(np.int32)

    quaternary_arr = np.where(
        (quaternary_arr < 0) | (quaternary_arr > QUATERNARY_NUM_CLASSES),
        0,
        quaternary_arr,
    )
    landuse_arr = np.where(
        (landuse_arr < 0) | (landuse_arr > LANDUSE_NUM_CLASSES),
        0,
        landuse_arr,
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
    print(f"[INFO] Removing channels from continuous branch: {sorted(continuous_remove_idx)}")
    X = X[..., keep_idx]
    print(f"[INFO] X shape after channel removal: {X.shape}")

if USE_CATEGORICAL_EMBEDDINGS:
    if quaternary.size == 0 or landuse.size == 0:
        raise RuntimeError("Categorical embeddings are enabled but categorical tensors were not prepared.")
    quaternary = np.asarray(quaternary, dtype=np.int32)
    landuse = np.asarray(landuse, dtype=np.int32)
    print(f"Shapes -> quaternary: {quaternary.shape}, landuse: {landuse.shape}")

num_classes = 3
n_channels = X.shape[-1]
print("X shape:", X.shape)
print(f"Number of channels: {n_channels}")
print(f"[DEBUG] X after loading - has NaN: {np.isnan(X).any()}, range: [{np.nanmin(X):.4f}, {np.nanmax(X):.4f}]")

# Store original valid mask before NaN replacement
nan_mask_original = np.isnan(X)
valid_data_mask = ~np.any(nan_mask_original, axis=-1)

has_nan = np.isnan(X).any()
if has_nan:
    n_nan = np.isnan(X).sum()
    pct_nan = (n_nan / X.size) * 100
    print(f"[INFO] Found {n_nan} NaN pixels ({pct_nan:.2f}% of data).")
    X = np.nan_to_num(X, nan=0.0)
    print(f"[INFO] Replaced NaN with 0 (missing data sentinel).")
else:
    print(f"[INFO] No NaN values found in data.")

print(f"[INFO] Data is already normalized. No additional standardization applied.")
print(f"[INFO] After preprocessing - has NaN: {np.isnan(X).any()}, range: [{np.min(X):.4f}, {np.max(X):.4f}]")
print(f"[INFO] Valid data mask will be applied to uncertainty outputs.")

# Build current-channel lookup and permutation plan AFTER all preprocessing
current_channel_rows = []
permutation_plan = []
channel_occlusion_plan = []
if channel_map_rows is not None:
    current_channel_rows = build_current_channel_lookup(
        channel_map_rows=channel_map_rows,
        dropped_channels=dropped_channels,
        excluded_channel_idx_orig=excluded_channel_idx_orig,
        remove_embed_channels=(
            USE_CATEGORICAL_EMBEDDINGS
            and embeddings_from_channel_map
            and REMOVE_EMBED_CHANNELS_FROM_CONTINUOUS
        ),
        quaternary_channel_idx_orig=quaternary_channel_idx_orig,
        landuse_channel_idx_orig=landuse_channel_idx_orig,
    )

    permutation_plan = build_permutation_plan(
        current_channel_rows=current_channel_rows,
        quaternary_predictor_name=QUATERNARY_PREDICTOR_NAME,
        landuse_predictor_name=LANDUSE_PREDICTOR_NAME,
        groups_to_run=PERMUTATION_GROUPS_TO_RUN,
    )

    if DO_PERMUTATION_TEST:
        print("\n=== PERMUTATION TEST PLAN ===")
        for item in permutation_plan:
            print(
                f"  {item['group_name']}: type={item['perm_type']}, "
                f"n_channels={len(item.get('channel_indices', []))}"
            )

    if DO_CHANNEL_OCCLUSION_TEST:
        channel_occlusion_plan = build_channel_occlusion_plan(
            current_channel_rows=current_channel_rows,
            groups_to_run=COS_GROUPS_TO_RUN,
            mode=COS_MODE,
            subgroup_size=COS_GROUP_SUBGROUP_SIZE,
        )
        print("\n=== CHANNEL OCCLUSION SENSITIVITY (COS) PLAN ===")
        print(f"  Mode: {COS_MODE}")
        print(f"  Number of tests: {len(channel_occlusion_plan)}")
        preview = channel_occlusion_plan[:15]
        for item in preview:
            print(
                f"  {item['test_name']}: group={item['group_name']}, "
                f"n_channels={item['n_channels']}"
            )
        if len(channel_occlusion_plan) > len(preview):
            print(f"  ... {len(channel_occlusion_plan) - len(preview)} more tests")


#######################################################################
# === LOSS FUNCTIONS & METRICS === #
#######################################################################

@tf.autograph.experimental.do_not_convert
def masked_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.cast(y_true > 0, tf.float32)
    y_shift = tf.where(mask > 0, y_true - 1, 0)
    y_pred_label = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
    correct = tf.cast(tf.equal(y_shift, y_pred_label), tf.float32)
    correct = correct * mask
    return tf.reduce_sum(correct) / (tf.reduce_sum(mask) + 1e-7)


def make_loss_functions(class_weights_tf):
    @tf.autograph.experimental.do_not_convert
    def masked_sparse_ce_weighted(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        mask = tf.cast(y_true > 0, tf.float32)

        y_shift = tf.where(mask > 0, y_true - 1, 0)
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_shift, y_pred)

        w = tf.gather(class_weights_tf, tf.clip_by_value(y_true, 0, 3))
        ce = ce * mask * w
        return tf.reduce_sum(ce) / (tf.reduce_sum(mask * w) + 1e-7)

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

    if not USE_ADVANCED_LOSS:
        loss_fn = masked_sparse_ce_weighted
        loss_name = "weighted_ce"
    else:
        if ADVANCED_LOSS_TYPE == "focal_tversky":
            loss_fn = focal_tversky_loss
            loss_name = "focal_tversky"
        else:
            loss_fn = dice_ce_loss
            loss_name = "dice_ce"

    return loss_fn, loss_name


#######################################################################
# === UNET ARCHITECTURE === #
#######################################################################

def conv_block(x, f):
    x = Conv2D(f, 3, padding="same", activation="relu")(x)
    if USE_MC_DROPOUT:
        x = Dropout(MC_DROPOUT_RATE)(x)
    x = Conv2D(f, 3, padding="same", activation="relu")(x)
    if USE_MC_DROPOUT:
        x = Dropout(MC_DROPOUT_RATE)(x)
    return x


def encoder_block(x, f):
    c = conv_block(x, f)
    return c, MaxPooling2D(2)(c)


def decoder_block(x, skip, f):
    x = UpSampling2D(2)(x)
    return conv_block(Concatenate()([x, skip]), f)


def build_unet(input_shape, num_classes):
    inp = Input(shape=input_shape)
    c1, p1 = encoder_block(inp, 32)
    c2, p2 = encoder_block(p1, 64)
    c3, p3 = encoder_block(p2, 128)
    c4, p4 = encoder_block(p3, 256)

    bn = conv_block(p4, 512)

    d1 = decoder_block(bn, c4, 256)
    d2 = decoder_block(d1, c3, 128)
    d3 = decoder_block(d2, c2, 64)
    d4 = decoder_block(d3, c1, 32)

    out = Conv2D(num_classes, 1, activation="softmax")(d4)
    return Model(inp, out)


def build_unet_with_embeddings(input_shape, num_classes):
    cont_in = Input(shape=input_shape, dtype=tf.float32, name="cont_in")
    qua_in = Input(shape=input_shape[:2] + (1,), dtype=tf.int32, name="quaternary_in")
    land_in = Input(shape=input_shape[:2] + (1,), dtype=tf.int32, name="landuse_in")

    qua_idx = Lambda(lambda t: tf.squeeze(t, axis=-1), name="quaternary_squeeze")(qua_in)
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


#######################################################################
# === CALLBACKS === #
#######################################################################

class F1MetricsCallback(Callback):
    def __init__(self, Xv, yv, out_txt, prefix="", Xv_qua=None, Xv_land=None, pred_batch_size=F1_PRED_BATCH_SIZE):
        super().__init__()
        self.Xv = Xv
        self.Xv_qua = Xv_qua
        self.Xv_land = Xv_land
        self.yv = yv
        self.out = out_txt
        self.prefix = prefix
        self.history = []
        self.pred_batch_size = max(1, int(pred_batch_size))

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        n = self.Xv.shape[0]
        y_pred_all = np.empty_like(self.yv, dtype=np.int32)

        for start in range(0, n, self.pred_batch_size):
            end = min(start + self.pred_batch_size, n)
            if self.Xv_qua is not None and self.Xv_land is not None:
                pred_input = {
                    "cont_in": self.Xv[start:end],
                    "quaternary_in": self.Xv_qua[start:end],
                    "landuse_in": self.Xv_land[start:end],
                }
            else:
                pred_input = self.Xv[start:end]

            y_prob = self.model.predict(pred_input, batch_size=self.pred_batch_size, verbose=0)
            y_pred_all[start:end] = np.argmax(y_prob, axis=-1).astype(np.int32) + 1
            del y_prob
            gc.collect()

        mask = self.yv > 0
        y_true = self.yv[mask]
        y_pred = y_pred_all[mask]

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

        print(
            f"  F1 scores: class1={ep_dict['f1_1']:.4f}, class2={ep_dict['f1_2']:.4f}, "
            f"class3={ep_dict['f1_3']:.4f}, macro_f1={ep_dict['macro_f1']:.4f}",
            flush=True
        )

        acc = np.sum(y_pred == y_true) / (len(y_true) + eps)
        ep_dict["masked_accuracy"] = acc
        logs["val_masked_accuracy"] = acc
        print(f"  Masked accuracy: {acc:.4f}", flush=True)

        recalls = []
        for cls in [1, 2, 3]:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            recall = tp / (tp + fn + eps)
            recalls.append(recall)

        balanced_acc = np.mean(recalls)
        ep_dict["balanced_accuracy"] = balanced_acc
        logs["val_balanced_accuracy"] = balanced_acc
        print(f"  Balanced accuracy: {balanced_acc:.4f}", flush=True)

        self.history.append(ep_dict)

        del y_pred_all, y_true, y_pred
        gc.collect()

    def on_train_end(self, logs=None):
        with open(self.out, "w") as f:
            for i, h in enumerate(self.history):
                f.write(f"Epoch {i+1}: {h}\n")
        print(f"Saved F1 log to {self.out}")



#######################################################################
# === UNCERTAINTY QUANTIFICATION === #
#######################################################################

def compute_mc_uncertainty(
    model,
    X,
    valid_mask=None,
    mc_iterations=MC_ITERATIONS,
    mc_batch_size=MC_BATCH_SIZE,
    X_qua=None,
    X_land=None,
):
    print("  [DEBUG] Enabling MC Dropout for inference...")
    print(f"  [DEBUG] Input X shape: {X.shape}, dtype: {X.dtype}")
    print(f"  [DEBUG] Input X range: [{np.min(X):.4f}, {np.max(X):.4f}]")
    print(f"  [DEBUG] Input has NaN: {np.isnan(X).any()}")

    n_samples = X.shape[0]
    mc_batch_size = max(1, int(mc_batch_size))
    print(f"  [DEBUG] MC batch size: {mc_batch_size}, samples: {n_samples}")

    mean_pred = np.zeros((n_samples, X.shape[1], X.shape[2], 3), dtype=np.float32)
    m2_pred = np.zeros_like(mean_pred)
    first_pass = None
    second_pass = None

    for i in tqdm(range(mc_iterations), desc="MC Dropout passes", unit="pass"):
        pass_pred = np.empty_like(mean_pred)

        for start_idx in range(0, n_samples, mc_batch_size):
            end_idx = min(start_idx + mc_batch_size, n_samples)

            X_batch = tf.convert_to_tensor(X[start_idx:end_idx], dtype=tf.float32)

            if X_qua is not None and X_land is not None:
                X_qua_batch = tf.convert_to_tensor(X_qua[start_idx:end_idx], dtype=tf.int32)
                X_land_batch = tf.convert_to_tensor(X_land[start_idx:end_idx], dtype=tf.int32)
                batch_pred = model(
                    {
                        "cont_in": X_batch,
                        "quaternary_in": X_qua_batch,
                        "landuse_in": X_land_batch,
                    },
                    training=True,
                ).numpy().astype(np.float32, copy=False)
                del X_qua_batch, X_land_batch
            else:
                batch_pred = model(X_batch, training=True).numpy().astype(np.float32, copy=False)

            pass_pred[start_idx:end_idx] = batch_pred
            del X_batch, batch_pred
            gc.collect()

        if i == 0:
            first_pass = pass_pred.copy()
            print(f"  [DEBUG] First prediction shape: {pass_pred.shape}, dtype: {pass_pred.dtype}")
            print(f"  [DEBUG] First prediction range: [{np.min(pass_pred):.4f}, {np.max(pass_pred):.4f}]")
            print(f"  [DEBUG] First prediction has NaN: {np.isnan(pass_pred).any()}")
        elif i == 1:
            second_pass = pass_pred.copy()

        delta = pass_pred - mean_pred
        mean_pred += delta / float(i + 1)
        delta2 = pass_pred - mean_pred
        m2_pred += delta * delta2
        del delta, delta2, pass_pred
        gc.collect()

    pred_variance = m2_pred / max(mc_iterations - 1, 1)
    std_pred = np.sqrt(np.maximum(pred_variance, 0.0)).astype(np.float32, copy=False)
    mean_variance = float(np.mean(pred_variance))
    print(f"  MC prediction variance (mean): {mean_variance:.6f}")

    if first_pass is not None and second_pass is not None:
        pred_diff = np.abs(first_pass - second_pass)
        max_diff = float(np.max(pred_diff))
        print(f"  Max diff between first two MC passes: {max_diff:.6f}")
        del pred_diff

    if mean_variance < 1e-8:
        print("  [WARNING] Very low variance across MC passes — dropout may not be active!")

    mean_pred_clamped = np.clip(mean_pred, 1e-10, 1.0 - 1e-10)
    mean_pred_norm = mean_pred_clamped / np.sum(mean_pred_clamped, axis=-1, keepdims=True)
    entropy = -np.sum(mean_pred_norm * np.log(mean_pred_norm), axis=-1)

    if valid_mask is not None:
        entropy_masked = np.where(valid_mask, entropy, np.nan)
        n_masked = (~valid_mask).sum()
        n_total = valid_mask.size
        print(f"  Applied mask: {n_masked}/{n_total} pixels ({n_masked/n_total*100:.1f}%) marked as no-data (NaN)")
    else:
        entropy_masked = entropy
        print("  [INFO] No mask applied - computing uncertainty for all pixels")

    valid_entropy = entropy_masked[~np.isnan(entropy_masked)]
    if len(valid_entropy) > 0:
        print(f"  Entropy range (valid pixels): [{np.min(valid_entropy):.4f}, {np.max(valid_entropy):.4f}]")
    else:
        print("  [WARNING] All entropy values are NaN!")

    del m2_pred, pred_variance
    gc.collect()

    return mean_pred, std_pred, entropy_masked


def save_uncertainty_maps(mean_pred, std_pred, entropy, fold_idx, data_type="cv"):
    out_dir = os.path.join(QC_PLOTS, f"uncertainty_{data_type}_fold{fold_idx}")
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "mean_pred.npy"), mean_pred)
    np.save(os.path.join(out_dir, "std_pred.npy"), std_pred)
    np.save(os.path.join(out_dir, "entropy.npy"), entropy)

    total_pixels = entropy.size
    nan_pixels = np.isnan(entropy).sum()
    valid_pixels = total_pixels - nan_pixels
    pct_valid = (valid_pixels / total_pixels) * 100
    pct_masked = (nan_pixels / total_pixels) * 100

    mean_entropy_per_sample = np.nanmean(entropy, axis=(1, 2))
    valid_entropy = mean_entropy_per_sample[~np.isnan(mean_entropy_per_sample)]

    print(f"  [DIAGNOSTIC] mean_entropy_per_sample shape: {mean_entropy_per_sample.shape}")
    print(f"  [DIAGNOSTIC] valid samples in mean_entropy_per_sample: {np.sum(~np.isnan(mean_entropy_per_sample))}")

    if len(valid_entropy) > 0:
        plt.figure(figsize=(10, 5))
        plt.hist(valid_entropy, bins=30, alpha=0.7, edgecolor="black")
        plt.xlabel("Mean Entropy per Sample")
        plt.ylabel("Frequency")
        plt.title(f"Predictive Uncertainty Distribution ({data_type.upper()} Fold {fold_idx})")
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
        print(f"[WARNING] No valid entropy values found for {data_type} fold {fold_idx}")
        print(f"  All {total_pixels} pixels are masked (no-data)")
        print(f"Saved uncertainty maps to {out_dir} (all NaN - check data mask)")


#######################################################################
# === DATASET CREATION === #
#######################################################################

def create_dataset(X, y, batch_size=2, shuffle=True, X_qua=None, X_land=None):
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    if X_qua is not None and X_land is not None:
        X_qua = X_qua.astype(np.int32)
        X_land = X_land.astype(np.int32)
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "cont_in": X,
                    "quaternary_in": X_qua,
                    "landuse_in": X_land,
                },
                y,
            )
        )
    else:
        dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(1000, len(X)), reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


#######################################################################
# === TRUE SPATIAL CV === #
#######################################################################

K.clear_session()
gc.collect()

if DO_SPATIAL_CV:
    print("\n=== TRUE SPATIAL BLOCK CV ===")
    print(f"Block size: {SPATIAL_BLOCK_SIZE_KM} km")
    print(f"Buffer blocks: {SPATIAL_BUFFER_BLOCKS}")

    tile_to_block, block_to_tiles = assign_spatial_blocks(
        tile_ids=tile_ids,
        tile_metadata_path=TILE_METADATA_PATH,
        block_size_km=SPATIAL_BLOCK_SIZE_KM,
    )

    block_to_fold = assign_blocks_to_folds(
        block_to_tiles=block_to_tiles,
        n_folds=N_FOLDS,
        random_seed=SPATIAL_RANDOM_SEED,
    )

    print(f"[INFO] Number of spatial blocks: {len(block_to_tiles)}")

    X_qua_all = quaternary if USE_CATEGORICAL_EMBEDDINGS else None
    X_land_all = landuse if USE_CATEGORICAL_EMBEDDINGS else None

    permutation_fold_dir = os.path.join(PERMUTATION_OUT, "fold_results")
    permutation_summary_dir = os.path.join(PERMUTATION_OUT, "summary")
    os.makedirs(permutation_fold_dir, exist_ok=True)
    os.makedirs(permutation_summary_dir, exist_ok=True)
    all_permutation_rows = []

    cos_out = os.path.join(os.path.dirname(PERMUTATION_OUT), "Channel_Occlusion_Sensitivity")
    cos_fold_dir = os.path.join(cos_out, "fold_results")
    cos_summary_dir = os.path.join(cos_out, "summary")
    os.makedirs(cos_fold_dir, exist_ok=True)
    os.makedirs(cos_summary_dir, exist_ok=True)
    all_cos_rows = []

    for fold in range(N_FOLDS):
        tr_idx, va_idx = make_spatial_fold_indices(
            tile_ids=tile_ids,
            tile_to_block=tile_to_block,
            block_to_fold=block_to_fold,
            fold_id=fold,
            buffer_blocks=SPATIAL_BUFFER_BLOCKS,
        )

        if len(tr_idx) == 0 or len(va_idx) == 0:
            raise RuntimeError(
                f"Fold {fold+1} has empty split: train={len(tr_idx)}, val={len(va_idx)}. "
                f"Adjust block size or buffer."
            )

        tr_tile_ids = [tile_ids[i] for i in tr_idx]
        va_tile_ids = [tile_ids[i] for i in va_idx]
        tr_blocks = {tile_to_block[t] for t in tr_tile_ids}
        va_blocks = {tile_to_block[t] for t in va_tile_ids}

        overlap_blocks = tr_blocks.intersection(va_blocks)
        if overlap_blocks:
            raise RuntimeError(f"Spatial leakage: fold {fold+1} train/val share blocks!")

        print(f"\n--- Spatial Fold {fold+1}/{N_FOLDS} ---")
        print(f"Train tiles: {len(tr_idx)} | Val tiles: {len(va_idx)}")
        print(f"Train blocks: {len(tr_blocks)} | Val blocks: {len(va_blocks)}")

        Xtr = X[tr_idx]
        Xva = X[va_idx]
        ytr = y[tr_idx]
        yva = y[va_idx]

        if USE_CATEGORICAL_EMBEDDINGS:
            Xtr_qua = X_qua_all[tr_idx]
            Xva_qua = X_qua_all[va_idx]
            Xtr_land = X_land_all[tr_idx]
            Xva_land = X_land_all[va_idx]
        else:
            Xtr_qua = Xva_qua = None
            Xtr_land = Xva_land = None

        print(
            f"  [DEBUG CV FOLD {fold+1}] Xva after split - has NaN: {np.isnan(Xva).any()}, "
            f"range: [{np.nanmin(Xva):.4f}, {np.nanmax(Xva):.4f}]"
        )

        Xva_clean = Xva.copy()
        Xva_qua_clean = Xva_qua.copy() if Xva_qua is not None else None
        Xva_land_clean = Xva_land.copy() if Xva_land is not None else None

        fold_weights_np = compute_class_weights_from_y(
            ytr,
            class2_multiplier=CLASS2_WEIGHT_MULTIPLIER,
        )
        fold_class_weights_tf = tf.constant(fold_weights_np, dtype=tf.float32)
        LOSS_FUNCTION, loss_name = make_loss_functions(fold_class_weights_tf)

        print(f"  Fold class weights: {fold_weights_np}")
        print(f"  Loss function: {loss_name}")

        if USE_CATEGORICAL_EMBEDDINGS:
            model = build_unet_with_embeddings(X.shape[1:], num_classes)
        else:
            model = build_unet(X.shape[1:], num_classes)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=LOSS_FUNCTION,
            metrics=[masked_accuracy],
        )


        model_path = os.path.join(MODEL_OUT, f"unet_spatial_cv_fold{fold+1}.keras")
        f1_log = os.path.join(QC_PLOTS, f"spatial_cv_fold{fold+1}_f1.txt")

        f1cb = F1MetricsCallback(
            Xva,
            yva,
            f1_log,
            prefix=f"Spatial CV Fold {fold+1}",
            Xv_qua=Xva_qua,
            Xv_land=Xva_land,
        )

        ckpt = ModelCheckpoint(
            filepath=model_path,
            monitor="val_macro_f1",
            mode="max",
            save_best_only=True
        )

        train_dataset = create_dataset(
            Xtr,
            ytr,
            batch_size=2,
            shuffle=True,
            X_qua=Xtr_qua,
            X_land=Xtr_land,
        )

        val_dataset = create_dataset(
            Xva,
            yva,
            batch_size=2,
            shuffle=False,
            X_qua=Xva_qua,
            X_land=Xva_land,
        )


        # === PATCHED: Add EarlyStopping callback (monitor val_macro_f1, patience=6) ===
        from keras.callbacks import EarlyStopping
        early_stop = EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1
        )
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=CV_EPOCHS,
            callbacks=[f1cb, ckpt, TerminateOnNaN(), early_stop],
            verbose=1,
        )

        plt.figure(figsize=(8, 5))
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curve Spatial CV Fold {fold+1}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(QC_PLOTS, f"loss_curve_spatial_cv_fold{fold+1}.png"))
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(history.history["masked_accuracy"], label="Train Masked Accuracy")
        plt.plot(history.history["val_masked_accuracy"], label="Val Masked Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Masked Accuracy")
        plt.title(f"Masked Accuracy Spatial CV Fold {fold+1}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(QC_PLOTS, f"masked_accuracy_spatial_cv_fold{fold+1}.png"))
        plt.close()

        if "val_macro_f1" in history.history:
            plt.figure(figsize=(8, 5))
            plt.plot(history.history["val_macro_f1"], label="Val Macro F1")
            plt.xlabel("Epoch")
            plt.ylabel("Macro F1")
            plt.title(f"Macro F1 Spatial CV Fold {fold+1}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(QC_PLOTS, f"macro_f1_spatial_cv_fold{fold+1}.png"))
            plt.close()

        print(
            f"  [DEBUG CV FOLD {fold+1}] After fit - Xva has NaN: {np.isnan(Xva).any()}, "
            f"Xva_clean has NaN: {np.isnan(Xva_clean).any()}"
        )

        # Reload BEST checkpoint before MC dropout

        print(f"  Reloading best checkpoint: {model_path}")
        del model
        K.clear_session()
        gc.collect()

        best_model = tf.keras.models.load_model(model_path, compile=False)

        if USE_MC_DROPOUT:
            print(f"\nComputing MC Dropout uncertainty (Spatial Fold {fold+1})...")
            va_mask = valid_data_mask[va_idx]

            mean_pred, std_pred, entropy = compute_mc_uncertainty(
                best_model,
                Xva_clean,
                valid_mask=va_mask,
                mc_iterations=MC_ITERATIONS,
                X_qua=Xva_qua_clean,
                X_land=Xva_land_clean,
            )

            save_uncertainty_maps(
                mean_pred,
                std_pred,
                entropy,
                fold_idx=fold+1,
                data_type="spatial_cv",
            )

        # =========================
        # CHANNEL OCCLUSION TEST PER FOLD
        # =========================
        if DO_CHANNEL_OCCLUSION_TEST and len(channel_occlusion_plan) > 0:
            print(f"\n=== Running channel occlusion test for Spatial Fold {fold+1} ===")

            fold_cos_dir = os.path.join(cos_fold_dir, f"fold_{fold+1}")
            os.makedirs(fold_cos_dir, exist_ok=True)

            baseline_metrics_cos, fold_cos_rows = run_channel_occlusion_test_for_fold(
                model=best_model,
                fold_idx=fold + 1,
                Xva=Xva_clean,
                yva=yva,
                occlusion_plan=channel_occlusion_plan,
                occlusion_out_dir=fold_cos_dir,
                fill_value=COS_FILL_VALUE,
                batch_size=COS_BATCH_SIZE,
                Xva_qua=Xva_qua_clean,
                Xva_land=Xva_land_clean,
            )

            all_cos_rows.extend(fold_cos_rows)
            print(f"[COS] Fold {fold+1}: wrote {len(fold_cos_rows)} occlusion rows")
        elif DO_CHANNEL_OCCLUSION_TEST:
            print(f"[COS] Fold {fold+1}: channel occlusion skipped because channel_occlusion_plan is empty")

        del best_model
        K.clear_session()
        gc.collect()

    if DO_PERMUTATION_TEST and len(all_permutation_rows) > 0:
        summary_csv = os.path.join(permutation_summary_dir, "permutation_importance_summary.csv")
        write_permutation_summary(all_permutation_rows, summary_csv)
        print(f"\n[PERM] Wrote cross-fold summary to: {summary_csv}")

    if DO_CHANNEL_OCCLUSION_TEST and len(all_cos_rows) > 0:
        cos_summary_csv = os.path.join(cos_summary_dir, "channel_occlusion_summary.csv")
        write_channel_occlusion_summary(all_cos_rows, cos_summary_csv)
        print(f"\n[COS] Wrote cross-fold summary to: {cos_summary_csv}")


#######################################################################
# === OPTIONAL MAIN TRAINING (kept disabled by default) === #
#######################################################################

if DO_MAIN_TRAIN:
    print("\n=== Main Training ===")

    if USE_CATEGORICAL_EMBEDDINGS:
        Xtr, Xva, Xtr_qua, Xva_qua, Xtr_land, Xva_land, ytr, yva = train_test_split(
            X, quaternary, landuse, y, test_size=0.2, random_state=42
        )
    else:
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)
        Xtr_qua = Xva_qua = None
        Xtr_land = Xva_land = None

    Xva_clean = Xva.copy()
    Xva_qua_clean = Xva_qua.copy() if Xva_qua is not None else None
    Xva_land_clean = Xva_land.copy() if Xva_land is not None else None

    print(
        f"  [DEBUG MAIN] Xva after split - has NaN: {np.isnan(Xva).any()}, "
        f"range: [{np.nanmin(Xva):.4f}, {np.nanmax(Xva):.4f}]"
    )

    fold_weights_np = compute_class_weights_from_y(
        ytr,
        class2_multiplier=CLASS2_WEIGHT_MULTIPLIER,
    )
    fold_class_weights_tf = tf.constant(fold_weights_np, dtype=tf.float32)
    LOSS_FUNCTION, loss_name = make_loss_functions(fold_class_weights_tf)

    print(f"  Main-train class weights: {fold_weights_np}")
    print(f"  Loss function: {loss_name}")

    if USE_CATEGORICAL_EMBEDDINGS:
        model = build_unet_with_embeddings(X.shape[1:], num_classes)
    else:
        model = build_unet(X.shape[1:], num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=LOSS_FUNCTION,
        metrics=[masked_accuracy],
    )

    ckpt_path = os.path.join(MODEL_OUT, "unet_best_model.keras")
    ckpt = ModelCheckpoint(
        ckpt_path,
        monitor="val_macro_f1",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    f1_main_log = os.path.join(QC_PLOTS, "main_train_f1.txt")
    f1cb = F1MetricsCallback(
        Xva,
        yva,
        f1_main_log,
        prefix="MAIN",
        Xv_qua=Xva_qua,
        Xv_land=Xva_land,
    )

    train_dataset = create_dataset(
        Xtr,
        ytr,
        batch_size=2,
        shuffle=True,
        X_qua=Xtr_qua,
        X_land=Xtr_land,
    )
    val_dataset = create_dataset(
        Xva,
        yva,
        batch_size=2,
        shuffle=False,
        X_qua=Xva_qua,
        X_land=Xva_land,
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=40,
        callbacks=[f1cb, ckpt, TerminateOnNaN()],
        verbose=1,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve Main Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(QC_PLOTS, "loss_curve_main_training.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["masked_accuracy"], label="Train Masked Accuracy")
    plt.plot(history.history["val_masked_accuracy"], label="Val Masked Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Masked Accuracy")
    plt.title("Masked Accuracy Curve Main Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(QC_PLOTS, "masked_accuracy_curve_main_training.png"))
    plt.close()

    if "val_macro_f1" in history.history:
        plt.figure(figsize=(8, 5))
        plt.plot(history.history["val_macro_f1"], label="Val Macro F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title("Macro F1 Curve Main Training")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(QC_PLOTS, "macro_f1_curve_main_training.png"))
        plt.close()

    del model
    K.clear_session()
    gc.collect()

    best_model = tf.keras.models.load_model(ckpt_path, compile=False)

    if USE_MC_DROPOUT:
        print("\nComputing MC Dropout uncertainty (Main Training)...")
        _, va_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
        va_mask_main = valid_data_mask[va_idx]

        mean_pred, std_pred, entropy = compute_mc_uncertainty(
            best_model,
            Xva_clean,
            valid_mask=va_mask_main,
            mc_iterations=MC_ITERATIONS,
            X_qua=Xva_qua_clean,
            X_land=Xva_land_clean,
        )
        save_uncertainty_maps(mean_pred, std_pred, entropy, fold_idx=0, data_type="main")

    del best_model
    K.clear_session()
    gc.collect()

    print("\n✓ Training completed successfully!")
    print("✓ Check the saved plots and F1 logs for results.")