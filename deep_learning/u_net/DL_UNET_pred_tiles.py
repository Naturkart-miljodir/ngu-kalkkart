# -*- coding: utf-8 -*-
"""
Prediction tiling for Kalk U-Net
- Clean, NaN-proof sliding window predictor
- Handles nodata safely
- No gaps in output rasters
- Generates per-class probability maps + class map
"""

import os
import sys
import glob
import numpy as np
import rasterio
from rasterio.windows import Window
import tensorflow as tf
from tqdm import tqdm

# Add zlib path for Windows (fixes zlibwapi.dll error)
if sys.platform == 'win32':
    conda_env = os.path.dirname(os.path.dirname(sys.executable))
    zlib_path = os.path.join(conda_env, 'Library', 'bin')
    if os.path.exists(zlib_path):
        # Add to PATH as fallback
        if zlib_path not in os.environ['PATH']:
            os.environ['PATH'] = zlib_path + os.pathsep + os.environ['PATH']
        # Python 3.8+: explicitly add DLL search path (more reliable)
        try:
            os.add_dll_directory(zlib_path)
        except AttributeError:
            pass  # Python < 3.8, PATH modification above is the fallback

# =====================================================
# GPU CONFIGURATION
# =====================================================
print("\n" + "=" * 70)
print("GPU CONFIGURATION")
print("=" * 70)

print(f"TensorFlow version: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

gpus = tf.config.list_physical_devices('GPU')
print(f"Number of GPUs available: {len(gpus)}")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Enabled memory growth for: {gpu.name}")
        if tf.test.gpu_device_name():
            print(f"✓ Default GPU device: {tf.test.gpu_device_name()}")
    except RuntimeError as e:
        print(f"✗ GPU configuration error: {e}")
else:
    print("⚠ No GPU found - predictions will run on CPU")

print("=" * 70)
print()


# =====================================================
# 1. Paths
# =====================================================

PREDICTOR_DIR = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\Covariates_to_model"
MODEL_OUT     = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\Models"
PRED_OUT      = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\Raster"

os.makedirs(PRED_OUT, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODEL_OUT, "unet_best_model.keras")

TILE_SIZE = 128     # match training tiles
OVERLAP   = 0.5
STEP      = int(TILE_SIZE * (1 - OVERLAP))

CLASS_ORDER = ["low", "medium", "high"]
NUM_CLASSES = len(CLASS_ORDER)


# =====================================================
# 2. Load model + custom losses
# =====================================================

print("Loading model:", BEST_MODEL_PATH)

@tf.autograph.experimental.do_not_convert
def masked_sparse_ce(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    mask   = tf.cast(y_true > 0, tf.float32)
    y_shift = tf.where(mask > 0, y_true - 1, 0)
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_shift, y_pred)
    ce = ce * mask
    return tf.reduce_sum(ce) / (tf.reduce_sum(mask) + 1e-7)

model = tf.keras.models.load_model(
    BEST_MODEL_PATH,
    custom_objects={"masked_sparse_ce": masked_sparse_ce},
    compile=False
)
print("✔ Model loaded.")


# =====================================================
# 3. Load predictor rasters
# =====================================================

tif_paths = sorted(glob.glob(os.path.join(PREDICTOR_DIR, "*.tif")))
datasets = [rasterio.open(p) for p in tif_paths]
ref = datasets[0]

width, height = ref.width, ref.height
transform = ref.transform
crs = ref.crs

print("Predictor rasters:")
for p in tif_paths:
    print("  ", os.path.basename(p))
print("\nReference size:", width, "x", height)


# =====================================================
# 4. Prepare output arrays
# =====================================================

prob_sum   = np.zeros((height, width, NUM_CLASSES), dtype=np.float32)
weight_sum = np.zeros((height, width), dtype=np.float32)


# =====================================================
# 5. Generate tile start positions
# =====================================================

def make_starts(size, tile, step):
    starts = list(range(0, size - tile + 1, step))
    last = size - tile
    if last not in starts:
        starts.append(last)
    return starts

row_starts = make_starts(height, TILE_SIZE, STEP)
col_starts = make_starts(width,  TILE_SIZE, STEP)

print("Row starts:", row_starts[:5])
print("Col starts:", col_starts[:5])


# =====================================================
# 6. Slide and predict (NaN-proof!)
# =====================================================

print("\nStarting prediction tiling...\n")
pbar = tqdm(total=len(row_starts) * len(col_starts), ncols=90)

for r in row_starts:
    for c in col_starts:
        pbar.update(1)

        win = Window(c, r, TILE_SIZE, TILE_SIZE)

        bands = []
        valid_mask = None

        for ds in datasets:
            arr = ds.read(1, window=win).astype(np.float32)
            nod = ds.nodatavals[0]

            # --- CLEAN NODATA + NAN ---
            invalid = np.isnan(arr)
            if nod is not None:
                invalid |= (arr == nod)

            arr[invalid] = 0.0  # safe fill

            if valid_mask is None:
                valid_mask = ~invalid
            else:
                valid_mask &= ~invalid

            bands.append(arr)

        # All predictors entirely nodata → skip
        if not np.any(valid_mask):
            continue

        # Stack & predict
        X_tile = np.stack(bands, axis=-1)
        preds = model.predict(X_tile[np.newaxis, ...], verbose=0)[0]

        # Zero probabilities where invalid
        preds *= valid_mask[..., None].astype(np.float32)

        prob_sum[r:r+TILE_SIZE, c:c+TILE_SIZE] += preds
        weight_sum[r:r+TILE_SIZE, c:c+TILE_SIZE] += valid_mask.astype(np.float32)

pbar.close()
print("\n✔ Tile prediction completed.")


# =====================================================
# 7. Final blended probability maps
# =====================================================

print("Blending tiles...")

ws = weight_sum.copy()
ws[ws == 0] = 1   # avoid division by zero

probs = prob_sum / ws[..., None]

# Pixels with no data everywhere → probs = 0
probs[weight_sum == 0] = 0.0


# =====================================================
# 8. Class map
# =====================================================

class_map = np.zeros((height, width), dtype=np.uint8)
valid = weight_sum > 0
class_map[valid] = np.argmax(probs[valid], axis=-1).astype(np.uint8) + 1


# =====================================================
# 9. Write probability rasters
# =====================================================

print("Writing probability rasters...")

for i, cname in enumerate(CLASS_ORDER):
    profile = ref.profile.copy()
    profile.update(dtype="float32", count=1, nodata=0.0)

    out = os.path.join(PRED_OUT, f"prob_{i+1}_{cname}.tif")
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(probs[:, :, i], 1)

    print(" ✔", out)


# =====================================================
# 10. Write class map raster
# =====================================================

profile = ref.profile.copy()
profile.update(dtype="uint8", count=1, nodata=0)

out_cm = os.path.join(PRED_OUT, "class_map_3class.tif")
with rasterio.open(out_cm, "w", **profile) as dst:
    dst.write(class_map, 1)

print("\n✔ Class map saved:", out_cm)
print("✔ Prediction DONE.")
