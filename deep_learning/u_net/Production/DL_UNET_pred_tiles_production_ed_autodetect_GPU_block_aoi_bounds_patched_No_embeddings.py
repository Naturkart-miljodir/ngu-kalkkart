# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
DL_UNET_pred_tiles_blockwise_aoi.py

Blockwise U-Net production prediction with optional AOI polygon masking.
Designed to avoid allocating full-scene arrays in RAM.

Main features
-------------
- Loads a trained Keras model (.keras or HDF5-backed file with .keras/.h5 name)
- Reads predictors in the same order as channel_map.csv
- Supports 3-input U-Net with embeddings:
    * cont_in
    * quaternary_in
    * landuse_in
- Predicts in super-blocks and tiles, writing outputs directly to disk
- Optional AOI vector polygon to restrict prediction area (e.g. Lofoten)
- Optional confidence and entropy rasters

Notes
-----
- This is the "basic" blockwise version:
    * probability rasters
    * class raster
    * confidence raster
    * entropy raster
- It does NOT include MC-dropout uncertainty yet.
"""

import os
import sys
import math
import glob
import json
import zipfile
import tempfile
import shutil
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import rasterio
from rasterio import shutil as rio_shutil
from rasterio.windows import Window
from rasterio.features import geometry_mask
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_geom
import fiona

import tensorflow as tf
import keras

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# =============================================================================
# USER SETTINGS
# =============================================================================

# ---- core paths ----
PREDICTOR_DIR = r"G:\Covariates_to_model"
MODEL_OUT = r"D:\Unet_models\MidNorge\models"
PRED_OUT = r"D:\Unet_models\MidNorge\250K\With_Ca_average\Ca_conc_aveg\Production"
CHANNEL_MAP_CSV = r"D:\Unet_models\MidNorge\250K\With_Ca_average\Ca_conc_aveg\predictor_list_used_for_training.csv"

# Optional stricter mapping file: columns predictor_name,filename
PREDICTOR_LIST_FILE = os.path.join(PREDICTOR_DIR, "predictor_file_list.csv")

# ---- model path ----
BEST_MODEL_PATH = r"D:\Unet_models\MidNorge\250K\With_Ca_average\Ca_conc_aveg\models\unet_best_model.keras"

# ---- optional AOI polygon ----
# Example:
# AOI_VECTOR_PATH = r"C:\Users\acosta_pedro\Documents\Lofoten\Lofoten.shp"
AOI_VECTOR_PATH = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Study_area\NordTrondelag_pol.shp"
AOI_LAYER = None  # set only for GPKG / multi-layer sources

# ---- output folders ----
PREDICTIONS_DIRNAME = "Predictions"
UNCERTAINTY_DIRNAME = "Uncertainty"

# ---- model/prediction parameters ----
NUM_CLASSES = 3
TILE_SIZE = 128
OVERLAP = 32                  # 16-32 is a good start
SUPERBLOCK_SIZE = 1536        # reduce to 1024 if RAM is still tight
BATCH_SIZE = 4                # reduce to 4 or 2 if needed
NODATA_FILL_VALUE = 0.0
SIZE_MISMATCH_TOL_PIXELS = 2

# ---- model input config ----
# Keep this aligned with training script:
# DL_Unet_GPU_embeddings_desktop_SpCV_final_No_embeddings.py
USE_CATEGORICAL_EMBEDDINGS = False
QUATERNARY_NAME = "quaternary_forenkletk_cog"
LANDUSE_NAME = "landuse_code_18_cog"
QUATERNARY_NUM_CLASSES = 22
LANDUSE_NUM_CLASSES = 33

# Channel exclusion tokens aligned with training
CHANNELS_EXCL = [
    # Using the model's saved predictor list as CHANNEL_MAP_CSV, so no extra exclusions.
]

# ---- output toggles ----
WRITE_CONFIDENCE = True
WRITE_ENTROPY = True
WRITE_CLASS = True

# ---- file names ----
PROB_PREFIX = "prob_class"
CLASS_NAME = "predicted_class.tif"
CONFIDENCE_NAME = "confidence.tif"
ENTROPY_NAME = "entropy.tif"

# ---- output format options ----
WRITE_COG = True
COG_COMPRESS = "LZW"
COG_BLOCKSIZE = 512

# ---- normalization settings (match tiling/training logic) ----
APPLY_TRAINING_STYLE_ZSCORE = True
NORMALIZE_IF_NAME_CONTAINS = (
    "Topo_",
    "Geophys_",
    "xgb_prediction_CaO_",
    "xgb_prediction_KESP_",
)
Z_N_SAMPLES_PER_RASTER = 2_000_000
Z_WINDOWS_PER_RASTER = 40
Z_WINDOW_SIZE = 512
Z_RANDOM_SEED = 42

# ---- fast QA mode (optional) ----
# Use this to run a tiny contiguous subset of the AOI and validate outputs quickly.
SMALL_TEST_MODE = False
SMALL_TEST_RADIUS_BLOCKS = 0  # 0=1 block, 1=3x3 blocks, 2=5x5 blocks
# Optional center in pixel coordinates (row, col). If None, AOI bbox center is used.
SMALL_TEST_CENTER_ROWCOL = None
# When True, write output rasters only for the selected block subset extent.
SMALL_TEST_CROP_OUTPUT = False
SMALL_TEST_OUTPUT_SUBDIR = "QA_small"

# ---- alphaearth VRT patching (optional) ----
PATCH_ALPHAEARTH_VRT = False
ALPHAEARTH_VRT_NAME = "alphaearth_dequant_national_epsg25833.vrt"
ALPHAEARTH_DIR = r"E:\Alpha_earth\dequant_images_all"

# =============================================================================
# GPU CONFIG
# =============================================================================

print("=" * 70)
print("GPU CONFIGURATION")
print("=" * 70)
print(f"TensorFlow version: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

gpus = tf.config.list_physical_devices("GPU")
print(f"Number of GPUs available: {len(gpus)}")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Enabled memory growth for: {gpu}")
        except Exception as e:
            print(f"⚠ Could not set memory growth for {gpu}: {e}")
    try:
        print(f"✓ Default GPU device: {tf.test.gpu_device_name()}")
    except Exception:
        pass
else:
    print("⚠ No GPU found - predictions will run on CPU")
print("=" * 70)

# =============================================================================
# HELPERS
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def is_zip_keras(path):
    try:
        if not os.path.isfile(path):
            return False
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
        return ("config.json" in names) and ("metadata.json" in names)
    except Exception:
        return False


def is_hdf5_file(path):
    try:
        if not os.path.isfile(path):
            return False
        with open(path, "rb") as f:
            sig = f.read(8)
        return sig == b"\x89HDF\r\n\x1a\n"
    except Exception:
        return False


def patch_alphaearth_vrt_if_requested():
    if not PATCH_ALPHAEARTH_VRT:
        return

    vrt_path = os.path.join(PREDICTOR_DIR, ALPHAEARTH_VRT_NAME)
    if not os.path.isfile(vrt_path):
        print(f"[INFO] AlphaEarth VRT not found, skipping patch: {vrt_path}")
        return

    try:
        with open(vrt_path, "r", encoding="utf-8") as f:
            txt = f.read()

        import re
        srcs = re.findall(r"<SourceFilename[^>]*>(.*?)</SourceFilename>", txt)
        if not srcs:
            print("[INFO] No SourceFilename entries found in AlphaEarth VRT.")
            return

        new_txt = txt
        for src in srcs:
            base = os.path.basename(src)
            new_src = os.path.join(ALPHAEARTH_DIR, base).replace("\\", "/")
            new_txt = new_txt.replace(src, new_src)

        patched = vrt_path.replace(".vrt", "_patched.vrt")
        with open(patched, "w", encoding="utf-8") as f:
            f.write(new_txt)

        print(f"[INFO] Patched AlphaEarth VRT: {patched}")
    except Exception as e:
        print(f"[WARN] Failed to patch AlphaEarth VRT: {e}")



def load_channel_map(channel_map_csv):
    df = pd.read_csv(channel_map_csv)

    possible_cols = [c for c in df.columns if c.lower() in ("channel_name", "predictor_name", "name")]
    if not possible_cols:
        raise ValueError("channel_map.csv must contain a predictor name column, e.g. channel_name or predictor_name.")
    name_col = possible_cols[0]

    band_col = None
    for c in df.columns:
        if c.lower() == "band":
            band_col = c
            break

    channel_idx_col = None
    for c in df.columns:
        if c.lower() in ("channel_idx", "channel", "index", "idx"):
            channel_idx_col = c
            break

    source_col = None
    for c in df.columns:
        if c.lower() in ("source_file", "filename", "file", "path", "source"):
            source_col = c
            break

    rows = []
    for row_idx, row in df.iterrows():
        predictor_name = str(row[name_col])
        band = 1
        if band_col and pd.notna(row[band_col]):
            try:
                band = int(row[band_col])
            except Exception:
                raise ValueError(
                    f"Invalid band value in channel_map.csv at row {row_idx}: {row[band_col]!r}"
                )

        channel_idx = row_idx
        if channel_idx_col and pd.notna(row[channel_idx_col]):
            try:
                channel_idx = int(row[channel_idx_col])
            except Exception:
                channel_idx = row[channel_idx_col]

        source_file = None
        if source_col and pd.notna(row[source_col]):
            source_file = str(row[source_col])

        rows.append({
            "row_idx": row_idx,
            "predictor_name": predictor_name,
            "band": band,
            "channel_idx": channel_idx,
            "source_file": source_file,
        })

    return rows, df


def normalize_name(s):
    s = str(s).strip().lower()
    s = s.replace(".tif", "").replace(".vrt", "")
    s = s.replace(" ", "_")
    return s


def should_normalize_source(path):
    stem_raw = os.path.splitext(os.path.basename(path))[0]
    return any(tag in stem_raw for tag in NORMALIZE_IF_NAME_CONTAINS)


def sample_band_for_zscore(ds, band, nwin, win_size, max_samples, rng):
    h = ds.height
    w = ds.width
    ww = min(win_size, w)
    hh = min(win_size, h)

    max_x = w - ww
    max_y = h - hh
    chunks = []

    nod = ds.nodata
    for _ in range(max(1, int(nwin))):
        x0 = rng.randint(0, max_x) if max_x > 0 else 0
        y0 = rng.randint(0, max_y) if max_y > 0 else 0
        arr = ds.read(
            band,
            window=Window(x0, y0, ww, hh),
            boundless=False,
            masked=False,
        ).astype(np.float32)
        if nod is not None:
            arr[arr == nod] = np.nan
        vals = arr[np.isfinite(arr)]
        if vals.size:
            chunks.append(vals)

    if not chunks:
        return 0.0, 1.0

    sample = np.concatenate(chunks)
    if sample.size > max_samples:
        idx = rng.choice(sample.size, size=max_samples, replace=False)
        sample = sample[idx]

    mean = float(np.mean(sample))
    std = float(np.std(sample))
    if std < 1e-6:
        std = 1.0
    return mean, std


def build_zscore_stats(datasets, channel_specs):
    if not APPLY_TRAINING_STYLE_ZSCORE:
        return {}

    rng = np.random.RandomState(Z_RANDOM_SEED)
    unique_pairs = []
    seen = set()
    for spec in channel_specs:
        p = spec["path"]
        b = int(spec.get("band", 1))
        key = (p, b)
        if key not in seen and should_normalize_source(p):
            seen.add(key)
            unique_pairs.append(key)

    if not unique_pairs:
        print("[NORM] No predictors matched z-score normalization rules.")
        return {}

    print(f"[NORM] Computing z-score stats for {len(unique_pairs)} channel(s)...")
    stats = {}
    iterator = unique_pairs
    if tqdm is not None:
        iterator = tqdm(unique_pairs, total=len(unique_pairs), desc="ZScore stats", unit="band", dynamic_ncols=True)

    for p, b in iterator:
        ds = datasets[p]
        m, s = sample_band_for_zscore(
            ds=ds,
            band=b,
            nwin=Z_WINDOWS_PER_RASTER,
            win_size=Z_WINDOW_SIZE,
            max_samples=Z_N_SAMPLES_PER_RASTER,
            rng=rng,
        )
        stats[(p, b)] = (m, s)

    print("[NORM] Z-score stats ready.")
    return stats


def build_predictor_lookup(predictor_dir):
    tif_paths = glob.glob(os.path.join(predictor_dir, "*.tif"))
    vrt_paths = glob.glob(os.path.join(predictor_dir, "*.vrt"))
    all_paths = tif_paths + vrt_paths

    by_norm = {}
    for p in all_paths:
        stem = normalize_name(os.path.basename(p))
        by_norm[stem] = p

    return by_norm, all_paths


def _match_single_predictor_path(pred_name, by_norm, predictor_paths, strict_lookup=None, predictor_dir=None):
    key = normalize_name(pred_name)

    if strict_lookup is not None:
        if key not in strict_lookup:
            raise RuntimeError(f"Missing predictor '{pred_name}' in predictor_file_list.csv")
        fn = strict_lookup[key]
        full = os.path.join(predictor_dir, fn)
        if not os.path.isfile(full):
            raise RuntimeError(f"Mapped file not found for predictor '{pred_name}': {full}")
        return full

    if key in by_norm:
        return by_norm[key]

    candidates = []
    for p in predictor_paths:
        stem = normalize_name(os.path.basename(p))
        if key in stem or stem in key:
            candidates.append(p)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(
            f"Ambiguous match for predictor '{pred_name}'. Candidates: {candidates}"
        )
    raise RuntimeError(f"Missing predictor '{pred_name}' in {predictor_dir}")


def match_predictor_files(predictor_dir, channel_map_csv, predictor_list_file=None, exclusion_tokens=None):
    channel_rows, df_map = load_channel_map(channel_map_csv)
    by_norm, predictor_paths = build_predictor_lookup(predictor_dir)

    dropped_rows = []
    if exclusion_tokens:
        tokens = [str(t).strip().lower() for t in exclusion_tokens if str(t).strip()]
        kept_rows = []
        for row in channel_rows:
            pname = str(row.get("predictor_name", ""))
            if any(tok in pname.lower() for tok in tokens):
                dropped_rows.append(pname)
                continue
            kept_rows.append(row)
        channel_rows = kept_rows

    strict_lookup = None
    if predictor_list_file and os.path.isfile(predictor_list_file):
        strict_df = pd.read_csv(predictor_list_file)
        required = {"predictor_name", "filename"}
        if not required.issubset(set(strict_df.columns)):
            raise ValueError("predictor_file_list.csv must contain columns: predictor_name,filename")

        strict_lookup = {
            normalize_name(r["predictor_name"]): r["filename"]
            for _, r in strict_df.iterrows()
        }

    ordered_specs = []
    ordered_names = []

    for row in channel_rows:
        pred_name = row["predictor_name"]
        matched_path = _match_single_predictor_path(
            pred_name,
            by_norm=by_norm,
            predictor_paths=predictor_paths,
            strict_lookup=strict_lookup,
            predictor_dir=predictor_dir,
        )

        spec = dict(row)
        spec["path"] = matched_path
        ordered_specs.append(spec)
        ordered_names.append(pred_name)

    return ordered_names, ordered_specs, dropped_rows


def apply_channel_exclusions(ordered_names, channel_specs, exclusion_tokens):
    if not exclusion_tokens:
        return ordered_names, channel_specs, []

    tokens = [str(t).strip().lower() for t in exclusion_tokens if str(t).strip()]
    keep_names = []
    keep_specs = []
    dropped = []

    for name, spec in zip(ordered_names, channel_specs):
        lname = str(name).lower()
        if any(tok in lname for tok in tokens):
            dropped.append((name, spec))
            continue
        keep_names.append(name)
        keep_specs.append(spec)

    return keep_names, keep_specs, dropped


def validate_raster_stack(channel_specs):
    if not channel_specs:
        raise ValueError("No channel specs provided.")

    unique_paths = []
    seen = set()
    for spec in channel_specs:
        p = spec["path"]
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    with rasterio.open(unique_paths[0]) as ref:
        width = ref.width
        height = ref.height
        transform = ref.transform
        crs = ref.crs
        profile = ref.profile.copy()

    for p in unique_paths[1:]:
        with rasterio.open(p) as ds:
            if ds.width != width or ds.height != height:
                dw = abs(ds.width - width)
                dh = abs(ds.height - height)
                if dw > SIZE_MISMATCH_TOL_PIXELS or dh > SIZE_MISMATCH_TOL_PIXELS:
                    raise ValueError(f"Raster size mismatch: {p}")
                print(
                    "[WARN] Minor raster size mismatch tolerated "
                    f"for {os.path.basename(p)}: {ds.width}x{ds.height} vs {width}x{height}"
                )
            if ds.transform != transform:
                raise ValueError(f"Raster transform mismatch: {p}")
            if ds.crs != crs:
                raise ValueError(f"Raster CRS mismatch: {p}")

    return width, height, transform, crs, profile


def open_predictor_datasets(channel_specs, ref_width, ref_height, ref_transform, ref_crs):
    unique_paths = []
    seen = set()
    for spec in channel_specs:
        p = spec["path"]
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    datasets = {}
    for p in unique_paths:
        ds = rasterio.open(p)
        if (
            ds.width != ref_width
            or ds.height != ref_height
            or ds.transform != ref_transform
            or ds.crs != ref_crs
        ):
            print(f"[INFO] Grid mismatch for {os.path.basename(p)}; using WarpedVRT to reference grid.")
            ds = WarpedVRT(
                ds,
                crs=ref_crs,
                transform=ref_transform,
                width=ref_width,
                height=ref_height,
                resampling=Resampling.bilinear,
            )
        datasets[p] = ds

    # validate requested band indices now, before long processing starts
    for spec in channel_specs:
        ds = datasets[spec["path"]]
        band = int(spec.get("band", 1))
        if band < 1 or band > ds.count:
            raise ValueError(
                f"Requested band {band} is out of range for predictor "
                f"'{spec['predictor_name']}' in file {spec['path']} (dataset has {ds.count} band(s))."
            )

    return datasets


def close_predictor_datasets(datasets):
    for ds in datasets.values():
        try:
            ds.close()
        except Exception:
            pass


def get_aoi_geometries(vector_path, layer=None):
    if vector_path is None:
        return None

    if not os.path.isfile(vector_path):
        raise FileNotFoundError(f"AOI vector not found: {vector_path}")

    geoms = []
    if layer:
        src = fiona.open(vector_path, layer=layer)
    else:
        src = fiona.open(vector_path)

    with src as shp:
        for feat in shp:
            geom = feat["geometry"]
            if geom is not None:
                geoms.append(geom)

    if len(geoms) == 0:
        raise ValueError(f"No valid geometries found in AOI: {vector_path}")
    return geoms


def reproject_geometries_to_crs(geoms, src_crs, dst_crs):
    if geoms is None:
        return None
    if src_crs is None or dst_crs is None:
        return geoms
    if str(src_crs) == str(dst_crs):
        return geoms

    reproj = []
    for geom in geoms:
        if geom is None:
            continue
        reproj.append(transform_geom(src_crs, dst_crs, geom, precision=6))
    return reproj




def _collect_geom_bounds_coords(coords, xs, ys):
    for item in coords:
        if isinstance(item[0], (float, int)):
            xs.append(item[0])
            ys.append(item[1])
        else:
            _collect_geom_bounds_coords(item, xs, ys)


def get_aoi_block_starts(geoms, transform, width, height, superblock_size):
    """Return row/col superblock starts limited to AOI bounding box.

    Keeps the precise per-block AOI mask test later, but drastically reduces
    the number of candidate blocks iterated over.
    """
    if geoms is None:
        return list(range(0, height, superblock_size)), list(range(0, width, superblock_size))

    xs, ys = [], []
    for g in geoms:
        if g is None:
            continue
        coords = g.get("coordinates")
        if coords is not None:
            _collect_geom_bounds_coords(coords, xs, ys)

    if not xs or not ys:
        return list(range(0, height, superblock_size)), list(range(0, width, superblock_size))

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    # rowcol expects (x, y). Use top-left and bottom-right style corners.
    r0, c0 = rasterio.transform.rowcol(transform, minx, maxy)
    r1, c1 = rasterio.transform.rowcol(transform, maxx, miny)

    rmin = max(0, min(r0, r1))
    rmax = min(height - 1, max(r0, r1))
    cmin = max(0, min(c0, c1))
    cmax = min(width - 1, max(c0, c1))

    # Snap outward to superblock boundaries and include one-buffer margin.
    row_start_min = max(0, (rmin // superblock_size) * superblock_size - superblock_size)
    row_start_max = min(height - 1, (rmax // superblock_size) * superblock_size + superblock_size)
    col_start_min = max(0, (cmin // superblock_size) * superblock_size - superblock_size)
    col_start_max = min(width - 1, (cmax // superblock_size) * superblock_size + superblock_size)

    row_starts = list(range(row_start_min, min(row_start_max + superblock_size, height), superblock_size))
    col_starts = list(range(col_start_min, min(col_start_max + superblock_size, width), superblock_size))

    return row_starts, col_starts


def apply_small_test_subset(row_starts, col_starts, geoms, transform, superblock_size):
    if not SMALL_TEST_MODE:
        return row_starts, col_starts

    if len(row_starts) == 0 or len(col_starts) == 0:
        return row_starts, col_starts

    if SMALL_TEST_CENTER_ROWCOL is not None:
        center_r, center_c = SMALL_TEST_CENTER_ROWCOL
    else:
        xs, ys = [], []
        if geoms is not None:
            for g in geoms:
                if g is None:
                    continue
                coords = g.get("coordinates")
                if coords is not None:
                    _collect_geom_bounds_coords(coords, xs, ys)
        if xs and ys:
            cx = 0.5 * (min(xs) + max(xs))
            cy = 0.5 * (min(ys) + max(ys))
            center_r, center_c = rasterio.transform.rowcol(transform, cx, cy)
        else:
            center_r = row_starts[len(row_starts) // 2]
            center_c = col_starts[len(col_starts) // 2]

    nearest_row = min(row_starts, key=lambda r: abs(r - center_r))
    nearest_col = min(col_starts, key=lambda c: abs(c - center_c))

    rmin = nearest_row - SMALL_TEST_RADIUS_BLOCKS * superblock_size
    rmax = nearest_row + SMALL_TEST_RADIUS_BLOCKS * superblock_size
    cmin = nearest_col - SMALL_TEST_RADIUS_BLOCKS * superblock_size
    cmax = nearest_col + SMALL_TEST_RADIUS_BLOCKS * superblock_size

    row_sub = [r for r in row_starts if rmin <= r <= rmax]
    col_sub = [c for c in col_starts if cmin <= c <= cmax]

    print(
        f"[INFO] SMALL_TEST_MODE active: radius={SMALL_TEST_RADIUS_BLOCKS}, "
        f"center_rowcol=({nearest_row},{nearest_col}), "
        f"subset={len(row_sub)} row starts x {len(col_sub)} col starts"
    )
    return row_sub, col_sub


def make_output_profile(ref_profile, dtype="float32", count=1, nodata=None, compress="lzw"):
    profile = ref_profile.copy()
    profile.update(
        dtype=dtype,
        count=count,
        compress=compress,
        tiled=True,
        BIGTIFF="IF_SAFER"
    )
    if nodata is not None:
        profile["nodata"] = nodata
    else:
        profile.pop("nodata", None)
    return profile


def create_output_rasters(base_profile, out_dir):
    outputs = {}

    pred_dir = ensure_dir(os.path.join(out_dir, PREDICTIONS_DIRNAME))
    unc_dir = ensure_dir(os.path.join(out_dir, UNCERTAINTY_DIRNAME))

    # class probability rasters
    prob_paths = []
    for k in range(NUM_CLASSES):
        path = os.path.join(pred_dir, f"{PROB_PREFIX}_{k+1}.tif")
        prof = make_output_profile(base_profile, dtype="float32", count=1, nodata=np.nan)
        prob_paths.append((path, rasterio.open(path, "w", **prof)))
    outputs["prob"] = prob_paths

    if WRITE_CLASS:
        path = os.path.join(pred_dir, CLASS_NAME)
        prof = make_output_profile(base_profile, dtype="uint8", count=1, nodata=0)
        outputs["class"] = (path, rasterio.open(path, "w", **prof))

    if WRITE_CONFIDENCE:
        path = os.path.join(pred_dir, CONFIDENCE_NAME)
        prof = make_output_profile(base_profile, dtype="float32", count=1, nodata=np.nan)
        outputs["confidence"] = (path, rasterio.open(path, "w", **prof))

    if WRITE_ENTROPY:
        path = os.path.join(unc_dir, ENTROPY_NAME)
        prof = make_output_profile(base_profile, dtype="float32", count=1, nodata=np.nan)
        outputs["entropy"] = (path, rasterio.open(path, "w", **prof))

    return outputs


def close_output_rasters(outputs):
    for item in outputs.get("prob", []):
        item[1].close()
    if "class" in outputs:
        outputs["class"][1].close()
    if "confidence" in outputs:
        outputs["confidence"][1].close()
    if "entropy" in outputs:
        outputs["entropy"][1].close()


def gaussian_like_weight(tile_size, overlap):
    # simple edge-tapering weight; avoids seams
    if overlap <= 0:
        return np.ones((tile_size, tile_size), dtype=np.float32)

    y = np.linspace(-1, 1, tile_size, dtype=np.float32)
    x = np.linspace(-1, 1, tile_size, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    w = np.ones((tile_size, tile_size), dtype=np.float32)

    edge = overlap / float(tile_size)
    dist_x = 1.0 - np.abs(xx)
    dist_y = 1.0 - np.abs(yy)
    taper_x = np.clip(dist_x / edge, 0.0, 1.0)
    taper_y = np.clip(dist_y / edge, 0.0, 1.0)
    w = np.minimum(taper_x, taper_y)

    w = np.clip(w, 1e-6, 1.0).astype(np.float32)
    return w


def compute_entropy(prob_block, eps=1e-7):
    p = np.clip(prob_block, eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1).astype(np.float32)


def infer_model_input_names(model):
    try:
        return [t.name.split(":")[0] for t in model.inputs]
    except Exception:
        return []


def load_model_auto(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model not found: {path}")

    print("\nLoading model...")
    print(f"[INFO] Model file extension: {os.path.splitext(path)[1].lower()}")
    print(f"[INFO] zip .keras archive: {is_zip_keras(path)}")
    print(f"[INFO] HDF5 model file: {is_hdf5_file(path)}")

    # TF 2.6 / older keras may not need this, but harmless if available
    try:
        keras.config.enable_unsafe_deserialization()
        print("[INFO] Enabled unsafe deserialization via keras.")
    except Exception:
        pass
    try:
        tf.keras.config.enable_unsafe_deserialization()
        print("[INFO] Enabled unsafe deserialization via tf.keras.")
    except Exception:
        pass

    # 1) native load attempt
    try:
        model = tf.keras.models.load_model(path, compile=False)
        return model
    except Exception as e1:
        print(f"[WARN] Native load failed: {e1}")

    # 2) if content is HDF5 but file extension is .keras, copy to temp .h5
    if is_hdf5_file(path):
        tmp_dir = tempfile.mkdtemp(prefix="keras_h5_fix_")
        tmp_h5 = os.path.join(tmp_dir, "temp_model.h5")
        shutil.copy2(path, tmp_h5)
        try:
            model = tf.keras.models.load_model(tmp_h5, compile=False)
            return model
        finally:
            # leave temp_dir cleanup to OS if file still in use
            pass

    raise RuntimeError(f"Failed to load model: {path}")


def read_block_stack(datasets, channel_specs, row0, row1, col0, col1, zscore_stats=None):
    h = row1 - row0
    w = col1 - col0
    block = np.zeros((h, w, len(channel_specs)), dtype=np.float32)

    win = Window(col0, row0, w, h)
    for i, spec in enumerate(channel_specs):
        p = spec["path"]
        ds = datasets[p]
        band = int(spec.get("band", 1))
        arr = ds.read(band, window=win)
        arr = arr.astype(np.float32)

        nod = ds.nodata
        if nod is not None:
            arr[arr == nod] = np.nan

        if zscore_stats is not None:
            z = zscore_stats.get((p, band))
            if z is not None:
                mean, std = z
                arr = (arr - mean) / std

        block[:, :, i] = arr

    return block


def extract_aoi_mask_for_block(geoms, transform, row0, row1, col0, col1, height, width):
    if geoms is None:
        return np.ones((row1-row0, col1-col0), dtype=bool)

    win = Window(col0, row0, col1-col0, row1-row0)
    win_transform = rasterio.windows.transform(win, transform)

    mask = geometry_mask(
        geoms,
        out_shape=(row1-row0, col1-col0),
        transform=win_transform,
        invert=True,
        all_touched=False
    )
    return mask


def split_inputs_for_model(block_stack, use_embeddings, quaternary_idx=None, landuse_idx=None):
    if not use_embeddings:
        return block_stack.astype(np.float32), None, None

    if quaternary_idx is None or landuse_idx is None:
        raise RuntimeError("Embedding mode requires quaternary_idx and landuse_idx.")

    # continuous branch excludes the categorical inputs
    n_bands = block_stack.shape[-1]
    keep = [i for i in range(n_bands) if i not in (quaternary_idx, landuse_idx)]
    cont = block_stack[:, :, keep].astype(np.float32)

    qua = block_stack[:, :, quaternary_idx]
    land = block_stack[:, :, landuse_idx]

    # fill NaN, round and clip categorical indices to match training-time cleaning
    qua = np.rint(np.nan_to_num(qua, nan=0.0)).astype(np.int32)
    land = np.rint(np.nan_to_num(land, nan=0.0)).astype(np.int32)
    qua = np.where((qua < 0) | (qua > QUATERNARY_NUM_CLASSES), 0, qua)
    land = np.where((land < 0) | (land > LANDUSE_NUM_CLASSES), 0, land)

    # expected input shape often (B,H,W,1)
    qua = qua[..., np.newaxis]
    land = land[..., np.newaxis]

    return cont, qua, land


def make_tile_origins(block_h, block_w, tile_size, overlap):
    step = tile_size - overlap
    if step <= 0:
        raise ValueError("OVERLAP must be smaller than TILE_SIZE")

    rows = list(range(0, max(block_h - tile_size + 1, 1), step))
    cols = list(range(0, max(block_w - tile_size + 1, 1), step))

    if len(rows) == 0:
        rows = [0]
    if len(cols) == 0:
        cols = [0]

    if rows[-1] != max(block_h - tile_size, 0):
        rows.append(max(block_h - tile_size, 0))
    if cols[-1] != max(block_w - tile_size, 0):
        cols.append(max(block_w - tile_size, 0))

    return rows, cols


def predict_block(
    model,
    block_stack,
    block_aoi_mask,
    use_embeddings,
    quaternary_idx=None,
    landuse_idx=None,
    tile_size=TILE_SIZE,
    overlap=OVERLAP,
    batch_size=BATCH_SIZE,
):
    block_h, block_w, n_bands = block_stack.shape

    prob_sum = np.zeros((block_h, block_w, NUM_CLASSES), dtype=np.float32)
    weight_sum = np.zeros((block_h, block_w), dtype=np.float32)

    rows, cols = make_tile_origins(block_h, block_w, tile_size, overlap)
    weight = gaussian_like_weight(tile_size, overlap)

    batch_cont = []
    batch_qua = []
    batch_land = []
    batch_meta = []

    def flush_batch():
        nonlocal prob_sum, weight_sum, batch_cont, batch_qua, batch_land, batch_meta
        if len(batch_meta) == 0:
            return

        x_cont = np.stack(batch_cont, axis=0)
        if use_embeddings:
            x_qua = np.stack(batch_qua, axis=0)
            x_land = np.stack(batch_land, axis=0)
            preds = model.predict(
                {
                    "cont_in": x_cont,
                    "quaternary_in": x_qua,
                    "landuse_in": x_land,
                },
                batch_size=len(batch_meta),
                verbose=0,
            )
        else:
            preds = model.predict(
                x_cont,
                batch_size=len(batch_meta),
                verbose=0,
            )

        preds = np.asarray(preds, dtype=np.float32)

        for i, (r, c, valid_mask) in enumerate(batch_meta):
            p = preds[i]  # (tile, tile, classes)
            w = weight.copy()

            # restrict to AOI + valid pixels
            local_mask = valid_mask.astype(np.float32)
            w = w * local_mask

            prob_sum[r:r+tile_size, c:c+tile_size, :] += p * w[..., np.newaxis]
            weight_sum[r:r+tile_size, c:c+tile_size] += w

        batch_cont = []
        batch_qua = []
        batch_land = []
        batch_meta = []

    for r in rows:
        for c in cols:
            tile = block_stack[r:r+tile_size, c:c+tile_size, :]
            tile_mask = block_aoi_mask[r:r+tile_size, c:c+tile_size]

            # require full tile shape
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                continue

            # skip tile entirely if AOI absent there
            if not np.any(tile_mask):
                continue

            # identify pixels with all finite continuous/categorical values
            valid_data = np.all(np.isfinite(tile), axis=-1)

            # only predict if there are any valid pixels inside AOI
            valid_mask = (valid_data & tile_mask)
            if not np.any(valid_mask):
                continue

            tile_filled = np.nan_to_num(tile, nan=NODATA_FILL_VALUE).astype(np.float32)
            cont, qua, land = split_inputs_for_model(
                tile_filled,
                use_embeddings=use_embeddings,
                quaternary_idx=quaternary_idx,
                landuse_idx=landuse_idx,
            )

            batch_cont.append(cont)
            if use_embeddings:
                batch_qua.append(qua)
                batch_land.append(land)
            batch_meta.append((r, c, valid_mask))

            if len(batch_meta) >= batch_size:
                flush_batch()

    flush_batch()

    out_prob = np.full((block_h, block_w, NUM_CLASSES), np.nan, dtype=np.float32)
    ok = weight_sum > 0
    out_prob[ok, :] = prob_sum[ok, :] / weight_sum[ok, np.newaxis]

    return out_prob


def write_block_outputs(outputs, row0, col0, prob_block, out_row0=0, out_col0=0):
    block_h, block_w, _ = prob_block.shape
    win = Window(col0 - out_col0, row0 - out_row0, block_w, block_h)

    # probs
    for k, (_, ds) in enumerate(outputs["prob"]):
        arr = prob_block[:, :, k].astype(np.float32)
        ds.write(arr, 1, window=win)

    # class
    if "class" in outputs:
        class_arr = np.zeros((block_h, block_w), dtype=np.uint8)
        valid = np.all(np.isfinite(prob_block), axis=-1)
        if np.any(valid):
            class_arr[valid] = np.argmax(prob_block[valid], axis=-1).astype(np.uint8) + 1
        outputs["class"][1].write(class_arr, 1, window=win)

    # confidence
    if "confidence" in outputs:
        conf = np.full((block_h, block_w), np.nan, dtype=np.float32)
        valid = np.all(np.isfinite(prob_block), axis=-1)
        if np.any(valid):
            conf[valid] = np.max(prob_block[valid], axis=-1).astype(np.float32)
        outputs["confidence"][1].write(conf, 1, window=win)

    # entropy
    if "entropy" in outputs:
        ent = np.full((block_h, block_w), np.nan, dtype=np.float32)
        valid = np.all(np.isfinite(prob_block), axis=-1)
        if np.any(valid):
            ent[valid] = compute_entropy(prob_block[valid])
        outputs["entropy"][1].write(ent, 1, window=win)


def gather_output_paths(outputs):
    out = []
    for path, _ in outputs.get("prob", []):
        out.append(path)
    if "class" in outputs:
        out.append(outputs["class"][0])
    if "confidence" in outputs:
        out.append(outputs["confidence"][0])
    if "entropy" in outputs:
        out.append(outputs["entropy"][0])
    return out


def convert_tifs_to_cog(tif_paths):
    if not tif_paths:
        return

    print("\n[COG] Converting output GeoTIFFs to Cloud-Optimized GeoTIFF...")
    iterator = tif_paths
    if tqdm is not None:
        iterator = tqdm(tif_paths, total=len(tif_paths), desc="COG", unit="file", dynamic_ncols=True)

    for src in iterator:
        tmp_cog = src + ".cog.tmp.tif"
        rio_shutil.copy(
            src,
            tmp_cog,
            driver="COG",
            compress=COG_COMPRESS,
            blocksize=COG_BLOCKSIZE,
            overview_resampling="nearest",
            BIGTIFF="IF_SAFER",
        )
        os.replace(tmp_cog, src)


def summarize_class_counts(class_raster_path):
    if not os.path.isfile(class_raster_path):
        print(f"[SUMMARY] Class raster not found: {class_raster_path}")
        return

    with rasterio.open(class_raster_path) as ds:
        arr = ds.read(1)

    unique, counts = np.unique(arr, return_counts=True)
    pairs = [(int(u), int(c)) for u, c in zip(unique, counts)]
    total = int(arr.size)

    print("\n[SUMMARY] Predicted class pixel counts")
    print(f"[SUMMARY] Total pixels: {total}")
    for cls, cnt in pairs:
        pct = (100.0 * cnt / total) if total > 0 else 0.0
        print(f"[SUMMARY] Class {cls}: {cnt} ({pct:.2f}%)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n=== PATHS ===")
    print(f"PREDICTOR_DIR: {PREDICTOR_DIR}")
    print(f"MODEL_OUT:     {MODEL_OUT}")
    print(f"PRED_OUT:      {PRED_OUT}")
    print(f"MODEL:         {BEST_MODEL_PATH}")
    print(f"CHANNEL_MAP:   {CHANNEL_MAP_CSV}")

    ensure_dir(PRED_OUT)
    run_output_dir = PRED_OUT
    if SMALL_TEST_MODE:
        run_output_dir = ensure_dir(os.path.join(PRED_OUT, SMALL_TEST_OUTPUT_SUBDIR))
        print(f"[INFO] Small-test output dir: {run_output_dir}")
    patch_alphaearth_vrt_if_requested()

    ordered_names, channel_specs, dropped_rows = match_predictor_files(
        PREDICTOR_DIR,
        CHANNEL_MAP_CSV,
        predictor_list_file=PREDICTOR_LIST_FILE,
        exclusion_tokens=CHANNELS_EXCL,
    )

    width, height, transform, crs, ref_profile = validate_raster_stack(channel_specs)
    print(f"\nRaster stack validated: {len(channel_specs)} channels after exclusions | size={width} x {height}")
    if dropped_rows:
        print(f"[INFO] Excluded channels by token matching: {len(dropped_rows)}")
    print("Predictor order used for production:")
    dup_counter = {}
    for spec in channel_specs:
        key = (spec["path"], int(spec.get("band", 1)))
        dup_counter[key] = dup_counter.get(key, 0) + 1
    for i, spec in enumerate(channel_specs):
        print(f"  {i:03d} | {spec['predictor_name']} | band {spec.get('band', 1)} | {os.path.basename(spec['path'])}")

    q_idx = None
    l_idx = None
    if USE_CATEGORICAL_EMBEDDINGS:
        for i, nm in enumerate(ordered_names):
            if normalize_name(nm) == normalize_name(QUATERNARY_NAME):
                q_idx = i
            if normalize_name(nm) == normalize_name(LANDUSE_NAME):
                l_idx = i

        if q_idx is None:
            raise RuntimeError(f"Could not find quaternary channel '{QUATERNARY_NAME}' in channel_map.")
        if l_idx is None:
            raise RuntimeError(f"Could not find landuse channel '{LANDUSE_NAME}' in channel_map.")

    print("\nDerived channels dropped: None")
    print(f"Excluded channels: {len(dropped_rows)}")
    print(f"Embedding mode (from config): {USE_CATEGORICAL_EMBEDDINGS}")
    if USE_CATEGORICAL_EMBEDDINGS:
        cont_count = len(ordered_names) - 2
        print(f"Quaternary channel idx: {q_idx}")
        print(f"Landuse channel idx:    {l_idx}")
        print(f"Continuous branch channel count after preprocessing: {cont_count}")
    else:
        print(f"Continuous branch channel count after preprocessing: {len(ordered_names)}")

    model = load_model_auto(BEST_MODEL_PATH)
    input_names = infer_model_input_names(model)
    print(f"Model inputs detected: {len(model.inputs)}")
    if input_names:
        print(f"Model input names: {input_names}")

    model_expects_embeddings = len(model.inputs) == 3
    if model_expects_embeddings != USE_CATEGORICAL_EMBEDDINGS:
        raise RuntimeError(
            "Model/input mismatch: script USE_CATEGORICAL_EMBEDDINGS "
            f"is {USE_CATEGORICAL_EMBEDDINGS}, but loaded model expects "
            f"{len(model.inputs)} input tensor(s)."
        )

    geoms = get_aoi_geometries(AOI_VECTOR_PATH, AOI_LAYER)
    aoi_crs = None
    if AOI_VECTOR_PATH is not None and os.path.isfile(AOI_VECTOR_PATH):
        if AOI_LAYER is not None:
            with fiona.open(AOI_VECTOR_PATH, layer=AOI_LAYER) as shp:
                aoi_crs = shp.crs
        else:
            with fiona.open(AOI_VECTOR_PATH) as shp:
                aoi_crs = shp.crs
    geoms = reproject_geometries_to_crs(geoms, aoi_crs, crs)
    if geoms is None:
        print("[INFO] No AOI polygon provided. Full extent will be processed.")
    else:
        print(f"[INFO] AOI loaded from: {AOI_VECTOR_PATH}")
        if aoi_crs is not None:
            print(f"[INFO] AOI reprojected from {aoi_crs} to {crs}")

    outputs = {}
    output_paths = []
    datasets = {}
    zscore_stats = None
    out_row0 = 0
    out_col0 = 0

    try:
        halo = OVERLAP
        row_starts, col_starts = get_aoi_block_starts(
            geoms, transform, width, height, SUPERBLOCK_SIZE
        )
        row_starts, col_starts = apply_small_test_subset(
            row_starts,
            col_starts,
            geoms,
            transform,
            SUPERBLOCK_SIZE,
        )

        # In small test mode, optionally crop output rasters to selected core extent.
        output_profile = ref_profile.copy()
        if SMALL_TEST_MODE and SMALL_TEST_CROP_OUTPUT and row_starts and col_starts:
            out_row0 = min(row_starts)
            out_col0 = min(col_starts)
            out_row1 = max(min(r + SUPERBLOCK_SIZE, height) for r in row_starts)
            out_col1 = max(min(c + SUPERBLOCK_SIZE, width) for c in col_starts)
            out_h = out_row1 - out_row0
            out_w = out_col1 - out_col0
            out_transform = rasterio.windows.transform(Window(out_col0, out_row0, out_w, out_h), transform)
            output_profile.update(width=out_w, height=out_h, transform=out_transform)
            print(
                f"[INFO] SMALL_TEST_CROP_OUTPUT active: output window rows {out_row0}:{out_row1}, "
                f"cols {out_col0}:{out_col1}, size={out_w}x{out_h}"
            )

        outputs = create_output_rasters(output_profile, run_output_dir)
        output_paths = gather_output_paths(outputs)
        datasets = open_predictor_datasets(channel_specs, width, height, transform, crs)
        zscore_stats = build_zscore_stats(datasets, channel_specs)

        total_blocks = len(row_starts) * len(col_starts)
        block_counter = 0

        if geoms is not None:
            print(
                f"[INFO] AOI-limited candidate superblocks: {total_blocks} "
                f"({len(row_starts)} row starts x {len(col_starts)} col starts)"
            )

        block_pairs = product(row_starts, col_starts)
        if tqdm is not None:
            block_pairs = tqdm(
                block_pairs,
                total=total_blocks,
                desc="Superblocks",
                unit="block",
                dynamic_ncols=True,
            )

        for row0_core, col0_core in block_pairs:
            row1_core = min(row0_core + SUPERBLOCK_SIZE, height)
            col1_core = min(col0_core + SUPERBLOCK_SIZE, width)
            block_counter += 1

            # expanded read window with halo
            row0 = max(0, row0_core - halo)
            row1 = min(height, row1_core + halo)
            col0 = max(0, col0_core - halo)
            col1 = min(width, col1_core + halo)

            if tqdm is None:
                print(
                    f"\n[BLOCK {block_counter}/{total_blocks}] "
                    f"read rows {row0}:{row1}, cols {col0}:{col1} | "
                    f"core rows {row0_core}:{row1_core}, cols {col0_core}:{col1_core}"
                )
            else:
                block_pairs.set_postfix_str(f"core r{row0_core}:{row1_core}, c{col0_core}:{col1_core}")

            block_stack = read_block_stack(
                datasets,
                channel_specs,
                row0,
                row1,
                col0,
                col1,
                zscore_stats=zscore_stats,
            )
            block_aoi_mask = extract_aoi_mask_for_block(
                geoms, transform, row0, row1, col0, col1, height, width
            )

            # skip block if AOI absent
            if not np.any(block_aoi_mask):
                if tqdm is None:
                    print("[INFO] Skipping block: outside AOI.")
                continue

            # skip block if no finite data
            if not np.any(np.isfinite(block_stack)):
                if tqdm is None:
                    print("[INFO] Skipping block: no valid raster data.")
                continue

            prob_block = predict_block(
                model=model,
                block_stack=block_stack,
                block_aoi_mask=block_aoi_mask,
                use_embeddings=USE_CATEGORICAL_EMBEDDINGS,
                quaternary_idx=q_idx,
                landuse_idx=l_idx,
                tile_size=TILE_SIZE,
                overlap=OVERLAP,
                batch_size=BATCH_SIZE,
            )

            # crop halo back to core
            top = row0_core - row0
            bottom = top + (row1_core - row0_core)
            left = col0_core - col0
            right = left + (col1_core - col0_core)

            prob_core = prob_block[top:bottom, left:right, :]

            write_block_outputs(
                outputs=outputs,
                row0=row0_core,
                col0=col0_core,
                prob_block=prob_core,
                out_row0=out_row0,
                out_col0=out_col0,
            )

        print("\n✔ Finished blockwise prediction.")

    finally:
        close_predictor_datasets(datasets)
        close_output_rasters(outputs)

    if WRITE_COG:
        convert_tifs_to_cog(output_paths)
        print("[COG] Conversion completed.")

    class_path = os.path.join(run_output_dir, PREDICTIONS_DIRNAME, CLASS_NAME)
    summarize_class_counts(class_path)

    print("\nOutputs written to:")
    print(os.path.join(run_output_dir, PREDICTIONS_DIRNAME))
    print(os.path.join(run_output_dir, UNCERTAINTY_DIRNAME))


if __name__ == "__main__":
    main()