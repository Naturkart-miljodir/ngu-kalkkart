"""
Build optimized regression matrix for Kalk RF workflow.
- Loads predictors and training polygons
- Rasterizes labels
- Extracts predictor values for labeled pixels using block/window reads
- Saves regression matrix to Modelling/RandForest/Regression
"""

import os
import re
import glob
import time
import shutil
import numpy as np
from tqdm import tqdm
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from rasterio.features import rasterize
import geopandas as gpd


def windows_path_to_wsl(path_text):
    """
    Convert Windows path to WSL mount path when needed.

    :param path_text: Input filesystem path.
    :return: WSL-compatible path for non-Windows OS.
    """
    normalized = path_text.replace("\\", "/")
    if len(normalized) >= 2 and normalized[1] == ":":
        drive_letter = normalized[0].lower()
        return f"/mnt/{drive_letter}{normalized[2:]}"
    return normalized


def make_vrt_absolute(vrt_path, source_dir, out_dir):
    """
    Create a VRT copy with absolute source file paths.

    :param vrt_path: Path to source VRT.
    :param source_dir: Directory containing source raster tiles.
    :param out_dir: Directory where patched VRT is written.
    :return: Path to patched VRT.
    """
    with open(vrt_path, "r", encoding="utf-8") as file_obj:
        txt = file_obj.read()

    def repl(match_obj):
        open_tag, old_path, close_tag = match_obj.groups()
        new_abs = os.path.join(source_dir, os.path.basename(old_path)).replace("\\", "/")
        open_tag = re.sub(r'relativeToVRT="[^"]*"', 'relativeToVRT="0"', open_tag)
        return f"{open_tag}{new_abs}{close_tag}"

    patched = re.sub(r"(<SourceFilename[^>]*>)([^<]+)(</SourceFilename>)", repl, txt)
    os.makedirs(out_dir, exist_ok=True)
    out_vrt = os.path.join(out_dir, os.path.basename(vrt_path))
    with open(out_vrt, "w", encoding="utf-8") as file_obj:
        file_obj.write(patched)
    return out_vrt


def assert_same_grid(ref_ds, other_ds, ref_name, other_name):
    """
    Validate that two rasters share identical grid definition.

    :param ref_ds: Reference raster dataset.
    :param other_ds: Candidate raster dataset.
    :param ref_name: Name for reference raster in error messages.
    :param other_name: Name for candidate raster in error messages.
    :return: None
    """
    if ref_ds.crs != other_ds.crs:
        raise RuntimeError(f"CRS mismatch: {ref_name}={ref_ds.crs} vs {other_name}={other_ds.crs}")
    if ref_ds.transform != other_ds.transform:
        raise RuntimeError(f"Transform mismatch: {ref_name} vs {other_name}")
    if ref_ds.width != other_ds.width or ref_ds.height != other_ds.height:
        raise RuntimeError(f"Dimensions mismatch: {ref_name} vs {other_name}")


def dataset_var_names(dataset_paths, datasets):
    """
    Build predictor names, expanding multi-band rasters to per-band names.

    :param dataset_paths: Predictor file paths in load order.
    :param datasets: Opened raster datasets in the same order.
    :return: List of predictor names.
    """
    names = []
    for path, ds in zip(dataset_paths, datasets):
        base_name = os.path.splitext(os.path.basename(path))[0]
        if ds.count == 1:
            names.append(base_name)
        else:
            for band_idx in range(1, ds.count + 1):
                names.append(f"{base_name}_b{band_idx:02d}")
    return names


def nodata_mask_for_dataset(data_3d, nodata_values):
    """
    Compute valid mask for one predictor dataset block.

    :param data_3d: Predictor block with shape (bands, h, w).
    :param nodata_values: Per-band nodata values from rasterio.
    :return: Boolean mask of shape (h, w), True where all bands are valid.
    """
    valid = np.all(np.isfinite(data_3d), axis=0)
    for band_idx, band_nodata in enumerate(nodata_values):
        if band_nodata is None:
            continue
        band_arr = data_3d[band_idx]
        if np.isnan(band_nodata):
            valid &= ~np.isnan(band_arr)
        else:
            valid &= band_arr != band_nodata
    return valid


PROJECT_DIR_WINDOWS = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project"

if os.name == "nt":
    PROJECT_DIR = PROJECT_DIR_WINDOWS
else:
    PROJECT_DIR = windows_path_to_wsl(PROJECT_DIR_WINDOWS)

PREDICTOR_DIR = r"G:\Covariates_to_model"
POLYGON_GPKG = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2025\kalk_prosjekt3.0\MDir_data\Data_2026\kalkkart_treningsdata_ed_June2026.gpkg"
POLYGON_LAYER = "NiN_all_agg"
REF_MASK = r"G:\Covariates_to_model\Topo_dtm_ch.tif"
OUT_DIR = r"D:\Classical_ML"
MATRIX_PATH = os.getenv(
    "RF_MATRIX_PATH",
    os.path.join(OUT_DIR, "regression_matrix_2026_total_with_clean_ed_2.npz"),
)
ALPHAEARTH_VRT_NAME = "alphaearth_dequant_national_epsg25833.vrt"
ALPHAEARTH_DIR = r"E:\Alpha_earth\dequant_images_all"

LABEL_FIELD = "KA_mean_weighted_category"
STATUS_FIELD = "KA_mean_weighted_category_status"
BACKGROUND_VALUE = 0
WINDOW_SIZE = int(os.getenv("RF_WINDOW_SIZE", "1024"))
ETA_EVERY_WINDOWS = int(os.getenv("RF_ETA_EVERY_WINDOWS", "25"))
RANDOM_SEED = 42
_max_samples_raw = os.getenv("RF_MAX_SAMPLES_TOTAL", "").strip()
MAX_SAMPLES_TOTAL = int(_max_samples_raw) if _max_samples_raw else None
KEEP_TMP_ARRAYS = os.getenv("RF_KEEP_TMP_ARRAYS", "0").strip() == "1"

np.random.seed(RANDOM_SEED)
os.makedirs(OUT_DIR, exist_ok=True)

with rasterio.open(REF_MASK) as ref_ds:
    width, height = ref_ds.width, ref_ds.height
    transform = ref_ds.transform
    crs = ref_ds.crs

print(f"Using snap raster: {REF_MASK}")
print(f"Dimensions: {width}x{height}, CRS: {crs}")

raster_paths = sorted(
    glob.glob(os.path.join(PREDICTOR_DIR, "*.tif"))
    + glob.glob(os.path.join(PREDICTOR_DIR, "*.vrt"))
)

# Avoid double-loading AlphaEarth when a pre-patched VRT is present in PREDICTOR_DIR.
raster_paths = [p for p in raster_paths if not p.lower().endswith("_absolute_paths.vrt")]

alphaearth_vrt = os.path.join(PREDICTOR_DIR, ALPHAEARTH_VRT_NAME)
if os.path.exists(alphaearth_vrt):
    tmp_vrt_dir = os.path.join(OUT_DIR, "_tmp_vrt")
    patched_alphaearth_vrt = make_vrt_absolute(alphaearth_vrt, ALPHAEARTH_DIR, tmp_vrt_dir)
    raster_paths = [patched_alphaearth_vrt if path == alphaearth_vrt else path for path in raster_paths]

if not raster_paths:
    raise FileNotFoundError(f"No predictor rasters found in: {PREDICTOR_DIR}")

datasets = []
with rasterio.open(REF_MASK) as ref_ds:
    for path in raster_paths:
        ds = rasterio.open(path)
        try:
            assert_same_grid(ref_ds, ds, "REF_MASK", os.path.basename(path))
        except RuntimeError:
            print(f"Grid mismatch for {os.path.basename(path)}; using WarpedVRT to REF grid.")
            ds = WarpedVRT(
                ds,
                crs=ref_ds.crs,
                transform=ref_ds.transform,
                width=ref_ds.width,
                height=ref_ds.height,
                resampling=Resampling.bilinear,
            )
        datasets.append(ds)

var_names = dataset_var_names(raster_paths, datasets)
print(f"Loaded {len(var_names)} predictors")

if POLYGON_LAYER:
    gdf = gpd.read_file(POLYGON_GPKG, layer=POLYGON_LAYER)
else:
    gdf = gpd.read_file(POLYGON_GPKG)
if gdf.crs != crs:
    gdf = gdf.to_crs(crs)

class_order = ["low", "medium", "high"]
class_map = {cls: idx + 1 for idx, cls in enumerate(class_order)}
status_order = ["clean", "dirty"]
status_map = {cls: idx + 1 for idx, cls in enumerate(status_order)}

if LABEL_FIELD not in gdf.columns:
    raise KeyError(f"Label field '{LABEL_FIELD}' not found in: {POLYGON_GPKG}")
if STATUS_FIELD not in gdf.columns:
    raise KeyError(f"Status field '{STATUS_FIELD}' not found in: {POLYGON_GPKG}")

gdf = gdf[gdf.geometry.notnull()]
gdf = gdf[~gdf.geometry.is_empty]
gdf = gdf[gdf[LABEL_FIELD].isin(class_order)]
gdf = gdf[gdf[STATUS_FIELD].isin(status_order)].copy()
gdf["label"] = gdf[LABEL_FIELD].map(class_map).astype(np.uint8)
gdf["status"] = gdf[STATUS_FIELD].map(status_map).astype(np.uint8)

label_shapes = [(geom, int(lbl)) for geom, lbl in zip(gdf.geometry, gdf["label"])]
status_shapes = [(geom, int(st)) for geom, st in zip(gdf.geometry, gdf["status"])]

y_raster = rasterize(
    shapes=label_shapes,
    out_shape=(height, width),
    transform=transform,
    fill=BACKGROUND_VALUE,
    dtype=np.uint8,
)
y_status_raster = rasterize(
    shapes=status_shapes,
    out_shape=(height, width),
    transform=transform,
    fill=BACKGROUND_VALUE,
    dtype=np.uint8,
)

total_labeled = int((y_raster > 0).sum())
if total_labeled == 0:
    raise RuntimeError("No labeled pixels found after rasterization.")

print("Rasterized polygons to labels")
print("Extracting regression matrix pixel data (window-based)...")
print(f"Labeled pixels total: {total_labeled}")

tmp_chunks_dir = os.path.join(OUT_DIR, f"_tmp_chunks_{int(time.time())}")
tmp_arrays_dir = os.path.join(OUT_DIR, f"_tmp_arrays_{int(time.time())}")
os.makedirs(tmp_chunks_dir, exist_ok=True)
os.makedirs(tmp_arrays_dir, exist_ok=True)

chunk_files = []

window_specs = []
for row_off in range(0, height, WINDOW_SIZE):
    for col_off in range(0, width, WINDOW_SIZE):
        win_h = min(WINDOW_SIZE, height - row_off)
        win_w = min(WINDOW_SIZE, width - col_off)
        window_specs.append((row_off, col_off, win_h, win_w))

start_time = time.time()
processed_labeled = 0
valid_collected = 0
chunk_idx = 0

for win_idx, (row_off, col_off, win_h, win_w) in enumerate(tqdm(window_specs), start=1):
    y_block = y_raster[row_off : row_off + win_h, col_off : col_off + win_w]
    y_status_block = y_status_raster[row_off : row_off + win_h, col_off : col_off + win_w]
    labeled_mask = y_block > 0
    labeled_count = int(labeled_mask.sum())

    if labeled_count == 0:
        continue

    processed_labeled += labeled_count

    predictor_blocks = []
    all_valid = labeled_mask.copy()
    window = Window(col_off, row_off, win_w, win_h)

    for ds in datasets:
        with np.errstate(over="ignore", invalid="ignore"):
            block = ds.read(window=window).astype(np.float32)
        predictor_blocks.append(block)
        all_valid &= nodata_mask_for_dataset(block, ds.nodatavals)

    valid_mask = all_valid & (y_status_block > 0)
    valid_count = int(valid_mask.sum())

    if valid_count == 0:
        continue

    stacked = np.concatenate(predictor_blocks, axis=0)
    rr, cc = np.where(valid_mask)

    x_chunk = stacked[:, rr, cc].T.astype(np.float32, copy=False)
    y_chunk = y_block[rr, cc].astype(np.int16, copy=False)
    y_status_chunk = y_status_block[rr, cc].astype(np.int16, copy=False)
    row_chunk = (rr + row_off).astype(np.int32, copy=False)
    col_chunk = (cc + col_off).astype(np.int32, copy=False)

    if MAX_SAMPLES_TOTAL is not None:
        remaining = MAX_SAMPLES_TOTAL - valid_collected
        if remaining <= 0:
            break
        if x_chunk.shape[0] > remaining:
            select_idx = np.random.choice(x_chunk.shape[0], size=remaining, replace=False)
            x_chunk = x_chunk[select_idx]
            y_chunk = y_chunk[select_idx]
            y_status_chunk = y_status_chunk[select_idx]
            row_chunk = row_chunk[select_idx]
            col_chunk = col_chunk[select_idx]

    chunk_path = os.path.join(tmp_chunks_dir, f"chunk_{chunk_idx:06d}.npz")
    np.savez_compressed(
        chunk_path,
        X=x_chunk,
        y=y_chunk,
        y_status=y_status_chunk,
        rows=row_chunk,
        cols=col_chunk,
    )
    chunk_files.append(chunk_path)
    chunk_idx += 1

    valid_collected += x_chunk.shape[0]

    if win_idx % ETA_EVERY_WINDOWS == 0:
        elapsed = time.time() - start_time
        speed = processed_labeled / elapsed if elapsed > 0 else 0.0
        remaining_labeled = max(total_labeled - processed_labeled, 0)
        eta_sec = remaining_labeled / speed if speed > 0 else float("inf")
        eta_hours = eta_sec / 3600 if np.isfinite(eta_sec) else float("inf")
        print(
            f"[ETA] windows={win_idx}/{len(window_specs)} processed_labeled={processed_labeled}/{total_labeled} valid={valid_collected} speed={speed:.1f} px/s eta={eta_hours:.2f}h"
        )

    if MAX_SAMPLES_TOTAL is not None and valid_collected >= MAX_SAMPLES_TOTAL:
        break

if not chunk_files:
    for ds in datasets:
        ds.close()
    shutil.rmtree(tmp_chunks_dir, ignore_errors=True)
    shutil.rmtree(tmp_arrays_dir, ignore_errors=True)
    raise RuntimeError("No valid labeled samples found. Check raster alignment and NoData values.")

total_samples = valid_collected
num_features = len(var_names)

X_memmap_path = os.path.join(tmp_arrays_dir, "X_all.npy")
y_memmap_path = os.path.join(tmp_arrays_dir, "y_all.npy")
y_status_memmap_path = os.path.join(tmp_arrays_dir, "y_status_all.npy")
rows_memmap_path = os.path.join(tmp_arrays_dir, "rows_all.npy")
cols_memmap_path = os.path.join(tmp_arrays_dir, "cols_all.npy")

X_all = np.lib.format.open_memmap(
    X_memmap_path, mode="w+", dtype=np.float32, shape=(total_samples, num_features)
)
y_all = np.lib.format.open_memmap(y_memmap_path, mode="w+", dtype=np.int16, shape=(total_samples,))
y_status_all = np.lib.format.open_memmap(
    y_status_memmap_path, mode="w+", dtype=np.int16, shape=(total_samples,)
)
rows_arr = np.lib.format.open_memmap(rows_memmap_path, mode="w+", dtype=np.int32, shape=(total_samples,))
cols_arr = np.lib.format.open_memmap(cols_memmap_path, mode="w+", dtype=np.int32, shape=(total_samples,))

write_offset = 0
for chunk_path in tqdm(chunk_files, desc="Assembling memmaps"):
    with np.load(chunk_path, allow_pickle=False) as chunk:
        n = chunk["y"].shape[0]
        next_offset = write_offset + n
        X_all[write_offset:next_offset, :] = chunk["X"]
        y_all[write_offset:next_offset] = chunk["y"]
        y_status_all[write_offset:next_offset] = chunk["y_status"]
        rows_arr[write_offset:next_offset] = chunk["rows"]
        cols_arr[write_offset:next_offset] = chunk["cols"]
        write_offset = next_offset

if write_offset != total_samples:
    for ds in datasets:
        ds.close()
    raise RuntimeError(
        f"Assembled sample count mismatch: expected {total_samples}, got {write_offset}"
    )

del chunk_files
shutil.rmtree(tmp_chunks_dir, ignore_errors=True)

X_all.flush()
y_all.flush()
y_status_all.flush()
rows_arr.flush()
cols_arr.flush()

print(f"Valid samples in matrix: {len(y_all)}")
print(f"Class distribution: {np.bincount(y_all)}")
print(f"Status distribution (1=clean, 2=dirty): {np.bincount(y_status_all)}")
print(f"X shape: {X_all.shape}")

np.savez_compressed(
    MATRIX_PATH,
    X=X_all,
    y=y_all,
    y_status=y_status_all,
    rows=rows_arr,
    cols=cols_arr,
    var_names=np.array(var_names, dtype=object),
    class_order=np.array(class_order, dtype=object),
    status_order=np.array(status_order, dtype=object),
    width=np.array([width], dtype=np.int32),
    height=np.array([height], dtype=np.int32),
    transform=np.array(
        [
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
        ],
        dtype=np.float64,
    ),
    crs_wkt=np.array([crs.to_wkt() if crs is not None else ""], dtype=object),
    ref_mask=np.array([REF_MASK], dtype=object),
)

if not KEEP_TMP_ARRAYS:
    shutil.rmtree(tmp_arrays_dir, ignore_errors=True)
else:
    print(f"Kept temporary arrays in: {tmp_arrays_dir}")

for ds in datasets:
    ds.close()

print("\n✔ Regression matrix saved:")
print(f"  - {MATRIX_PATH}")
print("\nDone. Now run RF_GPU_kalk.py to train from this matrix.")
