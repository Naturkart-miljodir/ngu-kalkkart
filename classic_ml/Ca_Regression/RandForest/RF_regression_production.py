#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Apply trained Random Forest regression model to raster covariates and produce prediction raster.

Requirements:
- Model: joblib file with 'model' and 'var_names'
- Covariates: directory with .tif/.vrt rasters (matching training)
- Output: GeoTIFF with predicted values

Authors: GitHub Copilot
"""



import os
import re
import pandas as pd
from pathlib import Path
import joblib
import rasterio
import numpy as np
from rasterio.windows import Window

# === Helper functions (must be defined before use) ===
def normalize_name(name: str) -> str:
    stem = Path(str(name)).stem.lower().strip()
    stem = stem.replace(" ", "_").replace("-", "_")
    while "__" in stem:
        stem = stem.replace("__", "_")
    return stem

def load_channel_map(channel_map_path: Path) -> pd.DataFrame:
    if not channel_map_path.exists():
        raise RuntimeError(f"Channel map CSV not found: {channel_map_path}")
    channel_map = pd.read_csv(channel_map_path).copy()
    required = {"predictor_name", "source_file", "band"}
    missing_cols = required - set(channel_map.columns)
    if missing_cols:
        raise RuntimeError(
            "Channel map CSV is missing required columns: " + ", ".join(sorted(missing_cols))
        )
    # Normalize predictor_name by stripping extension and applying normalize_name
    channel_map["predictor_name_norm"] = channel_map["predictor_name"].astype(str).apply(lambda x: normalize_name(Path(x).stem))
    channel_map["source_file"] = channel_map["source_file"].astype(str).str.strip()
    # Allow NaN for missing bands, do not cast to int if missing
    channel_map["band"] = pd.to_numeric(channel_map["band"], errors="coerce")
    return channel_map

def map_predictor_to_file_band(var_name, predictor_dir: Path, channel_map: pd.DataFrame):
    vn_l = normalize_name(var_name)
    # Try alphaearth VRT band pattern first (as in classification script)
    m = re.match(r"^(alphaearth_dequant_national_epsg25833)_b(\d{1,3})$", vn_l)
    if m:
        base = m.group(1)
        band_num = int(m.group(2))
        match = channel_map[
            (channel_map["predictor_name_norm"] == base)
            & (channel_map["band"] == band_num)
        ]
        if not match.empty:
            row = match.iloc[0]
            src_path = predictor_dir / row["source_file"]
            return src_path, int(row["band"])
    # Try direct match (normalized)
    match = channel_map[channel_map["predictor_name_norm"] == vn_l]
    if not match.empty:
        row = match.iloc[0]
        src_path = predictor_dir / row["source_file"]
        return src_path, int(row["band"])
    return None, None

# === USER SETTINGS ===
MODEL_PATH = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\RandForest\Ca_conc_modelling\Model\rf_regression_final_model.joblib"
COVARIATE_DIR = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\Covariates_to_model"
OUTPUT_PATH = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\RandForest\Ca_conc_modelling\Model\rf_prediction_cog.tif"
BLOCK_SHAPE = (2048, 2048)  # Block size for processing


# AOI polygon (set to None to disable)
AOI_POLYGON_PATH = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Study_area\Lofoten_kalk.shp"
AOI_POLYGON_LAYER = None  # Set to layer name if using GPKG, else None

# --- Channel map CSV for variable-to-file/band mapping ---
CHANNEL_MAP_PATH = Path(r"C:/Users/acosta_pedro/OneDrive - Norges geologiske undersøkelse/Geochemistry NGU_2026/Kalk_project/Modelling/RandForest/Channel_map/Ca_regresssion_var_ed.csv")

# === LOAD MODEL ===

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
var_names = bundle["var_names"]


# --- Map predictors to files/bands using channel map ---
channel_map = load_channel_map(CHANNEL_MAP_PATH)


# === DIAGNOSTIC: Compare normalized var_names and channel map predictor_names (normalized) ===
print("\n[DIAGNOSTIC] Checking predictor name normalization and mapping...")
model_varnames_norm = [normalize_name(vn) for vn in var_names]
channel_map_predictors_norm = channel_map["predictor_name_norm"].tolist()
print(f"Model var_names (normalized): {model_varnames_norm}")
print(f"Channel map predictor_names (normalized): {channel_map_predictors_norm}")
missing_in_channel_map = [vn for vn in model_varnames_norm if vn not in channel_map_predictors_norm]
if missing_in_channel_map:
    print(f"[WARNING] The following normalized model var_names are missing in channel map: {missing_in_channel_map}")
else:
    print("[OK] All normalized model var_names found in channel map.")

rasters = {}
bands = {}
for vn in var_names:
    src_path, band = map_predictor_to_file_band(vn, Path(COVARIATE_DIR), channel_map)
    if src_path is None or not src_path.exists():
        raise RuntimeError(f"Covariate raster not found for predictor: {vn} (expected file: {src_path})")

# --- Map predictors to files/bands using channel map ---
channel_map = load_channel_map(CHANNEL_MAP_PATH)
rasters = {}
bands = {}
for vn in var_names:
    src_path, band = map_predictor_to_file_band(vn, Path(COVARIATE_DIR), channel_map)
    if src_path is None or not src_path.exists():
        raise RuntimeError(f"Covariate raster not found for predictor: {vn} (expected file: {src_path})")
    rasters[vn] = rasterio.open(src_path)
    bands[vn] = band

ref_raster = next(iter(rasters.values()))
height, width = ref_raster.height, ref_raster.width
transform = ref_raster.transform
crs = ref_raster.crs

# --- AOI MASK GENERATION ---
aoi_gdf = None
if AOI_POLYGON_PATH is not None:
    print(f"[INFO] Loading AOI polygon: {AOI_POLYGON_PATH}")
    import geopandas as gpd
    from rasterio import features
    aoi_gdf = gpd.read_file(AOI_POLYGON_PATH, layer=AOI_POLYGON_LAYER)
    if aoi_gdf.crs != crs:
        print(f"[INFO] Reprojecting AOI from {aoi_gdf.crs} to {crs}")
        aoi_gdf = aoi_gdf.to_crs(crs)
    aoi_geoms = [geom for geom in aoi_gdf.geometry if geom is not None]
    print(f"[INFO] AOI polygons loaded: {len(aoi_geoms)}")

# --- COG profile ---
profile = ref_raster.profile.copy()

profile.update(
    dtype="float32",
    count=1,
    compress="deflate",
    predictor=3,
    tiled=True,
    blockxsize=BLOCK_SHAPE[1],
    blockysize=BLOCK_SHAPE[0],
    nodata=np.nan,
    driver="COG",
    BIGTIFF="IF_SAFER",
)


# --- Progress bar setup ---
try:
    from tqdm import tqdm
    _use_tqdm = True
except ImportError:
    _use_tqdm = False

total_blocks = ((height + BLOCK_SHAPE[0] - 1) // BLOCK_SHAPE[0]) * ((width + BLOCK_SHAPE[1] - 1) // BLOCK_SHAPE[1])
block_iter = (
    (row_off, col_off)
    for row_off in range(0, height, BLOCK_SHAPE[0])
    for col_off in range(0, width, BLOCK_SHAPE[1])
)
if _use_tqdm:
    block_iter = tqdm(block_iter, total=total_blocks, desc="Predicting blocks", unit="block")

with rasterio.open(OUTPUT_PATH, "w", **profile) as dst:
    for row_off, col_off in block_iter:
        win = Window(
            col_off,
            row_off,
            min(BLOCK_SHAPE[1], width - col_off),
            min(BLOCK_SHAPE[0], height - row_off),
        )
        block_h = int(win.height)
        block_w = int(win.width)
        # --- AOI block pre-check ---
        process_block = True
        aoi_block = None
        if aoi_gdf is not None:
            from rasterio.features import geometry_mask
            win_transform = rasterio.windows.transform(win, transform)
            aoi_block = geometry_mask(
                aoi_geoms,
                out_shape=(block_h, block_w),
                transform=win_transform,
                invert=True,
                all_touched=False,
            )
            if not np.any(aoi_block):
                process_block = False
        if not process_block:
            if not _use_tqdm:
                print(f"Skipped block row={row_off}:{row_off+block_h}, col={col_off}:{col_off+block_w} (outside AOI)")
            continue
        block_stack = []
        for vn in var_names:
            arr = rasters[vn].read(bands[vn], window=win, masked=True).astype(np.float32)
            arr = np.where(np.ma.getmaskarray(arr), np.nan, arr)
            block_stack.append(arr)
        block_stack = np.stack(block_stack, axis=-1)
        block_2d = block_stack.reshape(-1, block_stack.shape[-1])
        valid_mask = np.all(np.isfinite(block_2d), axis=1)
        preds = np.full(block_2d.shape[0], np.nan, dtype=np.float32)
        if np.any(valid_mask):
            preds[valid_mask] = model.predict(block_2d[valid_mask])
        preds_2d = preds.reshape(block_h, block_w)
        # --- AOI masking ---
        if aoi_block is not None:
            preds_2d[~aoi_block] = np.nan
        dst.write(preds_2d, 1, window=win)
        if not _use_tqdm:
            print(f"Wrote block row={row_off}:{row_off+block_h}, col={col_off}:{col_off+block_w}")


for r in rasters.values():
    r.close()

print(f"✔ Cloud Optimized GeoTIFF written to: {OUTPUT_PATH}")

# --- Crop output raster to AOI bounding box and save as COG ---
if AOI_POLYGON_PATH is not None:
    print(f"[INFO] Cropping output raster to AOI extent...")
    import geopandas as gpd
    from rasterio.mask import mask
    aoi_gdf = gpd.read_file(AOI_POLYGON_PATH, layer=AOI_POLYGON_LAYER)
    aoi_geom = [geom for geom in aoi_gdf.geometry if geom is not None]
    with rasterio.open(OUTPUT_PATH) as src:
        out_image, out_transform = mask(src, aoi_geom, crop=True, nodata=src.nodata)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "driver": "COG"
        })
    cropped_path = OUTPUT_PATH.replace(".tif", "_cropped.tif")
    with rasterio.open(cropped_path, "w", **out_meta) as dest:
        dest.write(out_image)
    print(f"✔ Cropped COG saved to: {cropped_path}")
