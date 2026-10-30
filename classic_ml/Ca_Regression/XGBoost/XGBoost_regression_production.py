#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Apply trained XGBoost regression model to raster covariates and produce prediction raster.

Requirements:
- Model: joblib file from XGBoost modelling with at least 'model' and 'var_names'
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
from rasterio import shutil as rio_shutil

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
    # Normalize predictor_name once; do not pre-strip stem here because normalize_name already does it.
    channel_map["predictor_name_norm"] = channel_map["predictor_name"].astype(str).apply(normalize_name)
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

    # Fallback 1: direct file lookup by predictor name (already often includes extension)
    direct_path = predictor_dir / str(var_name)
    if direct_path.exists():
        return direct_path, 1

    # Fallback 2: normalized stem matching against files in predictor directory
    # This handles minor naming differences between model predictor names and channel map entries.
    target_norm = normalize_name(var_name)
    for ext in ("*.tif", "*.vrt"):
        for p in predictor_dir.glob(ext):
            if normalize_name(p.name) == target_norm:
                return p, 1

    return None, None


def resolve_channel_map_path(candidates):
    for p in candidates:
        if p.exists():
            return p
    tried = "\n - " + "\n - ".join([str(p) for p in candidates])
    raise RuntimeError(f"No channel map CSV found. Tried:{tried}")


def build_channel_map_candidate_from_model(var_names, predictor_dir: Path, out_csv: Path) -> Path:
    predictor_dir = Path(predictor_dir)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    raster_files = sorted(list(predictor_dir.glob("*.tif")) + list(predictor_dir.glob("*.vrt")))
    stem_index = {normalize_name(p.name): p for p in raster_files}
    rows = []

    for var_name in var_names:
        norm_name = normalize_name(var_name)
        src_path = None
        band = np.nan

        band_match = re.match(r"^(.*)_b(\d{1,3})$", norm_name)
        if band_match:
            base_name = band_match.group(1)
            if base_name in stem_index:
                src_path = stem_index[base_name]
                band = int(band_match.group(2))

        if src_path is None and norm_name in stem_index:
            src_path = stem_index[norm_name]
            band = 1

        rows.append({
            "predictor_name": str(var_name),
            "source_file": src_path.name if src_path is not None else "",
            "band": band,
        })

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[PRECHECK] Wrote model-derived channel map candidate: {out_csv}")
    return out_csv


def preflight_predictor_mapping(var_names, predictor_dir: Path, channel_map: pd.DataFrame, diag_out_dir: Path):
    diag_out_dir.mkdir(parents=True, exist_ok=True)

    required_df = pd.DataFrame({
        "predictor": list(var_names),
        "predictor_norm": [normalize_name(v) for v in var_names],
    })
    required_csv = diag_out_dir / "required_predictors_from_model.csv"
    required_df.to_csv(required_csv, index=False)
    print(f"[PRECHECK] Required predictor list saved: {required_csv}")

    cm_cols = ["predictor_name_norm", "source_file", "band"]
    cm_view = channel_map[cm_cols].copy()
    cm_grouped = cm_view.groupby("predictor_name_norm", dropna=False)
    recon_rows = []
    for _, r in required_df.iterrows():
        pn = r["predictor_norm"]
        if pn in cm_grouped.groups:
            subset = cm_grouped.get_group(pn)
            source_list = sorted({str(x) for x in subset["source_file"].tolist()})
            band_list = sorted({int(x) for x in subset["band"].dropna().tolist()})
            recon_rows.append({
                "predictor": r["predictor"],
                "predictor_norm": pn,
                "in_channel_map": True,
                "channel_map_rows": int(len(subset)),
                "candidate_source_files": "|".join(source_list),
                "candidate_bands": "|".join([str(x) for x in band_list]),
            })
        else:
            recon_rows.append({
                "predictor": r["predictor"],
                "predictor_norm": pn,
                "in_channel_map": False,
                "channel_map_rows": 0,
                "candidate_source_files": "",
                "candidate_bands": "",
            })

    recon_df = pd.DataFrame(recon_rows)
    recon_csv = diag_out_dir / "required_vs_channel_map.csv"
    recon_df.to_csv(recon_csv, index=False)
    matched_count = int(recon_df["in_channel_map"].sum())
    print(f"[PRECHECK] Required vs channel-map table saved: {recon_csv}")
    print(f"[PRECHECK] Channel-map name coverage: {matched_count}/{len(recon_df)}")

    rows = []
    resolved = []

    for vn in var_names:
        src_path, band = map_predictor_to_file_band(vn, predictor_dir, channel_map)
        src_exists = bool(src_path is not None and Path(src_path).exists())
        rows.append({
            "predictor": vn,
            "source_file": str(src_path) if src_path is not None else "",
            "band": int(band) if band is not None else np.nan,
            "resolved": bool(src_path is not None),
            "source_exists": src_exists,
        })
        resolved.append((vn, src_path, band, src_exists))

    diag_df = pd.DataFrame(rows)
    diag_csv = diag_out_dir / "mapping_diagnostics.csv"
    diag_df.to_csv(diag_csv, index=False)

    ok_count = int(diag_df["source_exists"].sum())
    total = len(diag_df)
    coverage_pct = (100.0 * ok_count / total) if total > 0 else 0.0
    print(f"[PRECHECK] Predictor mapping coverage: {ok_count}/{total} ({coverage_pct:.1f}%)")
    print(f"[PRECHECK] Mapping diagnostics saved: {diag_csv}")

    unresolved = diag_df[~diag_df["source_exists"]]
    if not unresolved.empty:
        unresolved_csv = diag_out_dir / "mapping_unresolved_predictors.csv"
        unresolved.to_csv(unresolved_csv, index=False)
        first_missing = unresolved["predictor"].head(10).tolist()
        raise RuntimeError(
            "Precheck failed: unresolved predictor mappings detected. "
            f"Resolved {ok_count}/{total}. "
            f"First missing predictors: {first_missing}. "
            f"See diagnostics: {unresolved_csv}"
        )

    return resolved

# === USER SETTINGS ===
# Change only these three fields for most production runs.
TARGET_VARIABLE = "CaO"
MODEL_SUBSET = "Top_60"        # Empty for single-run tuned models; use e.g. "Top_60" for multi-run folders
MODEL_RUN_TAG = "Geo_topo_bedrock_modelling_tuned/250K"
PRODUCTION_RUN_TAG = "CaO_with_LOI_Top_60"

XGB_MODEL_ROOT = Path(r"C:/Users/acosta_pedro/OneDrive - Norges geologiske undersøkelse/Geochemistry NGU_2026/Kalk_project/Modelling/XGBoost/Ca_conc_modelling/Model_XRF_total_Lab/with_LOI")
PRODUCTION_OUT_DIR = Path(r"G:/National_maps/CaO_conc/250K/CaO_Top_60")

model_root = XGB_MODEL_ROOT / MODEL_RUN_TAG if str(MODEL_RUN_TAG).strip() else XGB_MODEL_ROOT
if str(MODEL_SUBSET).strip():
    MODEL_PATH = str(model_root / TARGET_VARIABLE / MODEL_SUBSET / f"xgb_final_model_{TARGET_VARIABLE}.joblib")
else:
    MODEL_PATH = str(model_root / TARGET_VARIABLE / f"xgb_final_model_{TARGET_VARIABLE}.joblib")
COVARIATE_DIR = r"G:\Covariates_to_model"
model_label = MODEL_SUBSET if str(MODEL_SUBSET).strip() else MODEL_RUN_TAG
OUTPUT_PATH = str(PRODUCTION_OUT_DIR / f"xgb_prediction_{TARGET_VARIABLE}_{model_label}.tif")
BLOCK_SHAPE = (2048, 2048)  # Block size for processing
os.makedirs(PRODUCTION_OUT_DIR, exist_ok=True)


# AOI polygon (set to None to disable)
AOI_POLYGON_LAYER = None  # Set to layer name if using GPKG, else None

AOI_POLYGON_PATH = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2025\kalk_prosjekt3.0\Mask\shape\Norge_mask.shp"

# Keep GTiff-first mode as default, then convert to COG after block inference.
DIRECT_WRITE_COG = False

# --- Channel map CSV for variable-to-file/band mapping ---
# First write a fresh candidate from the current model predictor list and the
# covariate folder. Keep the older mapping files as fallbacks for edge cases.
CHANNEL_MAP_CANDIDATES = [
    PRODUCTION_OUT_DIR / "channel_map_candidate_from_model.csv",
    Path(r"C:/Users/acosta_pedro/OneDrive - Norges geologiske undersøkelse/Geochemistry NGU_2026/Kalk_project/Modelling/XGBoost/Ca_conc_modelling/Production/channel_map_candidate_from_model.csv"),
    Path(r"C:/Users/acosta_pedro/OneDrive - Norges geologiske undersøkelse/Geochemistry NGU_2026/Kalk_project/Modelling/RandForest/Channel_map/Ca_regresssion_var_ed.csv"),
    Path(r"C:/Users/acosta_pedro/OneDrive - Norges geologiske undersøkelse/Geochemistry NGU_2026/Kalk_project/Modelling/RandForest/Channel_map/Ca_regresssion_var.csv"),
]

# === LOAD MODEL ===

bundle = joblib.load(MODEL_PATH)
if not isinstance(bundle, dict):
    raise RuntimeError(
        "Model file must be a joblib dictionary containing at least 'model' and 'var_names'."
    )

if "model" not in bundle or "var_names" not in bundle:
    raise RuntimeError(
        "Model bundle is missing required keys. Expected: 'model' and 'var_names'."
    )

model = bundle["model"]
var_names = bundle["var_names"]

model_type = str(bundle.get("model_type", "unknown")).lower()
if model_type not in {"unknown", "xgboost"}:
    print(f"[WARNING] model_type='{model_type}' in bundle (expected 'xgboost').")
else:
    print(f"[INFO] Loaded production model_type='{model_type}'.")

build_channel_map_candidate_from_model(
    var_names=var_names,
    predictor_dir=Path(COVARIATE_DIR),
    out_csv=CHANNEL_MAP_CANDIDATES[0],
)
CHANNEL_MAP_PATH = resolve_channel_map_path(CHANNEL_MAP_CANDIDATES)


# --- Map predictors to files/bands using channel map ---
channel_map = load_channel_map(CHANNEL_MAP_PATH)
resolved_predictors = preflight_predictor_mapping(
    var_names=var_names,
    predictor_dir=Path(COVARIATE_DIR),
    channel_map=channel_map,
    diag_out_dir=PRODUCTION_OUT_DIR,
)


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

# --- Map predictors to files/bands using channel map ---
rasters = {}
bands = {}
for vn, src_path, band, _src_exists in resolved_predictors:
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
    nodata=np.nan,
)

if DIRECT_WRITE_COG:
    profile.update(
        driver="COG",
        COMPRESS="DEFLATE",
        PREDICTOR="3",
        BLOCKSIZE="512",
        BIGTIFF="IF_SAFER",
        OVERVIEWS="AUTO",
        RESAMPLING="NEAREST",
    )
else:
    profile.update(
        driver="GTiff",
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=min(BLOCK_SHAPE[1], width),
        blockysize=min(BLOCK_SHAPE[0], height),
        BIGTIFF="IF_SAFER",
    )

tmp_output_path = OUTPUT_PATH if DIRECT_WRITE_COG else OUTPUT_PATH.replace(".tif", "_tmp_gtiff.tif")


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

with rasterio.open(tmp_output_path, "w", **profile) as dst:
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
            nodata_block = np.full((block_h, block_w), np.nan, dtype=np.float32)
            dst.write(nodata_block, 1, window=win)
            if not _use_tqdm:
                print(f"Skipped block row={row_off}:{row_off+block_h}, col={col_off}:{col_off+block_w} (outside AOI)")
            continue
        block_stack = []
        for vn in var_names:
            arr = rasters[vn].read(bands[vn], window=win, masked=True).astype(np.float32)
            arr = np.where(np.ma.getmaskarray(arr), np.nan, arr)
            block_stack.append(arr)
        block_array = np.stack(block_stack, axis=-1)
        block_2d = block_array.reshape(-1, block_array.shape[-1])
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

if not DIRECT_WRITE_COG:
    print("[INFO] Converting temporary GTiff to COG...")
    rio_shutil.copy(
        tmp_output_path,
        OUTPUT_PATH,
        driver="COG",
        COMPRESS="DEFLATE",
        PREDICTOR="3",
        BLOCKSIZE="512",
        BIGTIFF="IF_SAFER",
        OVERVIEWS="AUTO",
        RESAMPLING="NEAREST",
    )
    try:
        if os.path.exists(tmp_output_path):
            os.remove(tmp_output_path)
    except OSError:
        print(f"[WARNING] Could not remove temporary file: {tmp_output_path}")

print(f"✔ Cloud Optimized GeoTIFF written to: {OUTPUT_PATH}")
print("[INFO] Output uses full covariate extent with AOI masking outside polygon.")
