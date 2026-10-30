#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Apply trained MLP Klass_Ca classifier to raster covariates and produce:
  - class map         (predicted class 1=Kalkfattig / 2=Intermediær / 3=Kalkrik)
  - per-class probability maps (one GeoTIFF per class)
  - Shannon entropy uncertainty map

Requirements:
- Model: joblib bundle from MLP_window_Kalkklass_modelling.py with keys
                 'state_dict', 'architecture', 'scaler', 'var_names',
                 'label_offset', 'class_labels', 'n_classes'
- Covariates: directory with .tif/.vrt rasters (matching training)
- Output: Cloud Optimised GeoTIFFs (float32) for class map, probabilities, entropy

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
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def build_channel_map_from_importance(importance_path: Path, predictor_dir: Path, output_path: Path) -> Path:
    """Build a direct raster map when the shared channel map is unavailable."""
    if not importance_path.exists():
        raise RuntimeError(f"Variable importance CSV not found: {importance_path}")
    importance = pd.read_csv(importance_path)
    if "Variable" not in importance.columns:
        raise RuntimeError("Variable importance CSV must contain a 'Variable' column.")

    rows = []
    for variable in importance["Variable"].dropna().astype(str):
        direct = predictor_dir / variable
        matches = [direct] if direct.exists() else list(predictor_dir.rglob(variable))
        if not matches:
            raise RuntimeError(f"Could not resolve predictor '{variable}' under {predictor_dir}.")
        rows.append({"predictor_name": variable, "source_file": str(matches[0]), "band": 1})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"[INFO] Built channel map from variable importance: {output_path}")
    return output_path


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


class ResBlock(nn.Module):
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


class MLPNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_layers,
        dropout: float,
        use_batch_norm: bool,
        use_residual: bool,
    ):
        super().__init__()
        layers = []
        in_size = n_features
        for idx, h in enumerate(hidden_layers):
            if use_residual and idx > 0 and h == hidden_layers[idx - 1]:
                layers.append(ResBlock(h, dropout, use_batch_norm))
            else:
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


def infer_use_residual_from_state_dict(state_dict: dict) -> bool:
    # ResBlock creates keys with fc1/fc2 under sequential children.
    return any(".fc1.weight" in str(k) for k in state_dict.keys())


def load_mlp_from_bundle(bundle: dict, device: str):
    if "state_dict" not in bundle or "architecture" not in bundle:
        raise RuntimeError("MLP bundle missing required keys: 'state_dict' and/or 'architecture'.")

    arch = bundle["architecture"]
    state_np = bundle["state_dict"]
    state_torch = {k: torch.from_numpy(v) for k, v in state_np.items()}
    use_residual = infer_use_residual_from_state_dict(state_np)

    net = MLPNet(
        n_features=int(arch["n_features"]),
        n_classes=int(arch["n_classes"]),
        hidden_layers=list(arch["hidden_layers"]),
        dropout=float(arch.get("dropout", 0.0)),
        use_batch_norm=bool(arch.get("use_batch_norm", True)),
        use_residual=use_residual,
    ).to(device)
    net.load_state_dict(state_torch)
    net.eval()

    scaler = bundle.get("scaler", None)
    if scaler is None:
        raise RuntimeError(
            "MLP bundle does not contain 'scaler'. Re-save model with latest training script."
        )
    return net, scaler, use_residual


def predict_proba_mlp(
    net: MLPNet,
    scaler,
    X: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    X_scaled = scaler.transform(X).astype(np.float32, copy=False)
    out = np.empty((X_scaled.shape[0], net.net[-1].out_features), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, X_scaled.shape[0], batch_size):
            j = min(i + batch_size, X_scaled.shape[0])
            xb = torch.from_numpy(X_scaled[i:j]).to(device)
            proba = torch.softmax(net(xb), dim=1).cpu().numpy().astype(np.float32, copy=False)
            out[i:j] = proba
    return out

# === USER SETTINGS ===
# Model bundle produced by the current MLP training configuration:
# weighted cross-entropy with balanced class weights, SHAP enabled.
MODEL_VARIANT = "Only_Geovariabler/Balanced_SHAP_WCE_BalCW"

KLASS_CA_MODEL_ROOT = Path(r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\MLP\KlassCa_modelling")
MODEL_PATH = str(KLASS_CA_MODEL_ROOT / MODEL_VARIANT / "mlp_cls_final_model_Kalkklass.joblib")

COVARIATE_DIR = r"G:\Covariates_to_model"

PRODUCTION_OUT_DIR = Path(r"G:\National_maps\Ca_class\MLP\250K\Balanced_SHAP_WCE_BalCW")

BLOCK_SHAPE = (2048, 2048)  # Block size for processing
INFER_BATCH_SIZE = 65536     # valid pixels per MLP forward-pass batch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The current training model already contains the selected class-weighting strategy.
# Keep production probabilities identical to model probabilities by default.
USE_CLASS_SCORE_CALIBRATION = False
CLASS_SCORE_WEIGHTS = {
    1: 1.00,  # Kalkfattig (reference)
    2: 0.80,  # Intermediær (slightly increased vs v2)
    3: 0.56,  # Kalkrik (slightly stronger downweight vs v2)
}
CALIBRATION_SUFFIX = "_calibrated_v3"
os.makedirs(PRODUCTION_OUT_DIR, exist_ok=True)


# AOI polygon (set to None to disable)
AOI_POLYGON_PATH = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2025\kalk_prosjekt3.0\Mask\shape\Norge_mask.shp"
AOI_POLYGON_LAYER = None  # Set to layer name if using GPKG, else None

# --- Channel map CSV for variable-to-file/band mapping ---
# Channel map is shared across modelling workflows and currently stored under
# the existing RandForest folder in this project structure.
CHANNEL_MAP_CANDIDATES = [
    PRODUCTION_OUT_DIR / "channel_map_candidate_from_model.csv",
    Path(r"C:/Users/acosta_pedro/OneDrive - Norges geologiske undersøkelse/Geochemistry NGU_2026/Kalk_project/Modelling/RandomForest/Channel_map/Ca_regresssion_var_ed.csv"),
    Path(r"C:/Users/acosta_pedro/OneDrive - Norges geologiske undersøkelse/Geochemistry NGU_2026/Kalk_project/Modelling/RandomForest/Channel_map/Ca_regresssion_var.csv"),
]
try:
    CHANNEL_MAP_PATH = resolve_channel_map_path(CHANNEL_MAP_CANDIDATES)
except RuntimeError:
    IMPORTANCE_PATH = Path(
        r"C:/Users/acosta_pedro/OneDrive - Norges geologiske undersøkelse/"
        r"Geochemistry NGU_2026/Kalk_project/Modelling/MLP/KlassCa_modelling/"
        r"Only_Geovariabler/Balanced_SHAP_WCE_BalCW/variable_importance.csv"
    )
    CHANNEL_MAP_PATH = build_channel_map_from_importance(
        IMPORTANCE_PATH,
        Path(COVARIATE_DIR),
        PRODUCTION_OUT_DIR / "channel_map_from_variable_importance.csv",
    )

# === LOAD MODEL ===

bundle = joblib.load(MODEL_PATH)
if not isinstance(bundle, dict):
    raise RuntimeError(
        "Model file must be a joblib dictionary bundle from MLP training."
    )

if "state_dict" not in bundle or "architecture" not in bundle or "var_names" not in bundle:
    raise RuntimeError(
        "Model bundle is missing required keys. Expected: 'state_dict', 'architecture', 'var_names'."
    )

var_names = list(bundle["var_names"])
mlp_model, scaler, use_residual = load_mlp_from_bundle(bundle, DEVICE)

model_type = str(bundle.get("model_type", "unknown")).lower()
if model_type not in {"unknown", "mlp", "mlp_classifier"}:
    print(f"[WARNING] model_type='{model_type}' in bundle (expected 'mlp_classifier').")
else:
    print(f"[INFO] Loaded production model_type='{model_type}'.")
print(f"[INFO] Inference device={DEVICE} | use_residual={use_residual}")

label_offset = int(bundle.get("label_offset", 1))
class_labels = bundle.get("class_labels", {})
n_classes    = int(bundle.get("n_classes", len(class_labels)))
print(f"[INFO] n_classes={n_classes}, label_offset={label_offset}, class_labels={class_labels}")
if USE_CLASS_SCORE_CALIBRATION:
    print(f"[INFO] Class-score calibration enabled: {CLASS_SCORE_WEIGHTS}")
else:
    print("[INFO] Class-score calibration disabled.")

# --- Output paths derived from class_labels in bundle ---
base_class_name = f"KlassCa_class_map{CALIBRATION_SUFFIX}" if USE_CLASS_SCORE_CALIBRATION else "KlassCa_class_map"
base_entropy_name = f"KlassCa_entropy{CALIBRATION_SUFFIX}" if USE_CLASS_SCORE_CALIBRATION else "KlassCa_entropy"
OUTPUT_CLASS_MAP = str(PRODUCTION_OUT_DIR / f"{base_class_name}.tif")
OUTPUT_ENTROPY   = str(PRODUCTION_OUT_DIR / f"{base_entropy_name}.tif")
OUTPUT_PROB_PATHS = {}
for _cls_int, _cls_label in sorted(class_labels.items()):
    _safe = (_cls_label.replace("æ", "ae").replace("Æ", "Ae")
                       .replace("ø", "o").replace("Ø", "O").replace(" ", "_"))
    _name = f"KlassCa_prob_{_cls_int}_{_safe}{CALIBRATION_SUFFIX}" if USE_CLASS_SCORE_CALIBRATION else f"KlassCa_prob_{_cls_int}_{_safe}"
    OUTPUT_PROB_PATHS[_cls_int] = str(PRODUCTION_OUT_DIR / f"{_name}.tif")


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

# --- AOI-FIRST OUTPUT EXTENT ---
if AOI_POLYGON_PATH is None:
    raise RuntimeError("AOI_POLYGON_PATH must be set for AOI-first production mode.")

print(f"[INFO] Loading AOI polygon: {AOI_POLYGON_PATH}")
import geopandas as gpd
aoi_gdf = gpd.read_file(AOI_POLYGON_PATH, layer=AOI_POLYGON_LAYER)
if aoi_gdf.crs != crs:
    print(f"[INFO] Reprojecting AOI from {aoi_gdf.crs} to {crs}")
    aoi_gdf = aoi_gdf.to_crs(crs)
aoi_geoms = [geom for geom in aoi_gdf.geometry if geom is not None]
if len(aoi_geoms) == 0:
    raise RuntimeError("AOI contains no valid geometries.")
print(f"[INFO] AOI polygons loaded: {len(aoi_geoms)}")

aoi_bounds = aoi_gdf.total_bounds
aoi_window = rasterio.windows.from_bounds(*aoi_bounds, transform=transform)
aoi_window = aoi_window.round_offsets().round_lengths()

aoi_col_off = max(0, int(aoi_window.col_off))
aoi_row_off = max(0, int(aoi_window.row_off))
aoi_col_max = min(width, int(aoi_window.col_off + aoi_window.width))
aoi_row_max = min(height, int(aoi_window.row_off + aoi_window.height))
aoi_width = aoi_col_max - aoi_col_off
aoi_height = aoi_row_max - aoi_row_off
if aoi_width <= 0 or aoi_height <= 0:
    raise RuntimeError("AOI does not overlap raster extent.")

aoi_window = Window(aoi_col_off, aoi_row_off, aoi_width, aoi_height)
aoi_transform = rasterio.windows.transform(aoi_window, transform)
print(
    f"[INFO] AOI raster window: row_off={aoi_row_off}, col_off={aoi_col_off}, "
    f"height={aoi_height}, width={aoi_width}"
)

# --- Temporary tiled GeoTIFF profile ---
# Write blocks first; convert to COG only after all inference is complete.
profile = ref_raster.profile.copy()

profile.update(
    dtype="float32",
    count=1,
    compress="deflate",
    predictor=3,
    tiled=True,
    blockxsize=min(BLOCK_SHAPE[1], aoi_width),
    blockysize=min(BLOCK_SHAPE[0], aoi_height),
    nodata=np.nan,
    driver="GTiff",
    BIGTIFF="IF_SAFER",
    transform=aoi_transform,
    width=aoi_width,
    height=aoi_height,
)


# --- Progress bar setup ---
try:
    from tqdm import tqdm
    _use_tqdm = True
except ImportError:
    _use_tqdm = False

total_blocks = ((aoi_height + BLOCK_SHAPE[0] - 1) // BLOCK_SHAPE[0]) * ((aoi_width + BLOCK_SHAPE[1] - 1) // BLOCK_SHAPE[1])
block_iter = (
    (row_off, col_off)
    for row_off in range(0, aoi_height, BLOCK_SHAPE[0])
    for col_off in range(0, aoi_width, BLOCK_SHAPE[1])
)
if _use_tqdm:
    block_iter = tqdm(block_iter, total=total_blocks, desc="Predicting blocks", unit="block")

import contextlib

class_tmp_path = str(PRODUCTION_OUT_DIR / "KlassCa_class_map_tmp_gtiff.tif")
entropy_tmp_path = str(PRODUCTION_OUT_DIR / "KlassCa_entropy_tmp_gtiff.tif")
prob_tmp_paths = {
    cls_int: str(PRODUCTION_OUT_DIR / f"KlassCa_prob_{cls_int}_tmp_gtiff.tif")
    for cls_int in sorted(OUTPUT_PROB_PATHS)
}

with contextlib.ExitStack() as stack:
    dst_class   = stack.enter_context(rasterio.open(class_tmp_path, "w", **profile))
    dst_entropy = stack.enter_context(rasterio.open(entropy_tmp_path, "w", **profile))
    dst_probs   = {
        cls_int: stack.enter_context(rasterio.open(prob_tmp_paths[cls_int], "w", **profile))
        for cls_int in sorted(OUTPUT_PROB_PATHS)
    }
    for out_row_off, out_col_off in block_iter:
        out_win = Window(
            out_col_off,
            out_row_off,
            min(BLOCK_SHAPE[1], aoi_width - out_col_off),
            min(BLOCK_SHAPE[0], aoi_height - out_row_off),
        )
        src_win = Window(
            aoi_col_off + out_col_off,
            aoi_row_off + out_row_off,
            out_win.width,
            out_win.height,
        )
        block_h = int(out_win.height)
        block_w = int(out_win.width)
        # --- AOI block pre-check ---
        process_block = True
        aoi_block = None
        from rasterio.features import geometry_mask
        src_win_transform = rasterio.windows.transform(src_win, transform)
        aoi_block = geometry_mask(
            aoi_geoms,
            out_shape=(block_h, block_w),
            transform=src_win_transform,
            invert=True,
            all_touched=False,
        )
        if not np.any(aoi_block):
            process_block = False
        if not process_block:
            # Write explicit nodata for fully outside-AOI blocks so unwritten tiles do not appear as zeros.
            nodata_block = np.full((block_h, block_w), np.nan, dtype=np.float32)
            dst_class.write(nodata_block, 1, window=out_win)
            dst_entropy.write(nodata_block, 1, window=out_win)
            for cls_int in sorted(OUTPUT_PROB_PATHS):
                dst_probs[cls_int].write(nodata_block, 1, window=out_win)
            if not _use_tqdm:
                print(
                    f"Skipped block row={int(src_win.row_off)}:{int(src_win.row_off)+block_h}, "
                    f"col={int(src_win.col_off)}:{int(src_win.col_off)+block_w} (outside AOI)"
                )
            continue
        block_stack = []
        for vn in var_names:
            arr = rasters[vn].read(bands[vn], window=src_win, masked=True).astype(np.float32)
            arr = np.where(np.ma.getmaskarray(arr), np.nan, arr)
            block_stack.append(arr)
        block_stack = np.stack(block_stack, axis=-1)
        block_2d = block_stack.reshape(-1, block_stack.shape[-1])
        valid_mask = np.all(np.isfinite(block_2d), axis=1)
        n_pixels      = block_2d.shape[0]
        class_preds   = np.full(n_pixels, np.nan, dtype=np.float32)
        entropy_preds = np.full(n_pixels, np.nan, dtype=np.float32)
        prob_preds    = {cls_int: np.full(n_pixels, np.nan, dtype=np.float32)
                         for cls_int in sorted(OUTPUT_PROB_PATHS)}
        if np.any(valid_mask):
            proba = predict_proba_mlp(
                net=mlp_model,
                scaler=scaler,
                X=block_2d[valid_mask],
                device=DEVICE,
                batch_size=INFER_BATCH_SIZE,
            )

            if USE_CLASS_SCORE_CALIBRATION:
                cls_order = sorted(OUTPUT_PROB_PATHS)
                w = np.array([CLASS_SCORE_WEIGHTS.get(c, 1.0) for c in cls_order], dtype=np.float32)[None, :]
                proba_adj = proba * w
                row_sum = proba_adj.sum(axis=1, keepdims=True)
                bad = row_sum.squeeze() <= 1e-12
                if np.any(bad):
                    proba_adj[bad] = proba[bad]
                    row_sum = proba_adj.sum(axis=1, keepdims=True)
                proba = proba_adj / np.clip(row_sum, 1e-12, None)

            class_preds[valid_mask] = (np.argmax(proba, axis=1) + label_offset).astype(np.float32)
            for i, cls_int in enumerate(sorted(OUTPUT_PROB_PATHS)):
                prob_preds[cls_int][valid_mask] = proba[:, i].astype(np.float32)
            clipped = np.clip(proba, 1e-12, 1.0)
            entropy_preds[valid_mask] = (-np.sum(clipped * np.log2(clipped), axis=1)).astype(np.float32)
        class_2d   = class_preds.reshape(block_h, block_w)
        entropy_2d = entropy_preds.reshape(block_h, block_w)
        # --- AOI masking ---
        if aoi_block is not None:
            class_2d[~aoi_block]   = np.nan
            entropy_2d[~aoi_block] = np.nan
        dst_class.write(class_2d, 1, window=out_win)
        dst_entropy.write(entropy_2d, 1, window=out_win)
        for cls_int in sorted(OUTPUT_PROB_PATHS):
            prob_2d = prob_preds[cls_int].reshape(block_h, block_w)
            if aoi_block is not None:
                prob_2d[~aoi_block] = np.nan
            dst_probs[cls_int].write(prob_2d, 1, window=out_win)
        if not _use_tqdm:
            print(
                f"Wrote block row={int(src_win.row_off)}:{int(src_win.row_off)+block_h}, "
                f"col={int(src_win.col_off)}:{int(src_win.col_off)+block_w}"
            )


for r in rasters.values():
    r.close()


def convert_gtiff_to_cog(src_path: str, dst_path: str):
    print(f"[INFO] Converting completed raster to COG: {dst_path}")
    rio_shutil.copy(
        src_path,
        dst_path,
        driver="COG",
        COMPRESS="DEFLATE",
        PREDICTOR="3",
        BLOCKSIZE="512",
        BIGTIFF="IF_SAFER",
        OVERVIEWS="AUTO",
        RESAMPLING="NEAREST",
    )


print("[INFO] All blocks processed. Converting temporary GeoTIFFs to COGs...")
convert_gtiff_to_cog(class_tmp_path, OUTPUT_CLASS_MAP)
convert_gtiff_to_cog(entropy_tmp_path, OUTPUT_ENTROPY)
for cls_int in sorted(OUTPUT_PROB_PATHS):
    convert_gtiff_to_cog(prob_tmp_paths[cls_int], OUTPUT_PROB_PATHS[cls_int])

for tmp_path in [class_tmp_path, entropy_tmp_path, *prob_tmp_paths.values()]:
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except OSError:
        print(f"[WARNING] Could not remove temporary file: {tmp_path}")

print(f"\n✔ Class map written to:  {OUTPUT_CLASS_MAP}")
print(f"✔ Entropy map written to: {OUTPUT_ENTROPY}")
for cls_int, p in sorted(OUTPUT_PROB_PATHS.items()):
    lbl = class_labels.get(cls_int, str(cls_int))
    print(f"✔ Probability class {cls_int} ({lbl}): {p}")
print("[INFO] Outputs were written directly on AOI extent (no post-cropping step).")
