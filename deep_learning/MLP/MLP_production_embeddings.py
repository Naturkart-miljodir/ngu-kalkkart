# -*- coding: utf-8 -*-
"""
Blockwise production prediction for the Kalk MLP-with-embeddings model.

This script is intentionally pointwise, not tile-segmentation based.
It loads the saved Keras MLP model plus metadata from training, matches
predictor rasters in the trained predictor order, applies the same
continuous/categorical preprocessing logic, and writes production rasters.

Important note
--------------
The current training artifact does not persist final continuous-scaler
statistics. As a fallback, this script estimates production-time medians and
z-score stats directly from the predictor rasters using random sampling. That
matches the training preprocessing logic closely, but not perfectly.
"""

import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf
import geopandas as gpd
from shapely.geometry import box, shape
from rasterio import shutil as rio_shutil
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_geom
from rasterio.windows import Window

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# =============================================================================
# USER SETTINGS
# =============================================================================

PREDICTOR_DIR = r"G:\Covariates_to_model"
MODEL_DIR = r"D:\MultiLayerPerceptron_models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "mlp_final_model.keras")
MODEL_META_PATH = os.path.join(MODEL_DIR, "mlp_final_model_meta.joblib")
PRED_OUT = os.path.join(MODEL_DIR, "Production")

# Optional stricter mapping file with columns predictor_name, filename.
PREDICTOR_LIST_FILE = os.path.join(PREDICTOR_DIR, "predictor_file_list.csv")

AOI_VECTOR_PATH = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Study_area\NordTrondelag_pol.shp"
AOI_LAYER = None

SUPERBLOCK_SIZE = 2048
PRED_BATCH_SIZE = 131072
NODATA_FILL_VALUE = 0.0
SIZE_MISMATCH_TOL_PIXELS = 2
USE_AOI_BOUNDING_BOX_LIMIT = True

WRITE_CLASS = True
WRITE_CONFIDENCE = True
WRITE_ENTROPY = True
WRITE_COG = True
COG_COMPRESS = "LZW"
COG_BLOCKSIZE = 512

PROB_PREFIX = "prob_class"
CLASS_NAME = "predicted_class.tif"
CONFIDENCE_NAME = "confidence.tif"
ENTROPY_NAME = "entropy.tif"
PREDICTIONS_DIRNAME = "Predictions"
UNCERTAINTY_DIRNAME = "Uncertainty"

# Approximate continuous preprocessing stats from production rasters.
ENABLE_CONTINUOUS_STANDARDIZATION = True
CONTINUOUS_ZSCORE_CLIP = 8.0
LOG1P_CONTINUOUS_PREDICTORS: List[str] = []
NO_SCALE_CONTINUOUS_PREDICTORS: List[str] = [
    "KalsiumElvInnsjo4_10m_masked_cog",
]
STAT_SAMPLES_PER_RASTER = 500000
STAT_WINDOWS_PER_RASTER = 25
STAT_WINDOW_SIZE = 512
STAT_RANDOM_SEED = 42


# =============================================================================
# HELPERS
# =============================================================================

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def normalize_name(value: str) -> str:
    out = str(value).strip().lower()
    out = out.replace(".tif", "").replace(".vrt", "")
    out = out.replace(" ", "_")
    return out


def load_model_and_meta(model_path: str, meta_path: str):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Model metadata not found: {meta_path}")

    meta = joblib.load(meta_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model, meta


def infer_num_classes(meta: Dict[str, Any]) -> int:
    classes = meta.get("classes", [])
    if not classes:
        raise RuntimeError("Model metadata is missing 'classes'.")
    return int(len(classes))


def get_predictor_names(meta: Dict[str, Any]) -> List[str]:
    var_names = meta.get("var_names")
    if not var_names:
        raise RuntimeError("Model metadata is missing 'var_names'.")
    return [str(v) for v in var_names]


def get_embedding_config(meta: Dict[str, Any]) -> Dict[str, Any]:
    cat_lookup = meta.get("categorical_predictors", {}) or {}
    cat_names = list(cat_lookup.keys())
    qua_name = cat_names[0] if len(cat_names) >= 1 else "QuaternaryClass_id"
    land_name = cat_names[1] if len(cat_names) >= 2 else "landuseCode_18_cog"
    return {
        "use_embeddings": bool(meta.get("use_categorical_embeddings", False)),
        "quaternary_name": qua_name,
        "landuse_name": land_name,
        "quaternary_num_classes": int(meta.get("quaternary_num_classes_active", 1)),
        "landuse_num_classes": int(meta.get("landuse_num_classes_active", 1)),
    }


def build_predictor_lookup(predictor_dir: str):
    tif_paths = glob.glob(os.path.join(predictor_dir, "*.tif"))
    vrt_paths = glob.glob(os.path.join(predictor_dir, "*.vrt"))
    all_paths = tif_paths + vrt_paths
    by_norm: Dict[str, str] = {}
    for path in all_paths:
        by_norm[normalize_name(os.path.basename(path))] = path
    return by_norm, all_paths


def load_strict_lookup(predictor_list_file: Optional[str]) -> Optional[Dict[str, str]]:
    if not predictor_list_file or not os.path.isfile(predictor_list_file):
        return None

    strict_df = pd.read_csv(predictor_list_file)
    required = {"predictor_name", "filename"}
    if not required.issubset(set(strict_df.columns)):
        raise ValueError("predictor_file_list.csv must contain columns: predictor_name, filename")

    return {
        normalize_name(row["predictor_name"]): str(row["filename"])
        for _, row in strict_df.iterrows()
    }


def match_predictor_file(
    predictor_name: str,
    by_norm: Dict[str, str],
    predictor_paths: List[str],
    strict_lookup: Optional[Dict[str, str]],
    predictor_dir: str,
) -> str:
    key = normalize_name(predictor_name)

    if strict_lookup is not None:
        if key not in strict_lookup:
            raise RuntimeError(f"Missing predictor '{predictor_name}' in predictor_file_list.csv")
        full = os.path.join(predictor_dir, strict_lookup[key])
        if not os.path.isfile(full):
            raise RuntimeError(f"Mapped file not found for predictor '{predictor_name}': {full}")
        return full

    if key in by_norm:
        return by_norm[key]

    candidates = []
    for path in predictor_paths:
        stem = normalize_name(os.path.basename(path))
        if key in stem or stem in key:
            candidates.append(path)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(f"Ambiguous match for predictor '{predictor_name}'. Candidates: {candidates}")
    raise RuntimeError(f"Missing predictor '{predictor_name}' in {predictor_dir}")


def build_channel_specs(predictor_dir: str, predictor_names: List[str], predictor_list_file: Optional[str]):
    by_norm, predictor_paths = build_predictor_lookup(predictor_dir)
    strict_lookup = load_strict_lookup(predictor_list_file)

    specs = []
    for predictor_name in predictor_names:
        specs.append(
            {
                "predictor_name": predictor_name,
                "band": 1,
                "path": match_predictor_file(
                    predictor_name,
                    by_norm=by_norm,
                    predictor_paths=predictor_paths,
                    strict_lookup=strict_lookup,
                    predictor_dir=predictor_dir,
                ),
            }
        )
    return specs


def validate_raster_stack(channel_specs: List[Dict[str, Any]]):
    if not channel_specs:
        raise RuntimeError("No predictor rasters matched.")

    unique_paths: List[str] = []
    seen = set()
    for spec in channel_specs:
        path = spec["path"]
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)

    with rasterio.open(unique_paths[0]) as ref:
        width = ref.width
        height = ref.height
        transform = ref.transform
        crs = ref.crs
        profile = ref.profile.copy()

    for path in unique_paths[1:]:
        with rasterio.open(path) as ds:
            if ds.width != width or ds.height != height:
                dw = abs(ds.width - width)
                dh = abs(ds.height - height)
                if dw > SIZE_MISMATCH_TOL_PIXELS or dh > SIZE_MISMATCH_TOL_PIXELS:
                    raise RuntimeError(f"Raster size mismatch: {path}")
            if ds.transform != transform:
                raise RuntimeError(f"Raster transform mismatch: {path}")
            if ds.crs != crs:
                raise RuntimeError(f"Raster CRS mismatch: {path}")

    return width, height, transform, crs, profile


def open_predictor_datasets(channel_specs, ref_width, ref_height, ref_transform, ref_crs):
    datasets: Dict[str, Any] = {}
    seen = set()
    for spec in channel_specs:
        path = spec["path"]
        if path in seen:
            continue
        seen.add(path)
        ds = rasterio.open(path)
        if (
            ds.width != ref_width
            or ds.height != ref_height
            or ds.transform != ref_transform
            or ds.crs != ref_crs
        ):
            ds = WarpedVRT(
                ds,
                crs=ref_crs,
                transform=ref_transform,
                width=ref_width,
                height=ref_height,
                resampling=Resampling.bilinear,
            )
        datasets[path] = ds
    return datasets


def close_predictor_datasets(datasets):
    for ds in datasets.values():
        try:
            ds.close()
        except Exception:
            pass


def get_aoi_geometries(vector_path: Optional[str], layer: Optional[str]):
    if vector_path is None:
        return None, None
    if not os.path.isfile(vector_path):
        raise FileNotFoundError(f"AOI vector not found: {vector_path}")

    gdf = gpd.read_file(vector_path, layer=layer) if layer else gpd.read_file(vector_path)
    if gdf.empty:
        raise RuntimeError(f"No valid AOI geometries found in {vector_path}")
    geoms = [geom.__geo_interface__ for geom in gdf.geometry if geom is not None]
    return geoms, gdf.crs


def reproject_geometries_to_crs(geoms, src_crs, dst_crs):
    if geoms is None or src_crs is None or dst_crs is None or str(src_crs) == str(dst_crs):
        return geoms
    return [transform_geom(src_crs, dst_crs, geom, precision=6) for geom in geoms if geom is not None]


def _collect_geom_bounds_coords(coords, xs, ys):
    for item in coords:
        if isinstance(item[0], (float, int)):
            xs.append(item[0])
            ys.append(item[1])
        else:
            _collect_geom_bounds_coords(item, xs, ys)


def get_aoi_block_pairs(geoms, transform, width, height, superblock_size):
    if geoms is None or not USE_AOI_BOUNDING_BOX_LIMIT:
        row_starts = list(range(0, height, superblock_size))
        col_starts = list(range(0, width, superblock_size))
        return [(r, c) for r in row_starts for c in col_starts]

    xs: List[float] = []
    ys: List[float] = []
    for geom in geoms:
        if geom is None:
            continue
        coords = geom.get("coordinates")
        if coords is not None:
            _collect_geom_bounds_coords(coords, xs, ys)

    if not xs or not ys:
        row_starts = list(range(0, height, superblock_size))
        col_starts = list(range(0, width, superblock_size))
        return [(r, c) for r in row_starts for c in col_starts]

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    r0, c0 = rasterio.transform.rowcol(transform, minx, maxy)
    r1, c1 = rasterio.transform.rowcol(transform, maxx, miny)

    rmin = max(0, min(r0, r1))
    rmax = min(height - 1, max(r0, r1))
    cmin = max(0, min(c0, c1))
    cmax = min(width - 1, max(c0, c1))

    row_start_min = max(0, (rmin // superblock_size) * superblock_size - superblock_size)
    row_start_max = min(height - 1, (rmax // superblock_size) * superblock_size + superblock_size)
    col_start_min = max(0, (cmin // superblock_size) * superblock_size - superblock_size)
    col_start_max = min(width - 1, (cmax // superblock_size) * superblock_size + superblock_size)

    row_starts = list(range(row_start_min, min(row_start_max + superblock_size, height), superblock_size))
    col_starts = list(range(col_start_min, min(col_start_max + superblock_size, width), superblock_size))
    block_pairs = [(r, c) for r in row_starts for c in col_starts]

    # Refine the bounding-box subset using true AOI intersection so we avoid
    # scanning nearly-full extents when the AOI bbox is large.
    try:
        aoi_shapes = [shape(g) for g in geoms if g is not None]
        if aoi_shapes:
            aoi_union = aoi_shapes[0]
            for geom in aoi_shapes[1:]:
                aoi_union = aoi_union.union(geom)

            refined_pairs = []
            for row0, col0 in block_pairs:
                row1 = min(row0 + superblock_size, height)
                col1 = min(col0 + superblock_size, width)

                x_left, y_top = transform * (col0, row0)
                x_right, y_bottom = transform * (col1, row1)
                block_geom = box(
                    min(x_left, x_right),
                    min(y_top, y_bottom),
                    max(x_left, x_right),
                    max(y_top, y_bottom),
                )
                if aoi_union.intersects(block_geom):
                    refined_pairs.append((row0, col0))

            if refined_pairs:
                block_pairs = refined_pairs
    except Exception as exc:
        print(f"[WARN] AOI intersection refinement failed; falling back to bbox-only limiting: {exc}")

    return block_pairs


def extract_aoi_mask_for_block(geoms, transform, row0, row1, col0, col1):
    if geoms is None:
        return np.ones((row1 - row0, col1 - col0), dtype=bool)

    win = Window(col0, row0, col1 - col0, row1 - row0)
    win_transform = rasterio.windows.transform(win, transform)
    return geometry_mask(
        geoms,
        out_shape=(row1 - row0, col1 - col0),
        transform=win_transform,
        invert=True,
        all_touched=False,
    )


def make_output_profile(ref_profile, dtype="float32", count=1, nodata=None):
    profile = ref_profile.copy()
    profile.update(dtype=dtype, count=count, compress=COG_COMPRESS, tiled=True, BIGTIFF="IF_SAFER")
    if nodata is not None:
        profile["nodata"] = nodata
    else:
        profile.pop("nodata", None)
    return profile


def create_output_rasters(base_profile, out_dir: str, num_classes: int):
    outputs = {}
    pred_dir = ensure_dir(os.path.join(out_dir, PREDICTIONS_DIRNAME))
    unc_dir = ensure_dir(os.path.join(out_dir, UNCERTAINTY_DIRNAME))

    prob_paths = []
    for class_idx in range(num_classes):
        path = os.path.join(pred_dir, f"{PROB_PREFIX}_{class_idx + 1}.tif")
        prob_paths.append((path, rasterio.open(path, "w", **make_output_profile(base_profile, nodata=np.nan))))
    outputs["prob"] = prob_paths

    if WRITE_CLASS:
        path = os.path.join(pred_dir, CLASS_NAME)
        outputs["class"] = (path, rasterio.open(path, "w", **make_output_profile(base_profile, dtype="uint8", nodata=0)))
    if WRITE_CONFIDENCE:
        path = os.path.join(pred_dir, CONFIDENCE_NAME)
        outputs["confidence"] = (path, rasterio.open(path, "w", **make_output_profile(base_profile, nodata=np.nan)))
    if WRITE_ENTROPY:
        path = os.path.join(unc_dir, ENTROPY_NAME)
        outputs["entropy"] = (path, rasterio.open(path, "w", **make_output_profile(base_profile, nodata=np.nan)))
    return outputs


def close_output_rasters(outputs):
    for _, ds in outputs.get("prob", []):
        ds.close()
    for key in ["class", "confidence", "entropy"]:
        if key in outputs:
            outputs[key][1].close()


def gather_output_paths(outputs):
    out = [path for path, _ in outputs.get("prob", [])]
    for key in ["class", "confidence", "entropy"]:
        if key in outputs:
            out.append(outputs[key][0])
    return out


def convert_tifs_to_cog(tif_paths: List[str]):
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


def compute_entropy(prob_block: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(prob_block, eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1).astype(np.float32)


def sample_band_values(ds, band: int, nwin: int, win_size: int, max_samples: int, rng) -> np.ndarray:
    width = ds.width
    height = ds.height
    ww = min(win_size, width)
    hh = min(win_size, height)
    max_x = max(width - ww, 0)
    max_y = max(height - hh, 0)

    chunks = []
    for _ in range(max(1, int(nwin))):
        x0 = rng.randint(0, max_x + 1) if max_x > 0 else 0
        y0 = rng.randint(0, max_y + 1) if max_y > 0 else 0
        arr = ds.read(band, window=Window(x0, y0, ww, hh)).astype(np.float32)
        nod = ds.nodata
        if nod is not None:
            arr[arr == nod] = np.nan
        vals = arr[np.isfinite(arr)]
        if vals.size:
            chunks.append(vals)

    if not chunks:
        return np.array([], dtype=np.float32)

    sample = np.concatenate(chunks)
    if sample.size > max_samples:
        idx = rng.choice(sample.size, size=max_samples, replace=False)
        sample = sample[idx]
    return sample.astype(np.float32)


def build_continuous_stats(channel_specs, datasets, cont_idx, cont_var_names):
    rng = np.random.RandomState(STAT_RANDOM_SEED)
    log_targets = {normalize_name(v) for v in LOG1P_CONTINUOUS_PREDICTORS}
    no_scale_targets = {normalize_name(v) for v in NO_SCALE_CONTINUOUS_PREDICTORS}

    stats: Dict[int, Dict[str, float]] = {}
    iterator = zip(cont_idx, cont_var_names)
    iterator = list(iterator)
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(iterator), desc="MLP stats", unit="band", dynamic_ncols=True)

    for src_idx, predictor_name in iterator:
        spec = channel_specs[src_idx]
        ds = datasets[spec["path"]]
        sample = sample_band_values(
            ds,
            int(spec.get("band", 1)),
            nwin=STAT_WINDOWS_PER_RASTER,
            win_size=STAT_WINDOW_SIZE,
            max_samples=STAT_SAMPLES_PER_RASTER,
            rng=rng,
        )
        if normalize_name(predictor_name) in log_targets and normalize_name(predictor_name) not in no_scale_targets:
            sample = np.log1p(np.clip(sample, 0.0, None))

        if sample.size == 0:
            median = 0.0
            mean = 0.0
            std = 1.0
        else:
            median = float(np.median(sample))
            mean = float(np.mean(sample))
            std = float(np.std(sample))
            if std < 1e-6:
                std = 1.0

        stats[src_idx] = {
            "median": median,
            "mean": mean,
            "std": std,
            "log1p": normalize_name(predictor_name) in log_targets and normalize_name(predictor_name) not in no_scale_targets,
            "no_scale": normalize_name(predictor_name) in no_scale_targets,
        }

    return stats


def build_continuous_stats_from_meta(cont_src_idx: List[int], cont_var_names: List[str], meta: Dict[str, Any]):
    preprocessor = meta.get("continuous_preprocessor")
    if not preprocessor:
        return None

    saved_names = [str(v) for v in preprocessor.get("feature_names", [])]
    if saved_names != list(cont_var_names):
        raise RuntimeError(
            "Saved continuous preprocessor feature order does not match production predictor order."
        )

    med = np.asarray(preprocessor["median_raw"], dtype=np.float32)
    mean = np.asarray(preprocessor["mean"], dtype=np.float32)
    std = np.asarray(preprocessor["std"], dtype=np.float32)
    log1p_mask = np.asarray(preprocessor["log1p_mask"], dtype=bool)
    no_scale_mask = np.asarray(preprocessor["no_scale_mask"], dtype=bool)

    stats: Dict[int, Dict[str, float]] = {}
    for j, src_idx in enumerate(cont_src_idx):
        stats[src_idx] = {
            "median": float(med[j]),
            "mean": float(mean[j]),
            "std": float(std[j]) if float(std[j]) >= 1e-6 else 1.0,
            "log1p": bool(log1p_mask[j]),
            "no_scale": bool(no_scale_mask[j]),
        }
    return stats


def read_block_stack(datasets, channel_specs, row0, row1, col0, col1):
    h = row1 - row0
    w = col1 - col0
    block = np.zeros((h, w, len(channel_specs)), dtype=np.float32)
    win = Window(col0, row0, w, h)
    for i, spec in enumerate(channel_specs):
        ds = datasets[spec["path"]]
        arr = ds.read(int(spec.get("band", 1)), window=win).astype(np.float32)
        nod = ds.nodata
        if nod is not None:
            arr[arr == nod] = np.nan
        block[:, :, i] = arr
    return block


def preprocess_continuous_block(X_cont: np.ndarray, cont_src_idx: List[int], cont_stats: Dict[int, Dict[str, float]]) -> np.ndarray:
    out = np.asarray(X_cont, dtype=np.float32).copy()
    for j, src_idx in enumerate(cont_src_idx):
        cfg = cont_stats[src_idx]
        col = out[:, j]
        col = np.where(np.isfinite(col), col, np.nan)
        if cfg["log1p"]:
            col = np.log1p(np.clip(col, 0.0, None))
        col = np.where(np.isfinite(col), col, cfg["median"])
        if ENABLE_CONTINUOUS_STANDARDIZATION and not cfg["no_scale"]:
            col = (col - cfg["mean"]) / cfg["std"]
            if CONTINUOUS_ZSCORE_CLIP is not None:
                clip_v = float(CONTINUOUS_ZSCORE_CLIP)
                col = np.clip(col, -clip_v, clip_v)
        out[:, j] = col.astype(np.float32)
    return out


def sanitize_embedding_ids(values: np.ndarray, max_id: int) -> np.ndarray:
    out = np.rint(np.where(np.isfinite(values), values, 0)).astype(np.int32)
    out = np.where((out < 0) | (out > max_id), 0, out)
    return out


def predict_block(
    model,
    block_stack: np.ndarray,
    block_aoi_mask: np.ndarray,
    use_embeddings: bool,
    cont_src_idx: List[int],
    cont_stats: Dict[int, Dict[str, float]],
    quaternary_idx: Optional[int],
    landuse_idx: Optional[int],
    quaternary_num_classes: int,
    landuse_num_classes: int,
    batch_size: int,
    num_classes: int,
):
    h, w, _ = block_stack.shape
    out_prob = np.full((h, w, num_classes), np.nan, dtype=np.float32)

    valid = block_aoi_mask & np.any(np.isfinite(block_stack), axis=-1)
    if not np.any(valid):
        return out_prob

    rows, cols = np.where(valid)
    X_valid = block_stack[rows, cols, :]

    X_cont = X_valid[:, cont_src_idx] if cont_src_idx else np.zeros((len(rows), 1), dtype=np.float32)
    X_cont = preprocess_continuous_block(X_cont, cont_src_idx, cont_stats)

    if use_embeddings and (quaternary_idx is None or landuse_idx is None):
        raise RuntimeError("Embedding mode requires categorical predictor indices.")

    total = len(rows)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        Xc = X_cont[start:end]
        if use_embeddings:
            Xq = sanitize_embedding_ids(X_valid[start:end, quaternary_idx], quaternary_num_classes)
            Xl = sanitize_embedding_ids(X_valid[start:end, landuse_idx], landuse_num_classes)
            preds = model([Xc, Xq, Xl], training=False).numpy()
        else:
            preds = model([Xc], training=False).numpy()
        out_prob[rows[start:end], cols[start:end], :] = np.asarray(preds, dtype=np.float32)

    return out_prob


def write_block_outputs(outputs, row0, col0, prob_block):
    h, w, _ = prob_block.shape
    win = Window(col0, row0, w, h)
    for k, (_, ds) in enumerate(outputs["prob"]):
        ds.write(prob_block[:, :, k].astype(np.float32), 1, window=win)

    valid = np.all(np.isfinite(prob_block), axis=-1)
    if "class" in outputs:
        class_arr = np.zeros((h, w), dtype=np.uint8)
        if np.any(valid):
            class_arr[valid] = np.argmax(prob_block[valid], axis=-1).astype(np.uint8) + 1
        outputs["class"][1].write(class_arr, 1, window=win)
    if "confidence" in outputs:
        conf = np.full((h, w), np.nan, dtype=np.float32)
        if np.any(valid):
            conf[valid] = np.max(prob_block[valid], axis=-1).astype(np.float32)
        outputs["confidence"][1].write(conf, 1, window=win)
    if "entropy" in outputs:
        ent = np.full((h, w), np.nan, dtype=np.float32)
        if np.any(valid):
            ent[valid] = compute_entropy(prob_block[valid])
        outputs["entropy"][1].write(ent, 1, window=win)


def summarize_class_counts(class_raster_path: str):
    if not os.path.isfile(class_raster_path):
        return
    with rasterio.open(class_raster_path) as ds:
        arr = ds.read(1)
    unique, counts = np.unique(arr, return_counts=True)
    total = int(arr.size)
    print("\n[SUMMARY] Predicted class pixel counts")
    print(f"[SUMMARY] Total pixels: {total}")
    for cls, cnt in zip(unique, counts):
        pct = 100.0 * int(cnt) / total if total > 0 else 0.0
        print(f"[SUMMARY] Class {int(cls)}: {int(cnt)} ({pct:.2f}%)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n=== LOADING MODEL ===")
    model, meta = load_model_and_meta(BEST_MODEL_PATH, MODEL_META_PATH)
    predictor_names = get_predictor_names(meta)
    embed_cfg = get_embedding_config(meta)
    num_classes = infer_num_classes(meta)

    print(f"Model: {BEST_MODEL_PATH}")
    print(f"Metadata: {MODEL_META_PATH}")
    print(f"Predictors in trained order: {len(predictor_names)}")
    print(f"Embedding mode: {embed_cfg['use_embeddings']}")

    channel_specs = build_channel_specs(PREDICTOR_DIR, predictor_names, PREDICTOR_LIST_FILE)
    width, height, transform, crs, ref_profile = validate_raster_stack(channel_specs)
    print(f"Validated raster stack: {len(channel_specs)} predictors | size={width} x {height}")

    q_idx = None
    l_idx = None
    if embed_cfg["use_embeddings"]:
        for i, name in enumerate(predictor_names):
            if normalize_name(name) == normalize_name(embed_cfg["quaternary_name"]):
                q_idx = i
            if normalize_name(name) == normalize_name(embed_cfg["landuse_name"]):
                l_idx = i
        if q_idx is None or l_idx is None:
            raise RuntimeError("Could not resolve categorical predictor indices for embeddings.")

    cont_src_idx = [i for i in range(len(predictor_names)) if i not in {q_idx, l_idx}]
    cont_var_names = [predictor_names[i] for i in cont_src_idx]

    geoms, aoi_crs = get_aoi_geometries(AOI_VECTOR_PATH, AOI_LAYER)
    geoms = reproject_geometries_to_crs(geoms, aoi_crs, crs)

    ensure_dir(PRED_OUT)
    outputs = {}
    output_paths: List[str] = []
    datasets = {}
    try:
        outputs = create_output_rasters(ref_profile, PRED_OUT, num_classes)
        output_paths = gather_output_paths(outputs)
        datasets = open_predictor_datasets(channel_specs, width, height, transform, crs)
        cont_stats = build_continuous_stats_from_meta(cont_src_idx, cont_var_names, meta)
        if cont_stats is not None:
            print("[INFO] Using saved continuous preprocessing stats from model metadata.")
        else:
            print("[INFO] Saved preprocessing stats not found in metadata. Estimating from rasters.")
            cont_stats = build_continuous_stats(channel_specs, datasets, cont_src_idx, cont_var_names)

        block_pairs = get_aoi_block_pairs(geoms, transform, width, height, SUPERBLOCK_SIZE)
        total_blocks = len(block_pairs)
        if geoms is not None and USE_AOI_BOUNDING_BOX_LIMIT:
            row_count = len({r for r, _ in block_pairs})
            col_count = len({c for _, c in block_pairs})
            print(
                f"[INFO] AOI-limited candidate superblocks: {total_blocks} "
                f"({row_count} unique row starts x {col_count} unique col starts)"
            )
        iterator = block_pairs
        if tqdm is not None:
            iterator = tqdm(iterator, total=total_blocks, desc="MLP production", unit="block", dynamic_ncols=True)

        for row0, col0 in iterator:
            row1 = min(row0 + SUPERBLOCK_SIZE, height)
            col1 = min(col0 + SUPERBLOCK_SIZE, width)
            block_stack = read_block_stack(datasets, channel_specs, row0, row1, col0, col1)
            block_aoi_mask = extract_aoi_mask_for_block(geoms, transform, row0, row1, col0, col1)
            if not np.any(block_aoi_mask):
                continue

            prob_block = predict_block(
                model=model,
                block_stack=block_stack,
                block_aoi_mask=block_aoi_mask,
                use_embeddings=embed_cfg["use_embeddings"],
                cont_src_idx=cont_src_idx,
                cont_stats=cont_stats,
                quaternary_idx=q_idx,
                landuse_idx=l_idx,
                quaternary_num_classes=embed_cfg["quaternary_num_classes"],
                landuse_num_classes=embed_cfg["landuse_num_classes"],
                batch_size=PRED_BATCH_SIZE,
                num_classes=num_classes,
            )
            write_block_outputs(outputs, row0, col0, prob_block)

        print("\n✔ Finished blockwise MLP production prediction.")
    finally:
        close_predictor_datasets(datasets)
        close_output_rasters(outputs)

    if WRITE_COG:
        convert_tifs_to_cog(output_paths)

    class_path = os.path.join(PRED_OUT, PREDICTIONS_DIRNAME, CLASS_NAME)
    summarize_class_counts(class_path)

    print("\nOutputs written to:")
    print(os.path.join(PRED_OUT, PREDICTIONS_DIRNAME))
    print(os.path.join(PRED_OUT, UNCERTAINTY_DIRNAME))


if __name__ == "__main__":
    main()