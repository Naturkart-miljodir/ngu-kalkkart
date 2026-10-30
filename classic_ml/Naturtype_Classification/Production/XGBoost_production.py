"""
XGBoost production export with hard-coded paths, no CLI arguments needed.

WINDOW-FRIENDLY VERSION adapted for XGBoost inference.

What is new:
- supports XGBoost models trained from the categorical-maps workflow
- detects engineered predictors like:
    base_predictor_mean_w5
    base_predictor_std_w11
- computes those moving-window features on the fly during raster prediction
- uses halo-aware window reading so focal stats are correct at block edges
- still respects AOI and still predicts blockwise
- keeps progress reporting and valid-pixel reporting

Notes:
- all original pixel predictors are still used
- only engineered window predictors get focal mean/std computation
"""

from pathlib import Path
import math
import re
import time
import warnings

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.shutil import copy as rio_copy
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rasterio.windows import transform as window_transform
from shapely.geometry import box

warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in cast")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.")


# ============================================================
# HARD-CODED PATHS
# ============================================================

MODEL_PATH = Path(
    r"D:\Classical_ML_models\Test\XGBoost\Categorical_maps\Ca_prob\clean\Tunned\Top30\xgb_final_model.joblib"
)

PREDICTOR_DIR = Path(
    r"G:\Covariates_to_model"
)


OUTPUT_DIR = Path(
    r"D:\Classical_ML_models\Test\XGBoost\Categorical_maps\Ca_prob\clean\Tunned\Top30\Production"
)

CHANNEL_MAP_PATH = Path(
    r"C:/Users/acosta_pedro/OneDrive - Norges geologiske undersøkelse/Geochemistry NGU_2026/Kalk_project/Modelling/RandForest/Channel_map/var_names_list_ed2.csv"
)

AOI_CANDIDATES = [
    Path(
        r"C:/Users/acosta_pedro/OneDrive - Norges geologiske undersøkelse/Geochemistry NGU_2026/Kalk_project/Study_area/NordTrondelag_pol.shp"
    ),
    Path(
        r"C:/Users/acosta_pedro/OneDrive - Norges geologiske undersøkelse/Geochemistry NGU_2026/Kalk_project/Modelling/aoi.gpkg"
    ),
]

# prediction block size for output
BLOCK_SHAPE = (1536, 1536)

# For speed testing, keep this False to skip COG conversion and write plain GeoTIFFs.
# Set to True after QA to produce final Cloud Optimized GeoTIFF outputs.
RUN_COG_CONVERSION = False

# used only if model bundle does not contain these keys
DEFAULT_WINDOW_MIN_VALID_FRAC = 0.20

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================
def normalize_name(name: str) -> str:
    stem = Path(str(name)).stem.lower().strip()
    stem = stem.replace(" ", "_").replace("-", "_")
    while "__" in stem:
        stem = stem.replace("__", "_")
    return stem


def collect_predictor_files(predictor_dir: Path):
    files = []
    for pattern in ("*.tif", "*.tiff", "*.vrt"):
        files.extend(predictor_dir.glob(pattern))
    return sorted(set(files))


def find_aoi_path(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None


def load_channel_map(channel_map_path: Path) -> pd.DataFrame:
    if not channel_map_path.exists():
        raise RuntimeError(f"Channel-map CSV not found: {channel_map_path}")

    channel_map = pd.read_csv(channel_map_path).copy()
    required = {"predictor_name", "source_file", "band"}
    missing_cols = required - set(channel_map.columns)
    if missing_cols:
        raise RuntimeError(
            "Channel-map CSV is missing required columns: "
            + ", ".join(sorted(missing_cols))
        )

    channel_map["predictor_name"] = (
        channel_map["predictor_name"].astype(str).str.lower().str.strip()
    )
    channel_map["source_file"] = channel_map["source_file"].astype(str).str.strip()
    channel_map["band"] = pd.to_numeric(channel_map["band"], errors="coerce")

    if channel_map["band"].isna().any():
        bad = channel_map[channel_map["band"].isna()]
        raise RuntimeError(
            "Invalid 'band' values found in channel-map CSV for rows:\n"
            + bad[["predictor_name", "source_file"]].to_string(index=False)
        )

    channel_map["band"] = channel_map["band"].astype(int)
    return channel_map


def map_raw_predictor(raw_name: str, predictor_dir: Path, channel_map: pd.DataFrame, file_lookup: dict = None):
    vn_l = normalize_name(raw_name)

    m = re.match(r"^(alphaearth_dequant_national_epsg25833)_b(\d{1,3})$", vn_l)
    if m:
        base = m.group(1)
        band_num = int(m.group(2))
        match = channel_map[
            (channel_map["predictor_name"] == base)
            & (channel_map["band"] == band_num)
        ]
        if not match.empty:
            row = match.iloc[0]
            src_path = predictor_dir / row["source_file"]
            return src_path, int(row["band"])

    match = channel_map[channel_map["predictor_name"] == vn_l]
    if not match.empty:
        row = match.iloc[0]
        src_path = predictor_dir / row["source_file"]
        return src_path, int(row["band"])

    # Fallback: direct filename mapping by normalized stem when channel_map misses an entry.
    if file_lookup:
        direct = file_lookup.get(vn_l)
        if direct is not None:
            return direct, 1

    return None, None


def validate_selected_sources(raw_sources):
    if not raw_sources:
        raise RuntimeError("No raw predictor sources were selected.")

    for vn, (src_path, band) in raw_sources.items():
        with rasterio.open(src_path) as src:
            if band < 1 or band > src.count:
                raise RuntimeError(
                    f"Requested band {band} for predictor '{vn}' exceeds band count {src.count} in {src_path.name}"
                )


def load_aoi_geometry(ref_meta, aoi_path: Path):
    if aoi_path is None:
        print("AOI file not found in candidate list (processing full extent)")
        return None

    print(f"Reading AOI polygon from {aoi_path}")
    aoi_gdf = gpd.read_file(aoi_path)

    if aoi_gdf.empty:
        raise RuntimeError(f"AOI file has no geometries: {aoi_path}")
    if aoi_gdf.crs is None:
        raise RuntimeError(f"AOI CRS is missing: {aoi_path}")

    ref_crs = ref_meta["crs"]
    if aoi_gdf.crs != ref_crs:
        print(f"Reprojecting AOI from {aoi_gdf.crs} to {ref_crs}")
        aoi_gdf = aoi_gdf.to_crs(ref_crs)

    aoi_geom = aoi_gdf.geometry.union_all()
    if aoi_geom is None or aoi_geom.is_empty:
        raise RuntimeError(f"AOI union is empty: {aoi_path}")

    print("AOI loaded successfully (window-based masking enabled)")
    return aoi_geom


def _collect_geom_bounds_coords(coords, xs, ys):
    for item in coords:
        if isinstance(item[0], (float, int)):
            xs.append(item[0])
            ys.append(item[1])
        else:
            _collect_geom_bounds_coords(item, xs, ys)


def get_aoi_block_starts(geoms, transform, width, height, block_size):
    if geoms is None:
        return list(range(0, height, block_size)), list(range(0, width, block_size))

    if hasattr(geoms, "bounds"):
        minx, miny, maxx, maxy = geoms.bounds
    else:
        xs, ys = [], []
        for geom in geoms:
            if geom is None:
                continue
            coords = geom.get("coordinates")
            if coords is not None:
                _collect_geom_bounds_coords(coords, xs, ys)

        if not xs or not ys:
            return list(range(0, height, block_size)), list(range(0, width, block_size))

        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)

    r0, c0 = rasterio.transform.rowcol(transform, minx, maxy)
    r1, c1 = rasterio.transform.rowcol(transform, maxx, miny)

    rmin = max(0, min(r0, r1))
    rmax = min(height - 1, max(r0, r1))
    cmin = max(0, min(c0, c1))
    cmax = min(width - 1, max(c0, c1))

    row_start_min = max(0, (rmin // block_size) * block_size - block_size)
    row_start_max = min(height - 1, (rmax // block_size) * block_size + block_size)
    col_start_min = max(0, (cmin // block_size) * block_size - block_size)
    col_start_max = min(width - 1, (cmax // block_size) * block_size + block_size)

    row_starts = list(range(row_start_min, min(row_start_max + block_size, height), block_size))
    col_starts = list(range(col_start_min, min(col_start_max + block_size, width), block_size))
    return row_starts, col_starts


def build_aoi_mask_for_window(aoi_geom, ref_transform, win: Window):
    win_height = int(win.height)
    win_width = int(win.width)

    if aoi_geom is None:
        return np.ones((win_height, win_width), dtype=bool)

    wb = window_bounds(win, ref_transform)
    win_box = box(*wb)
    if not aoi_geom.intersects(win_box):
        return np.zeros((win_height, win_width), dtype=bool)

    return geometry_mask(
        [aoi_geom],
        transform=window_transform(win, ref_transform),
        invert=True,
        out_shape=(win_height, win_width),
        all_touched=True,
    )


def read_window_as_float32(src, band: int, win: Window, boundless: bool = False):
    arr_m = src.read(band, window=win, masked=True, boundless=boundless)

    if np.ma.isMaskedArray(arr_m):
        raw = np.asarray(arr_m.data)
        mask = np.ma.getmaskarray(arr_m).copy()
    else:
        raw = np.asarray(arr_m)
        mask = np.zeros(raw.shape, dtype=bool)

    nodata_val = src.nodata
    if nodata_val is not None:
        try:
            with np.errstate(invalid="ignore"):
                mask |= (raw == nodata_val)
        except Exception:
            pass

    if np.issubdtype(raw.dtype, np.floating):
        with np.errstate(invalid="ignore"):
            mask |= ~np.isfinite(raw)

    with np.errstate(over="ignore", invalid="ignore"):
        arr = raw.astype(np.float32, copy=False)
    arr[mask] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    return arr


def focal_mean_std_simple(arr: np.ndarray, win_size: int, min_valid_frac: float = 0.2):
    from numpy.lib.stride_tricks import sliding_window_view

    if win_size % 2 == 0:
        raise ValueError("Window size must be odd")

    r = win_size // 2
    arr_pad = np.pad(arr, ((r, r), (r, r)), mode="constant", constant_values=np.nan)
    windows = sliding_window_view(arr_pad, (win_size, win_size))

    valid_count = np.sum(np.isfinite(windows), axis=(-2, -1))
    min_valid = max(1, int(np.ceil(win_size * win_size * float(min_valid_frac))))

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_arr = np.nanmean(windows, axis=(-2, -1)).astype(np.float32)
        std_arr = np.nanstd(windows, axis=(-2, -1)).astype(np.float32)

    mean_arr[valid_count < min_valid] = np.nan
    std_arr[valid_count < min_valid] = np.nan
    return mean_arr, std_arr


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def predict_proba_xgb_or_orf(model, X_valid: np.ndarray):
    """Return class probabilities for either a plain XGBoost model or ORF-style model bundle."""
    if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
        probs = model.predict_proba(X_valid).astype(np.float32)
        classes = np.asarray(model.classes_, dtype=np.int32)
        # Training remaps labels to 0..K-1 internally; convert back to 1..K for output rasters.
        if classes.size > 0 and np.array_equal(classes, np.arange(classes.size, dtype=np.int32)):
            classes = classes + 1
        return probs, classes

    if isinstance(model, dict) and model.get("model_type") in {"xgb_orf_style", "orf_style"}:
        classes = np.asarray(model.get("classes", []), dtype=np.int32)
        model_records = model.get("models", [])

        if classes.size < 2 or len(model_records) != (classes.size - 1):
            raise RuntimeError("Invalid ORF model bundle: classes/models structure is inconsistent")

        cumulative_probs = []
        for rec in model_records:
            const_p = rec.get("constant_prob")
            if const_p is not None:
                p_gt = np.full(X_valid.shape[0], float(const_p), dtype=np.float32)
            else:
                sub_model = rec.get("model")
                if sub_model is None or not hasattr(sub_model, "predict_proba"):
                    raise RuntimeError("Invalid ORF model bundle: missing binary sub-model")
                sub_prob = sub_model.predict_proba(X_valid)
                sub_classes = list(sub_model.classes_)
                if 1 in sub_classes:
                    pos_idx = sub_classes.index(1)
                    p_gt = sub_prob[:, pos_idx].astype(np.float32)
                else:
                    p_gt = np.zeros(X_valid.shape[0], dtype=np.float32)
            cumulative_probs.append(p_gt)

        cum = np.column_stack(cumulative_probs).astype(np.float32)
        cum = np.clip(cum, 0.0, 1.0)
        for j in range(1, cum.shape[1]):
            cum[:, j] = np.minimum(cum[:, j], cum[:, j - 1])

        n = X_valid.shape[0]
        k = classes.size
        probs = np.zeros((n, k), dtype=np.float32)
        probs[:, 0] = 1.0 - cum[:, 0]
        for cls_idx in range(1, k - 1):
            probs[:, cls_idx] = cum[:, cls_idx - 1] - cum[:, cls_idx]
        probs[:, -1] = cum[:, -1]
        probs = np.clip(probs, 0.0, 1.0)
        denom = probs.sum(axis=1, keepdims=True)
        denom[denom <= 0] = 1.0
        probs = probs / denom
        return probs.astype(np.float32), classes

    raise RuntimeError("Unsupported model type: expected RF with predict_proba or ORF bundle dict")


def convert_to_cog(src_path: Path, dst_path: Path):
    with rasterio.open(src_path) as src:
        rio_copy(
            src,
            dst_path,
            driver="COG",
            compress="LZW",
            blocksize=1024,
            overview_resampling="nearest",
        )


def print_progress(done, total, start_time, processed_blocks, skipped_blocks, written_blocks, total_valid_pixels, last_valid_pixels):
    elapsed = time.time() - start_time
    frac = done / total if total else 1.0
    eta = (elapsed / frac - elapsed) if frac > 0 else float("inf")
    bar_len = 28
    filled = min(bar_len, int(round(bar_len * frac)))
    bar = "#" * filled + "-" * (bar_len - filled)
    print(
        f"[{bar}] {done}/{total} windows ({100*frac:5.1f}%) | "
        f"processed={processed_blocks} skipped={skipped_blocks} written={written_blocks} | "
        f"last_valid={last_valid_pixels:,} total_valid={total_valid_pixels:,} | "
        f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}",
        flush=True,
    )


def parse_feature_specs(var_names):
    """
    Returns:
      feature_specs: list of dicts in model var_names order
      raw_predictors_needed: sorted list of base/raw predictor names needed
      max_radius: maximum halo radius needed
    """
    feature_specs = []
    raw_needed = set()
    max_radius = 0

    pat = re.compile(r"^(.*)_(mean|std)_w(\d+)$", re.IGNORECASE)

    for vn in var_names:
        vn_str = str(vn)
        m = pat.match(vn_str)
        if m:
            base_name = m.group(1)
            stat = m.group(2).lower()
            win_size = int(m.group(3))
            feature_specs.append(
                {
                    "type": "window",
                    "name": vn_str,
                    "base_name": base_name,
                    "stat": stat,
                    "win_size": win_size,
                }
            )
            raw_needed.add(base_name)
            max_radius = max(max_radius, win_size // 2)
        else:
            feature_specs.append(
                {
                    "type": "raw",
                    "name": vn_str,
                    "base_name": vn_str,
                }
            )
            raw_needed.add(vn_str)

    return feature_specs, sorted(raw_needed), max_radius


def build_feature_cube_for_window(
    raw_arrays_expanded: dict,
    feature_specs: list,
    core_h: int,
    core_w: int,
    halo: int,
    min_valid_frac: float,
):
    """
    Builds a 3D cube (core_h, core_w, n_features) in exact model var_names order.
    """
    feature_layers = []
    cache = {}

    for spec in feature_specs:
        if spec["type"] == "raw":
            arr_exp = raw_arrays_expanded[spec["base_name"]]
            if halo > 0:
                layer = arr_exp[halo:halo + core_h, halo:halo + core_w]
            else:
                layer = arr_exp[:core_h, :core_w]
            feature_layers.append(layer)
        else:
            key = (spec["base_name"], spec["stat"], spec["win_size"])
            if key not in cache:
                arr_exp = raw_arrays_expanded[spec["base_name"]]
                mean_arr, std_arr = focal_mean_std_simple(
                    arr_exp,
                    spec["win_size"],
                    min_valid_frac=min_valid_frac,
                )
                cache[(spec["base_name"], "mean", spec["win_size"])] = mean_arr
                cache[(spec["base_name"], "std", spec["win_size"])] = std_arr

            full_layer = cache[key]
            if halo > 0:
                layer = full_layer[halo:halo + core_h, halo:halo + core_w]
            else:
                layer = full_layer[:core_h, :core_w]
            feature_layers.append(layer)

    return np.stack(feature_layers, axis=-1)


def main():
    print("=== RUNNING XGBOOST PRODUCTION (WINDOW-FRIENDLY) ===")
    print("Model:     ", MODEL_PATH)
    print("Predictors:", PREDICTOR_DIR)
    print("ChannelMap:", CHANNEL_MAP_PATH)
    print("Output:    ", OUTPUT_DIR)

    aoi_path = find_aoi_path(AOI_CANDIDATES)
    if aoi_path is not None:
        print("AOI:       ", aoi_path)
    else:
        print("AOI:        not found in candidate list")

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    if not PREDICTOR_DIR.exists():
        raise RuntimeError(f"Predictor directory not found: {PREDICTOR_DIR}")

    all_files = collect_predictor_files(PREDICTOR_DIR)
    print(f"Found {len(all_files)} candidate raster files (.tif/.tiff/.vrt)")
    file_lookup = {normalize_name(p.name): p for p in all_files}

    channel_map = load_channel_map(CHANNEL_MAP_PATH)
    loaded = joblib.load(MODEL_PATH)

    if isinstance(loaded, dict):
        print("Loaded bundle dict")
        model = loaded.get("model")
        var_names = loaded.get("var_names")
        categorical_mode = loaded.get("categorical_mode")
        excluded_predictors = loaded.get("excluded_predictors", [])
        window_sizes = loaded.get("window_sizes_px", [])
        window_include_predictors = loaded.get("window_include_predictors", [])
        window_min_valid_frac = float(loaded.get("window_min_valid_frac", DEFAULT_WINDOW_MIN_VALID_FRAC))
        print("Bundle keys:", list(loaded.keys()))
        print("categorical_mode:", categorical_mode)
        if excluded_predictors:
            print("excluded_predictors:", excluded_predictors)
        if window_sizes:
            print("window_sizes_px:", window_sizes)
        if window_include_predictors:
            print("window_include_predictors:", window_include_predictors)
    else:
        model = loaded
        var_names = None
        window_min_valid_frac = DEFAULT_WINDOW_MIN_VALID_FRAC
        print("Loaded bare model (no bundle dict)")

    if model is None:
        raise RuntimeError("No model found in joblib bundle")
    if var_names is None:
        raise RuntimeError("No var_names found in model bundle; cannot map predictors reliably.")

    print("Model type:", type(model).__name__)
    print(f"Model expects {len(var_names)} predictors")

    feature_specs, raw_predictors_needed, halo = parse_feature_specs(var_names)
    print(f"Detected {sum(fs['type']=='window' for fs in feature_specs)} window-derived predictors")
    print(f"Unique raw predictors needed: {len(raw_predictors_needed)}")
    print(f"Maximum halo radius required: {halo} pixels")

    raw_sources = {}
    missing = []
    for raw_name in raw_predictors_needed:
        src_path, band = map_raw_predictor(raw_name, PREDICTOR_DIR, channel_map, file_lookup=file_lookup)
        if src_path is None or band is None:
            missing.append(raw_name)
        elif not src_path.exists():
            missing.append(f"{raw_name} (file not found: {src_path.name})")
        else:
            raw_sources[raw_name] = (src_path, band)

    if missing:
        raise RuntimeError(
            "Some raw predictors could not be mapped via channel_map:\n - "
            + "\n - ".join(map(str, missing))
        )

    print("\nRaw predictors that will be used:")
    for vn, (src_path, band) in raw_sources.items():
        print(f" - {vn} -> {src_path.name} (band {band})")

    validate_selected_sources(raw_sources)

    raw_to_path = {vn: src_path for vn, (src_path, _) in raw_sources.items()}
    unique_paths = sorted({str(p) for p in raw_to_path.values()})
    srcs_by_path = {p: rasterio.open(p) for p in unique_paths}
    srcs_aligned_by_path = {}
    try:
        bands = {vn: band for vn, (_, band) in raw_sources.items()}
        ref_src = next(iter(srcs_by_path.values()))
        ref_meta = ref_src.meta.copy()
        rows, cols = ref_meta["height"], ref_meta["width"]

        ref_transform = ref_meta["transform"]
        ref_crs = ref_meta["crs"]

        # Align all sources to the same grid using warp (same approach as U-Net production workflows).
        srcs_aligned_by_path = {}
        for src_path, src in srcs_by_path.items():
            needs_warp = (
                (src.height != rows)
                or (src.width != cols)
                or (src.transform != ref_transform)
                or (src.crs != ref_crs)
            )
            if needs_warp:
                print(f"[INFO] Warping {Path(src.name).name} to reference grid")
                srcs_aligned_by_path[src_path] = WarpedVRT(
                    src,
                    crs=ref_crs,
                    transform=ref_transform,
                    width=cols,
                    height=rows,
                    resampling=Resampling.nearest,
                )
            else:
                srcs_aligned_by_path[src_path] = src

        # Probe first sample to infer class ordering for plain XGBoost and ORF paths alike.
        probe = np.zeros((1, len(var_names)), dtype=np.float32)
        _, class_labels = predict_proba_xgb_or_orf(model, probe)
        n_classes = len(class_labels)

        aoi_geom = load_aoi_geometry(ref_meta, aoi_path)
        row_starts, col_starts = get_aoi_block_starts(aoi_geom, ref_meta["transform"], cols, rows, BLOCK_SHAPE[0])

        out_meta = ref_meta.copy()
        out_meta.update(dtype="float32", count=1, compress="lzw", nodata=np.nan)

        prob_paths = [OUTPUT_DIR / f"xgb_prob_class_{i+1}.tif" for i in range(n_classes)]
        maxprob_path = OUTPUT_DIR / "xgb_prob_max.tif"
        class_path = OUTPUT_DIR / "xgb_class.tif"

        prob_tmp_paths = [OUTPUT_DIR / f"_tmp_xgb_prob_class_{i+1}.tif" for i in range(n_classes)]
        maxprob_tmp_path = OUTPUT_DIR / "_tmp_xgb_prob_max.tif"
        class_tmp_path = OUTPUT_DIR / "_tmp_xgb_class.tif"

        if RUN_COG_CONVERSION:
            prob_write_paths = prob_tmp_paths
            maxprob_write_path = maxprob_tmp_path
            class_write_path = class_tmp_path
        else:
            prob_write_paths = prob_paths
            maxprob_write_path = maxprob_path
            class_write_path = class_path

        prob_dsts = [rasterio.open(path, "w", **out_meta) for path in prob_write_paths]
        maxprob_dst = rasterio.open(maxprob_write_path, "w", **out_meta)
        class_dst = rasterio.open(class_write_path, "w", **out_meta)

        try:
            total_windows = len(row_starts) * len(col_starts)

            print(f"\nGrid size: {rows:,} rows x {cols:,} cols")
            print(f"Block size: {BLOCK_SHAPE[0]} x {BLOCK_SHAPE[1]}")
            print(f"Total windows to scan: {total_windows:,}")
            print("Starting blockwise prediction...", flush=True)

            start_time = time.time()
            window_counter = 0
            processed_blocks = 0
            skipped_blocks = 0
            written_blocks = 0
            total_valid_pixels = 0
            last_progress_time = start_time

            for row_off in row_starts:
                for col_off in col_starts:
                    window_counter += 1
                    core_win = Window(
                        col_off,
                        row_off,
                        min(BLOCK_SHAPE[1], cols - col_off),
                        min(BLOCK_SHAPE[0], rows - row_off),
                    )

                    core_h = int(core_win.height)
                    core_w = int(core_win.width)

                    block_aoi_2d = build_aoi_mask_for_window(
                        aoi_geom=aoi_geom,
                        ref_transform=ref_meta["transform"],
                        win=core_win,
                    )
                    aoi_pixels = int(block_aoi_2d.sum())

                    if aoi_pixels == 0:
                        skipped_blocks += 1
                        if (
                            window_counter == 1
                            or window_counter % 25 == 0
                            or (time.time() - last_progress_time) >= 30
                        ):
                            print_progress(
                                window_counter,
                                total_windows,
                                start_time,
                                processed_blocks,
                                skipped_blocks,
                                written_blocks,
                                total_valid_pixels,
                                0,
                            )
                            last_progress_time = time.time()
                        continue

                    # expanded/halo window for any focal features
                    if halo > 0:
                        expanded_win = Window(
                            col_off - halo,
                            row_off - halo,
                            core_w + 2 * halo,
                            core_h + 2 * halo,
                        )
                    else:
                        expanded_win = core_win

                    raw_arrays_expanded = {}
                    for raw_name, src_path in raw_to_path.items():
                        src = srcs_aligned_by_path[str(src_path)]
                        read_boundless = True
                        if isinstance(src, WarpedVRT):
                            # WarpedVRT does not support boundless reads.
                            read_boundless = False
                        arr_exp = read_window_as_float32(
                            src,
                            bands[raw_name],
                            expanded_win,
                            boundless=read_boundless,
                        )
                        raw_arrays_expanded[raw_name] = arr_exp

                    block_stack = build_feature_cube_for_window(
                        raw_arrays_expanded=raw_arrays_expanded,
                        feature_specs=feature_specs,
                        core_h=core_h,
                        core_w=core_w,
                        halo=halo,
                        min_valid_frac=window_min_valid_frac,
                    )

                    block_bands = block_stack.shape[2]
                    block_stack_2d = block_stack.reshape(-1, block_bands)
                    block_aoi = block_aoi_2d.reshape(-1)

                    block_nodata = np.any(~np.isfinite(block_stack_2d), axis=1)
                    valid_mask = block_aoi & (~block_nodata)
                    valid_pixels = int(valid_mask.sum())

                    processed_blocks += 1
                    last_valid_pixels = valid_pixels

                    if valid_pixels == 0:
                        if (
                            processed_blocks <= 3
                            or processed_blocks % 10 == 0
                        ):
                            print(f"    AOI pixels in current window: {aoi_pixels:,} | valid pixels: {valid_pixels:,}")
                        if (
                            window_counter == 1
                            or window_counter % 25 == 0
                            or (time.time() - last_progress_time) >= 30
                        ):
                            print_progress(
                                window_counter,
                                total_windows,
                                start_time,
                                processed_blocks,
                                skipped_blocks,
                                written_blocks,
                                total_valid_pixels,
                                last_valid_pixels,
                            )
                            last_progress_time = time.time()
                        continue

                    X_valid = block_stack_2d[valid_mask]
                    probs, _ = predict_proba_xgb_or_orf(model, X_valid)

                    prob_cube = np.full((core_h * core_w, n_classes), np.nan, dtype=np.float32)
                    prob_cube[valid_mask, :] = probs
                    prob_cube = prob_cube.reshape(core_h, core_w, n_classes)

                    maxprob = np.nanmax(prob_cube, axis=2).astype(np.float32)
                    pred_class = np.full((core_h, core_w), np.nan, dtype=np.float32)
                    pred_class_flat = pred_class.reshape(-1)
                    pred_class_flat[valid_mask] = class_labels[np.argmax(probs, axis=1)].astype(np.float32)
                    pred_class = pred_class_flat.reshape(core_h, core_w)

                    for i, dst in enumerate(prob_dsts):
                        dst.write(prob_cube[:, :, i], 1, window=core_win)
                    maxprob_dst.write(maxprob, 1, window=core_win)
                    class_dst.write(pred_class, 1, window=core_win)

                    written_blocks += 1
                    total_valid_pixels += valid_pixels

                    if (
                        processed_blocks <= 3
                        or processed_blocks % 10 == 0
                    ):
                        print(f"    AOI pixels in current window: {aoi_pixels:,} | valid pixels: {valid_pixels:,}")

                    if (
                        window_counter == 1
                        or window_counter % 25 == 0
                        or (time.time() - last_progress_time) >= 30
                    ):
                        print_progress(
                            window_counter,
                            total_windows,
                            start_time,
                            processed_blocks,
                            skipped_blocks,
                            written_blocks,
                            total_valid_pixels,
                            last_valid_pixels,
                        )
                        last_progress_time = time.time()

            print_progress(
                total_windows,
                total_windows,
                start_time,
                processed_blocks,
                skipped_blocks,
                written_blocks,
                total_valid_pixels,
                0,
            )
            print("\nAll probability rasters written (blockwise, AOI respected)")
            print(f"Processed AOI-intersecting windows: {processed_blocks:,}")
            print(f"Skipped non-AOI windows: {skipped_blocks:,}")
            print(f"Windows with at least one valid predicted pixel: {written_blocks:,}")
            print(f"Total valid predicted pixels: {total_valid_pixels:,}")
            print(f"Saved: {maxprob_path.name}")
            print(f"Saved: {class_path.name}")
            for path in prob_paths:
                print(f"Saved: {path.name}")

        finally:
            for dst in prob_dsts:
                dst.close()
            maxprob_dst.close()
            class_dst.close()

        if RUN_COG_CONVERSION:
            # Convert temporary GTiff outputs to Cloud Optimized GeoTIFFs.
            print("\nConverting outputs to COG...")
            for tmp_path, final_path in zip(prob_tmp_paths, prob_paths):
                convert_to_cog(tmp_path, final_path)
                tmp_path.unlink(missing_ok=True)
            convert_to_cog(maxprob_tmp_path, maxprob_path)
            maxprob_tmp_path.unlink(missing_ok=True)
            convert_to_cog(class_tmp_path, class_path)
            class_tmp_path.unlink(missing_ok=True)
            print("COG conversion completed.")
        else:
            print("\n[INFO] RUN_COG_CONVERSION=False: wrote GeoTIFF outputs directly (COG conversion skipped).")

    finally:
        # Close each dataset object only once (some aligned handles may alias base handles).
        closed = set()
        for src in srcs_aligned_by_path.values():
            if id(src) in closed:
                continue
            src.close()
            closed.add(id(src))
        for src in srcs_by_path.values():
            if id(src) in closed:
                continue
            src.close()
            closed.add(id(src))


if __name__ == "__main__":
    main()
