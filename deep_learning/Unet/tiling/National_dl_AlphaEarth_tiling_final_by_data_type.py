"""
National DL tiling for AlphaEarth run (final version).

Rules:
1) Uses only AlphaEarth, topographic, and categorical inputs.
2) Z-score normalize ONLY topographic predictors.
3) Keep these raw (no normalization):
   - alphaearth_dequant_national_epsg25833.vrt
   - landuse_Code_18_cog
   - geol_Ca_cog
   - marine_limit_cog
   - quaternary_cog
"""

import os
import re
import csv
import glob
import shutil
import random
import time
from typing import Optional
import numpy as np
import rasterio
import geopandas as gpd
from datetime import datetime, timedelta
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from tqdm import tqdm
from shapely.geometry import box
from rasterio.windows import Window
from rasterio.features import rasterize

# ============================================================
# USER PATHS - NATIONAL ALPHAEARTH
# ============================================================
PREDICTOR_DIR = r"G:\Covariates_to_model"
POLYGON_GPKG = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2025\kalk_prosjekt3.0\MDir_data\Data_2026\kalkkart_treningsdata_ed_June2026.gpkg"
POLYGON_LAYER = "NiN_all_agg"
OUT_DIR = r"D:\DL_AE_chips\With_status_class"
TILE_METADATA_DIR = os.path.join(OUT_DIR, "tile_metadata")

# Reference grid (snap raster with water as NoData)
REF_MASK = r"G:\Covariates_to_model\Topo_dtm_ch.tif"

# AlphaEarth source tile folder (used to patch VRT source paths to absolute)
ALPHAEARTH_DIR = r"E:\Alpha_earth\dequant_images_all"

LABEL_FIELD = "KA_mean_weighted_category"
STATUS_FIELD = "KA_mean_weighted_category_status"
BACKGROUND_VALUE = 0
ONLY_CLEAN_DATA = True
CLEAN_OUTPUT_DIR_BEFORE_RUN = True

# Normalize only predictors whose names include one of these substrings.
# Matching is case-sensitive and checks the original filename stem.
NORMALIZE_IF_NAME_CONTAINS = (
    "Topo_",
    "Geophys_",
    "xgb_prediction_CaO_",
    "xgb_prediction_KESP_",
)

# ============================================================
# TILE SETTINGS
# ============================================================
TILE_SIZE = 128
MIN_LABEL_RATIO = 0.05  # Custom value
STRIDE = int(0.8 * TILE_SIZE)  # Custom stride

# Z-score estimation for topographic rasters only
Z_N_SAMPLES_PER_RASTER = 2_000_000
Z_WINDOWS_PER_RASTER = 40
Z_WINDOW_SIZE = 512

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# End-of-run visual QA previews
PREVIEW_COUNT = 3
PREVIEW_BANDS_1BASED = [1, 40, 64]
PREVIEW_ALPHA = 0.40
PREVIEW_DIR = os.path.join(OUT_DIR, "chip_previews")

# ============================================================
# OUTPUT FOLDERS
# ============================================================
X_DIR = os.path.join(OUT_DIR, "X")
Y_DIR = os.path.join(OUT_DIR, "y")
Y_STATUS_DIR = os.path.join(OUT_DIR, "y_status")
os.makedirs(X_DIR, exist_ok=True)
os.makedirs(Y_DIR, exist_ok=True)
os.makedirs(Y_STATUS_DIR, exist_ok=True)
os.makedirs(TILE_METADATA_DIR, exist_ok=True)


def clear_existing_outputs():
    for d in [X_DIR, Y_DIR, Y_STATUS_DIR]:
        for p in glob.glob(os.path.join(d, "*.npy")):
            os.remove(p)

    for p in [
        os.path.join(TILE_METADATA_DIR, "tile_metadata.csv"),
        os.path.join(TILE_METADATA_DIR, "channel_map.csv"),
    ]:
        if os.path.exists(p):
            os.remove(p)

    if os.path.exists(PREVIEW_DIR):
        shutil.rmtree(PREVIEW_DIR)


if CLEAN_OUTPUT_DIR_BEFORE_RUN:
    print("[INFO] Cleaning previous chip outputs before run...")
    clear_existing_outputs()

# ============================================================
# HELPERS
# ============================================================
def predictor_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0].lower()


def predictor_stem_raw(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def canonical_predictor_stem(path: str) -> str:
    stem = predictor_stem(path)
    if stem.endswith("_absolute_paths"):
        return stem[: -len("_absolute_paths")]
    return stem


def canonical_predictor_stem_raw(path: str) -> str:
    stem = predictor_stem_raw(path)
    if stem.endswith("_absolute_paths"):
        return stem[: -len("_absolute_paths")]
    return stem


def predictor_group(path: str) -> str:
    stem = canonical_predictor_stem(path)
    stem_raw = canonical_predictor_stem_raw(path)
    if stem == "alphaearth_dequant_national_epsg25833":
        return "alphaearth"
    if any(tag in stem_raw for tag in NORMALIZE_IF_NAME_CONTAINS):
        return "topographic"
    return "other"


def should_normalize(path: str) -> bool:
    # Normalize only predictors with explicit name patterns.
    stem = canonical_predictor_stem_raw(path)
    return any(tag in stem for tag in NORMALIZE_IF_NAME_CONTAINS)


def assert_same_grid(ds_a, ds_b, name_a="A", name_b="B"):
    if ds_a.crs != ds_b.crs:
        raise RuntimeError(f"CRS mismatch: {name_a}={ds_a.crs} vs {name_b}={ds_b.crs}")
    if ds_a.transform != ds_b.transform:
        raise RuntimeError(f"Transform mismatch: {name_a} vs {name_b}")
    if ds_a.width != ds_b.width or ds_a.height != ds_b.height:
        raise RuntimeError(f"Dimensions mismatch: {name_a} vs {name_b}")


def find_predictors(predictor_dir: str):
    tifs = sorted(glob.glob(os.path.join(predictor_dir, "*.tif")))
    vrts = sorted(glob.glob(os.path.join(predictor_dir, "*.vrt")))
    rasters = tifs + vrts
    if not rasters:
        raise RuntimeError(f"No predictors found in: {predictor_dir}")
    return rasters


def make_vrt_absolute(vrt_path: str, source_dir: str, out_dir: str):
    """
    Create a copy of VRT with absolute SourceFilename paths based on basename.
    """
    with open(vrt_path, "r", encoding="utf-8") as f:
        txt = f.read()

    def repl(m):
        open_tag, old_path, close_tag = m.groups()
        new_abs = os.path.join(source_dir, os.path.basename(old_path)).replace(
            "\\", "/"
        )
        open_tag = re.sub(r'relativeToVRT="[^"]*"', 'relativeToVRT="0"', open_tag)
        return f"{open_tag}{new_abs}{close_tag}"

    txt2 = re.sub(r"(<SourceFilename[^>]*>)([^<]+)(</SourceFilename>)", repl, txt)
    out_vrt = os.path.join(
        out_dir, os.path.basename(vrt_path).replace(".vrt", "_absolute_paths.vrt")
    )

    with open(out_vrt, "w", encoding="utf-8") as f:
        f.write(txt2)

    return out_vrt


def sample_band_for_zscore(ds, band: int, nwin: int, win_size: int, max_samples: int):
    """
    Random-window sampling to estimate mean/std for one raster band.
    """
    H, W = ds.height, ds.width
    w = min(win_size, W)
    h = min(win_size, H)
    if w <= 0 or h <= 0:
        return 0.0, 1.0

    nodata = ds.nodata
    vals = []

    max_x = W - w
    max_y = H - h

    for _ in range(nwin):
        x0 = np.random.randint(0, max_x + 1) if max_x > 0 else 0
        y0 = np.random.randint(0, max_y + 1) if max_y > 0 else 0
        arr = ds.read(band, window=Window(x0, y0, w, h)).astype(np.float32)

        if nodata is not None:
            arr[arr == nodata] = np.nan

        v = arr[~np.isnan(arr)]
        if v.size > 0:
            vals.append(v)

    if not vals:
        return 0.0, 1.0

    sample = np.concatenate(vals)
    if sample.size > max_samples:
        idx = np.random.choice(sample.size, size=max_samples, replace=False)
        sample = sample[idx]

    mean = float(np.mean(sample))
    std = float(np.std(sample))
    if std < 1e-6:
        std = 1.0
    return mean, std


def build_land_mask(ref_arr: np.ndarray, ref_nodata):
    if ref_nodata is None:
        return ~np.isnan(ref_arr)
    return (~np.isnan(ref_arr)) & (ref_arr != ref_nodata)


def format_seconds(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def save_preview_png(tile_name: str, band_1based: int, out_png: str):
    """
    Save one visual QA PNG for a tile: grayscale predictor band + class overlay.

    NaN pixels are rendered as white for quick visual checks.
    """
    import matplotlib.pyplot as plt

    x_path = os.path.join(X_DIR, f"{tile_name}.npy")
    y_path = os.path.join(Y_DIR, f"{tile_name}.npy")
    X = np.load(x_path)
    y = np.load(y_path)

    band_idx = max(0, min(band_1based - 1, X.shape[0] - 1))
    band = X[band_idx].astype(np.float32)

    finite = np.isfinite(band)
    if np.any(finite):
        vmin = float(np.percentile(band[finite], 2))
        vmax = float(np.percentile(band[finite], 98))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        gray = np.clip((band - vmin) / (vmax - vmin), 0.0, 1.0)
    else:
        gray = np.zeros_like(band, dtype=np.float32)

    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb[~finite] = 1.0  # NaN -> white

    class_colors = {
        1: np.array([0.20, 0.75, 0.20], dtype=np.float32),
        2: np.array([0.98, 0.75, 0.25], dtype=np.float32),
        3: np.array([0.90, 0.25, 0.25], dtype=np.float32),
    }

    overlay = rgb.copy()
    for cls, color in class_colors.items():
        m = y == cls
        if np.any(m):
            overlay[m] = (1.0 - PREVIEW_ALPHA) * overlay[m] + PREVIEW_ALPHA * color

    plt.figure(figsize=(5, 5), dpi=150)
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close()


def create_end_run_previews(tile_names):
    """
    Create a small set of preview PNGs for visual confirmation.
    """
    if not tile_names:
        print("Preview PNGs skipped: no tiles created.")
        return []

    os.makedirs(PREVIEW_DIR, exist_ok=True)
    n = min(PREVIEW_COUNT, len(tile_names))
    sampled = random.sample(tile_names, k=n)

    preview_files = []
    for i, tile_name in enumerate(sampled, start=1):
        desired_band = PREVIEW_BANDS_1BASED[(i - 1) % len(PREVIEW_BANDS_1BASED)]
        out_png = os.path.join(
            PREVIEW_DIR,
            f"preview_{i:02d}_{tile_name}_band{desired_band:02d}.png",
        )
        save_preview_png(tile_name=tile_name, band_1based=desired_band, out_png=out_png)
        preview_files.append(out_png)

    return preview_files

# ============================================================
# LOAD REFERENCE GRID
# ============================================================
print("=" * 70)
print("LOADING REFERENCE GRID")
print("=" * 70)

ref_ds = rasterio.open(REF_MASK)
ref_crs = ref_ds.crs
ref_transform = ref_ds.transform
ref_nodata = ref_ds.nodata
W, H = ref_ds.width, ref_ds.height

print(f"REF_MASK: {REF_MASK}")
print(f"Size: {W} x {H} | CRS: {ref_crs} | NoData: {ref_nodata}")

# ============================================================
# LOAD PREDICTORS
# ============================================================
print("\n" + "=" * 70)
print("LOADING PREDICTORS")
print("=" * 70)

predictor_paths = find_predictors(PREDICTOR_DIR)

# Patch AlphaEarth VRT to absolute paths, if present
patched_paths = []
for p in predictor_paths:
    if os.path.basename(p).lower() == "alphaearth_dequant_national_epsg25833.vrt":
        original_vrt = p
        patched_vrt = make_vrt_absolute(original_vrt, ALPHAEARTH_DIR, OUT_DIR)
        use_patched = True
        try:
            with rasterio.open(patched_vrt) as ds_test:
                if (
                    ds_test.crs != ref_ds.crs
                    or ds_test.transform != ref_ds.transform
                    or ds_test.width != ref_ds.width
                    or ds_test.height != ref_ds.height
                ):
                    use_patched = False
        except Exception as e:
            print(f"Patched AlphaEarth VRT open failed, using original VRT: {e}")
            use_patched = False

        if use_patched:
            p = patched_vrt
            print(f"Patched AlphaEarth VRT: {p}")
        else:
            p = original_vrt
            print("Patched AlphaEarth VRT grid mismatch; using original VRT instead.")
    patched_paths.append(p)

predictor_paths = patched_paths

predictors = []
normalized_predictor_files = []
raw_predictor_files = []
for p in predictor_paths:
    ds = rasterio.open(p)
    try:
        assert_same_grid(ds, ref_ds, os.path.basename(p), "REF_MASK")
    except RuntimeError:
        # Align mismatched predictors (e.g., AlphaEarth VRT) to the reference grid on the fly.
        print(f"Grid mismatch for {os.path.basename(p)}; using WarpedVRT to REF grid.")
        ds = WarpedVRT(
            ds,
            crs=ref_crs,
            transform=ref_transform,
            width=W,
            height=H,
            resampling=Resampling.bilinear,
        )
    predictors.append((p, ds))
    norm_flag = should_normalize(p)
    norm_mode = "Z-score" if norm_flag else "RAW"
    print(f"{os.path.basename(p):45s} bands={ds.count:2d} norm={norm_mode}")
    if norm_flag:
        normalized_predictor_files.append(os.path.basename(p))
    else:
        raw_predictor_files.append(os.path.basename(p))

print("\n" + "-" * 70)
print("NORMALIZATION SUMMARY")
print("-" * 70)
print("Rules (case-sensitive substrings):")
for tag in NORMALIZE_IF_NAME_CONTAINS:
    print(f"  - {tag}")
print(f"Z-score predictors ({len(normalized_predictor_files)}):")
for name in normalized_predictor_files:
    print(f"  - {name}")
print(f"RAW predictors ({len(raw_predictor_files)}):")
for name in raw_predictor_files:
    print(f"  - {name}")

# Build and save channel mapping for downstream training/embeddings
channel_map = []
channel_idx = 0
for p, ds in predictors:
    stem = canonical_predictor_stem(p)
    group = predictor_group(p)
    normalized = should_normalize(p)
    for band in range(1, ds.count + 1):
        channel_map.append(
            {
                "channel_idx": channel_idx,
                "predictor_name": stem,
                "source_file": os.path.basename(p),
                "band": band,
                "group": group,
                "normalized": int(normalized),
            }
        )
        channel_idx += 1

channel_map_csv = os.path.join(TILE_METADATA_DIR, "channel_map.csv")
with open(channel_map_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(channel_map[0].keys()))
    writer.writeheader()
    writer.writerows(channel_map)

print(f"Channel map saved: {channel_map_csv}")
print(f"Total channels in X: {len(channel_map)}")

# ============================================================
# LOAD LABELS
# ============================================================
print("\n" + "=" * 70)
print("LOADING LABEL POLYGONS")
print("=" * 70)

gdf = gpd.read_file(POLYGON_GPKG, layer=POLYGON_LAYER)
if gdf.crs != ref_crs:
    gdf = gdf.to_crs(ref_crs)

class_order = ["low", "medium", "high"]
class_map = {k: i + 1 for i, k in enumerate(class_order)}
status_order = ["clean", "dirty"]
status_map = {k: i + 1 for i, k in enumerate(status_order)}

if LABEL_FIELD not in gdf.columns:
    raise KeyError(f"Label field '{LABEL_FIELD}' not found in: {POLYGON_GPKG}")
if STATUS_FIELD not in gdf.columns:
    raise KeyError(f"Status field '{STATUS_FIELD}' not found in: {POLYGON_GPKG}")

gdf["label"] = gdf[LABEL_FIELD].map(class_map)
gdf["status"] = gdf[STATUS_FIELD].map(status_map)

# Drop polygons with missing/unmapped label or status
before = len(gdf)
gdf = gdf.dropna(subset=["label", "status"]).copy()
gdf["label"] = gdf["label"].astype(np.uint8)
gdf["status"] = gdf["status"].astype(np.uint8)
print(f"Labels/status loaded: {before} -> {len(gdf)} after mapping")

if ONLY_CLEAN_DATA:
    before_clean = len(gdf)
    gdf = gdf[gdf["status"] == status_map["clean"]].copy()
    print(f"Clean-only filter enabled: {before_clean} -> {len(gdf)} polygons")

# ============================================================
# COMPUTE Z-SCORE STATS (TOPOGRAPHIC ONLY)
# ============================================================
print("\n" + "=" * 70)
print("COMPUTING Z-SCORE STATS (TOPOGRAPHIC ONLY)")
print("=" * 70)

z_stats: dict[str, Optional[list[tuple[float, float]]]] = {}
for p, ds in predictors:
    if not should_normalize(p):
        z_stats[p] = None
        continue

    per_band = []
    for b in range(1, ds.count + 1):
        m, s = sample_band_for_zscore(
            ds=ds,
            band=b,
            nwin=Z_WINDOWS_PER_RASTER,
            win_size=Z_WINDOW_SIZE,
            max_samples=Z_N_SAMPLES_PER_RASTER,
        )
        per_band.append((m, s))
    z_stats[p] = per_band

# ============================================================
# TILE GENERATION
# ============================================================
print("\n" + "=" * 70)
print("GENERATING TILES")
print("=" * 70)

tile_id = 0
tiles_filtered = 0
tiles_skipped = 0
metadata = []

row_starts = list(range(0, H, STRIDE))
col_starts = list(range(0, W, STRIDE))
total_rows = len(row_starts)
ETA_REPORT_EVERY_ROWS = 25
run_start_ts = time.time()

for row_idx, row in enumerate(tqdm(row_starts, desc="Rows"), start=1):
    for col in col_starts:
        if row + TILE_SIZE > H or col + TILE_SIZE > W:
            tiles_skipped += 1
            continue

        win = Window(col, row, TILE_SIZE, TILE_SIZE)
        bounds = rasterio.windows.bounds(win, ref_transform)
        tile_geom = box(*bounds)

        sub = gdf[gdf.intersects(tile_geom)]
        if sub.empty:
            tiles_skipped += 1
            continue

        # Rasterize labels
        y = rasterize(
            shapes=[(geom, int(lbl)) for geom, lbl in zip(sub.geometry, sub["label"])],
            out_shape=(TILE_SIZE, TILE_SIZE),
            transform=ref_ds.window_transform(win),
            fill=BACKGROUND_VALUE,
            dtype=np.uint8,
        )
        y_status = rasterize(
            shapes=[(geom, int(st)) for geom, st in zip(sub.geometry, sub["status"])],
            out_shape=(TILE_SIZE, TILE_SIZE),
            transform=ref_ds.window_transform(win),
            fill=BACKGROUND_VALUE,
            dtype=np.uint8,
        )

        label_ratio = float(np.mean(y > 0))
        if label_ratio < MIN_LABEL_RATIO:
            tiles_filtered += 1
            continue

        # Land mask from reference raster
        ref_arr = ref_ds.read(1, window=win).astype(np.float32)
        if ref_nodata is not None:
            ref_arr[ref_arr == ref_nodata] = np.nan
        land = build_land_mask(ref_arr, ref_nodata)

        if int(np.sum(land)) < 50:
            tiles_filtered += 1
            continue

        channels = []
        for p, ds in predictors:
            arr = ds.read(list(range(1, ds.count + 1)), window=win).astype(np.float32)

            nod = ds.nodata
            if nod is not None:
                arr[arr == nod] = np.nan

            # mask water
            arr[:, ~land] = np.nan

            # normalize only topographic
            stats = z_stats[p]
            if stats is not None:
                for bi in range(arr.shape[0]):
                    mean, std = stats[bi]
                    arr[bi] = (arr[bi] - mean) / std

            for bi in range(arr.shape[0]):
                channels.append(arr[bi])

        X = np.stack(channels, axis=0).astype(np.float32)

        # If any channel is NaN on pixel => all channels NaN + background label
        any_nan = np.isnan(X).any(axis=0)
        X[:, any_nan] = np.nan
        y[any_nan] = BACKGROUND_VALUE
        y_status[any_nan] = BACKGROUND_VALUE

        if np.all(np.isnan(X)):
            tiles_filtered += 1
            continue

        tile_name = f"tile_{tile_id:06d}"
        np.save(os.path.join(X_DIR, f"{tile_name}.npy"), X)
        np.save(os.path.join(Y_DIR, f"{tile_name}.npy"), y)
        np.save(os.path.join(Y_STATUS_DIR, f"{tile_name}.npy"), y_status)

        clean_ratio = float(np.mean(y_status == status_map["clean"]))
        dirty_ratio = float(np.mean(y_status == status_map["dirty"]))

        metadata.append(
            {
                "tile_id": tile_name,
                "col": col,
                "row": row,
                "xmin": bounds[0],
                "ymin": bounds[1],
                "xmax": bounds[2],
                "ymax": bounds[3],
                "label_ratio": label_ratio,
                "clean_ratio": clean_ratio,
                "dirty_ratio": dirty_ratio,
                "valid_pixels": int(np.sum(~any_nan)),
            }
        )

        tile_id += 1

    if row_idx % ETA_REPORT_EVERY_ROWS == 0 or row_idx == total_rows:
        elapsed_sec = time.time() - run_start_ts
        rows_per_sec = row_idx / max(elapsed_sec, 1e-9)
        remaining_rows = total_rows - row_idx
        eta_sec = remaining_rows / rows_per_sec if rows_per_sec > 0 else 0.0
        finish_dt = datetime.now() + timedelta(seconds=eta_sec)
        print(
            f"[ETA] rows {row_idx}/{total_rows} | elapsed {format_seconds(elapsed_sec)} "
            f"| remaining {format_seconds(eta_sec)} | finish ~ {finish_dt:%Y-%m-%d %H:%M}"
        )

# ============================================================
# SAVE METADATA
# ============================================================
meta_csv = os.path.join(TILE_METADATA_DIR, "tile_metadata.csv")
if metadata:
    with open(meta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata[0].keys()))
        writer.writeheader()
        writer.writerows(metadata)

preview_pngs = []
try:
    preview_pngs = create_end_run_previews([m["tile_id"] for m in metadata])
except Exception as e:
    print(f"Preview PNG generation failed: {e}")

print("\nDone.")
print(f"Tiles created : {tile_id}")
print(f"Tiles filtered: {tiles_filtered}")
print(f"Tiles skipped : {tiles_skipped}")
print(f"Output folder : {OUT_DIR}")
print(f"Metadata CSV  : {meta_csv if metadata else 'No tiles created'}")
print(f"Status tiles  : {Y_STATUS_DIR}")
if preview_pngs:
    print(f"Preview PNGs : {len(preview_pngs)} in {PREVIEW_DIR}")
