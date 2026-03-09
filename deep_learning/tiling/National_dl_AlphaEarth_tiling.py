"""
National DL tiling for AlphaEarth run.

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
import random
import time
import numpy as np
import rasterio
import geopandas as gpd
from datetime import datetime, timedelta

from tqdm import tqdm
from shapely.geometry import box
from rasterio.windows import Window
from rasterio.features import rasterize

# ============================================================
# USER PATHS - NATIONAL ALPHAEARTH
# ============================================================
PREDICTOR_DIR = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\Covariates_to_model"
POLYGON_GPKG = r"C:\Users\acosta_pedro\Miljødirektoratet\Endre Grüner Ofstad - kalkkart\Data\kalkkart_treningsdata.gpkg"
POLYGON_LAYER = "NiN_all_agg"
OUT_DIR = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\DL_AE_chips"
TILE_METADATA_DIR = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\DL_chips_spatial_location"

# Reference grid (snap raster with water as NoData)
REF_MASK = r"E:\Alpha_earth\qc\alphaearth_mosaic_epsg25833_band1_qc_cog.tif"

# AlphaEarth source tile folder (used to patch VRT source paths to absolute)
ALPHAEARTH_DIR = r"E:\Alpha_earth\dequant_images_all"

LABEL_FIELD = "KA_mean_weighted_category"
BACKGROUND_VALUE = 0

# Do NOT normalize these predictors
NON_NORMALIZED_PREDICTOR_STEMS = {
    "alphaearth_dequant_national_epsg25833",
    "landuse_code_18_cog",
    "geology_ca_icp_coe_cog",
    "geology_logca_icp_cog",
    "geol_ca_cog",
    "marine_limit_cog",
    "quaternary_cog",
    "quaternary_forenkletk_cog",
}

# ============================================================
# TILE SETTINGS
# ============================================================
TILE_SIZE = 128
MIN_LABEL_RATIO = 0.15

# Z-score estimation for topographic rasters only
Z_N_SAMPLES_PER_RASTER = 2_000_000
Z_WINDOWS_PER_RASTER = 40
Z_WINDOW_SIZE = 512

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ============================================================
# OUTPUT FOLDERS
# ============================================================
X_DIR = os.path.join(OUT_DIR, "X")
Y_DIR = os.path.join(OUT_DIR, "y")
os.makedirs(X_DIR, exist_ok=True)
os.makedirs(Y_DIR, exist_ok=True)
os.makedirs(TILE_METADATA_DIR, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================
def predictor_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0].lower()


def canonical_predictor_stem(path: str) -> str:
    stem = predictor_stem(path)
    if stem.endswith("_absolute_paths"):
        return stem[: -len("_absolute_paths")]
    return stem


def predictor_group(path: str) -> str:
    stem = canonical_predictor_stem(path)
    if stem == "alphaearth_dequant_national_epsg25833":
        return "alphaearth"
    if stem in {
        "landuse_code_18_cog",
        "geol_ca_cog",
        "marine_limit_cog",
        "quaternary_cog",
        "quaternary_forenkletk_cog",
        "geology_ca_icp_coe_cog",
        "geology_logca_icp_cog",
    }:
        return "categorical"
    return "topographic"


def should_normalize(path: str) -> bool:
    # Only topographic predictors are normalized.
    stem = predictor_stem(path)
    if stem in NON_NORMALIZED_PREDICTOR_STEMS:
        return False
    if stem.endswith("_absolute_paths"):
        original_stem = stem[: -len("_absolute_paths")]
        if original_stem in NON_NORMALIZED_PREDICTOR_STEMS:
            return False
    return True


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
        p = make_vrt_absolute(p, ALPHAEARTH_DIR, OUT_DIR)
        print(f"Patched AlphaEarth VRT: {p}")
    patched_paths.append(p)

predictor_paths = patched_paths

predictors = []
for p in predictor_paths:
    ds = rasterio.open(p)
    assert_same_grid(ds, ref_ds, os.path.basename(p), "REF_MASK")
    predictors.append((p, ds))
    norm_mode = "Z-score" if should_normalize(p) else "RAW"
    print(f"{os.path.basename(p):45s} bands={ds.count:2d} norm={norm_mode}")

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
gdf["label"] = gdf[LABEL_FIELD].map(class_map)

# Drop polygons with missing/unmapped label
before = len(gdf)
gdf = gdf.dropna(subset=["label"]).copy()
gdf["label"] = gdf["label"].astype(np.uint8)
print(f"Labels loaded: {before} -> {len(gdf)} after mapping")

# ============================================================
# COMPUTE Z-SCORE STATS (TOPOGRAPHIC ONLY)
# ============================================================
print("\n" + "=" * 70)
print("COMPUTING Z-SCORE STATS (TOPOGRAPHIC ONLY)")
print("=" * 70)

z_stats: dict[str, list[tuple[float, float]] | None] = {}
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

row_positions = list(range(0, H, TILE_SIZE))
total_rows = len(row_positions)
ETA_REPORT_EVERY_ROWS = 25
run_start_ts = time.time()

for row_idx, row in enumerate(tqdm(row_positions, desc="Rows"), start=1):
    for col in range(0, W, TILE_SIZE):
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

        if np.all(np.isnan(X)):
            tiles_filtered += 1
            continue

        tile_name = f"tile_{tile_id:06d}"
        np.save(os.path.join(X_DIR, f"{tile_name}.npy"), X)
        np.save(os.path.join(Y_DIR, f"{tile_name}.npy"), y)

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

print("\nDone.")
print(f"Tiles created : {tile_id}")
print(f"Tiles filtered: {tiles_filtered}")
print(f"Tiles skipped : {tiles_skipped}")
print(f"Output folder : {OUT_DIR}")
print(f"Metadata CSV  : {meta_csv if metadata else 'No tiles created'}")
