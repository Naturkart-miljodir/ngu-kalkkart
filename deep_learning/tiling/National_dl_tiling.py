"""
Deep Learning Tiling Script for National Test Area
Adapted from Trondelag_dl_tiling.py

Combines:
- Normalized Sentinel bands (from robustly-computed stats)
- Topographic predictors (z-score normalized)
- Sentinel spectral indices (NDVI, NDWI, NDMI, etc.)

Output: X (features) and y (labels) as .npy files
"""
import os
import glob
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import random

# ============================================================
# USER PATHS - NATIONAL TEST
# ============================================================
PREDICTOR_DIR = r"E:\Test\National_test\Predictors"
POLYGON_GPKG  = r"C:\Users\acosta_pedro\Miljødirektoratet\Endre Grüner Ofstad - kalkkart\Data\kalkkart_treningsdata.gpkg"
POLYGON_LAYER = "NiN_all_agg"
OUT_DIR       = r"E:\Test\National_test\DL_tiles"

# Reference grid (snap raster with water as NoData)
DTM_MASK = r"E:\Test\National_test\Mask\National_mask_aligned.tif"

# Robust stats for Sentinel bands (computed by National_compute_normalization_stats.py)
SENTINEL_STATS_CSV = r"E:\Test\National_test\stats_norm\global_robust_stats_landonly_trimmed_weighted.csv"

# Directory containing masked Sentinel tiles (separate from PREDICTOR_DIR)
SENTINEL_MASKED_DIR = r"E:\Test\National_test\masked_snow"

LABEL_FIELD = "KA_mean_weighted_category"
BACKGROUND_VALUE = 0

# Auto-detect Sentinel VRT from PREDICTOR_DIR (or specify explicitly)
SENTINEL_VRT_EXPLICIT = None

# ============================================================
# TILE SETTINGS
# ============================================================
TILE_SIZE = 128
MIN_LABEL_RATIO = 0.15

# Sentinel values: treat 0 as NoData (outside coverage / mosaic edges)
SENTINEL_ZERO_IS_NODATA = True

# Z-score normalization for non-sentinel predictors
Z_N_SAMPLES_PER_RASTER = 2_000_000
Z_WINDOWS_PER_RASTER = 40
Z_WINDOW_SIZE = 512

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ============================================================
# OUTPUT FOLDERS
# ============================================================
os.makedirs(os.path.join(OUT_DIR, "X"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "y"), exist_ok=True)
RANDOM_TILE_DIR = os.path.join(OUT_DIR, "Random_tile_pic")
os.makedirs(RANDOM_TILE_DIR, exist_ok=True)

# ============================================================
# SENTINEL BAND MAPPING
# ============================================================
# Adjust if your 10-band Sentinel stack differs
S2 = {
    "BLUE":  2,   # Band_2
    "GREEN": 3,   # Band_3
    "RED":   4,   # Band_4
    "NIR":   8,   # Band_8
    "SWIR1": 9,   # Band_9
    "SWIR2": 10,  # Band_10
}

# ============================================================
# HELPERS
# ============================================================
def safe_div(a, b, eps=1e-8, max_val=1.0):
    """
    Safe division for spectral indices.
    
    - Uses small epsilon (1e-8) to stabilize division.
    - Clips result to [-max_val, max_val]; values outside are set to NaN.
    - Handles NaN propagation: if either input is NaN, output is NaN.
    
    Args:
        a: numerator (array or scalar)
        b: denominator (array or scalar)
        eps: small constant to avoid division by zero (default 1e-8)
        max_val: clip result to interval [-max_val, max_val]; default 1.0
    
    Returns:
        result: clipped array/scalar, with out-of-bounds values set to NaN
    """
    # Ensure both are float for NaN handling
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    
    # Compute division, preserving NaN from inputs
    with np.errstate(divide='ignore', invalid='ignore'):
        result = a / (b + eps)
    
    # Clip to physical bounds and mark out-of-bounds as NaN
    result[(result < -max_val) | (result > max_val)] = np.nan
    
    return result

def load_sentinel_stats(csv_path, min_iqr=1e-6):
    """Load robust normalization stats (median/IQR) from CSV."""
    stats = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            b = int(row["band"])
            med = float(row["median"])
            iqr = max(float(row["iqr"]), min_iqr)
            stats[b] = (med, iqr)
    return stats

def find_predictors(predictor_dir, sentinel_vrt_explicit=None):
    """Locate all predictor rasters and identify Sentinel VRT."""
    tifs = sorted(glob.glob(os.path.join(predictor_dir, "*.tif")))
    vrts = sorted(glob.glob(os.path.join(predictor_dir, "*.vrt")))

    sentinel_vrt = None
    if sentinel_vrt_explicit:
        sentinel_vrt = sentinel_vrt_explicit
    else:
        # Auto-detect: first .vrt containing "sentinel" or "Sentinel"
        for v in vrts:
            if "sentinel" in os.path.basename(v).lower():
                sentinel_vrt = v
                break
        if sentinel_vrt is None and vrts:
            sentinel_vrt = vrts[0]  # Fallback to first VRT

    if sentinel_vrt is None:
        raise RuntimeError(
            "Could not find Sentinel VRT. "
            "Set SENTINEL_VRT_EXPLICIT or ensure a .vrt exists in PREDICTOR_DIR."
        )

    # Other predictors: all TIFs + non-Sentinel VRTs
    other_predictors = tifs + [v for v in vrts if v != sentinel_vrt]

    return other_predictors, sentinel_vrt

def assert_same_grid(ds_a, ds_b, name_a="A", name_b="B"):
    """Verify that two datasets share CRS, pixel size, and geotransform."""
    if ds_a.crs != ds_b.crs:
        raise RuntimeError(
            f"CRS mismatch: {name_a} vs {name_b}\n"
            f"  {ds_a.crs}\n  {ds_b.crs}"
        )
    if (ds_a.transform.a != ds_b.transform.a) or (ds_a.transform.e != ds_b.transform.e):
        raise RuntimeError(f"Pixel size mismatch: {name_a} vs {name_b}")
    if ds_a.transform != ds_b.transform:
        raise RuntimeError(f"Transform mismatch (not snapped): {name_a} vs {name_b}")

def land_mask_from_dtm(mask_arr, mask_nodata):
    """Create boolean mask: True = land (not nodata and not nan)."""
    if mask_nodata is None:
        return ~np.isnan(mask_arr)
    return (~np.isnan(mask_arr)) & (mask_arr != mask_nodata)

def sample_raster_for_zscore(ds, ref_ds, nwin=40, win=512, max_samples=2_000_000):
    """
    Estimate global mean/std via random window sampling.
    Used for Z-score normalization of non-Sentinel predictors.
    """
    H, W = ref_ds.height, ref_ds.width
    w = min(win, W)
    h = min(win, H)
    max_x0 = W - w
    max_y0 = H - h
    
    if max_x0 < 0 or max_y0 < 0:
        return (0.0, 1.0)

    nodata = ds.nodata
    vals = []

    for _ in range(nwin):
        col0 = np.random.randint(0, max_x0 + 1)
        row0 = np.random.randint(0, max_y0 + 1)
        window = Window(col0, row0, w, h)
        arr = ds.read(1, window=window).astype("float32")

        if nodata is not None:
            arr[arr == nodata] = np.nan

        v = arr.ravel()
        v = v[~np.isnan(v)]
        if v.size:
            vals.append(v)

    if not vals:
        return (0.0, 1.0)

    allv = np.concatenate(vals)
    if allv.size > max_samples:
        idx = np.random.choice(allv.size, size=max_samples, replace=False)
        allv = allv[idx]

    mean = float(np.mean(allv))
    std = float(np.std(allv))
    if std < 1e-6:
        std = 1.0
    
    return (mean, std)

# ============================================================
# LOAD REFERENCE GRID (DTM MASK)
# ============================================================
print("=" * 70)
print("LOADING REFERENCE GRID (DTM MASK)")
print("=" * 70)

mask_ds = rasterio.open(DTM_MASK)
ref = mask_ds

width, height = ref.width, ref.height
transform = ref.transform
crs = ref.crs
mask_nodata = ref.nodata

print(f"Reference grid: {DTM_MASK}")
print(f"  Size: {width} x {height} pixels")
print(f"  Tile size: {TILE_SIZE}")
print(f"  CRS: {crs}")
print(f"  NoData value: {mask_nodata}")

# ============================================================
# LOAD PREDICTORS
# ============================================================
print("\n" + "=" * 70)
print("LOADING PREDICTORS")
print("=" * 70)

other_paths, sentinel_vrt = find_predictors(PREDICTOR_DIR, SENTINEL_VRT_EXPLICIT)

# CRITICAL: If using masked Sentinel directory, update VRT paths to absolute paths
if os.path.exists(SENTINEL_MASKED_DIR):
    print(f"Updating VRT paths to use SENTINEL_MASKED_DIR: {SENTINEL_MASKED_DIR}")
    with open(sentinel_vrt, 'r') as f:
        vrt_content = f.read()
    
    # Create a temp VRT with absolute paths in OUT_DIR (not PREDICTOR_DIR to avoid confusion)
    temp_vrt_name = os.path.basename(sentinel_vrt).replace('.vrt', '_absolute_paths.vrt')
    temp_vrt = os.path.join(OUT_DIR, temp_vrt_name)
    import re
    # Replace relative paths like "National_tile_XX_YY_masked_snow.tif" with full paths
    updated_content = re.sub(
        r'(SourceFilename.*?)National_tile_\d{2}_\d{2}_masked_snow\.tif',
        lambda m: m.group(1) + os.path.join(SENTINEL_MASKED_DIR, os.path.basename(m.group(0)).replace(m.group(1), '')),
        vrt_content
    )
    # Also try to replace any relative path with absolute
    updated_content = re.sub(
        r'SourceFilename relativeToVRT="[01]">([^<]+National_tile[^<]*\.tif)',
        lambda m: f'SourceFilename relativeToVRT="0">{os.path.join(SENTINEL_MASKED_DIR, os.path.basename(m.group(1)))}',
        updated_content
    )
    with open(temp_vrt, 'w') as f:
        f.write(updated_content)
    sentinel_vrt = temp_vrt
    print(f"  Using temporary VRT with absolute paths: {temp_vrt}")

print(f"\nSentinel VRT: {sentinel_vrt}")
print(f"Other predictors: {len(other_paths)}")

# Open Sentinel
sentinel_ds = rasterio.open(sentinel_vrt)
assert_same_grid(sentinel_ds, ref, "Sentinel VRT", "DTM mask")
sent_band_count = sentinel_ds.count
print(f"  Sentinel bands: {sent_band_count}")
print(f"  ✓ Reading ALL {sent_band_count} bands from Sentinel VRT")

# Open other predictors
other_ds = []
for p in other_paths:
    ds = rasterio.open(p)
    assert_same_grid(ds, ref, os.path.basename(p), "DTM mask")
    other_ds.append((p, ds))
    print(f"  {os.path.basename(p):40s}: {ds.count} band(s)")

# ============================================================
# LOAD POLYGONS (LABELS)
# ============================================================
print("\n" + "=" * 70)
print("LOADING LABELS")
print("=" * 70)

# Read from GeoPackage with specific layer
gdf = gpd.read_file(POLYGON_GPKG, layer=POLYGON_LAYER)

# Check and reproject if needed
if gdf.crs != crs:
    print(f"⚠ Polygon CRS mismatch!")
    print(f"  Polygon CRS: {gdf.crs}")
    print(f"  Covariate CRS: {crs}")
    print(f"  Reprojecting polygons to {crs}...")
    gdf = gdf.to_crs(crs)
else:
    print(f"Labels CRS: {crs} ✓")

CLASS_ORDER = ["low", "medium", "high"]
CLASS_MAP = {cls: i + 1 for i, cls in enumerate(CLASS_ORDER)}
gdf["label"] = gdf[LABEL_FIELD].map(CLASS_MAP).astype(np.uint8)
print(f"Classes: {CLASS_ORDER}")
print(f"Loaded {len(gdf)} label polygons")

# ============================================================
# LOAD SENTINEL ROBUST STATS
# ============================================================
print("\n" + "=" * 70)
print("LOADING SENTINEL NORMALIZATION STATS")
print("=" * 70)

sent_stats = load_sentinel_stats(SENTINEL_STATS_CSV)
print(f"Loaded stats for {len(sent_stats)} bands")

if max(sent_stats.keys()) != sent_band_count:
    print(f"\n⚠ WARNING: Sentinel stats bands ({max(sent_stats.keys())}) != "
          f"Sentinel VRT bands ({sent_band_count})")

for b in sorted(sent_stats.keys()):
    med, iqr = sent_stats[b]
    print(f"  Band {b:02d}: median={med:10.3f}, iqr={iqr:10.3f}")

# ============================================================
# COMPUTE Z-SCORE STATS FOR OTHER PREDICTORS
# ============================================================
print("\n" + "=" * 70)
print("ESTIMATING Z-SCORE STATS (OTHER PREDICTORS)")
print("=" * 70)

z_stats = {}
for p, ds in other_ds:
    name = os.path.basename(p)
    if ds.count != 1:
        print(f"{name}: {ds.count} bands (will z-score each using band1 stats)")
    
    mean, std = sample_raster_for_zscore(
        ds, ref, 
        nwin=Z_WINDOWS_PER_RASTER, 
        win=Z_WINDOW_SIZE, 
        max_samples=Z_N_SAMPLES_PER_RASTER
    )
    z_stats[p] = (mean, std)
    print(f"  {name:40s}: mean={mean:10.4f}, std={std:10.4f}")

# ============================================================
# TILE GENERATION
# ============================================================
print("\n" + "=" * 70)
print("GENERATING TILES")
print("=" * 70)
print(f"Grid: {width} x {height} pixels, tile size: {TILE_SIZE}")
print(f"Total possible tiles: {(width // TILE_SIZE) * (height // TILE_SIZE)}")
print()

pixel_counts = {0: 0, 1: 0, 2: 0, 3: 0}
tile_id = 0
tiles_skipped = 0
tiles_filtered = 0

# Initialize metadata tracking for spatial block CV
tile_metadata = []

# Calculate total iterations for progress bar
total_rows = (height + TILE_SIZE - 1) // TILE_SIZE

with tqdm(total=total_rows, desc="Processing rows", unit="row") as pbar_rows:
    for row in range(0, height, TILE_SIZE):
        for col in range(0, width, TILE_SIZE):

            # Skip partial tiles at edges
            if (row + TILE_SIZE > height) or (col + TILE_SIZE > width):
                tiles_skipped += 1
                continue

            win = Window(col_off=col, row_off=row, width=TILE_SIZE, height=TILE_SIZE)
            bounds = rasterio.windows.bounds(win, transform)
            tile_geom = box(*bounds)

            # Find labels in this tile
            sub = gdf[gdf.intersects(tile_geom)]
            if len(sub) == 0:
                tiles_skipped += 1
                continue

            shapes = [(geom, int(lbl)) for geom, lbl in zip(sub.geometry, sub["label"])]

            # Rasterize labels
            y = rasterize(
                shapes=shapes,
                out_shape=(TILE_SIZE, TILE_SIZE),
                transform=ref.window_transform(win),
                fill=BACKGROUND_VALUE,
                dtype=np.uint8
            )

            # Count pixels BEFORE filtering
            unique, counts = np.unique(y, return_counts=True)
            for u, c in zip(unique, counts):
                pixel_counts[int(u)] += int(c)

            # Skip tiles with too little labelled area
            label_ratio = np.sum(y > 0) / (TILE_SIZE * TILE_SIZE)
            if label_ratio < MIN_LABEL_RATIO:
                tiles_filtered += 1
                continue

            # Read DTM mask and build land mask
            dtm_arr = mask_ds.read(1, window=win).astype("float32")
            if mask_nodata is not None:
                dtm_arr[dtm_arr == mask_nodata] = np.nan
            land = land_mask_from_dtm(dtm_arr, mask_nodata)

            # Skip if too little land
            if np.sum(land) < 50:
                tiles_filtered += 1
                continue

            # ------------------------------------------
            # 1) OTHER PREDICTORS (Z-SCORE)
            # ------------------------------------------
            bands = []
            valid_all_covariates = np.ones((TILE_SIZE, TILE_SIZE), dtype=bool)

            for p, ds in other_ds:
                mean, std = z_stats[p]

                if ds.count == 1:
                    arr = ds.read(1, window=win).astype("float32")
                    nod = ds.nodata
                    if nod is not None:
                        arr[arr == nod] = np.nan

                    arr[~land] = np.nan
                    
                    # Track which pixels are valid for this covariate
                    valid_cov = ~np.isnan(arr)
                    valid_all_covariates = valid_all_covariates & valid_cov
                    
                    arr = (arr - mean) / std
                    bands.append(arr)

                else:
                    # Multiband: z-score each band using band1 stats
                    arr = ds.read(list(range(1, ds.count + 1)), window=win).astype("float32")
                    nod = ds.nodata
                    if nod is not None:
                        arr[arr == nod] = np.nan
                    arr[:, ~land] = np.nan
                    
                    # All bands must be valid for a pixel to be included
                    for b in range(arr.shape[0]):
                        valid_cov = ~np.isnan(arr[b])
                        valid_all_covariates = valid_all_covariates & valid_cov
                    
                    arr = (arr - mean) / std
                    bands.extend([arr[i] for i in range(arr.shape[0])])

            # ------------------------------------------
            # 2) SENTINEL BANDS (ROBUST NORMALIZE)
            # ------------------------------------------
            sent = sentinel_ds.read(
                list(range(1, sent_band_count + 1)), 
                window=win
            ).astype("float32")

            # Handle NoData
            snod = sentinel_ds.nodata
            if snod is not None:
                sent[sent == snod] = np.nan

            # Treat 0 as NoData (mosaic edges)
            if SENTINEL_ZERO_IS_NODATA:
                sent[sent == 0] = np.nan

            # Apply land mask
            sent[:, ~land] = np.nan

            # Track valid pixels: ALL Sentinel bands must be valid
            for b in range(sent_band_count):
                valid_sent = ~np.isnan(sent[b])
                valid_all_covariates = valid_all_covariates & valid_sent

            # Robust normalize each band: (raw - median) / iqr
            sentN = np.empty_like(sent, dtype="float32")
            for bi in range(1, sent_band_count + 1):
                med, iqr = sent_stats.get(bi, (0.0, 1.0))
                sentN[bi - 1] = (sent[bi - 1] - med) / iqr

            # Append normalized Sentinel bands
            for i in range(sentN.shape[0]):
                bands.append(sentN[i])

            # ------------------------------------------
            # 3) SENTINEL SPECTRAL INDICES (5)
            # ------------------------------------------
            # CRITICAL: Compute indices from RAW (pre-normalized) Sentinel bands
            # to avoid division instability when normalized bands cross zero.
            # After computing indices, they will inherit the land mask.
            
            def get_band_raw(arr10, key):
                """Extract band from RAW (unnormalized) Sentinel stack."""
                idx = S2[key] - 1
                if idx < 0 or idx >= arr10.shape[0]:
                    raise RuntimeError(
                        f"Band mapping for {key} out of range. "
                        f"Edit S2 dict to match your 10-band stack."
                    )
                return arr10[idx]

            BLUE_raw  = get_band_raw(sent, "BLUE")
            GREEN_raw = get_band_raw(sent, "GREEN")
            RED_raw   = get_band_raw(sent, "RED")
            NIR_raw   = get_band_raw(sent, "NIR")
            SW1_raw   = get_band_raw(sent, "SWIR1")
            SW2_raw   = get_band_raw(sent, "SWIR2")

            # Compute 5 indices from RAW bands
            # safe_div guards division stability and clips to [-1.0, 1.0]
            NDVI = safe_div((NIR_raw - RED_raw),   (NIR_raw + RED_raw), max_val=1.0)
            NDWI = safe_div((GREEN_raw - NIR_raw), (GREEN_raw + NIR_raw), max_val=1.0)
            NDMI = safe_div((NIR_raw - SW1_raw),   (NIR_raw + SW1_raw), max_val=1.0)
            NDSI = safe_div((GREEN_raw - SW1_raw), (GREEN_raw + SW1_raw), max_val=1.0)
            NBR  = safe_div((NIR_raw - SW2_raw),   (NIR_raw + SW2_raw), max_val=1.0)

            # Mask indices (track valid pixels across indices)
            for idx_arr in [NDVI, NDWI, NDMI, NDSI, NBR]:
                idx_arr[~land] = np.nan
                valid_idx = ~np.isnan(idx_arr)
                valid_all_covariates = valid_all_covariates & valid_idx
                bands.append(idx_arr.astype("float32"))

            # ------------------------------------------
            # APPLY COMPREHENSIVE NODATA MASK
            # ------------------------------------------
            # Stack all bands into (C, H, W)
            X = np.stack(bands, axis=0).astype("float32")
            
            # CRITICAL: Ensure pixel consistency across all channels
            # If ANY channel has NaN at a pixel, ALL channels get NaN there
            # This prevents mixed data (e.g., valid DTM but NaN Sentinel)
            has_any_nan = np.isnan(X).any(axis=0)  # Shape: (H, W)
            X[:, has_any_nan] = np.nan  # Mask all channels where ANY is NaN
            
            # Count valid pixels
            n_valid = np.sum(~has_any_nan)
            if n_valid == 0:
                tiles_filtered += 1
                continue  # Skip tiles with no valid pixels
            
            # Also mask labels where covariates are invalid
            y[has_any_nan] = BACKGROUND_VALUE

            tile_name = f"tile_{tile_id:06d}"
            np.save(os.path.join(OUT_DIR, "X", f"{tile_name}.npy"), X)
            np.save(os.path.join(OUT_DIR, "y", f"{tile_name}.npy"), y)

            # Record metadata for spatial block CV
            # Calculate geographic center and bounds
            center_lon = (bounds[0] + bounds[2]) / 2
            center_lat = (bounds[1] + bounds[3]) / 2
            
            # Count labels (excluding background)
            label_counts = {cls: 0 for cls in range(1, len(CLASS_ORDER) + 1)}
            unique_labels, label_pixel_counts = np.unique(y[y > 0], return_counts=True)
            for lbl, cnt in zip(unique_labels, label_pixel_counts):
                label_counts[int(lbl)] = int(cnt)
            
            tile_metadata.append({
                "tile_id": tile_name,
                "pixel_col": col,
                "pixel_row": row,
                "lon_min": bounds[0],
                "lat_min": bounds[1],
                "lon_max": bounds[2],
                "lat_max": bounds[3],
                "center_lon": center_lon,
                "center_lat": center_lat,
                "n_valid_pixels": int(n_valid),
                "label_ratio": float(label_ratio),
                "n_label_low": label_counts.get(1, 0),
                "n_label_medium": label_counts.get(2, 0),
                "n_label_high": label_counts.get(3, 0)
            })

            tile_id += 1
            
            # Update progress bar
            pbar_rows.set_description(
                f"Processing rows | Created: {tile_id} | Filtered: {tiles_filtered} | Skipped: {tiles_skipped}"
            )
        
        pbar_rows.update(1)

print(f"\n✓ Tile generation complete!")
print(f"✓ Total tiles created: {tile_id}")
print(f"✓ Tiles filtered (insufficient labels/data): {tiles_filtered}")

# Save metadata for spatial block CV
if tile_metadata:
    import pandas as pd
    metadata_df = pd.DataFrame(tile_metadata)
    metadata_csv = os.path.join(OUT_DIR, "tile_metadata.csv")
    metadata_df.to_csv(metadata_csv, index=False)
    print(f"\n✓ Tile metadata saved: {metadata_csv}")
    print(f"  Use this for spatial block CV and regional subsetting")
print(f"✓ Tiles skipped (edge/no polygons): {tiles_skipped}")
print(f"✓ Saved to: {OUT_DIR}")

# ============================================================
# PIXEL COUNT SUMMARY
# ============================================================
summary_path = os.path.join(OUT_DIR, "class_pixel_counts.txt")
with open(summary_path, "w") as f:
    f.write("Pixel counts across ALL tiles (before filtering):\n\n")
    for cls, count in pixel_counts.items():
        label = "background" if cls == 0 else CLASS_ORDER[cls - 1] if cls - 1 < len(CLASS_ORDER) else "unknown"
        f.write(f"  Class {cls} ({label}): {count}\n")
    total_pix = sum(pixel_counts.values())
    f.write(f"\nTotal pixels analysed: {total_pix}\n")

print(f"\n✔ Pixel counts: {summary_path}")
print(pixel_counts)

# ============================================================
# POST-TILING AUDIT: Verify Index Computation Fix
# ============================================================
def audit_tiles(x_dir, sample_fraction=0.25, n_channels_expected=16):
    """
    Audit output tiles to verify index computation fix.
    Reports per-channel statistics, focusing on indices (CH11-CH15).
    """
    print("\n" + "=" * 80)
    print("POST-TILING AUDIT: Index Computation Verification")
    print("=" * 80)
    
    tile_paths = sorted(glob.glob(os.path.join(x_dir, "*.npy")))
    if not tile_paths:
        print("[WARNING] No tiles found in audit directory.")
        return
    
    # Sample tiles for audit (to avoid processing all for time reasons)
    n_tiles = len(tile_paths)
    sample_size = max(1, int(n_tiles * sample_fraction))
    sample_idx = np.random.choice(n_tiles, size=sample_size, replace=False)
    audit_paths = [tile_paths[i] for i in sorted(sample_idx)]
    
    print(f"\nAuditing {sample_size}/{n_tiles} tiles ({sample_fraction*100:.0f}% sample)...")
    
    # Initialize accumulators
    nan_counts = np.zeros(n_channels_expected, dtype=np.int64)
    finite_counts = np.zeros(n_channels_expected, dtype=np.int64)
    mins = np.full(n_channels_expected, np.inf, dtype=np.float32)
    maxs = np.full(n_channels_expected, -np.inf, dtype=np.float32)
    
    # Index channels (CH11-CH15 in 1-based, 10-14 in 0-based)
    index_ch_0based = [10, 11, 12, 13, 14]
    outside_bounds = np.zeros(len(index_ch_0based), dtype=np.int64)
    index_finite = np.zeros(len(index_ch_0based), dtype=np.int64)
    
    for tile_path in tqdm(audit_paths, desc="Auditing tiles", unit="tile"):
        X_tile = np.load(tile_path)
        
        # Ensure channels-last format for consistent indexing
        if X_tile.ndim == 3:
            if X_tile.shape[0] < X_tile.shape[1]:
                # channels-first: convert to channels-last view
                X_ch = np.transpose(X_tile, (1, 2, 0))
            else:
                # already channels-last
                X_ch = X_tile
        else:
            print(f"[WARNING] Unexpected shape {X_tile.shape} in {os.path.basename(tile_path)}")
            continue
        
        if X_ch.shape[-1] != n_channels_expected:
            print(f"[WARNING] Channel count mismatch: expected {n_channels_expected}, got {X_ch.shape[-1]}")
            continue
        
        # Per-channel statistics
        for c in range(n_channels_expected):
            ch_data = X_ch[..., c]
            is_nan = np.isnan(ch_data)
            nan_counts[c] += np.sum(is_nan)
            
            finite_data = ch_data[~is_nan]
            finite_counts[c] += len(finite_data)
            
            if len(finite_data) > 0:
                mins[c] = min(mins[c], float(np.min(finite_data)))
                maxs[c] = max(maxs[c], float(np.max(finite_data)))
        
        # Index outlier check (CH11-CH15)
        for i, c in enumerate(index_ch_0based):
            ch_data = X_ch[..., c]
            finite_data = ch_data[~np.isnan(ch_data)]
            index_finite[i] += len(finite_data)
            
            if len(finite_data) > 0:
                outliers = np.sum((finite_data < -1.0) | (finite_data > 1.0))
                outside_bounds[i] += outliers
    
    # Print results
    print("\n" + "-" * 80)
    print("PER-CHANNEL SUMMARY (1-based channel ID):")
    print("-" * 80)
    for c in range(n_channels_expected):
        total_px = nan_counts[c] + finite_counts[c]
        nan_pct = (nan_counts[c] / total_px * 100.0) if total_px > 0 else 0.0
        ch_type = "TOPO" if c == 0 else f"SENT_{c}" if c < 11 else ["NDVI", "NDWI", "NDMI", "NDSI", "NBR"][c - 11]
        
        if np.isfinite(mins[c]):
            print(f"  CH{c+1:02d} ({ch_type:>10s}): "
                  f"finite={finite_counts[c]:>12,d}  nan={nan_counts[c]:>12,d}  "
                  f"nan%={nan_pct:6.2f}  min={mins[c]:10.4f}  max={maxs[c]:10.4f}")
        else:
            print(f"  CH{c+1:02d} ({ch_type:>10s}): [NO DATA]")
    
    # Index-specific check
    print("\n" + "-" * 80)
    print("SPECTRAL INDEX CHECK (CH11-CH15, expected range: [-1.0, 1.0]):")
    print("-" * 80)
    index_names = ["NDVI", "NDWI", "NDMI", "NDSI", "NBR"]
    for i, c in enumerate(index_ch_0based):
        finite = index_finite[i]
        out_of_bounds = outside_bounds[i]
        
        if finite > 0:
            outlier_pct = (out_of_bounds / finite) * 100.0
            status = "✓ PASS" if outlier_pct < 1.0 else "✗ FAIL"
            print(f"  CH{c+1:02d} ({index_names[i]:>5s}): {finite:>12,d} valid  "
                  f"{out_of_bounds:>12,d} out-of-bounds  ({outlier_pct:6.2f}%)  {status}")
        else:
            print(f"  CH{c+1:02d} ({index_names[i]:>5s}): [NO DATA]")
    
    print("\n" + "-" * 80)
    print("INTERPRETATION:")
    print("-" * 80)
    print("✓ EXPECTED (after fix):  Outlier % < 1.0% for all indices (CH11-CH15)")
    print("✗ PROBLEM (before fix): Outlier % of 15%-42% for CH12-CH15")
    print("\nIf most indices show < 1% outliers, the raw-band fix is working correctly.")
    print("=" * 80 + "\n")

# Run audit on output tiles
audit_tiles(os.path.join(OUT_DIR, "X"), sample_fraction=0.25)
# RANDOM TILE PREVIEW
# ============================================================
X_paths = sorted(glob.glob(os.path.join(OUT_DIR, "X", "*.npy")))
y_paths = sorted(glob.glob(os.path.join(OUT_DIR, "y", "*.npy")))

if len(X_paths) > 0:
    idx = random.randint(0, len(X_paths) - 1)
    X_rand = np.load(X_paths[idx])
    y_rand = np.load(y_paths[idx])

    # Display first channel (typically topographic covariate)
    bg = X_rand[0]

    plt.figure(figsize=(8, 8))
    plt.imshow(bg, cmap="gray")
    plt.imshow(np.ma.masked_where(y_rand == 0, y_rand), cmap="autumn", alpha=0.6)
    plt.axis("off")
    plt.title("Random Tile Preview (channel 0)")

    out_png = os.path.join(RANDOM_TILE_DIR, "random_tile.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✔ Random tile preview: {out_png}")
else:
    print("⚠ No tiles created — random preview not saved.")

print("\n" + "=" * 70)
print("TILING COMPLETE")
print("=" * 70)
