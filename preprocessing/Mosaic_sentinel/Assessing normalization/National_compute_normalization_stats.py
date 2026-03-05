"""
Compute robust normalization stats (median, IQR) for Sentinel masked tiles
Samples from valid (non-NaN) pixels to compute per-band statistics
Output: CSV with median/IQR for each band (used by Tiles training code)
"""

import os
import csv
import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# SETTINGS - EDIT THESE
# ============================================================

# Input: folder with masked Sentinel tiles (water/snow = NaN)
MASKED_TILES_DIR = r"E:\Test\National_test\masked_snow"

# Output: where to save the stats CSV
OUT_FOLDER = r"E:\Test\National_test\stats_norm"
OUT_CSV = os.path.join(OUT_FOLDER, "global_robust_stats_landonly_trimmed_weighted.csv")

# Sampling parameters - OPTIMIZED FOR NATIONWIDE (33 tiles)
# Reduced per-tile sampling to keep total runtime ~30 min (instead of hours)
WINDOW_SIZE = 512                              # Size of sampling windows
BASE_WINDOWS_PER_TILE_PER_BAND = 8            # Baseline windows per tile (was 40)
MIN_WINDOWS_PER_TILE_PER_BAND = 5             # Minimum (for low-coverage tiles)
MAX_WINDOWS_PER_TILE_PER_BAND = 30            # Maximum (for large tiles, was 120)
MIN_VALID_FRACTION_PER_WINDOW = 0.02          # Skip windows < 2% valid pixels
MAX_TRIES_PER_WINDOW = 200                    # Attempts to find valid window

# Trimming for robust stats
TRIM_LOW = 2                                   # Trim bottom 2%
TRIM_HIGH = 98                                 # Trim top 2%

print("=" * 70)
print("COMPUTE ROBUST NORMALIZATION STATS (LAND-ONLY)")
print("=" * 70)

# Create output directory
os.makedirs(OUT_FOLDER, exist_ok=True)

# Find all masked tiles
tiles = sorted([
    os.path.join(MASKED_TILES_DIR, f)
    for f in os.listdir(MASKED_TILES_DIR)
    if f.lower().endswith(('.tif', '.tiff'))
])

if not tiles:
    raise RuntimeError(f"No tiles found in {MASKED_TILES_DIR}")

print(f"\nFound {len(tiles)} masked tiles")

# Get band count from first tile
with rasterio.open(tiles[0]) as src:
    band_count = src.count
    print(f"Band count: {band_count}")

# ============================================================
# SAMPLING FUNCTION
# ============================================================

def sample_band_windows(tile_paths, band_index, num_windows):
    """
    Sample valid pixels from band across multiple tiles using random windows.
    Returns list of valid pixel values.
    """
    vals_list = []
    
    for tile_path in tile_paths:
        try:
            with rasterio.open(tile_path) as ds:
                H, W = ds.height, ds.width
                w = min(WINDOW_SIZE, W)
                h = min(WINDOW_SIZE, H)
                
                if w < 1 or h < 1:
                    continue
                
                sampling_attempts = 0
                successful_windows = 0
                
                while successful_windows < num_windows and sampling_attempts < MAX_TRIES_PER_WINDOW:
                    sampling_attempts += 1
                    
                    # Random window position
                    x_off = np.random.randint(0, max(1, W - w + 1))
                    y_off = np.random.randint(0, max(1, H - h + 1))
                    
                    try:
                        window = Window(x_off, y_off, w, h)
                        data = ds.read(band_index, window=window).astype('float32')
                        
                        # Find valid pixels (not NaN)
                        valid = ~np.isnan(data)
                        frac_valid = float(np.sum(valid)) / float(valid.size)
                        
                        # Skip if too few valid pixels
                        if frac_valid < MIN_VALID_FRACTION_PER_WINDOW:
                            continue
                        
                        # Extract valid values
                        vals = data[valid]
                        vals_list.extend(vals.tolist())
                        successful_windows += 1
                        
                    except Exception as e:
                        continue
        
        except Exception as e:
            print(f"  Warning: Could not read {os.path.basename(tile_path)}")
            continue
    
    return np.array(vals_list, dtype='float32')

# ============================================================
# COMPUTE STATS
# ============================================================

print(f"\nComputing stats for {band_count} bands...")
print("(This may take a few minutes)\n")

stats_rows = []

for band_idx in range(1, band_count + 1):
    print(f"  Band {band_idx}...", end=' ', flush=True)
    
    # Determine number of windows to sample (scaled by tile count)
    num_windows = max(
        MIN_WINDOWS_PER_TILE_PER_BAND,
        min(
            MAX_WINDOWS_PER_TILE_PER_BAND,
            BASE_WINDOWS_PER_TILE_PER_BAND * len(tiles)
        )
    )
    
    # Sample pixels
    vals = sample_band_windows(tiles, band_idx, num_windows)
    
    if len(vals) == 0:
        print("NO DATA")
        continue
    
    # Compute robust stats (trimmed percentiles)
    median = np.percentile(vals, 50)
    iqr = np.percentile(vals, TRIM_HIGH) - np.percentile(vals, TRIM_LOW)
    
    # Ensure IQR is positive
    if iqr <= 0:
        iqr = 1.0
        print(f"(IQR={iqr:.2f})")
    else:
        print(f"(n={len(vals)}, median={median:.1f}, IQR={iqr:.1f})")
    
    stats_rows.append({
        'band': band_idx,
        'median': median,
        'iqr': iqr,
        'n_samples': len(vals),
        'min': float(np.min(vals)),
        'max': float(np.max(vals)),
        'mean': float(np.mean(vals)),
        'std': float(np.std(vals)),
    })

# ============================================================
# WRITE CSV
# ============================================================

if len(stats_rows) == 0:
    print("\n⚠️  ERROR: No stats computed!")
    exit(1)

with open(OUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['band', 'median', 'iqr', 'n_samples', 'min', 'max', 'mean', 'std'])
    writer.writeheader()
    for row in stats_rows:
        writer.writerow(row)

print(f"\n{'=' * 70}")
print(f"✓ Stats saved to: {OUT_CSV}")
print(f"{'=' * 70}")
print(f"\nNext step: Use this CSV in Tiles_Spyder_ed_2026.py for training")
