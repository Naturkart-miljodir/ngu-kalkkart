"""
Quick diagnostic: Assess Sentinel normalization quality
Checks for centering, spread, outliers, and NaN handling
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import warnings
warnings.filterwarnings('ignore')

# Configuration
SENTINEL_VRT = r"E:\Test\Trondelag_test\Predictors\Trondelag_mosaic.vrt"
DTM_FILE = r"E:\Test\Trondelag_test\Predictors\Trondelag_DTM_aligned.tif"  # Use DTM as land reference
STATS_CSV = r"E:\Test\Trondelag_test\stats_norm\global_robust_stats_landonly_trimmed_weighted.csv"
N_SAMPLES = 3000  # Samples per band

print("=" * 70)
print("SENTINEL NORMALIZATION QUALITY ASSESSMENT")
print("=" * 70)

# Load stats
stats_df = pd.read_csv(STATS_CSV)
print(f"\nLoaded stats for {len(stats_df)} bands")

# Open VRT and DTM (as land mask reference)
sentinel_ds = rasterio.open(SENTINEL_VRT)
mask_ds = rasterio.open(DTM_FILE)
H, W = sentinel_ds.height, sentinel_ds.width
print(f"Raster size: {H} x {W}")

# Sample random windows
print(f"\nSampling {N_SAMPLES} pixels per band from random locations...")
results = []

for band_idx in range(1, sentinel_ds.count + 1):
    raw_vals = []
    norm_vals = []
    
    for _ in range(N_SAMPLES):
        # Random pixel
        x = np.random.randint(0, W)
        y = np.random.randint(0, H)
        
        # Read single pixel
        try:
            window = Window(x, y, 1, 1)
            raw_pixel = sentinel_ds.read(band_idx, window=window)[0, 0]
            mask_pixel = mask_ds.read(1, window=window)[0, 0]
            
            # Only use land pixels (mask != nodata and not nan)
            if mask_pixel > 0 and not np.isnan(mask_pixel) and not np.isnan(raw_pixel):
                raw_vals.append(raw_pixel)
                
                # Apply normalization
                stats_row = stats_df[stats_df['band'] == band_idx].iloc[0]
                median = stats_row['median']
                iqr = stats_row['iqr']
                
                if iqr > 0:
                    norm_val = (raw_pixel - median) / iqr
                    norm_vals.append(norm_val)
        except:
            continue
    
    if len(raw_vals) > 0:
        raw_arr = np.array(raw_vals)
        norm_arr = np.array(norm_vals)
        
        # Statistics
        stats = {
            'Band': band_idx,
            'N_samples': len(raw_vals),
            'Raw_mean': np.mean(raw_arr),
            'Raw_std': np.std(raw_arr),
            'Raw_min': np.min(raw_arr),
            'Raw_max': np.max(raw_arr),
            'Norm_mean': np.mean(norm_arr),
            'Norm_std': np.std(norm_arr),
            'Norm_min': np.min(norm_arr),
            'Norm_max': np.max(norm_arr),
            'Norm_q95': np.percentile(norm_arr, 95),
            'Norm_q05': np.percentile(norm_arr, 5),
            'Outliers_pos': np.sum(norm_arr > 5),
            'Outliers_neg': np.sum(norm_arr < -5),
        }
        results.append(stats)

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("NORMALIZATION RESULTS")
print("=" * 70)

print("\nBand-by-band summary:")
print(results_df[['Band', 'N_samples', 'Norm_mean', 'Norm_std', 'Norm_min', 'Norm_max']].to_string(index=False))

print("\n" + "-" * 70)
print("QUALITY CHECKS:")
print("-" * 70)

# Check 1: Centering (normalized mean should be ~0)
centering_issue = results_df['Norm_mean'].abs() > 0.5
if centering_issue.any():
    print(f"\n⚠️  CENTERING ISSUE: {centering_issue.sum()} bands not centered at 0")
    print(results_df[centering_issue][['Band', 'Norm_mean']])
else:
    print(f"\n✓ CENTERING OK: All bands centered near 0")

# Check 2: Spread (normalized std should be ~1)
spread_issue = (results_df['Norm_std'] < 0.7) | (results_df['Norm_std'] > 1.5)
if spread_issue.any():
    print(f"\n⚠️  SPREAD ISSUE: {spread_issue.sum()} bands have unusual std dev")
    print(results_df[spread_issue][['Band', 'Norm_std']])
else:
    print(f"\n✓ SPREAD OK: All bands have std ~1")

# Check 3: Outliers (too many extreme values)
outlier_issue = (results_df['Outliers_pos'] > 10) | (results_df['Outliers_neg'] > 10)
if outlier_issue.any():
    print(f"\n⚠️  OUTLIER ISSUE: {outlier_issue.sum()} bands have many extreme values")
    print(results_df[outlier_issue][['Band', 'Outliers_pos', 'Outliers_neg']])
else:
    print(f"\n✓ OUTLIERS OK: All bands within reasonable bounds")

# Check 4: Range coverage
range_issue = (results_df['Norm_q95'] - results_df['Norm_q05']) < 3
if range_issue.any():
    print(f"\n⚠️  RANGE ISSUE: {range_issue.sum()} bands have low variability")
    print(results_df[range_issue][['Band', 'Norm_q05', 'Norm_q95']])
else:
    print(f"\n✓ RANGE OK: All bands well-distributed")

print("\n" + "=" * 70)
print("SUMMARY: Normalization is", "✓ HEALTHY" if not (centering_issue.any() or spread_issue.any() or outlier_issue.any()) else "⚠️ NEEDS ATTENTION")
print("=" * 70)

sentinel_ds.close()
mask_ds.close()
