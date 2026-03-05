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
MASKED_DIR = r"E:\Test\Trondelag_test\masked"
STATS_CSV = r"E:\Test\Trondelag_test\stats_norm\global_robust_stats_landonly_trimmed_weighted.csv"
N_SAMPLES = 20000  # Samples per band (more stable)
WINDOW_SIZE = 256
MIN_VALID_FRACTION = 0.02
MAX_ATTEMPTS = 4000

print("=" * 70)
print("SENTINEL NORMALIZATION QUALITY ASSESSMENT")
print("=" * 70)

# Load stats
stats_df = pd.read_csv(STATS_CSV)
print(f"\nLoaded stats for {len(stats_df)} bands")

# List masked tiles
tile_paths = [
    os.path.join(MASKED_DIR, fn)
    for fn in os.listdir(MASKED_DIR)
    if fn.lower().endswith(".tif") and "_masked" in fn.lower()
]
if not tile_paths:
    raise RuntimeError(f"No masked tiles found in: {MASKED_DIR}")

# Open tiles once for efficiency
tile_datasets = [rasterio.open(p) for p in tile_paths]
band_count = tile_datasets[0].count
print(f"Found {len(tile_datasets)} masked tiles with {band_count} bands")

# Sample random windows
print(f"\nSampling {N_SAMPLES} pixels per band from masked tiles...")
results = []

for band_idx in range(1, band_count + 1):
    raw_vals = []

    attempts = 0
    while len(raw_vals) < N_SAMPLES and attempts < MAX_ATTEMPTS:
        attempts += 1
        ds = tile_datasets[np.random.randint(0, len(tile_datasets))]

        w = min(WINDOW_SIZE, ds.width)
        h = min(WINDOW_SIZE, ds.height)
        if w < 1 or h < 1:
            continue

        x = np.random.randint(0, max(1, ds.width - w + 1))
        y = np.random.randint(0, max(1, ds.height - h + 1))

        try:
            window = Window(x, y, w, h)
            data = ds.read(band_idx, window=window).astype("float32")
            valid = ~np.isnan(data)
            frac = float(np.sum(valid)) / float(valid.size)
            if frac < MIN_VALID_FRACTION:
                continue

            vals = data[valid]
            need = N_SAMPLES - len(raw_vals)
            if vals.size > need:
                idx = np.random.choice(vals.size, size=need, replace=False)
                vals = vals[idx]

            raw_vals.extend(vals.tolist())
        except Exception:
            continue
    
    print(f"  Band {band_idx}: {len(raw_vals)} valid pixels sampled")
    
    if len(raw_vals) > 10:  # Lower threshold
        raw_arr = np.array(raw_vals, dtype="float32")

        stats_row = stats_df[stats_df['band'] == band_idx].iloc[0]
        median = float(stats_row['median'])
        iqr = float(stats_row['iqr'])
        if iqr <= 0:
            print(f"  ⚠️  Band {band_idx}: IQR <= 0, skipping normalization stats")
            continue

        norm_arr = (raw_arr - median) / iqr

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

if len(results_df) == 0:
    print("⚠️  No valid samples collected! Check data/paths.")
else:
    print("\nBand-by-band summary:")
    print(results_df[['Band', 'N_samples', 'Norm_mean', 'Norm_std', 'Norm_min', 'Norm_max']].to_string(index=False))

if len(results_df) == 0:
    for ds in tile_datasets:
        ds.close()
    raise SystemExit(1)

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

# Save results
output_dir = os.path.dirname(os.path.abspath(__file__))
results_csv = os.path.join(output_dir, "normalization_results.csv")
summary_txt = os.path.join(output_dir, "normalization_summary.txt")
results_df.to_csv(results_csv, index=False)

summary_lines = []
summary_lines.append("Band-by-band summary:")
summary_lines.append(results_df[['Band', 'N_samples', 'Norm_mean', 'Norm_std', 'Norm_min', 'Norm_max']].to_string(index=False))
summary_lines.append("")

if centering_issue.any():
    summary_lines.append(f"CENTERING ISSUE: {centering_issue.sum()} bands not centered at 0")
    summary_lines.append(results_df[centering_issue][['Band', 'Norm_mean']].to_string(index=False))
else:
    summary_lines.append("CENTERING OK: All bands centered near 0")

summary_lines.append("")
if spread_issue.any():
    summary_lines.append(f"SPREAD ISSUE: {spread_issue.sum()} bands have unusual std dev")
    summary_lines.append(results_df[spread_issue][['Band', 'Norm_std']].to_string(index=False))
else:
    summary_lines.append("SPREAD OK: All bands have std ~1")

summary_lines.append("")
if outlier_issue.any():
    summary_lines.append(f"OUTLIER ISSUE: {outlier_issue.sum()} bands have many extreme values")
    summary_lines.append(results_df[outlier_issue][['Band', 'Outliers_pos', 'Outliers_neg']].to_string(index=False))
else:
    summary_lines.append("OUTLIERS OK: All bands within reasonable bounds")

summary_lines.append("")
if range_issue.any():
    summary_lines.append(f"RANGE ISSUE: {range_issue.sum()} bands have low variability")
    summary_lines.append(results_df[range_issue][['Band', 'Norm_q05', 'Norm_q95']].to_string(index=False))
else:
    summary_lines.append("RANGE OK: All bands well-distributed")

summary_lines.append("")
summary_lines.append("SUMMARY: Normalization is " + ("✓ HEALTHY" if not (centering_issue.any() or spread_issue.any() or outlier_issue.any()) else "⚠️ NEEDS ATTENTION"))

with open(summary_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print(f"\nSaved results to: {results_csv}")
print(f"Saved summary to: {summary_txt}")

for ds in tile_datasets:
    ds.close()
