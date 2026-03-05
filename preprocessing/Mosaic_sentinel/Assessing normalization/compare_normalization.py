"""
Compare normalization: water-only vs water+snow
Shows improvement from snow removal
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
import os

# Load stats
WATER_STATS = r"E:\Test\Trondelag_test\stats_norm\global_robust_stats_landonly_trimmed_weighted.csv"
SNOW_STATS = r"E:\Test\Trondelag_test\stats_norm\global_robust_stats_landonly_snowmasked.csv"
MASKED_SNOW_DIR = r"E:\Test\Trondelag_test\masked_snow"

water_df = pd.read_csv(WATER_STATS)
snow_df = pd.read_csv(SNOW_STATS)

print("=" * 70)
print("COMPARING: WATER-ONLY vs WATER+SNOW NORMALIZATION")
print("=" * 70)

# Sample from snow-masked tiles for normalization comparison
tiles = sorted([
    os.path.join(MASKED_SNOW_DIR, f)
    for f in os.listdir(MASKED_SNOW_DIR)
    if f.lower().endswith('.tif')
])

N_SAMPLES = 50000
band_count = 10

print(f"\nSampling {N_SAMPLES} pixels per band...")

all_norm_water = {}
all_norm_snow = {}

for band_idx in range(1, band_count + 1):
    raw_vals = []
    
    # Sample from snow-masked tiles
    attempts = 0
    while len(raw_vals) < N_SAMPLES and attempts < 5000:
        attempts += 1
        ds = rasterio.open(tiles[np.random.randint(0, len(tiles))])
        
        W, H = ds.width, ds.height
        w = min(256, W)
        h = min(256, H)
        
        x = np.random.randint(0, max(1, W - w + 1))
        y = np.random.randint(0, max(1, H - h + 1))
        
        try:
            window = Window(x, y, w, h)
            data = ds.read(band_idx, window=window).astype('float32')
            valid = ~np.isnan(data)
            if np.sum(valid) < 10:
                ds.close()
                continue
            
            vals = data[valid]
            need = N_SAMPLES - len(raw_vals)
            if vals.size > need:
                idx = np.random.choice(vals.size, size=need, replace=False)
                vals = vals[idx]
            
            raw_vals.extend(vals.tolist())
            ds.close()
        except:
            ds.close()
            continue
    
    raw_vals = np.array(raw_vals[:N_SAMPLES], dtype='float32')
    
    # Normalize with both stats
    water_row = water_df[water_df['band'] == band_idx].iloc[0]
    snow_row = snow_df[snow_df['band'] == band_idx].iloc[0]
    
    norm_water = (raw_vals - water_row['median']) / water_row['iqr'] if water_row['iqr'] > 0 else raw_vals
    norm_snow = (raw_vals - snow_row['median']) / snow_row['iqr'] if snow_row['iqr'] > 0 else raw_vals
    
    all_norm_water[band_idx] = norm_water
    all_norm_snow[band_idx] = norm_snow
    
    print(f"  Band {band_idx}: sampled {len(raw_vals)} pixels")

# ============================================================
# PLOT: Box-whisker comparison (water vs water+snow)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

bands_to_plot = list(range(1, band_count + 1))

# Water-only
data_water = [all_norm_water[b] for b in bands_to_plot]
ax = axes[0]
bp1 = ax.boxplot(data_water, labels=bands_to_plot, patch_artist=True,
                  widths=0.6, showfliers=True,
                  boxprops=dict(facecolor='#ff7f0e', alpha=0.7, edgecolor='black', linewidth=1.5),
                  medianprops=dict(color='red', linewidth=2.5),
                  whiskerprops=dict(color='black', linewidth=1.5),
                  capprops=dict(color='black', linewidth=1.5),
                  flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.5, linestyle='none'))
ax.axhline(y=0, color='black', linestyle='-', linewidth=2, label='Target Mean = 0')
ax.axhline(y=1, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label='±1σ')
ax.axhline(y=-1, color='orange', linestyle='--', linewidth=1.5, alpha=0.6)
ax.set_xlabel('Band', fontsize=13, fontweight='bold')
ax.set_ylabel('Normalized Value', fontsize=13, fontweight='bold')
ax.set_title('Water-Only Masking\n(Original)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(-5, 8)

# Water+Snow
data_snow = [all_norm_snow[b] for b in bands_to_plot]
ax = axes[1]
bp2 = ax.boxplot(data_snow, labels=bands_to_plot, patch_artist=True,
                  widths=0.6, showfliers=True,
                  boxprops=dict(facecolor='#2ca02c', alpha=0.7, edgecolor='black', linewidth=1.5),
                  medianprops=dict(color='red', linewidth=2.5),
                  whiskerprops=dict(color='black', linewidth=1.5),
                  capprops=dict(color='black', linewidth=1.5),
                  flierprops=dict(marker='o', markerfacecolor='darkgreen', markersize=4, alpha=0.5, linestyle='none'))
ax.axhline(y=0, color='black', linestyle='-', linewidth=2, label='Target Mean = 0')
ax.axhline(y=1, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label='±1σ')
ax.axhline(y=-1, color='orange', linestyle='--', linewidth=1.5, alpha=0.6)
ax.set_xlabel('Band', fontsize=13, fontweight='bold')
ax.set_ylabel('Normalized Value', fontsize=13, fontweight='bold')
ax.set_title('Water + Snow Masking\n(Improved)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(-5, 8)

plt.suptitle('Normalization Improvement: Snow Removal Impact on Distribution', fontsize=16, fontweight='bold')
plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), 'comparison_water_vs_snowmasked.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_path}")
plt.close()

# ============================================================
# TABLE: Statistics comparison
# ============================================================
print("\n" + "=" * 70)
print("STATISTICS COMPARISON")
print("=" * 70)

comparison_data = []
for band in bands_to_plot:
    water_mean = np.mean(all_norm_water[band])
    water_std = np.std(all_norm_water[band])
    water_outliers = np.sum(np.abs(all_norm_water[band]) > 5)
    
    snow_mean = np.mean(all_norm_snow[band])
    snow_std = np.std(all_norm_snow[band])
    snow_outliers = np.sum(np.abs(all_norm_snow[band]) > 5)
    
    outlier_reduction = ((water_outliers - snow_outliers) / max(water_outliers, 1)) * 100
    
    comparison_data.append({
        'Band': band,
        'Water_Mean': f"{water_mean:.3f}",
        'Snow_Mean': f"{snow_mean:.3f}",
        'Water_Std': f"{water_std:.3f}",
        'Snow_Std': f"{snow_std:.3f}",
        'Water_Outliers': water_outliers,
        'Snow_Outliers': snow_outliers,
        'Outlier_Reduction_%': f"{outlier_reduction:.1f}%"
    })

comp_df = pd.DataFrame(comparison_data)
print(comp_df.to_string(index=False))

print("\n" + "=" * 70)
print("KEY FINDINGS")
print("=" * 70)

total_water_outliers = sum([np.sum(np.abs(all_norm_water[b]) > 5) for b in bands_to_plot])
total_snow_outliers = sum([np.sum(np.abs(all_norm_snow[b]) > 5) for b in bands_to_plot])
total_reduction = ((total_water_outliers - total_snow_outliers) / max(total_water_outliers, 1)) * 100

print(f"Total outliers (water-only): {total_water_outliers:,}")
print(f"Total outliers (water+snow): {total_snow_outliers:,}")
print(f"Overall outlier reduction: {total_reduction:.1f}%")
print("=" * 70)
