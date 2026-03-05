"""
Pixel-Based Random Forest Variable Selection for Kalk Project
- Loads rasters and polygons directly
- Rasterizes polygons to pixels
- Samples pixels for training
- Trains Random Forest classifier
- Outputs variable importance rankings
- Saves results to Modelling/RandForest
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GroupKFold
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import rasterio
from rasterio.features import rasterize
import geopandas as gpd

# ----------------------------------------------------
# PATHS (same as tile generator)
# ----------------------------------------------------
PREDICTOR_DIR = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\Covariates_to_model"
POLYGON_SHP   = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Naturtype\NordTrond_Train_MDir_ed.shp"
SNAP_RASTER   = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Mask\Mask_10m_NortdTrond.tif"
OUT_DIR       = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\RandForest"

LABEL_FIELD = "KA_3Class"
BACKGROUND_VALUE = 0

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------
# LOAD SNAP RASTER (reference for alignment)
# ----------------------------------------------------
snap_ds = rasterio.open(SNAP_RASTER)
width, height = snap_ds.width, snap_ds.height
transform = snap_ds.transform
crs = snap_ds.crs

print(f"Using snap raster: {SNAP_RASTER}")
print(f"Dimensions: {width}x{height}, CRS: {crs}")

# ----------------------------------------------------
# LOAD PREDICTOR RASTERS
# ----------------------------------------------------
tif_paths = sorted(glob.glob(os.path.join(PREDICTOR_DIR, "*.tif")))
datasets = [rasterio.open(p) for p in tif_paths]

var_names = [os.path.splitext(os.path.basename(p))[0] for p in tif_paths]

print(f"Loaded {len(var_names)} predictors: {var_names}")

# ----------------------------------------------------
# LOAD POLYGONS AND RASTERIZE
# ----------------------------------------------------
gdf = gpd.read_file(POLYGON_SHP)
if gdf.crs != crs:
    gdf = gdf.to_crs(crs)

CLASS_ORDER = ["low", "medium", "high"]
CLASS_MAP = {cls: i + 1 for i, cls in enumerate(CLASS_ORDER)}
gdf["label"] = gdf[LABEL_FIELD].map(CLASS_MAP).astype(np.uint8)

shapes = [(geom, int(lbl)) for geom, lbl in zip(gdf.geometry, gdf["label"])]

y_raster = rasterize(
    shapes=shapes,
    out_shape=(height, width),
    transform=transform,
    fill=BACKGROUND_VALUE,
    dtype=np.uint8
)

print("Rasterized polygons to labels.")

# ----------------------------------------------------
# EXTRACT PIXEL DATA
# ----------------------------------------------------
print("Extracting pixel data...")

# Find pixels with labels (much more efficient than looping through all pixels)
labeled_pixels = np.where(y_raster > 0)
rows, cols = labeled_pixels
n_labeled = len(rows)

print(f"Found {n_labeled} labeled pixels out of {height * width} total pixels")

features = []
labels = []
rows_list = []
cols_list = []

for i in tqdm(range(n_labeled)):
    row, col = rows[i], cols[i]

    # Get coordinate for this pixel
    x, y = rasterio.transform.xy(transform, row, col)

    pixel_features = []
    valid = True

    # Sample each predictor at this coordinate
    for ds in datasets:
        try:
            sampled = list(ds.sample([(x, y)]))
            val = sampled[0][0]  # First (and only) sample
        except:
            valid = False
            break

        nodata = ds.nodatavals[0]
        if nodata is not None and (np.isnan(val) or val == nodata):
            valid = False
            break
        pixel_features.append(float(val))

    if valid:
        features.append(pixel_features)
        labels.append(int(y_raster[row, col]))
        rows_list.append(row)
        cols_list.append(col)

X_all = np.array(features)
y_all = np.array(labels, dtype=int)

rows_arr = np.array(rows_list, dtype=int)
cols_arr = np.array(cols_list, dtype=int)

print(f"Total valid labeled pixels: {len(y_all)}")
if len(y_all) > 0:
    print(f"Class distribution: {np.bincount(y_all)}")
else:
    print("No valid pixels found. Check raster alignment and NoData values.")

# ----------------------------------------------------
# SAMPLE DATA (balanced if possible) + Spatial-block CV
# ----------------------------------------------------
SAMPLE_SIZE = 50000

# Define spatial block size in pixels (adjust if needed)
BLOCK_SIZE = 500

if len(y_all) > SAMPLE_SIZE:
    sss = StratifiedShuffleSplit(n_splits=1, train_size=SAMPLE_SIZE, random_state=42)
    idx = next(sss.split(X_all, y_all))[0]
    X_sample = X_all[idx]
    y_sample = y_all[idx]
    rows_sample = rows_arr[idx]
    cols_sample = cols_arr[idx]
else:
    X_sample, y_sample = X_all, y_all
    rows_sample, cols_sample = rows_arr, cols_arr

print(f"Sampled {len(y_sample)} pixels")

# Assign block IDs based on pixel indices
n_block_cols = int(np.ceil(width / BLOCK_SIZE))
block_rows = rows_sample // BLOCK_SIZE
block_cols = cols_sample // BLOCK_SIZE
groups = block_rows * n_block_cols + block_cols

print(f"Using spatial blocks of {BLOCK_SIZE} px; {np.unique(groups).size} blocks in sample")

# ----------------------------------------------------
# Spatial-block cross-validation (GroupKFold)
# ----------------------------------------------------
print("\nRunning spatial-block cross-validation...")

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

importances_list = []
y_true_all = []
y_pred_all = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_sample, y_sample, groups)):
    print(f"Fold {fold + 1}/{n_splits}: train groups={np.unique(groups[train_idx]).size}, test groups={np.unique(groups[test_idx]).size}")

    X_tr, X_te = X_sample[train_idx], X_sample[test_idx]
    y_tr, y_te = y_sample[train_idx], y_sample[test_idx]

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )

    rf.fit(X_tr, y_tr)

    y_pred = rf.predict(X_te)
    y_true_all.extend(y_te.tolist())
    y_pred_all.extend(y_pred.tolist())
    importances_list.append(rf.feature_importances_)

# Aggregate results across folds
print("\nCross-validated Classification Report:")
print(classification_report(y_true_all, y_pred_all, target_names=["low", "medium", "high"]))

mean_importances = np.mean(importances_list, axis=0)

# Train final model on full sampled data to save a final importance ranking as well
final_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
final_rf.fit(X_sample, y_sample)
final_importances = final_rf.feature_importances_

# Use mean_importances for cross-validated importance ranking
importances = mean_importances

# ----------------------------------------------------
# VARIABLE IMPORTANCE (cross-validated)
# ----------------------------------------------------
importances = mean_importances
importance_df = pd.DataFrame({
    'Variable': var_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nTop Variables by Importance:")
print(importance_df.head(10))

# ----------------------------------------------------
# SAVE RESULTS
# ----------------------------------------------------
importance_path = os.path.join(OUT_DIR, "variable_importance.csv")
importance_df.to_csv(importance_path, index=False)

report_path = os.path.join(OUT_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write("Random Forest Cross-validated Classification Report\n")
    f.write("=" * 40 + "\n")
    f.write(classification_report(y_true_all, y_pred_all, target_names=["low", "medium", "high"]))
    f.write(f"\nTotal samples (sampled): {len(y_sample)}\n")
    f.write(f"Total test predictions (CV): {len(y_pred_all)}\n")

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Variable'][:15], importance_df['Importance'][:15])
plt.xlabel('Importance')
plt.title('Top Variable Importances (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()

plot_path = os.path.join(OUT_DIR, "variable_importance.png")
plt.savefig(plot_path, dpi=150)
plt.close()

# Close all datasets
snap_ds.close()
for ds in datasets:
    ds.close()

print("\n✔ Results saved to:")
print(f"  - {importance_path}")
print(f"  - {report_path}")
print(f"  - {plot_path}")

print("\nWorkflow complete! Use top variables for U-Net input.")