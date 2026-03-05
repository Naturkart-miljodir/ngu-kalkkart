"""
Build Regression Matrix for Kalk RF workflow
- Loads predictors and training polygons
- Rasterizes labels
- Extracts predictor values for labeled pixels
- Saves regression matrix to Modelling/RandForest for re-use
"""

import os
import glob
import numpy as np
from tqdm import tqdm
import rasterio
from rasterio.features import rasterize
import geopandas as gpd


def windows_path_to_wsl(path_text):
    normalized = path_text.replace("\\", "/")
    if len(normalized) >= 2 and normalized[1] == ":":
        drive_letter = normalized[0].lower()
        return f"/mnt/{drive_letter}{normalized[2:]}"
    return normalized


PROJECT_DIR_WINDOWS = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project"

if os.name == "nt":
    PROJECT_DIR = PROJECT_DIR_WINDOWS
else:
    PROJECT_DIR = windows_path_to_wsl(PROJECT_DIR_WINDOWS)

PREDICTOR_DIR = os.path.join(PROJECT_DIR, "Modelling", "Covariates_to_model")
POLYGON_SHP = os.path.join(PROJECT_DIR, "Naturtype", "NordTrond_Train_MDir_ed.shp")
SNAP_RASTER = os.path.join(PROJECT_DIR, "Mask", "Mask_10m_NortdTrond.tif")
OUT_DIR = os.path.join(PROJECT_DIR, "Modelling", "RandForest")
MATRIX_PATH = os.path.join(OUT_DIR, "regression_matrix.npz")

LABEL_FIELD = "KA_3Class"
BACKGROUND_VALUE = 0

os.makedirs(OUT_DIR, exist_ok=True)

snap_ds = rasterio.open(SNAP_RASTER)
width, height = snap_ds.width, snap_ds.height
transform = snap_ds.transform
crs = snap_ds.crs

print(f"Using snap raster: {SNAP_RASTER}")
print(f"Dimensions: {width}x{height}, CRS: {crs}")

tif_paths = sorted(glob.glob(os.path.join(PREDICTOR_DIR, "*.tif")))
datasets = [rasterio.open(p) for p in tif_paths]
var_names = [os.path.splitext(os.path.basename(p))[0] for p in tif_paths]

if len(var_names) == 0:
    raise FileNotFoundError(f"No predictor rasters found in: {PREDICTOR_DIR}")

print(f"Loaded {len(var_names)} predictors")

gdf = gpd.read_file(POLYGON_SHP)
if gdf.crs != crs:
    gdf = gdf.to_crs(crs)

class_order = ["low", "medium", "high"]
class_map = {cls: i + 1 for i, cls in enumerate(class_order)}

if LABEL_FIELD not in gdf.columns:
    raise KeyError(f"Label field '{LABEL_FIELD}' not found in: {POLYGON_SHP}")

gdf["label"] = gdf[LABEL_FIELD].map(class_map).astype(np.uint8)

shapes = [(geom, int(lbl)) for geom, lbl in zip(gdf.geometry, gdf["label"])]

y_raster = rasterize(
    shapes=shapes,
    out_shape=(height, width),
    transform=transform,
    fill=BACKGROUND_VALUE,
    dtype=np.uint8
)

print("Rasterized polygons to labels")
print("Extracting regression matrix pixel data...")

rows, cols = np.where(y_raster > 0)
n_labeled = len(rows)
print(f"Found {n_labeled} labeled pixels")

features = []
labels = []
rows_list = []
cols_list = []

for i in tqdm(range(n_labeled)):
    row, col = rows[i], cols[i]
    x, y = rasterio.transform.xy(transform, row, col)

    pixel_features = []
    valid = True

    for ds in datasets:
        try:
            sampled = list(ds.sample([(x, y)]))
            val = sampled[0][0]
        except Exception:
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

X_all = np.array(features, dtype=np.float32)
y_all = np.array(labels, dtype=np.int16)
rows_arr = np.array(rows_list, dtype=np.int32)
cols_arr = np.array(cols_list, dtype=np.int32)

if len(y_all) == 0:
    raise RuntimeError("No valid labeled pixels found. Check raster alignment and NoData values.")

print(f"Valid samples in matrix: {len(y_all)}")
print(f"Class distribution: {np.bincount(y_all)}")

np.savez_compressed(
    MATRIX_PATH,
    X=X_all,
    y=y_all,
    rows=rows_arr,
    cols=cols_arr,
    var_names=np.array(var_names, dtype=object),
    width=np.array([width], dtype=np.int32),
    height=np.array([height], dtype=np.int32)
)

snap_ds.close()
for ds in datasets:
    ds.close()

print("\n✔ Regression matrix saved:")
print(f"  - {MATRIX_PATH}")
print("\nDone. Now run RF_GPU_kalk.py to train from this matrix.")