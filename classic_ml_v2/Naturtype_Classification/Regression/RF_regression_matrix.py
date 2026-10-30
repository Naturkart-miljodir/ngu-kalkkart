"""
Build Regression Matrix for Kalk RF workflow.
- Loads predictors and training polygons
- Rasterizes labels
- Extracts predictor values for labeled pixels
- Saves regression matrix to Modelling/RandForest for re-use
"""

import os
import glob
import re
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

PREDICTOR_DIR = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\Covariates_to_model"
POLYGON_GPKG = r"C:\Users\acosta_pedro\Miljødirektoratet\Endre Grüner Ofstad - kalkkart\Data\kalkkart_treningsdata.gpkg"
POLYGON_LAYER = "NiN_all_agg"
REF_MASK = r"E:\Alpha_earth\qc\alphaearth_mosaic_epsg25833_band1_qc_cog.tif"
OUT_DIR = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\RandForest\Regression"
MATRIX_PATH = os.path.join(OUT_DIR, "regression_matrix.npz")
ALPHAEARTH_VRT_NAME = "alphaearth_dequant_national_epsg25833.vrt"
ALPHAEARTH_DIR = r"E:\Alpha_earth\dequant_images_all"

LABEL_FIELD = "KA_mean_weighted_category"
BACKGROUND_VALUE = 0


def make_vrt_absolute(vrt_path, source_dir, out_dir):
    with open(vrt_path, "r", encoding="utf-8") as f:
        txt = f.read()

    def repl(match_obj):
        open_tag, old_path, close_tag = match_obj.groups()
        new_abs = os.path.join(source_dir, os.path.basename(old_path)).replace("\\", "/")
        open_tag = re.sub(r'relativeToVRT="[^"]*"', 'relativeToVRT="0"', open_tag)
        return f"{open_tag}{new_abs}{close_tag}"

    patched = re.sub(r"(<SourceFilename[^>]*>)([^<]+)(</SourceFilename>)", repl, txt)
    os.makedirs(out_dir, exist_ok=True)
    out_vrt = os.path.join(out_dir, os.path.basename(vrt_path))
    with open(out_vrt, "w", encoding="utf-8") as f:
        f.write(patched)
    return out_vrt

os.makedirs(OUT_DIR, exist_ok=True)

snap_ds = rasterio.open(REF_MASK)
width, height = snap_ds.width, snap_ds.height
transform = snap_ds.transform
crs = snap_ds.crs

print(f"Using snap raster: {REF_MASK}")
print(f"Dimensions: {width}x{height}, CRS: {crs}")

raster_paths = sorted(
    glob.glob(os.path.join(PREDICTOR_DIR, "*.tif"))
    + glob.glob(os.path.join(PREDICTOR_DIR, "*.vrt"))
)

alphaearth_vrt = os.path.join(PREDICTOR_DIR, ALPHAEARTH_VRT_NAME)
if os.path.exists(alphaearth_vrt):
    tmp_vrt_dir = os.path.join(OUT_DIR, "_tmp_vrt")
    patched_alphaearth_vrt = make_vrt_absolute(alphaearth_vrt, ALPHAEARTH_DIR, tmp_vrt_dir)
    raster_paths = [patched_alphaearth_vrt if p == alphaearth_vrt else p for p in raster_paths]

datasets = [rasterio.open(p) for p in raster_paths]
dataset_base_names = [os.path.splitext(os.path.basename(p))[0] for p in raster_paths]
var_names = []
for base_name, ds in zip(dataset_base_names, datasets):
    if ds.count == 1:
        var_names.append(base_name)
    else:
        for band_idx in range(1, ds.count + 1):
            var_names.append(f"{base_name}_b{band_idx:02d}")

if len(var_names) == 0:
    raise FileNotFoundError(f"No predictor rasters found in: {PREDICTOR_DIR}")

print(f"Loaded {len(var_names)} predictors")

gdf = gpd.read_file(POLYGON_GPKG, layer=POLYGON_LAYER)
if gdf.crs != crs:
    gdf = gdf.to_crs(crs)

class_order = ["low", "medium", "high"]
class_map = {cls: i + 1 for i, cls in enumerate(class_order)}

if LABEL_FIELD not in gdf.columns:
    raise KeyError(f"Label field '{LABEL_FIELD}' not found in: {POLYGON_GPKG}")

gdf = gdf[gdf.geometry.notnull()]
gdf = gdf[~gdf.geometry.is_empty]
gdf = gdf[gdf[LABEL_FIELD].isin(class_order)].copy()
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
            vals = np.asarray(sampled[0], dtype=np.float32)
        except Exception:
            valid = False
            break

        if vals.size != ds.count:
            valid = False
            break

        if not np.all(np.isfinite(vals)):
            valid = False
            break

        for band_val, band_nodata in zip(vals, ds.nodatavals):
            if band_nodata is not None:
                if np.isnan(band_nodata):
                    if np.isnan(band_val):
                        valid = False
                        break
                elif band_val == band_nodata:
                    valid = False
                    break
        if not valid:
            break

        pixel_features.extend(vals.tolist())

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