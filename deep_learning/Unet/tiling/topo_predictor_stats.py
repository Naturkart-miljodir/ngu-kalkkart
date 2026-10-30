import rasterio
import numpy as np
import glob
import os

# Predictor directory
predictor_dir = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\Covariates_to_model"

# Topographic predictors (from NON_NORMALIZED_PREDICTOR_STEMS logic)
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

def predictor_stem(path):
    return os.path.splitext(os.path.basename(path))[0].lower()

def should_normalize(path):
    stem = predictor_stem(path)
    if stem in NON_NORMALIZED_PREDICTOR_STEMS:
        return False
    if stem.endswith("_absolute_paths"):
        original_stem = stem[: -len("_absolute_paths")]
        if original_stem in NON_NORMALIZED_PREDICTOR_STEMS:
            return False
    return True

# Thresholds for outlier detection
MIN_VALID = -1e4
MAX_VALID = 1e4

for path in glob.glob(os.path.join(predictor_dir, "*.tif")):
    if not should_normalize(path):
        continue  # Skip non-topographic
    with rasterio.open(path) as ds:
        for band in range(1, ds.count + 1):
            minv = np.inf
            maxv = -np.inf
            n_nan = 0
            n_total = 0
            for ji, window in ds.block_windows(band):
                arr = ds.read(band, window=window).astype(np.float32)
                arr[arr == ds.nodata] = np.nan
                n_total += arr.size
                n_nan += np.isnan(arr).sum()
                block_min = np.nanmin(arr)
                block_max = np.nanmax(arr)
                if block_min < minv:
                    minv = block_min
                if block_max > maxv:
                    maxv = block_max
            outlier = (minv < MIN_VALID) or (maxv > MAX_VALID)
            print(f"{os.path.basename(path)} band {band}: min={minv:.2f}, max={maxv:.2f}, nan={n_nan}/{n_total}",
                  "<-- OUTLIER" if outlier else "")
