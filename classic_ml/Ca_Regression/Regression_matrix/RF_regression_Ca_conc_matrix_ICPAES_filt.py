#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build regression matrix for Ca modelling from point samples + raster covariates.
"""
def load_points(path, layer=None):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        if CSV_X_COL not in df.columns or CSV_Y_COL not in df.columns:
            raise ValueError(f"CSV must contain {CSV_X_COL} and {CSV_Y_COL}")
        gdf = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(df[CSV_X_COL], df[CSV_Y_COL]),
            crs=CSV_CRS
        )
    else:
        gdf = gpd.read_file(path, layer=layer)
    if FILTER_ANALYSEMET_VALUES and "analysemet" in gdf.columns:
        gdf = gdf[gdf["analysemet"].isin(FILTER_ANALYSEMET_VALUES)]
    return gdf

def is_vrt(path):
    return os.path.splitext(path)[1].lower() == ".vrt"

def get_predictor_defs(predictor_dir):
    import glob
    import rasterio
    all_paths = []
    for ext in ("*.tif", "*.vrt", "*.img"):
        all_paths.extend(glob.glob(os.path.join(predictor_dir, ext)))
    predictors = []
    for p in all_paths:
        base = os.path.basename(p)
        if is_vrt(p):
            with rasterio.open(p) as ds:
                for b in range(1, ds.count + 1):
                    predictors.append({
                        "predictor_name": f"{base}_band{b}",
                        "path": p,
                        "band": b
                    })
        else:
            predictors.append({
                "predictor_name": base,
                "path": p,
                "band": 1
            })
    return predictors

def extract_values_for_points(gdf, predictor_defs):
    import numpy as np
    import rasterio
    from rasterio.errors import RasterioIOError
    xs = gdf.geometry.x.to_numpy()
    ys = gdf.geometry.y.to_numpy()
    X_cols = []
    var_names = []
    for i, d in enumerate(predictor_defs, start=1):
        pred_name = d["predictor_name"]
        path = d["path"]
        band = d["band"]
        print(f"[{i}/{len(predictor_defs)}] Extracting {pred_name} | band {band} | {os.path.basename(path)}")
        try:
            with rasterio.open(path) as ds:
                if gdf.crs is None:
                    raise RuntimeError("Point CRS is undefined")
                if str(gdf.crs) != str(ds.crs):
                    gdf_use = gdf.to_crs(ds.crs)
                    xx = gdf_use.geometry.x.to_numpy()
                    yy = gdf_use.geometry.y.to_numpy()
                else:
                    xx = xs
                    yy = ys
                if band < 1 or band > ds.count:
                    raise RuntimeError(f"Requested band {band} but dataset has {ds.count} bands for {path}")
                vals = np.array([v[0] for v in ds.sample(np.column_stack([xx, yy]), indexes=band)], dtype=np.float32)
                nod = ds.nodata
                if nod is not None:
                    vals[vals == nod] = np.nan
        except RasterioIOError as exc:
            raise RuntimeError(f"Raster read failed for {pred_name} | {path} | {exc}")
        X_cols.append(vals)
        var_names.append(pred_name)
    X = np.column_stack(X_cols).astype(np.float32)
    return X, var_names


import os
import pandas as pd
import geopandas as gpd
import numpy as np


# =======================
# USER SETTINGS
# =======================
FILTER_ANALYSEMET_VALUES = []  # e.g. ["XRF_H"], ["ICP", "XRF_H"], or [] for all
POINT_DATA_PATH = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2025\kalk_prosjekt3.0\Geochemistry\Berggrunns\shape\Komplett_datasett LITO_Feb2025.shp"  # or .csv
POINT_LAYER = None  # For GPKG, etc. Set to None for shapefile
CSV_X_COL = "UTM_E32"
CSV_Y_COL = "UTM_N32"
CSV_CRS = "EPSG:32632"
PREDICTOR_DIR = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\Covariates_to_model"
OUT_DIR = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\RandForest\Ca_conc_modelling"
SAVE_CSV = True
SAVE_PARQUET = True

# Toggle for target variable(s) to include in regression matrix
VERDI_NAMES = ["Ca_icpAES"]  # e.g., ["Ca"], ["Na"], ["Ca", "Na"]


# =======================
# MAIN
# =======================

def main():

    print("\n=== LOADING POINTS ===")
    gdf = load_points(POINT_DATA_PATH, layer=POINT_LAYER)


    print(f"Loaded points: {len(gdf)}")
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()
    print(f"After removing missing geometry: {len(gdf)}")


    # --- STRICT TARGET FILTERING: Only keep rows with valid, numeric, nonzero Ca_icpAES ---
    for verdi_col in VERDI_NAMES:
        if verdi_col in gdf.columns:
            gdf[verdi_col] = pd.to_numeric(gdf[verdi_col], errors="coerce")
        else:
            print(f"[WARNING] Target variable '{verdi_col}' not found in input data.")
    before_target = len(gdf)
    gdf = gdf.dropna(subset=VERDI_NAMES)
    print(f"After filtering non-numeric target values: {len(gdf)} (removed {before_target - len(gdf)})")
    before_zero = len(gdf)
    for verdi_col in VERDI_NAMES:
        if verdi_col in gdf.columns:
            gdf = gdf[gdf[verdi_col] != 0]
    print(f"After excluding zero target values: {len(gdf)} (removed {before_zero - len(gdf)})")

    print("\n=== BUILDING PREDICTOR LIST ===")
    predictor_defs = get_predictor_defs(PREDICTOR_DIR)
    print(f"Predictors found: {len(predictor_defs)}")
    for i, d in enumerate(predictor_defs[:15]):
        print(f"  {i:03d} | {d['predictor_name']} | band {d['band']} | {os.path.basename(d['path'])}")


    print("\n=== EXTRACTING RASTER VALUES ===")
    X, var_names = extract_values_for_points(gdf, predictor_defs)
    print(f"Initial matrix shape (all valid targets): {X.shape}")

    # --- DIAGNOSTIC: Show how many NaNs per predictor before filtering ---

    nan_per_predictor = np.isnan(X).sum(axis=0)
    print("\nMissing values per predictor (before filtering):")
    missing_data = []
    for i, (name, n_nan) in enumerate(zip(var_names, nan_per_predictor)):
        if n_nan > 0:
            print(f"  {i:03d} | {name:40} : {n_nan} missing")
        missing_data.append({"index": i, "predictor": name, "n_missing": n_nan})
    print(f"Total predictors with missing values: {(nan_per_predictor > 0).sum()} / {len(var_names)}")

    # Save missing value counts as CSV
    missing_csv_path = os.path.join(OUT_DIR, "predictor_missing_values.csv")
    pd.DataFrame(missing_data).to_csv(missing_csv_path, index=False)
    print(f"Saved missing value summary: {missing_csv_path}")
    # Plotting code removed as requested to avoid memory errors.

    # --- FILTER: Only keep rows where ALL predictors are valid (no NaN) ---
    before_pred_nan = X.shape[0]
    valid_rows = ~np.isnan(X).any(axis=1)
    X = X[valid_rows]
    gdf = gdf.iloc[valid_rows].copy()
    print(f"After filtering rows with any missing predictor: {X.shape[0]} (removed {before_pred_nan - X.shape[0]})")


    npz_path = os.path.join(OUT_DIR, "regression_matrix.npz")
    # Prepare y, rows, cols for compatibility
    # Use the first target variable in VERDI_NAMES
    target_col = VERDI_NAMES[0] if len(VERDI_NAMES) > 0 else None
    if target_col and target_col in gdf.columns:
        y = gdf[target_col].to_numpy(dtype=np.float32)
    else:
        print(f"[WARNING] Target variable '{target_col}' not found in gdf. Saving y as all NaN.")
        y = np.full(X.shape[0], np.nan, dtype=np.float32)
    rows = np.full(X.shape[0], np.nan, dtype=np.float32)
    cols = np.full(X.shape[0], np.nan, dtype=np.float32)
    np.savez_compressed(
        npz_path,
        X=X,
        y=y,
        rows=rows,
        cols=cols,
        var_names=np.array(var_names, dtype=object)
    )
    print(f"Saved: {npz_path}")

    # Add extra metadata columns from the shapefile to the output DataFrame
    extra_cols = ["beskrivels", "koordinats", "analysemet", "analyte", "enhet", "x", "y"]

    df_out = pd.DataFrame(X, columns=var_names)


    # Add selected target variable columns from gdf if present
    for verdi_col in VERDI_NAMES:
        if verdi_col in gdf.columns:
            df_out[verdi_col] = gdf[verdi_col].to_numpy()
        else:
            print(f"[WARNING] Target variable '{verdi_col}' not found in input data.")

    # Add requested extra columns from input shapefile if present
    for col in extra_cols:
        if col in gdf.columns:
            df_out[col] = gdf[col].to_numpy()
        elif col == "x":
            df_out["x"] = gdf.geometry.x.to_numpy()
        elif col == "y":
            df_out["y"] = gdf.geometry.y.to_numpy()

    if SAVE_CSV:
        csv_path = os.path.join(OUT_DIR, "regression_matrix.csv")
        df_out.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    if SAVE_PARQUET:
        try:
            pq_path = os.path.join(OUT_DIR, "regression_matrix.parquet")
            df_out.to_parquet(pq_path, index=False)
            print(f"Saved: {pq_path}")
        except Exception as exc:
            print(f"[WARNING] Could not save parquet: {exc}")

    # Save predictor names to a CSV file
    predictors_csv_path = os.path.join(OUT_DIR, "predictor_names.csv")
    pd.DataFrame({"predictor_name": var_names}).to_csv(predictors_csv_path, index=False)
    print(f"Saved: {predictors_csv_path}")

    with open(os.path.join(OUT_DIR, "regression_matrix_info.txt"), "w", encoding="utf-8") as f:
        f.write("Regression matrix for Ca modelling\n")
        f.write("=" * 60 + "\n")
        f.write(f"POINT_DATA_PATH: {POINT_DATA_PATH}\n")
        f.write(f"PREDICTOR_DIR: {PREDICTOR_DIR}\n")
        f.write(f"Rows: {X.shape[0]}\n")
        f.write(f"Columns: {X.shape[1]}\n")
        f.write("\nPredictors:\n")
        for i, name in enumerate(var_names):
            f.write(f"{i:03d} | {name}\n")

    print("\n✔ Done.")
    print(f"Outputs written to: {OUT_DIR}")

if __name__ == "__main__":
    main()