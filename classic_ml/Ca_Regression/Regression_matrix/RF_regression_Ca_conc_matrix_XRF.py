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


def predictor_group_name(predictor_name):
    name = predictor_name.lower()
    if name.startswith("aspect_"):
        return "aspect"
    if name.startswith("bedrock_"):
        return "bedrock"
    if name.startswith("biok_"):
        return "biok"
    if name.startswith("convergence_"):
        return "convergence"
    if name.startswith("grav_") or name.startswith("hg_grav_") or name.startswith("tdr_grav_") or name.startswith("vdc_grav_"):
        return "gravity"
    if name.startswith("hg_magnetic_") or name.startswith("magnetic_") or name.startswith("tdr_magnetic_") or name.startswith("vdc_magnetic_"):
        return "magnetic"
    if name.startswith("hillshade_"):
        return "hillshade"
    if name.startswith("kalsiumelvinnsjo4_"):
        return "kalsiumelvinnsjo4"
    if name.startswith("alphaearth_"):
        return "alphaearth"
    if name.startswith("dtm_") or name == "slope.tif":
        return "terrain"
    if name.startswith("geochronology_"):
        return "geochronology"
    if name.startswith("landuse_"):
        return "landuse"
    if name.startswith("metamorphic_"):
        return "metamorphic"
    if name.startswith("quaternary_"):
        return "quaternary"
    if name.startswith("tectonic_"):
        return "tectonic"
    if name.startswith("total_curvature_") or name.startswith("vd1_"):
        return "terrain"
    return "other"

def get_predictor_defs(predictor_dir):
    import rasterio
    if not os.path.isdir(predictor_dir):
        raise FileNotFoundError(
            f"Predictor directory not found or not accessible: {predictor_dir}"
        )

    all_paths = []
    for ext in ("*.tif", "*.tiff", "*.vrt", "*.img"):
        all_paths.extend(
            os.path.join(root, f)
            for root, _, files in os.walk(predictor_dir)
            for f in files
            if f.lower().endswith(ext.replace("*", ""))
        )

    all_paths = sorted(set(all_paths))

    if not all_paths:
        raise FileNotFoundError(
            f"No predictor rasters found under: {predictor_dir} (searched recursively for tif/tiff/vrt/img)"
        )
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


def get_reference_predictor_crs(predictor_defs):
    import rasterio
    if not predictor_defs:
        raise RuntimeError("Predictor list is empty; cannot determine predictor CRS.")

    ref_path = predictor_defs[0]["path"]
    with rasterio.open(ref_path) as ds:
        if ds.crs is None:
            raise RuntimeError(f"Predictor CRS is undefined for: {ref_path}")
        return ds.crs


def align_points_to_predictor_crs(gdf, predictor_crs):
    if gdf.crs is None:
        raise RuntimeError("Point CRS is undefined")

    if gdf.crs != predictor_crs:
        print(f"[CRS] Reprojecting points from {gdf.crs} to {predictor_crs}")
        return gdf.to_crs(predictor_crs)

    print(f"[CRS] Points already in predictor CRS: {predictor_crs}")
    return gdf


def sample_predictor_for_points(gdf, predictor_def):
    import rasterio

    path = predictor_def["path"]
    band = predictor_def["band"]

    xs = gdf.geometry.x.to_numpy()
    ys = gdf.geometry.y.to_numpy()

    with rasterio.open(path) as ds:
        if gdf.crs is None:
            raise RuntimeError("Point CRS is undefined")
        if ds.crs is None:
            raise RuntimeError(f"Predictor CRS is undefined for {path}")

        if gdf.crs != ds.crs:
            gdf_use = gdf.to_crs(ds.crs)
            xx = gdf_use.geometry.x.to_numpy()
            yy = gdf_use.geometry.y.to_numpy()
        else:
            xx = xs
            yy = ys

        if band < 1 or band > ds.count:
            raise RuntimeError(f"Requested band {band} but dataset has {ds.count} bands for {path}")

        vals = np.array(
            [v[0] for v in ds.sample(np.column_stack([xx, yy]), indexes=band)],
            dtype=np.float32,
        )

        nod = ds.nodata
        if nod is not None:
            vals[vals == nod] = np.nan

    return vals


def precheck_first_predictor_coverage(gdf, predictor_defs, min_valid_fraction=None):
    if not predictor_defs:
        raise RuntimeError("Predictor list is empty; cannot run first-predictor precheck.")

    first_def = predictor_defs[0]
    first_name = first_def["predictor_name"]
    first_path = first_def["path"]
    first_band = first_def["band"]

    print("\n=== PRECHECK: FIRST PREDICTOR COVERAGE ===")
    print(f"Checking predictor: {first_name} | band {first_band} | {os.path.basename(first_path)}")

    vals = sample_predictor_for_points(gdf, first_def)
    valid_mask = np.isfinite(vals)
    total = int(valid_mask.size)
    valid_n = int(valid_mask.sum())
    na_n = total - valid_n
    valid_frac = (valid_n / total) if total > 0 else 0.0

    print(
        f"First predictor valid points: {valid_n}/{total} "
        f"({valid_frac * 100:.1f}%), NA points: {na_n}/{total} ({(1.0 - valid_frac) * 100:.1f}%)"
    )

    if min_valid_fraction is not None and valid_frac < min_valid_fraction:
        raise RuntimeError(
            "Stopping early: first predictor valid fraction "
            f"{valid_frac * 100:.1f}% is below threshold {min_valid_fraction * 100:.1f}%."
        )

def extract_values_for_points(gdf, predictor_defs):
    import numpy as np
    from rasterio.errors import RasterioIOError
    X_cols = []
    var_names = []
    for i, d in enumerate(predictor_defs, start=1):
        pred_name = d["predictor_name"]
        path = d["path"]
        band = d["band"]
        print(f"[{i}/{len(predictor_defs)}] Extracting {pred_name} | band {band} | {os.path.basename(path)}")
        try:
            vals = sample_predictor_for_points(gdf, d)
        except RasterioIOError as exc:
            raise RuntimeError(f"Raster read failed for {pred_name} | {path} | {exc}")
        X_cols.append(vals)
        var_names.append(pred_name)
    if not X_cols:
        raise RuntimeError("No predictor values extracted; predictor list is empty.")
    X = np.column_stack(X_cols).astype(np.float32)
    return X, var_names


import os
import pandas as pd
import geopandas as gpd
import numpy as np


# =======================
# USER SETTINGS
# =======================

FILTER_ANALYSEMET_VALUES = []  # Only keep analysemet == 'XRF_H'
FILTER_ANALYTE_VALUES = []        # Only keep analyte == 'CaO'
POINT_DATA_PATH = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2025\kalk_prosjekt3.0\Geochemistry\Berggrunns\shape\Berggrunn_XRF_ICAPAES_data_sept2026.shp"
POINT_LAYER = None
CSV_X_COL = "x_utm33_ko"
CSV_Y_COL = "y_utm33_ko"
CSV_CRS = "EPSG:32633"
PREDICTOR_DIR = r"G:\Covariates_to_model"
OUT_DIR = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\Regression_matrix\Regression_matrix_XRFICP_sept2026"
SAVE_CSV = True
SAVE_PARQUET = True
RUN_FIRST_PREDICTOR_PRECHECK = True
MIN_FIRST_PREDICTOR_VALID_FRACTION = 0.20

# Target variable(s) to include in regression matrix
# Keep text labels, but use encoded integer labels for modelling.
KALKKLASS_TEXT_COL = "Kalkklass"
KALKKLASS_INT_COL = "Kalkklass_int"
KALKKLASS_CLASS_TO_INT = {
    "Klasse_1_Karbonat": 1,
    "Klasse_2_Ultramafisk_basepreg": 2,
    "Klasse_3_Ca-rik_silikatisk_kile": 3,
    "Klasse_4_Mafisk_intermediær_basepåvirket": 4,
    "Klasse_4_Mafisk_intermediaer_basepaavirket": 4,
    "Klasse_5_Sursilikat": 5,
    "Klasse_5_Sur_silikat": 5,
    "Uklassifisert": 0,
}

KLASS_CA_TEXT_COL = "Ca_klasse"
KLASS_CA_INT_COL = "Ca_klasse_int"
KLASS_CA_CLASS_TO_INT = {
    "Lav": 1,
    "Middels": 2,
    "Høy": 3,
    "Uklassifisert": 0,
}

CA_PREDPPM_COL = "Ca_predppm"
CA_PREDLOG_COL = "Ca_predlog"
MODEL_ATTRIBUTE_COLS = [CA_PREDPPM_COL, CA_PREDLOG_COL, KLASS_CA_TEXT_COL, KLASS_CA_INT_COL]

# This shapefile contains Ca_klasse, not the older derived target columns.
VERDI_NAMES = [KLASS_CA_INT_COL]
NUMERIC_VERDI_NAMES = [KLASS_CA_INT_COL]


# =======================
# MAIN
# =======================

def main():

    print("\n=== LOADING POINTS ===")

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output directory ready: {OUT_DIR}")

    gdf = load_points(POINT_DATA_PATH, layer=POINT_LAYER)
    print(f"Loaded points: {len(gdf)}")
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()
    print(f"After removing missing geometry: {len(gdf)}")

    # Optional filtering by analysemet/analyte. By default, keep all rows.
    if "analysemet" in gdf.columns:
        if FILTER_ANALYSEMET_VALUES:
            before = len(gdf)
            gdf = gdf[gdf["analysemet"].isin(FILTER_ANALYSEMET_VALUES)]
            print(
                f"After filtering analysemet in {FILTER_ANALYSEMET_VALUES}: {len(gdf)} "
                f"(removed {before - len(gdf)})"
            )
        else:
            print("No analysemet filter applied (FILTER_ANALYSEMET_VALUES is empty).")
    else:
        print("[WARNING] 'analysemet' column not found in input data.")

    # Support wildcards for analyte filter (e.g., CaO, CaO*)
    if "analyte" in gdf.columns:
        if FILTER_ANALYTE_VALUES:
            analyte_vals = FILTER_ANALYTE_VALUES

            def analyte_match(val):
                for pattern in analyte_vals:
                    if pattern.endswith("*"):
                        if str(val).startswith(pattern[:-1]):
                            return True
                    else:
                        if str(val) == pattern:
                            return True
                return False

            before = len(gdf)
            gdf = gdf[gdf["analyte"].apply(analyte_match)]
            print(
                f"After filtering analyte in {FILTER_ANALYTE_VALUES}: {len(gdf)} "
                f"(removed {before - len(gdf)})"
            )
        else:
            print("No analyte filter applied (FILTER_ANALYTE_VALUES is empty).")
    else:
        print("[WARNING] 'analyte' column not found in input data.")

    # Encode Kalkklass text labels to stable integer IDs for classification models.
    if KALKKLASS_TEXT_COL in gdf.columns:
        kalkklass_raw = gdf[KALKKLASS_TEXT_COL].astype(str).str.strip()
        gdf[KALKKLASS_INT_COL] = kalkklass_raw.map(KALKKLASS_CLASS_TO_INT)
        missing_map = int(gdf[KALKKLASS_INT_COL].isna().sum())
        if missing_map > 0:
            print(
                f"[WARNING] {missing_map} rows in '{KALKKLASS_TEXT_COL}' are not in class map and were set to NaN."
            )
    else:
        print(f"[WARNING] '{KALKKLASS_TEXT_COL}' column not found in input data.")

    # Encode Klass_Ca text labels to stable integer IDs for classification models.
    if KLASS_CA_TEXT_COL in gdf.columns:
        klass_ca_raw = gdf[KLASS_CA_TEXT_COL].astype(str).str.strip()
        gdf[KLASS_CA_INT_COL] = klass_ca_raw.map(KLASS_CA_CLASS_TO_INT)
        missing_map = int(gdf[KLASS_CA_INT_COL].isna().sum())
        if missing_map > 0:
            print(
                f"[WARNING] {missing_map} rows in '{KLASS_CA_TEXT_COL}' are not in class map and were set to NaN."
            )
    else:
        print(f"[WARNING] '{KLASS_CA_TEXT_COL}' column not found in input data.")

    for ca_col in (CA_PREDPPM_COL, CA_PREDLOG_COL):
        if ca_col in gdf.columns:
            gdf[ca_col] = pd.to_numeric(gdf[ca_col], errors="coerce")
        else:
            print(f"[WARNING] '{ca_col}' column not found in input data.")

    # --- STRICT TARGET FILTERING: Only keep rows with valid, numeric, nonzero 'verdi' ---
    for verdi_col in VERDI_NAMES:
        if verdi_col in gdf.columns:
            if verdi_col in NUMERIC_VERDI_NAMES:
                gdf[verdi_col] = pd.to_numeric(gdf[verdi_col], errors="coerce")
        else:
            print(f"[WARNING] Target variable '{verdi_col}' not found in input data.")
    before_target = len(gdf)
    gdf = gdf.dropna(subset=VERDI_NAMES)
    print(f"After filtering non-numeric target values: {len(gdf)} (removed {before_target - len(gdf)})")
    before_zero = len(gdf)
    for verdi_col in NUMERIC_VERDI_NAMES:
        if verdi_col in gdf.columns:
            gdf = gdf[gdf[verdi_col] != 0]
    print(f"After excluding zero target values: {len(gdf)} (removed {before_zero - len(gdf)})")

    print("\n=== BUILDING PREDICTOR LIST ===")
    predictor_defs = get_predictor_defs(PREDICTOR_DIR)
    print(f"Predictors found: {len(predictor_defs)}")
    for i, d in enumerate(predictor_defs[:15]):
        print(f"  {i:03d} | {d['predictor_name']} | band {d['band']} | {os.path.basename(d['path'])}")

    predictor_crs = get_reference_predictor_crs(predictor_defs)
    gdf = align_points_to_predictor_crs(gdf, predictor_crs)

    if RUN_FIRST_PREDICTOR_PRECHECK:
        precheck_first_predictor_coverage(
            gdf,
            predictor_defs,
            min_valid_fraction=MIN_FIRST_PREDICTOR_VALID_FRACTION,
        )


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

    group_rows = []
    group_order = []
    group_to_indices = {}
    for idx, name in enumerate(var_names):
        group = predictor_group_name(name)
        group_to_indices.setdefault(group, []).append(idx)
        if group not in group_order:
            group_order.append(group)

    for group in group_order:
        indices = group_to_indices[group]
        group_nan_cells = int(nan_per_predictor[indices].sum())
        group_missing_predictors = int((nan_per_predictor[indices] > 0).sum())
        group_rows_with_missing = int(np.isnan(X[:, indices]).any(axis=1).sum())
        group_rows_without_missing = int((~np.isnan(X[:, indices]).any(axis=1)).sum())
        group_rows.append(
            {
                "group": group,
                "n_predictors": len(indices),
                "n_predictors_with_missing": group_missing_predictors,
                "n_missing_cells": group_nan_cells,
                "rows_with_any_missing_in_group": group_rows_with_missing,
                "rows_without_missing_in_group": group_rows_without_missing,
            }
        )

    group_summary = pd.DataFrame(group_rows).sort_values(
        ["rows_with_any_missing_in_group", "n_missing_cells"], ascending=False
    )
    group_summary_path = os.path.join(OUT_DIR, "predictor_missing_by_group.csv")
    group_summary.to_csv(group_summary_path, index=False)
    print("\nMissing values by predictor group (top groups):")
    print(group_summary.head(12).to_string(index=False))
    print(f"Saved group missing-value summary: {group_summary_path}")

    # Save missing value counts as CSV
    missing_csv_path = os.path.join(OUT_DIR, "predictor_missing_values.csv")
    pd.DataFrame(missing_data).to_csv(missing_csv_path, index=False)
    print(f"Saved missing value summary: {missing_csv_path}")
    # Plotting code removed as requested to avoid memory errors.

    # --- FILTER: Only keep rows where ALL predictors are valid (no NaN) ---
    before_pred_nan = X.shape[0]
    valid_rows = ~np.isnan(X).any(axis=1)
    outside_mask_gdf = gdf.iloc[~valid_rows].copy()
    X = X[valid_rows]
    gdf = gdf.iloc[valid_rows].copy()
    print(f"After filtering rows with any missing predictor: {X.shape[0]} (removed {before_pred_nan - X.shape[0]})")

    # --- SAVE: Samples outside mask ---
    if len(outside_mask_gdf) > 0:
        outside_mask_dir = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\RandForest\Ca_conc_modelling\Regression_matrix_XRFtotal_lab\Samples_outside_mask"
        os.makedirs(outside_mask_dir, exist_ok=True)
        outside_mask_df = outside_mask_gdf.drop(columns=["geometry"], errors="ignore").copy()
        outside_mask_df["x"] = outside_mask_gdf.geometry.x.to_numpy()
        outside_mask_df["y"] = outside_mask_gdf.geometry.y.to_numpy()
        outside_mask_csv = os.path.join(outside_mask_dir, "samples_outside_mask.csv")
        outside_mask_df.to_csv(outside_mask_csv, index=False)
        print(f"Saved {len(outside_mask_df)} samples outside mask: {outside_mask_csv}")
    else:
        print("No samples outside mask.")


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
    for verdi_col in VERDI_NAMES + [KALKKLASS_TEXT_COL, KLASS_CA_TEXT_COL, CA_PREDPPM_COL, CA_PREDLOG_COL]:
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