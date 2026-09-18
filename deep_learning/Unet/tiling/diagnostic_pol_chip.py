# Name:           diagnostic_pol_chip.py
# Authors:        GitHub Copilot
# Description:    Diagnostics for DL chip spatial coverage. Compares tile metadata and chip files, identifies missing polygons.
# Requirements:   Python 3.9+, geopandas, pandas, shapely
# - python env:   kalk_unet
# - packages:     geopandas, pandas, shapely

"""
Diagnostics for DL chip spatial coverage.

- Compares tile metadata and chip files
- Identifies missing polygons/chips
"""

import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

# --- CONFIG ---
METADATA_CSV = "filtered_tile_metadata.csv"  # Path to tile metadata
TILE_FOOTPRINTS_SHP = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\DL_chips_spatial_location\tile_footprints.shp"  # Shapefile with chip/tile footprints
OUTPUT_DIAGNOSTIC = "tile_coverage_diagnostic.csv"  # Output CSV for diagnostics
OUTPUT_MISSING_SHAPEFILE = "missing_tile_footprints.shp"  # Shapefile for missing polygons
TILE_SIZE = 1280  # meters (adjust as needed)

# --- STEP 1: Load tile metadata ---
def load_metadata(csv_path):
    """
    Load tile metadata CSV.
    :param csv_path: Path to CSV file.
    :return: DataFrame with tile metadata.
    """
    return pd.read_csv(csv_path)

# --- STEP 2: List chip files ---
def load_tile_footprints(shp_path):
    """
    Load tile footprints shapefile.
    :param shp_path: Path to shapefile.
    :return: GeoDataFrame with tile footprints.
    """
    return gpd.read_file(shp_path)

# --- STEP 3: Compare metadata and chips ---
def compare_metadata_footprints(metadata_df, footprints_gdf):
    """
    Compare tile metadata polygons with tile footprints.
    :param metadata_df: DataFrame with tile metadata.
    :param footprints_gdf: GeoDataFrame with tile footprints.
    :return: DataFrame with diagnostic info, missing polygons.
    """
    # Create polygons from metadata extents
    polygons = [box(row['xmin'], row['ymin'], row['xmax'], row['ymax']) for _, row in metadata_df.iterrows()]
    metadata_gdf = gpd.GeoDataFrame(metadata_df, geometry=polygons, crs=footprints_gdf.crs)
    # Ensure CRS match
    if metadata_gdf.crs != footprints_gdf.crs:
        metadata_gdf = metadata_gdf.to_crs(footprints_gdf.crs)
    # Spatial join: find polygons fully within any footprint
    joined = gpd.sjoin(metadata_gdf, footprints_gdf, how="left", predicate="within")
    joined = joined.reset_index(drop=True)
    has_chip_mask = ~joined['index_right'].isna()
    missing_chip_mask = joined['index_right'].isna()
    diagnostic = joined.copy()
    diagnostic['has_chip'] = has_chip_mask
    diagnostic['missing_chip'] = missing_chip_mask
    missing_polygons = diagnostic[missing_chip_mask]
    return diagnostic, missing_polygons

# --- STEP 4: Export missing polygons as shapefile ---
def export_missing_polygons(missing_gdf, output_shapefile):
    """
    Export missing tile polygons as shapefile.
    :param missing_gdf: GeoDataFrame with missing polygons.
    :param output_shapefile: Path to output shapefile.
    """
    missing_gdf.to_file(output_shapefile)

# --- STEP 5: Main workflow ---
def main():
    metadata_df = load_metadata(METADATA_CSV)
    footprints_gdf = load_tile_footprints(TILE_FOOTPRINTS_SHP)
    diagnostic, missing_gdf = compare_metadata_footprints(metadata_df, footprints_gdf)
    diagnostic.to_csv(OUTPUT_DIAGNOSTIC, index=False)
    print(f"Total polygons: {len(metadata_df)}")
    print(f"Polygons contained in chips: {len(diagnostic) - len(missing_gdf)}")
    print(f"Polygons NOT contained in any chip: {len(missing_gdf)}")
    if len(missing_gdf) > 0:
        export_missing_polygons(missing_gdf, OUTPUT_MISSING_SHAPEFILE)
        print(f"Missing polygons exported to {OUTPUT_MISSING_SHAPEFILE}")
    else:
        print("All polygons are contained within chips.")

if __name__ == "__main__":
    main()
