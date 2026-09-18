import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import os


# Input CSV and output shapefile paths (updated as requested)
meta_csv = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\DL_AE_test_chips\tile_metadata\tile_metadata.csv"
out_shp = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\DL_AE_test_chips\chip_previews\chip_footprints_5pct.shp"

# Read tile metadata
meta = pd.read_csv(meta_csv)

# Create geometry for each tile
geoms = [box(row['xmin'], row['ymin'], row['xmax'], row['ymax']) for _, row in meta.iterrows()]
gdf = gpd.GeoDataFrame(meta, geometry=geoms, crs="EPSG:25833")

# Save to shapefile
os.makedirs(os.path.dirname(out_shp), exist_ok=True)
gdf.to_file(out_shp)

print(f"Chip footprints saved to: {out_shp}")
