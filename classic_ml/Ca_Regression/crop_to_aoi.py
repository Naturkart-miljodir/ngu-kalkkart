"""
Crop a raster (COG) to the AOI polygon extent, saving a new COG with minimal bounding box.

Usage:
    python crop_to_aoi.py <input_cog> <aoi_shapefile> <output_cog>

Example:
    python crop_to_aoi.py rf_prediction_cog.tif Lofoten_kalk.shp rf_prediction_cog_cropped.tif
"""
import sys
import rasterio
from rasterio.mask import mask
import geopandas as gpd

if len(sys.argv) != 4:
    print("Usage: python crop_to_aoi.py <input_cog> <aoi_shapefile> <output_cog>")
    sys.exit(1)

input_cog = sys.argv[1]
aoi_shp = sys.argv[2]
output_cog = sys.argv[3]

aoi_gdf = gpd.read_file(aoi_shp)
aoi_geom = [geom for geom in aoi_gdf.geometry if geom is not None]

with rasterio.open(input_cog) as src:
    out_image, out_transform = mask(src, aoi_geom, crop=True, nodata=src.nodata)
    out_meta = src.meta.copy()
    out_meta.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "driver": "COG"
    })

with rasterio.open(output_cog, "w", **out_meta) as dest:
    dest.write(out_image)

print(f"✔ Cropped COG saved to: {output_cog}")
