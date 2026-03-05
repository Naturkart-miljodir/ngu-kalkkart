"""
Export a small test area from Sentinel_masked_final.vrt to GeoTIFF
for visual verification in QGIS.
"""
import os
os.add_dll_directory(r'C:\ProgramData\anaconda3\envs\Kalk_DL_tiling\Library\bin')

from osgeo import gdal

print("Creating test GeoTIFF from masked VRT...")

vrt_path = r'E:\Test\sentinel_reprojected\Sentinel_masked_final.vrt'
output_path = r'E:\Test\sentinel_reprojected\test_area_masked.tif'

# Export a 2000x2000 pixel area from the center (20km x 20km)
translate_options = gdal.TranslateOptions(
    format='GTiff',
    srcWin=[13000, 14000, 2000, 2000],  # xoff, yoff, width, height
    creationOptions=['COMPRESS=LZW', 'TILED=YES']
)

ds = gdal.Translate(output_path, vrt_path, options=translate_options)
if ds is None:
    print("ERROR: Failed to export test area!")
else:
    print(f"✓ Test area exported to: {output_path}")
    print(f"  Size: 2000x2000 pixels (20km x 20km)")
    print(f"  Bands: {ds.RasterCount}")
    print("\nOpen this file in QGIS to verify:")
    print("  - Land areas have color (non-zero values)")
    print("  - Water areas are black (zero values)")
    ds = None
