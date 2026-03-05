"""
Export test area from UNMASKED VRT
"""
import os
os.add_dll_directory(r'C:\ProgramData\anaconda3\envs\Kalk_DL_tiling\Library\bin')

from osgeo import gdal

print("Exporting test area from UNMASKED Sentinel VRT...\n")

vrt_path = r'E:\Test\sentinel_reprojected\Sentinel_no_mask.vrt'
output_path = r'E:\Test\sentinel_reprojected\test_area_no_mask.tif'

translate_options = gdal.TranslateOptions(
    format='GTiff',
    srcWin=[13000, 14000, 2000, 2000],
    creationOptions=['COMPRESS=LZW', 'TILED=YES']
)

ds = gdal.Translate(output_path, vrt_path, options=translate_options)
if ds is None:
    print("ERROR: Failed to export")
else:
    print(f"✓ Exported: {output_path}")
    print(f"  Size: {ds.RasterXSize} x {ds.RasterYSize}")
    print(f"  Bands: {ds.RasterCount}")
    print(f"\nOpen in QGIS and compare with masked version")
    print(f"If this shows proper Sentinel values, then the mask is the problem")
    ds = None
