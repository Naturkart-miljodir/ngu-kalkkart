"""
Test if Sentinel data is accessible without masking
"""
import os
os.add_dll_directory(r'C:\ProgramData\anaconda3\envs\Kalk_DL_tiling\Library\bin')

from osgeo import gdal

print("Testing Sentinel mosaic directly (no masking)...\n")

# Export test area directly from temp_mosaic.vrt without mask
vrt_path = r'E:\Test\sentinel_reprojected\temp_mosaic.vrt'
output_path = r'E:\Test\sentinel_reprojected\test_area_no_mask.tif'

print(f"Opening: {vrt_path}")
ds = gdal.Open(vrt_path)
if ds is None:
    print("ERROR: Cannot open VRT")
    exit(1)

print(f"VRT opened successfully")
print(f"Bands: {ds.RasterCount}")

# Export sample
translate_options = gdal.TranslateOptions(
    format='GTiff',
    srcWin=[13000, 14000, 2000, 2000],
    creationOptions=['COMPRESS=LZW']
)

result = gdal.Translate(output_path, vrt_path, options=translate_options)
if result is None:
    print("ERROR: Failed to export")
else:
    print(f"✓ Exported: {output_path}")
    print(f"  Size: {result.RasterXSize} x {result.RasterYSize}")
    print(f"  Bands: {result.RasterCount}")
    result = None

ds = None
