"""
Check mask values and create a 10-band version for VRT multiplication
"""
import os
os.add_dll_directory(r'C:\ProgramData\anaconda3\envs\Kalk_DL_tiling\Library\bin')

from osgeo import gdal
import numpy as np

print("Creating 10-band mask for VRT...\n")

mask_input = r'E:\Test\sentinel_reprojected\dtm_mask_etrs89.tif'
mask_output = r'E:\Test\sentinel_reprojected\dtm_mask_10band.tif'

print("[1] Reading mask...")
mask_ds = gdal.Open(mask_input)
band = mask_ds.GetRasterBand(1)
data = band.ReadAsArray()

print(f"    Mask shape: {data.shape}")
print(f"    Mask dtype: {data.dtype}")
print(f"    Unique values: {np.unique(data)}")
print(f"    Sample values (center): {data[10000:10010, 10000:10010]}")

# Check if predominantly 0 or 1
zero_count = (data == 0).sum()
one_count = (data == 1).sum()
other_count = data.size - zero_count - one_count

print(f"\n    Value distribution:")
print(f"      0: {zero_count} pixels ({zero_count/data.size*100:.1f}%)")
print(f"      1: {one_count} pixels ({one_count/data.size*100:.1f}%)")
print(f"      Other: {other_count} pixels ({other_count/data.size*100:.1f}%)")

print("\n[2] Creating 10-band version...")

# Create 10-band GeoTIFF
driver = gdal.GetDriverByName('GTiff')
out_ds = driver.Create(
    mask_output,
    mask_ds.RasterXSize,
    mask_ds.RasterYSize,
    10,  # 10 bands
    band.DataType,
    options=['COMPRESS=LZW', 'TILED=YES']
)

# Copy geotransform and projection
out_ds.SetGeoTransform(mask_ds.GetGeoTransform())
out_ds.SetProjection(mask_ds.GetProjection())

# Write same data to all 10 bands
for b in range(1, 11):
    out_band = out_ds.GetRasterBand(b)
    out_band.WriteArray(data)
    out_band.FlushCache()

out_ds = None
band = None
mask_ds = None

print(f"    ✓ Saved to: {mask_output}")
print("\nNext step: Use dtm_mask_10band.tif in the VRT instead of dtm_mask_etrs89.tif")
