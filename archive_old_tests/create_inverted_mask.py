"""
Create an inverted mask (1=land, 0=water becomes 0=land, 1=water)
and test which version gives correct values
"""
import os
os.add_dll_directory(r'C:\ProgramData\anaconda3\envs\Kalk_DL_tiling\Library\bin')

from osgeo import gdal

print("Testing mask orientation...\n")

mask_input = r'E:\Test\sentinel_reprojected\dtm_mask_etrs89.tif'
mask_inverted = r'E:\Test\sentinel_reprojected\dtm_mask_etrs89_inverted.tif'

print("[1] Creating inverted mask...")
# Read original mask
mask_ds = gdal.Open(mask_input)
band = mask_ds.GetRasterBand(1)
data = band.ReadAsArray()
print(f"    Original mask unique values: {set(data.ravel().tolist())}")

# Invert: 1 becomes 0, 0 becomes 1
inverted_data = 1 - data
print(f"    Inverted mask unique values: {set(inverted_data.ravel().tolist())}")

# Write inverted mask
driver = gdal.GetDriverByName('GTiff')
out_ds = driver.Create(
    mask_inverted,
    mask_ds.RasterXSize,
    mask_ds.RasterYSize,
    1,
    band.DataType,
    options=['COMPRESS=LZW', 'TILED=YES']
)
out_ds.SetGeoTransform(mask_ds.GetGeoTransform())
out_ds.SetProjection(mask_ds.GetProjection())
out_band = out_ds.GetRasterBand(1)
out_band.WriteArray(inverted_data)
out_band.FlushCache()
out_ds = None
band = None
mask_ds = None

print(f"    ✓ Saved to: {mask_inverted}")
print("\nNow testing with BOTH original and inverted mask...")
print("Update your VRT to use the inverted mask and check if Sentinel values appear correctly")
