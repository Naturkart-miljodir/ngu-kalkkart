"""
Create an INVERTED mask (1-mask) and test if it fixes the issue
"""
import os
os.add_dll_directory(r'C:\ProgramData\anaconda3\envs\Kalk_DL_tiling\Library\bin')

from osgeo import gdal

print("Creating inverted mask...\n")

mask_input = r'E:\Test\sentinel_reprojected\dtm_mask_etrs89.tif'
mask_inverted = r'E:\Test\sentinel_reprojected\dtm_mask_etrs89_inverted.tif'

# Read original
ds_in = gdal.Open(mask_input)
band_in = ds_in.GetRasterBand(1)
data = band_in.ReadAsArray().astype('uint8')

print(f"Original mask unique values: {set(data.ravel().tolist())}")

# Invert: 1 becomes 0, 0 becomes 1
data_inverted = 1 - data

print(f"Inverted mask unique values: {set(data_inverted.ravel().tolist())}")

# Write inverted
driver = gdal.GetDriverByName('GTiff')
ds_out = driver.Create(
    mask_inverted,
    ds_in.RasterXSize,
    ds_in.RasterYSize,
    1,
    gdal.GDT_Byte,
    options=['COMPRESS=LZW']
)

ds_out.SetGeoTransform(ds_in.GetGeoTransform())
ds_out.SetProjection(ds_in.GetProjection())
ds_out.GetRasterBand(1).WriteArray(data_inverted)
ds_out = None

print(f"\n✓ Created inverted mask: {mask_inverted}")
print(f"\nNow testing both masks:")
print(f"  - dtm_mask_etrs89.tif (original)")
print(f"  - dtm_mask_etrs89_inverted.tif (inverted)")
print(f"\nWe'll update the VRT to use the inverted version")

ds_in = None
