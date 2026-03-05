"""
Create inverted mask and update VRT to use it
"""
import os
os.add_dll_directory(r'C:\ProgramData\anaconda3\envs\Kalk_DL_tiling\Library\bin')

from osgeo import gdal
import numpy as np

print("Creating inverted mask...\n")

mask_input = r'E:\Test\sentinel_reprojected\dtm_mask_etrs89.tif'
mask_inverted = r'E:\Test\sentinel_reprojected\dtm_mask_inverted.tif'

# Read original as uint8
ds_in = gdal.Open(mask_input)
band_in = ds_in.GetRasterBand(1)
data = band_in.ReadAsArray().astype('uint8')

print(f"Original mask - Value distribution:")
unique, counts = np.unique(data, return_counts=True)
for val, count in zip(unique, counts):
    print(f"  {val}: {count/data.size*100:.1f}%")

# Invert
data_inv = 1 - data

print(f"\nInverted mask - Value distribution:")
unique, counts = np.unique(data_inv, return_counts=True)
for val, count in zip(unique, counts):
    print(f"  {val}: {count/data_inv.size*100:.1f}%")

# Write inverted
driver = gdal.GetDriverByName('GTiff')
ds_out = driver.Create(
    mask_inverted,
    ds_in.RasterXSize,
    ds_in.RasterYSize,
    1,
    gdal.GDT_Byte,
    options=['COMPRESS=LZW', 'TILED=YES']
)

ds_out.SetGeoTransform(ds_in.GetGeoTransform())
ds_out.SetProjection(ds_in.GetProjection())
ds_out.GetRasterBand(1).WriteArray(data_inv)
ds_out = None
ds_in = None

print(f"\n✓ Inverted mask created: {mask_inverted}")
