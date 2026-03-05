"""
Create a clean binary mask: 1=land (keep), 0=NoData (exclude)
Then apply it to set water pixels to NoData
"""
import os
os.add_dll_directory(r'C:\ProgramData\anaconda3\envs\Kalk_DL_tiling\Library\bin')

from osgeo import gdal
import numpy as np

print("Creating clean binary mask with NoData...\n")

mask_input = r'E:\Test\sentinel_reprojected\dtm_mask_etrs89.tif'
mask_output = r'E:\Test\sentinel_reprojected\dtm_mask_clean.tif'

# Read original mask
ds_in = gdal.Open(mask_input)
band_in = ds_in.GetRasterBand(1)
data = band_in.ReadAsArray().astype('uint8')

print(f"Input mask value distribution:")
unique, counts = np.unique(data, return_counts=True)
for val, count in zip(unique, counts):
    print(f"  {val}: {count/data.size*100:.1f}%")

# Check if mask is inverted (0=land or 1=land)
if (data == 1).sum() > (data == 0).sum():
    print(f"\n→ Mask is: 1=land (majority), 0=water")
    data_clean = data  # Use as-is
else:
    print(f"\n→ Mask is INVERTED: 0=land (majority), 1=water")
    data_clean = 1 - data  # Invert it

print(f"\nClean mask value distribution:")
unique, counts = np.unique(data_clean, return_counts=True)
for val, count in zip(unique, counts):
    print(f"  {val}: {count/data_clean.size*100:.1f}%")

# Create output with NoData=0
driver = gdal.GetDriverByName('GTiff')
ds_out = driver.Create(
    mask_output,
    ds_in.RasterXSize,
    ds_in.RasterYSize,
    1,
    gdal.GDT_Byte,
    options=['COMPRESS=LZW', 'TILED=YES']
)

ds_out.SetGeoTransform(ds_in.GetGeoTransform())
ds_out.SetProjection(ds_in.GetProjection())

band_out = ds_out.GetRasterBand(1)
band_out.SetNoDataValue(0)  # 0 = NoData (water pixels)
band_out.WriteArray(data_clean)
band_out = None
ds_out = None
ds_in = None

print(f"\n✓ Clean mask created: {mask_output}")
print(f"  1 = land pixels (keep)")
print(f"  0 = NoData (water - will be excluded)")
print(f"\nNext: Use this mask to set water pixels to NoData in Sentinel mosaic")
