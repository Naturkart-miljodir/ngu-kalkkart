"""
Try a completely different approach: Use gdalwarp with cutline
This is the proper way to apply a binary mask in GDAL
"""
import subprocess
import os

os.chdir(r'E:\Test\sentinel_reprojected')

print("Creating masked mosaic using gdalwarp -cutline...\n")

vrt = r'E:\Test\sentinel_reprojected\Sentinel_no_mask.vrt'
mask = r'E:\Test\sentinel_reprojected\dtm_mask_etrs89.tif'
output = r'E:\Test\sentinel_reprojected\Sentinel_masked_with_cutline.tif'

# gdalwarp with -cutline will use mask where mask=1 means keep, mask=0 means exclude
cmd = [
    'gdalwarp',
    '-cutline', mask,
    '-crop_to_cutline',
    '-co', 'COMPRESS=LZW',
    '-co', 'TILED=YES',
    '-co', 'BIGTIFF=YES',
    vrt,
    output
]

print(f"Running gdalwarp with -cutline...")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print(f"✓ Success! Created: {output}")
    print(f"\nThis approach uses gdalwarp -cutline which properly handles masking")
    print(f"The -cutline method is more reliable than VRT pixel functions")
else:
    print(f"ERROR: {result.stderr}")
