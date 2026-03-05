"""
Simple solution: Use gdalwarp to mask the unmasked VRT with proper mask logic
"""
import subprocess

print("Creating properly masked GeoTIFF using gdalwarp...\n")

vrt_input = r'E:\Test\sentinel_reprojected\Sentinel_no_mask.vrt'
mask_file = r'E:\Test\sentinel_reprojected\dtm_mask_etrs89.tif'
output_tif = r'E:\Test\sentinel_reprojected\Sentinel_masked_proper.tif'

# Use gdalwarp with cutline (mask)
# Note: cutline inverts - pixels WHERE mask=1 (land) are kept, WHERE mask=0 (water) are excluded
cmd = [
    'gdalwarp',
    '-cutline', mask_file,
    '-crop_to_cutline',
    '-co', 'COMPRESS=LZW',
    '-co', 'TILED=YES',
    '-co', 'BIGTIFF=IF_SAFER',
    vrt_input,
    output_tif
]

print(f"Running: {' '.join(cmd[:3])}")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode != 0:
    print(f"ERROR: {result.stderr}")
else:
    print(f"✓ Created: {output_tif}")
    print(f"\nThis uses gdalwarp -cutline which properly handles mask/water exclusion")
