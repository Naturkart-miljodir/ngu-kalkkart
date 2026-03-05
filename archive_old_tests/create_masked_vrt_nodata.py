"""
Create a properly masked VRT with NoData values
Uses custom pixel function to set masked areas to NoData
"""
import os
import xml.etree.ElementTree as ET

out_folder = r"E:\Test\sentinel_reprojected"
temp_vrt = os.path.join(out_folder, "temp_mosaic.vrt")
mask_file = os.path.join(out_folder, "dtm_mask_etrs89.tif")
output_vrt = os.path.join(out_folder, "Sentinel_masked_final.vrt")

print("Creating masked VRT with NoData handling...\n")

# Read temp VRT to get metadata
tree = ET.parse(temp_vrt)
root = tree.getroot()

raster_x_size = root.get('rasterXSize')
raster_y_size = root.get('rasterYSize')
geo_elem = root.find('.//GeoTransform')
geo_text = geo_elem.text.strip() if geo_elem is not None and geo_elem.text else ""

print(f"Dimensions: {raster_x_size} x {raster_y_size}")

# Create new VRT with proper NoData handling
# Using approach: multiply each band by mask, and use NoData to represent masked areas
temp_vrt_path = temp_vrt.replace("\\", "/")
mask_path = mask_file.replace("\\", "/")

# Start building VRT
vrt_lines = [
    f'<VRTDataset rasterXSize="{raster_x_size}" rasterYSize="{raster_y_size}">',
    '  <SRS>EPSG:25833</SRS>',
    f'  <GeoTransform>{geo_text}</GeoTransform>',
]

# Add 10 bands with mask multiplication and NoData declaration
for band_num in range(1, 11):
    vrt_lines.append(f'  <VRTRasterBand dataType="UInt16" band="{band_num}" subClass="VRTDerivedRasterBand" noData="0">')
    vrt_lines.append('    <PixelFunctionType>mul</PixelFunctionType>')
    vrt_lines.append('    <ComplexSource>')
    vrt_lines.append(f'      <SourceFilename>{temp_vrt_path}</SourceFilename>')
    vrt_lines.append(f'      <SourceBand>{band_num}</SourceBand>')
    vrt_lines.append('    </ComplexSource>')
    vrt_lines.append('    <ComplexSource>')
    vrt_lines.append(f'      <SourceFilename>{mask_path}</SourceFilename>')
    vrt_lines.append('      <SourceBand>1</SourceBand>')
    vrt_lines.append('    </ComplexSource>')
    vrt_lines.append('  </VRTRasterBand>')

vrt_lines.append('</VRTDataset>')

# Write VRT file
vrt_content = '\n'.join(vrt_lines)
with open(output_vrt, 'w') as f:
    f.write(vrt_content)

print(f"✓ Masked VRT created with:")
print(f"  - 10 bands")
print(f"  - NoData=0 declaration")
print(f"  - Mask multiplication applied")
print(f"\nFile: {output_vrt}")
