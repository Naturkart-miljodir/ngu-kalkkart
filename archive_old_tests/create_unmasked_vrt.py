"""
Create a VRT that just reads Sentinel mosaic WITHOUT masking
to verify the underlying data is accessible
"""
import os
import xml.etree.ElementTree as ET

out_folder = r"E:\Test\sentinel_reprojected"
temp_vrt = os.path.join(out_folder, "temp_mosaic.vrt")
output_vrt = os.path.join(out_folder, "Sentinel_no_mask.vrt")

print("Creating unmasked Sentinel VRT for testing...\n")

# Read temp VRT metadata
tree = ET.parse(temp_vrt)
root = tree.getroot()

raster_x_size = root.get('rasterXSize')
raster_y_size = root.get('rasterYSize')
geo_elem = root.find('.//GeoTransform')
geo_text = geo_elem.text.strip() if geo_elem is not None and geo_elem.text else ""

temp_vrt_path = temp_vrt.replace("\\", "/")

# Create simple pass-through VRT (no masking)
vrt_xml = f'''<VRTDataset rasterXSize="{raster_x_size}" rasterYSize="{raster_y_size}">
  <SRS>EPSG:25833</SRS>
  <GeoTransform>{geo_text}</GeoTransform>'''

# Add 10 bands - just pass through Sentinel data
for band_num in range(1, 11):
    vrt_xml += f'''
  <VRTRasterBand dataType="UInt16" band="{band_num}">
    <ComplexSource>
      <SourceFilename>{temp_vrt_path}</SourceFilename>
      <SourceBand>{band_num}</SourceBand>
    </ComplexSource>
  </VRTRasterBand>'''

vrt_xml += '\n</VRTDataset>'

with open(output_vrt, 'w') as f:
    f.write(vrt_xml)

print(f"✓ Unmasked VRT created: {output_vrt}")
print(f"  This will show raw Sentinel values for comparison")
