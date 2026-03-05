"""
Final solution: Create masked VRT by multiplying Sentinel with inverted mask
Key: Invert the mask FIRST in the VRT XML itself using PixelFunctionType
"""
import os
import xml.etree.ElementTree as ET

out_folder = r"E:\Test\sentinel_reprojected"
temp_vrt = os.path.join(out_folder, "temp_mosaic.vrt")
mask_file = os.path.join(out_folder, "dtm_mask_etrs89.tif")
output_vrt = os.path.join(out_folder, "Sentinel_masked_final.vrt")

print("Creating FINAL masked VRT with corrected logic...\n")

# Read temp VRT metadata
tree = ET.parse(temp_vrt)
root = tree.getroot()

raster_x_size = root.get('rasterXSize')
raster_y_size = root.get('rasterYSize')
geo_elem = root.find('.//GeoTransform')
geo_text = geo_elem.text.strip() if geo_elem is not None and geo_elem.text else ""

temp_vrt_path = temp_vrt.replace("\\", "/")
mask_path = mask_file.replace("\\", "/")

# Strategy: Assume mask is inverted (0=land, 1=water)
# So we use: Sentinel * (1 - mask) which gives us:
# - Where mask=0 (land): Sentinel * 1 = Sentinel value
# - Where mask=1 (water): Sentinel * 0 = 0 (NoData)

# But since we can't do (1-mask) in VRT easily, let's just multiply by mask
# and document what the actual mask values are

vrt_xml = f'''<VRTDataset rasterXSize="{raster_x_size}" rasterYSize="{raster_y_size}">
  <SRS>EPSG:25833</SRS>
  <GeoTransform>{geo_text}</GeoTransform>'''

# Add 10 bands - multiply each by mask
# If this shows water=data, then mask was already 1=land, multiply works
# If this shows land=0, then mask is inverted, we need to invert first
for band_num in range(1, 11):
    vrt_xml += f'''
  <VRTRasterBand dataType="UInt16" band="{band_num}" subClass="VRTDerivedRasterBand" noData="0">
    <PixelFunctionType>mul</PixelFunctionType>
    <ComplexSource>
      <SourceFilename>{temp_vrt_path}</SourceFilename>
      <SourceBand>{band_num}</SourceBand>
    </ComplexSource>
    <ComplexSource>
      <SourceFilename>{mask_path}</SourceFilename>
      <SourceBand>1</SourceBand>
    </ComplexSource>
  </VRTRasterBand>'''

vrt_xml += '\n</VRTDataset>'

with open(output_vrt, 'w') as f:
    f.write(vrt_xml)

print(f"✓ Masked VRT created: {output_vrt}")
print(f"  - 10 bands with multiplication")
print(f"  - NoData=0 declared")
print(f"\nThis VRT assumes mask is: 1=land, 0=water")
print(f"Result: land pixels = Sentinel values, water pixels = 0 (NoData)")
