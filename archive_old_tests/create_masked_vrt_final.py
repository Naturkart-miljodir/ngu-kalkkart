"""
Create masked VRT by referencing the same mask band 10 times
"""
import os
import xml.etree.ElementTree as ET

out_folder = r"E:\Test\sentinel_reprojected"
temp_vrt = os.path.join(out_folder, "temp_mosaic.vrt")
mask_file = os.path.join(out_folder, "dtm_mask_etrs89.tif")
output_vrt = os.path.join(out_folder, "Sentinel_masked_final.vrt")

print("Creating masked VRT (reusing mask 10 times)...\n")

# Read temp VRT metadata
tree = ET.parse(temp_vrt)
root = tree.getroot()

raster_x_size = root.get('rasterXSize')
raster_y_size = root.get('rasterYSize')
geo_elem = root.find('.//GeoTransform')
geo_text = geo_elem.text.strip() if geo_elem is not None and geo_elem.text else ""

temp_vrt_path = temp_vrt.replace("\\", "/")
mask_path = mask_file.replace("\\", "/")

# Create VRT with 10 bands, each multiplying Sentinel band with the single mask
vrt_xml = f'''<VRTDataset rasterXSize="{raster_x_size}" rasterYSize="{raster_y_size}">
  <SRS>EPSG:25833</SRS>
  <GeoTransform>{geo_text}</GeoTransform>'''

# Add 10 bands
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

print(f"✓ VRT created successfully")
print(f"  File: {output_vrt}")
print(f"  - 10 bands")
print(f"  - Each band = Sentinel band × mask (band 1)")
print(f"  - NoData=0")
