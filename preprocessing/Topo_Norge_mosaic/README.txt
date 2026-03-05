These are jupyter notebooks created using ArcGIS Pro:

These codes make a mosaic of different topographical derivates e.g., slope, tri, etc. 

The topo data has tewo sources: 

1) tiles that need to be clipped by polygons representing hydrological basins prior making a mosaic (Masking_clip_Mosaic_Norge and Masking_clip_Norge)
2) tiles that don't need clipping before making the mosaic (Mosaic_noclip_Norge)

In both cases, Norway scale mosaics were created as geotifs (eventually these should be created as COG) for each topo derivate

The Reproj_to_25833 code is to reproject/align these topo geotifs (Norway mosaics) to the Alpha Earth data