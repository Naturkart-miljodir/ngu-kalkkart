These codes comprise objectives: 

1) mask Sentinel images for water and snow (Snow_masking folder)
2) There 3 codes that do the masking: mask_national_tiles_snow_ndsi (all imgaes available for Norway; masks water and snow),
mask_trondelag_tiles_snow_ndsi (all imgaes available for trondelag (local); masks water and snow), and mask_trondelag_tiles_no_snow (all images available for trondelag (local); ONLY masks water)
3) A VRT mosaic is also created after masking

Assessing normalization codes: 1) They calculate (for National and Local("trondelag") modelling) the statistics used to normalized the Sentinel images to create DL chips


