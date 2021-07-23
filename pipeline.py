"""
A class pipeline to lay out the file handling and function calling
for a complete image analysis of confocal images of colloid data

Pipeline steps

(1) Initialize class with file paths for data and metaData
(2) Denoise confocal images with pyACsN
(4) Deconvolution with Deconvoltuion Lab 2 (java library) on spatially hashed
    image
(5) Machine learning of particle locations

-Zsolt Jul 23 2021
"""
