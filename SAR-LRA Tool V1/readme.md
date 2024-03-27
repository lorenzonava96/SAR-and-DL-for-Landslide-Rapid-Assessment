# SAR-LRA Tool V1
Due to conflicts in library dependencies, it is necessary to create two separate Anaconda environments. The first environment is designated for executing the 01_SAR-LRA_Sentinel-1_Image_Acquisition.ipynb script, specifically tailored for processing and acquiring SAR data from Google Earth Engine. Conversely, the second environment is dedicated to running scripts 02_Models_Deployment.ipynb, which involve deploying models and storing predictions as shapefiles.

First environment library versions:
leafmap: 0.22.0
geemap: 0.24.1
Earth Engine API: 0.1.358
numpy: 1.25.0
pandas: 2.0.3

Second environment library versions:
Rasterio version: 1.2.10
TensorFlow version: 2.10.0
NumPy version: 1.21.5
OpenCV version: 4.5.5
