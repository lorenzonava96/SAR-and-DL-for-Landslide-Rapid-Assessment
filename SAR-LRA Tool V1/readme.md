# SAR-LRA Tool V1
It is necessary to create two separate Anaconda environments. The first environment is designated for executing the 01_SAR-LRA_Sentinel-1_Image_Acquisition.ipynb script, specifically tailored for processing and acquiring SAR data from Google Earth Engine. Conversely, the second environment is dedicated to running scripts 02_Models_Deployment.ipynb, which involve deploying models and storing predictions as shapefiles.

01_SAR-LRA_Sentinel-1_Image_Acquisition.ipynb libraries versions:
leafmap: 0.22.0;
geemap: 0.24.1;
Earth Engine API: 0.1.358;
numpy: 1.25.0;
pandas: 2.0.3;

02_Models_Deployment.ipynb libraries versions:
Rasterio version: 1.2.10;
TensorFlow version: 2.10.0;
NumPy version: 1.21.5;
OpenCV version: 4.5.5;

You can use 01_SAR-LRA_Sentinel-1_Image_Acquisition.ipynb to process and download the SAR imagery for a give event. Select the dates and digitise manually the Area of Interest. Default AoI and dates for the Sumatra event is available in the notebook for testing.
