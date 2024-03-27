# SAR-based Landslide Rapid Assessment (SAR-LRA) Tool
In this repository, we introduce a practical, all-weather, day-night SAR-based Landslide Rapid Assessment tool (SAR-LRA). Utilizing Deep Neural Networks (DNN), this tool is tailored for landslide detection during earthquake-triggered multiple landslide events (MLEs). For further details, please consult the preprint "Sentinel-1 SAR-based Globally Distributed Landslide Detection by Deep Neural Networks" by [Nava et al (2024)](link).

The repository provides files enabling the deployment of DNNs trained on Sentinel-1 Ascending and Descending orbits separately within a specified area. The code utilizes Google Earth Engine (GEE) to acquire and process satellite imagery and deploys the Deep Neural Networks (DNNs) on your local machine.

Due to conflicts in library dependencies, it is necessary to create two separate Anaconda environments. The first environment is designated for executing the 01_SAR-LRA_Sentinel-1_Image_Acquisition.ipynb script, specifically tailored for processing and acquiring SAR data from Google Earth Engine. Conversely, the second environment is dedicated to running scripts 02 and 03, which involve deploying models and storing predictions as shapefiles.

Please note that this tool is currently in its beta version and will undergo continuous improvement as new inventories become available. Additionally, we aim to expand its functionality to include rainfall-induced multiple landslide events in the future.

Contact emails: lorenava996@gmail.com

***SAR-LRA Tool***
![SAR-LRA Tool](https://github.com/lorenzonava96/SAR-and-DL-for-Landslide-Rapid-Assessment/blob/main/object%20detection.png)
