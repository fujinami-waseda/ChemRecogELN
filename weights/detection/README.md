# ChemRecogELN


## Introduction
This repository demonstrates the integrated application of computer vision schemes to chemical laboratory experiments.
The details are shown in an academic paper. The paper is currently under review.

## Installation
It is recommended to create an isolated environment (python==3.11).  
The following libraries need to be additionally installed to the default python packages:

```commandline
streamlit==1.37.0  
pytorch==2.3.1  
ultralytics==8.0.122  
scikit-learn==1.5.1  
opencv==4.10.0  
matplotlib==3.9.1  
gstreamer==1.24.5  
lxml==5.2.2  
joblib==1.4.2  
h5py==3.11.0   
beautifulsoup4==4.12.3  
pyzbar==0.1.9   
(zbar is additionally required and shold be connected to pyzbar)  
```

## Data location
Trained models for object detection "yolov8x.pt" and action recognition "save_3218.pth" are available on the [online drive](https://drive.google.com/drive/folders/11rnudTBXG4axoF9jx2BLnaM7z9wptjEH?usp=drive_link).  
Codes and data structure is following. Please locate the downloaded models in "./weithgts/detection" and "./weights/action".  
In the current version sample videos should be in root directory "./". Sample images for barcode recognition are in ./sample_barcode.

```commandline
~/
  app.py

  weights/
    detection/
      yolo8x.pt
    action/
      save_3218.pth

  fig6a.mp4
  fig6b.mp4
  sample_barcode/

  ultralytics/
  3D-ResNets-PyTorch-master/
  tempwork/
```

## Run
Executing the following command will launch the GUI on your web browser.

```commandline
Python3 -m streamlit run app.py
```
