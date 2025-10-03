# Data Preparation
If you want to have a good model that can generalize tasks robustly, then you need to prepare a properly annotated large dataset. However, this stage is time-consuming and can take at least one month. Thus, instead of doing so, you will use our available dataset from the laboratory. However, you will only be using 4 raw frames, each with a corresponding 20 augmented images.

Download the subset of the dataset in this [Google Drive](https://drive.google.com/drive/folders/1r70YybRAv3wUB_SGhOCKff3DZpFAYTDn?usp=sharing) folder. The directory structure must be the same as below. Important: Each image should have a txt file containing the labels.

```
segmentation_yolo11/
├── data/
│   ├── custom.yaml/
│   └── datasets/
│       ├── train/
|       |   ├── images/
|       |   └── labels/
|       ├── val/
|       |   ├── images/
|       |   └── labels/
|       └── test/
|           ├── images/
|           └── labels/
├── assets/
├── train.py
├── test.py
├── deploy.py
└── deploy_webbrowser.py
```