import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    model = YOLO("runs/segment/train4/weights//best.pt")  # load a custom model

    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    print(metrics.box.map)  # map50-95
    print(metrics.box.map50)  # map50
    print(metrics.box.map75)  # map75
    print(metrics.box.maps)  # a list contains map50-95 of each category