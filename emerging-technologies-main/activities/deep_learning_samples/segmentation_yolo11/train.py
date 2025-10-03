import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    #model = YOLO("yolo11n-seg.pt") 
    model = YOLO("yolo11x-seg.pt")  
    
    # Train the model
    results = model.train(
        data="data/custom.yaml",# the config file for dataset is in yaml
        epochs=30,               # number of epochs for training
        imgsz=640,              # imgsz is also input_size
        workers=8,              # in case of RunTimeError, reduce this value until you found enough workers
        batch=8                 # batch is batch_size; reduce if necessary
        )