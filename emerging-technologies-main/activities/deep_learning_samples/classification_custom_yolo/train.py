from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    #model = YOLO("yolo11n-cls.pt")
    model = YOLO("yolo11m-cls.pt")  
    
    # Train the model
    results = model.train(
        data="data/custom", # this is the parent directory of the dataset
        #epochs=2,          # number of epochs for training
        epochs=10,
        #imgsz=32,           # imgsz is also input_size
        imgsz=224,
        workers=8           # in case of RunTimeError, reduce this value until you found enough workers
        )