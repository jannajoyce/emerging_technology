import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import cv2
from ultralytics import YOLO
import supervision as sv


if __name__ == '__main__':
    # Load a model
    model = YOLO("runs/segment/train4/weights//best.pt")  # load a custom model
    # model = YOLO("assets/pave-yolo8n-seg.pt")

    # Read an image using OpenCV
    source = cv2.imread("data/datasets/test/images/Jampason Initao Dataset 2_frame07771_0000.png")
    
    # Run inference on the source
    results = model(source)  # list of Results objects

    # Let us visualize the predictions
    detections = sv.Detections.from_ultralytics(results[0])
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)
    annotated_image = source.copy()
    annotated_image = box_annotator.annotate(annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(annotated_image, detections=detections)
    cv2.imshow("Prediction", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

