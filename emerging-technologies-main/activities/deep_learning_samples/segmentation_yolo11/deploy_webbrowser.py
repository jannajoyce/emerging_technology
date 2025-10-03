import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ultralytics import solutions



if __name__ == '__main__':
    # Pass a model as an argument
    #solutions.inference(model="runs/segment/train4/weights//best.pt")
    solutions.inference(model="assets/pave-yolo8n-seg.pt")
    ### Make sure to run the file using command `streamlit run <file-name.py>`