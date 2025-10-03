import torchvision.transforms as transforms
import cv2
import torch
from PIL import Image


# Define image transformations (preprocessing)
def transform_image(input_size):
    mean, std = get_imagenet_mean_std()
    transform = transforms.Compose([
        transforms.Resize(input_size),  
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)  
        ])
    return transform
    
# Common values used for pre-trained models on ImageNet
# ImageNet: https://doi.org/10.1109/CVPR.2009.5206848
def get_imagenet_mean_std():
    mean = [0.485, 0.456, 0.406]   
    std = [0.229, 0.224, 0.225] 
    return (mean, std) 


# Function to preprocess an image
def preprocess_image(image, input_size, device):
    # Step 3.1: Convert the image from BGR to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Step 3.2: Convert the NumPy array to a PIL image
    image = Image.fromarray(image)
    # Step 3.3: Transform the image
    transform = transform_image(input_size)
    image_tensor = transform(image)
    # Step 3.4: Add a batch dimension (1, C, H, W)
    image_tensor = image_tensor.unsqueeze(0)
    # Step 3.5: Move the tensor to the specified device
    image_tensor = image_tensor.to(device)
    return image_tensor



if __name__ == "__main__":
    # We can simulate the deployment stage by predicting one sample image
    # Step 1. Load sample image
    image = cv2.imread('assets/cifar10_sample_image.png')
    # Step 2. Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Step 3. Perform pre-processing techniques
    # Important: You need to perform similar processes when you prepare data for training
    image_tensor = preprocess_image(image, input_size=(224, 224), device=device)
    # Step 4. Load model
    model = torch.load('runs/train_1/best_model.pt')
    model.to(device)    
    model.eval()
    # Step 5. Predict image
    labels = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }
    prediction = model(image_tensor)
    pred_label_index = torch.argmax(prediction, dim=1).item()
    pred_label = labels[pred_label_index]
    print(f'The image is predicted as {pred_label}')


    ###########################################################
    # That is the end of deployment steps. Nothing follows...
    # In real-world application, you can provide a loop for 
    # live data gathering and prediction.
    ###########################################################
    # So now, let us verify if our prediction is correct.
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()