from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch


# Load standardized CIFAR-1O dataset
def prepare_cifar10(input_size, batch_size):
    transform = transform_image(input_size)
    train_val_datasets = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_dataset, val_dataset = train_val_split(train_val_datasets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return (train_loader, val_loader, test_loader)    


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


# For stratified splitting of training and validation sets
def train_val_split(data):
    # Extract labels from the dataset
    targets = np.array(data.targets)

    # Stratified split (80% train, 20% validation)
    split = StratifiedShuffleSplit(train_size=0.8)
    train_idx, val_idx = next(split.split(np.zeros(len(targets)), targets))
    
    # Create train and validation datasets using the indices
    train_dataset = torch.utils.data.Subset(data, train_idx)
    val_dataset = torch.utils.data.Subset(data, val_idx)
    
    return (train_dataset, val_dataset)
