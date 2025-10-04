'''
    Pre-trained weights available at:

    1. https://huggingface.co/timm
        format: 
            model = timm.create_model(<model_name>, pretrained=True, num_classes=num_classes)
        sample usage:
            model = timm.create_model('mobilenetv3_small_050', pretrained=True, num_classes=num_classes)

    2. https://pytorch.org/vision/0.9/models.html
        format:
            model = torchvision.models.<model_name>(pretrained=True, num_classes=num_classes)
        sample usage:
            model = torchvision.models.resnet18(pretrained=True, num_classes=num_classes)

'''

from prepare_data import prepare_custom_dataset
import timm
import torchvision
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import time


def select_model(num_classes):
    #model = timm.create_model('efficientvit_m5.r224_in1k', pretrained=True, num_classes=num_classes)
    model = timm.create_model('mobilenetv3_large_100.ra_in1k', pretrained=True, num_classes=num_classes)
    return model


def config(model=None):
    # Define what device to use (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    if model is not None: # for training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        return device, criterion, optimizer
    else: # for testing
        return device, criterion



def train_model(train_loader, val_loader, epochs, num_classes, base_dir='runs'):
    # Configure training 
    model = select_model(num_classes)    
    device, criterion, optimizer = config(model)
    model.to(device)
    start_time = time.time()
    # Save losses and accuracies per epoch to generate plots later
    history = {
        'epochs': np.array([]).astype(int),
        'train_losses': np.array([]),
        'train_accuracies': np.array([]),
        'val_losses': np.array([]),
        'val_accuracies': np.array([]),
               }
    # Create a new subfolder for the current run
    save_dir = create_next_folder(base_dir, prefix='train_')
    # Training loop 
    for epoch in range(epochs):
        # Training step on batches
        model, train_loss, train_acc = train(model, train_loader, device, optimizer, criterion)
        # Validation step
        model, val_loss, val_acc = validate(model, val_loader, device, criterion)
        # Save parameters
        elapsed_time = get_elapsed_time(start_time)
        history = update_history(history, epoch, epochs, train_loss, train_acc, val_loss, val_acc, model, elapsed_time, save_dir)  
        # Plot loss and accuracy graphs to judge training convergence
        plot_loss_accuracy(history, save_dir)
    return history



def train(model, train_loader, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    # Train on training set
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    # Calculate training loss and accuracy
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    return (model, train_loss, train_acc)




# For validation step
def validate(model, val_loader, device, criterion):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation for validation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model(inputs)
            running_loss += criterion(outputs, labels).item()
            # Validation accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # Calculate validation loss and accuracy
    loss = running_loss / len(val_loader)
    acc = 100 * correct / total
    return (model, loss, acc)



# For saving parameters in history variable
def update_history(history, epoch, epochs, train_loss, train_acc, val_loss, val_acc, model, elapsed_time, save_dir):
    history['epochs'] = np.append(history['epochs'], epoch + 1)
    history['train_losses'] = np.append(history['train_losses'], train_loss)
    history['train_accuracies'] = np.append(history['train_accuracies'], train_acc)
    history['val_losses'] = np.append(history['val_losses'], val_loss)
    history['val_accuracies'] = np.append(history['val_accuracies'], val_acc)
    print(f"Epoch [{epoch + 1}/{epochs}] @{elapsed_time}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_acc={train_acc:.2f}%, val_acc={val_acc:.2f}%")
    if len(history['val_losses']) > 1:
        if val_loss < np.min(history['val_losses'][:-1]):
            # save model locally
            model_file_path = os.path.join(save_dir, 'best_model.pt')
            torch.save(model, model_file_path)
            print(f"Epoch [{epoch + 1}/{epochs}] @{elapsed_time}: Saved best model")
    return history



def plot_loss_accuracy(history, save_dir):
    # Creating subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # Plot losses on the left subplot
    ax[0].plot(history['epochs'], history['train_losses'], label='Training Loss', marker='o', color='blue')
    ax[0].plot(history['epochs'], history['val_losses'], label='Validation Loss', marker='o', color='red')
    ax[0].axvline(x=np.argmin(history['val_losses'])+1, label='Best Model', linestyle='--', linewidth=2, color='green')
    ax[0].set_title('Learning Curves based on Losses')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].grid(True, alpha=0.5)
    # Plot accuracies on the right subplot
    ax[1].plot(history['epochs'], history['train_accuracies'], label='Training Accuracy', marker='o', color='blue')
    ax[1].plot(history['epochs'], history['val_accuracies'], label='Validation Accuracy', marker='o', color='red')
    ax[1].axvline(x=np.argmin(history['val_losses'])+1, label='Best Model', linestyle='--', linewidth=2, color='green')
    ax[1].set_title('Learning Curves based on Accuracies')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    ax[1].grid(True, alpha=0.5)
    # Display the plot
    plt.tight_layout()
    # Save the figure
    plot_file_path = os.path.join(save_dir, 'learning_curves.png')
    plt.savefig(plot_file_path)
    plt.close() 


# Get the elapsed time to DD:HH:MM:SS format
def get_elapsed_time(start_time):
    end_time = time.time()
    elapsed_time = end_time - start_time
    days = int(elapsed_time // (24 * 3600))
    hours = int((elapsed_time % (24 * 3600)) // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    return (f"elapsed_time={days:02}:{hours:02}:{minutes:02}:{seconds:02}")


# Function to get the next folder number
def get_next_folder_number(save_dir, prefix):
    # List all items in the base directory
    if os.path.exists(save_dir):
        items = os.listdir(save_dir)
    else:
        os.makedirs(save_dir)
        items = []
    # Filter folders with the correct prefix (train_ or test_)
    numbers = []
    for item in items:
        if item.startswith(prefix) and item[len(prefix):].isdigit():
            numbers.append(int(item[len(prefix):]))
    # If there are no folders yet, return 1, otherwise return the next number
    return max(numbers) + 1 if numbers else 1
    

# Function to create the next subfolder
def create_next_folder(base_dir, prefix):
    next_number = get_next_folder_number(base_dir, prefix)
    folder_name = f"{prefix}{next_number}"
    folder_path = os.path.join(base_dir, folder_name)    
    # Create the next folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path





if __name__ == "__main__":
    # Data preparation
    # Important: You should set the input_size to what the model requires, 
    #            regardless the fact that CIFAR-10 images have 32x32 resolutions.
    #            Check the model card details for the input_size requirement.
    train_loader, val_loader, _ = prepare_custom_dataset(input_size=(224, 224), batch_size=64, train_dir='./data/custom/train', val_dir='./data/custom/val', test_dir='./data/custom/test')


    # Training
    history = train_model(train_loader, val_loader, num_classes=2, epochs=10)