from ultralytics import YOLO
from prepare_data import prepare_test_dataset
import torch
import os
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



def test_model(test_loader, model, num_classes, k=1):
    device, criterion = config()
    model.to(device)
    model = evaluate(model, test_loader, device, criterion, num_classes, k)


def config():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    return device, criterion


def evaluate(model, data_loader, device, criterion, num_classes, k, base_dir='runs/classify'):
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = torch.tensor([], dtype=torch.long).to(device)  # Initialize empty tensor for labels
    all_preds = torch.tensor([], dtype=torch.long).to(device)   # Initialize empty tensor for predictions
    all_probs = torch.tensor([], dtype=torch.float).to(device)  # For probabilities (needed for AUC/PRC)     
    with torch.no_grad():  # Disable gradient computation for validation
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model.model(inputs)
            running_loss += criterion(outputs, labels).item()
            # Validation accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Collect all predictions, labels, and probabilities
            all_labels = torch.cat((all_labels, labels))
            all_preds = torch.cat((all_preds, predicted))
            all_probs = torch.cat((all_probs, torch.softmax(outputs, dim=1)))  # Softmax for probabilities
    # Calculate validation loss and accuracy
    loss = running_loss / len(data_loader)
    acc = 100 * correct / total
    # Move tensors to CPU for metric calculations
    all_labels_cpu = all_labels.cpu().numpy()
    all_preds_cpu = all_preds.cpu().numpy()
    all_probs_cpu = all_probs.cpu().numpy()  # Probabilities for AUC and Precision-Recall Curve
    # Create a new subfolder for the current run
    save_dir = create_next_folder(base_dir, prefix='test')
    # Show summary of loss and accuracy
    show_loss_accuracy(loss, acc, num_classes, all_labels_cpu, all_probs_cpu, all_preds_cpu, k, save_dir)
    # Compute precision, recall, and F1 score (weighted average)
    get_classification_report(all_labels_cpu, all_preds_cpu, save_dir)
    # Compute confusion matrix
    get_confusion_matrix(all_labels_cpu, all_preds_cpu, save_dir)
    # Compute AUC for ROC curve (One-vs-Rest for multiclass)
    get_roc_auc_score(num_classes, all_labels_cpu, all_probs_cpu, save_dir)
    # Compute Area under the Precision-Recall Curve (One-vs-Rest for multiclass)
    get_precision_recall_curve(num_classes, all_labels_cpu, all_probs_cpu, save_dir)
    return model


# Function to create the next subfolder
def create_next_folder(base_dir, prefix):
    next_number = get_next_folder_number(base_dir, prefix)
    folder_name = f"{prefix}{next_number}"
    folder_path = os.path.join(base_dir, folder_name)    
    # Create the next folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


# Function to get the next folder number
def get_next_folder_number(save_dir, prefix):
    # List all items in the base directory
    if os.path.exists(save_dir):
        items = os.listdir(save_dir)
    else:
        os.makedirs(save_dir)
        items = []
    # Filter folders with the correct prefix 
    numbers = []
    for item in items:
        if item.startswith(prefix) and item[len(prefix):].isdigit():
            numbers.append(int(item[len(prefix):]))
    # If there are no folders yet, return 1, otherwise return the next number
    return max(numbers) + 1 if numbers else 1


def show_loss_accuracy(loss, acc, num_classes, all_labels, all_probs, all_preds, k, save_dir):
    # Compute top-k accuracy (e.g., top-5 accuracy)
    top_k_acc = metrics.top_k_accuracy_score(all_labels, all_probs, k=k, labels=range(num_classes))
    # Compute MCC (Matthews Correlation Coefficient)
    mcc = metrics.matthews_corrcoef(all_labels, all_preds)
    # Compute balanced accuracy
    bal_acc = metrics.balanced_accuracy_score(all_labels, all_preds)
    # Put in one container
    metrics_data = {
        'Loss': [loss],
        'Accuracy': [acc],
        f'Top-{k} Accuracy': [top_k_acc],
        'Balanced Accuracy': [bal_acc],
        'MCC': [mcc]
    }
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 4))  # Adjust size as needed
    ax.axis('tight')
    ax.axis('off')
    # Create the table
    table_data = [[metric, value[0]] for metric, value in metrics_data.items()]
    table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'], cellLoc='center', loc='center')
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)  # Scale the table for better visibility
    # Add title
    title = 'Model Evaluation Metrics'
    plt.title(title, fontsize=14)
    # Save the figure
    plot_file_path = os.path.join(save_dir, f'{title}.png')
    plt.savefig(plot_file_path)
    plt.close() 
    

def get_classification_report(all_labels, all_preds, save_dir):
    # Generate classification report
    report = metrics.classification_report(all_labels, all_preds, output_dict=True)
    # Convert the report into a pandas dataframe
    report_df = pd.DataFrame(report).transpose()
    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="Blues", fmt='.3f')
    # Add labels and titles
    title = 'Classification Report'
    plt.title(title)
    plt.ylabel('Classes')
    plt.xlabel('Metrics')
    # Save the figure
    plot_file_path = os.path.join(save_dir, f'{title}.png')
    plt.savefig(plot_file_path)
    plt.close() 


def get_confusion_matrix(all_labels, all_preds, save_dir):
    # Assuming conf_matrix is already created
    conf_matrix = metrics.confusion_matrix(all_labels, all_preds)
    # Plot confusion matrix
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap=plt.cm.Blues)  # You can change the color map if desired
    # Add labels and titles
    title = 'Confusion Matrix'
    plt.title(title)
    # Save the figure
    plot_file_path = os.path.join(save_dir, f'{title}.png')
    plt.savefig(plot_file_path)
    plt.close() 
    


def get_roc_auc_score(num_classes, all_labels, all_probs, save_dir):
    # Binarize the labels for multi-class ROC
    y_bin = label_binarize(all_labels, classes=[i for i in range(num_classes)])
    # Compute ROC curve and ROC AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    # Plot ROC curve for each class
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    # Plot diagonal line for reference
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
    # Add labels and titles
    title = 'ROC Curve (One-vs-Rest)'
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    # Save the figure
    plot_file_path = os.path.join(save_dir, f'{title}.png')
    plt.savefig(plot_file_path)
    plt.close() 
    


def get_precision_recall_curve(num_classes, all_labels, all_probs, save_dir):
    # Binarize the labels for multiclass precision-recall
    all_labels_bin = label_binarize(all_labels, classes=[i for i in range(num_classes)])
    # Plot Precision-Recall curves and compute AUC for each class
    plt.figure()
    for i in range(num_classes):
        precision, recall, _ = metrics.precision_recall_curve(all_labels_bin[:, i], all_probs[:, i])
        # Compute AUC for the precision-recall curve
        auc_score = metrics.auc(recall, precision)
        # Plot the precision-recall curve for each class
        plt.plot(recall, precision, lw=2, label=f'Class {i} (AUC = {auc_score:.2f})')
    # Add labels and title to the plot
    title = 'Precision-Recall Curve for Multiclass with AUC'
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='best')
    # Save the figure
    plot_file_path = os.path.join(save_dir, f'{title}.png')
    plt.savefig(plot_file_path)
    plt.close() 






if __name__ == '__main__':
    # Data Preparation
    test_loader = prepare_test_dataset(test_dir='data/custom/test', input_size=(224,224), batch_size=32) 
    
    # Testing 
    model = YOLO("runs/classify/train/weights/best.pt")  
    test_model(test_loader, model, num_classes=2, k=1)
