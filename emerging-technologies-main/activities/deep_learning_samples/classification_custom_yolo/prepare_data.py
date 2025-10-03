from torchvision import datasets, transforms
from torch.utils.data import DataLoader



def prepare_test_dataset(batch_size, test_dir, input_size):
    transform = transform_image(input_size)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


# Define image transformations (preprocessing)
def transform_image(input_size):
    transform = transforms.Compose([
        transforms.Resize(input_size),  
        transforms.ToTensor(),
        ])
    return transform
