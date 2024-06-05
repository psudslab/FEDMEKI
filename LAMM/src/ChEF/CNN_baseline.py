import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
import json
from PIL import Image
from sklearn.model_selection import train_test_split
import os
from open_clip import create_model_from_pretrained, get_tokenizer

# Custom Dataset class
class DICOMDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        # Load DICOM image
        dicom = pydicom.dcmread(path)
        image = apply_voi_lut(dicom.pixel_array, dicom)
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            image = np.max(image) - image
        image = Image.fromarray(image).convert('L')  # Convert to grayscale PIL Image
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)
class DICOMDatasetForCLIP(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels
        _, self.clip_image_processor = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load DICOM image
        dicom = pydicom.dcmread(image_path)
        image = apply_voi_lut(dicom.pixel_array, dicom)

        if dicom.PhotometricInterpretation == "MONOCHROME1":
            image = np.max(image) - image

        image = Image.fromarray(image).convert("RGB")  # Convert to RGB PIL Image

        # Process image through CLIP processor
        processed_image = self.clip_image_processor(image)

        return processed_image, torch.tensor(label, dtype=torch.float32)

class CustomSubset(Subset):
    """
    Subset of a dataset at specified indices.
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

def sample_dataset(dataset, sample_len=1000, sample_seed=0):
    """
    Sample a subset of a given dataset.
    """
    if sample_len == -1 or sample_len >= len(dataset):
        return dataset
    np.random.seed(sample_seed)
    random_indices = np.random.choice(len(dataset), sample_len, replace=False)
    return CustomSubset(dataset, random_indices)


def parse_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    images, labels = [], []
    for item in data:
        image_path = item['image']
        # Check for data format and extract classification accordingly
        if 'conversations' in item:
            classification = item['conversations'][1]['value'].split(': ')[1].strip().lower()
        elif 'label' in item:
            classification = item['label'].strip().lower()
        else:
            continue  # Skip if neither format is found

        label = 1 if classification == 'lung opacity' else 0
        images.append(image_path)
        labels.append(label)
    return images, labels


# Model Definition
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),  # Adjust size based on the output of your conv layers
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x
class SimpleBaseline(nn.Module):
    def __init__(self, pretrained_encoder_path):
        super(SimpleBaseline, self).__init__()
        # Load the pre-trained model
        # _, self.visual_preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.encoder = torch.load(pretrained_encoder_path)
        self.encoder.eval()  # Set the encoder to evaluation mode
        for param in self.encoder.parameters():
            param.requires_grad = False  # Freeze the parameters
        
        # Define the classification layer
        self.classifier = nn.Linear(512, 1)  # Assuming the encoder outputs 768-dimensional vectors
        
    def forward(self, x):
        with torch.no_grad():  # Ensure no gradients are computed for the encoder
            x = self.encoder.encode_image(x)  # Get the 768-dimensional vector
        x = self.classifier(x)  # Classify the vector
        return torch.sigmoid(x)  # Use sigmoid to output a probability
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=torch.device('cpu')):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)  # Adjust labels' shape
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)  # Adjust labels' shape
                outputs = model(images)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Accuracy: {100 * correct/total:.2f}%')

def main():
    # Define file paths
    training_data_path = "/home/xmw5190/FedMFM/lung_opacity_classification_server.json" # Update this path
    test_data_path = "/home/xmw5190/FedMFM/transformed_data.json" # Update this path
    

    # Parse data
    train_images, train_labels = parse_data(training_data_path)
    test_images, test_labels = parse_data(test_data_path)

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Datasets
    # train_dataset = DICOMDataset(train_images, train_labels, transform=transform)
    # test_dataset = DICOMDataset(test_images, test_labels, transform=transform)

    # Datasets
    train_dataset = DICOMDatasetForCLIP(train_images, train_labels)
    test_dataset = DICOMDatasetForCLIP(test_images, test_labels)


    # Sample the test dataset
    sampled_test_dataset = sample_dataset(test_dataset, sample_len=100, sample_seed=0)

    print("data prepared")
    # DataLoaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(sampled_test_dataset, batch_size=batch_size, shuffle=False)

    # Model setup
    # model = SimpleCNN()
    model = SimpleBaseline("/data/xiaochen/CSP_checkpoints/csp_clip.pt")
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training
    num_epochs = 4
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=num_epochs, device=device)

if __name__ == '__main__':
    main()