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
import torch
from torch.utils.data import Dataset
# from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor
from CLIP import load as load_clip
# import openclip
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, DeiTForImageClassificationWithTeacher


# Adjusted parse_data function to accommodate both formats
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

class DICOMDatasetForCLIP(Dataset):
    def __init__(self, image_paths, labels, clip_image_processor):
        self.image_paths = image_paths
        self.labels = labels
        self.clip_image_processor = clip_image_processor

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
        processed_image = self.clip_image_processor(image, return_tensors="pt")['pixel_values'].squeeze()


        return processed_image.half(), torch.tensor(label, dtype=torch.half)
# Assuming all other class and function definitions from the previous snippets remain unchanged
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=torch.device('cuda:0')):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation for accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                outputs = model(images)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Accuracy: {100 * correct/total:.2f}%')
class Clip(nn.Module):
    def __init__(self, encoder):
        super(SimpleBaseline, self).__init__()
        # Load the pre-trained model
        self.encoder = encoder
        self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(192, 128), 
                nn.ReLU(),           
                nn.Linear(128, 1),   
                nn.Sigmoid()         
            )
    def forward(self, x):
        # with torch.no_grad():  # Ensure no gradients are computed for the encoder
        x = self.encoder(x, output_hidden_states = True).hidden_states[-1].permute(0, 2, 1)  # Get the 768-dimensional vector
        x = self.classifier(x)  # Classify the vector
        return torch.sigmoid(x)  # Use sigmoid to output a probability





def main():


    training_data_path = "/home/xmw5190/FedMFM/data/RSNA/train_client.json"
    test_data_path = "/home/xmw5190/FedMFM/data/RSNA/test.json"

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/deit-tiny-distilled-patch16-224')
    model = DeiTForImageClassificationWithTeacher.from_pretrained('facebook/deit-tiny-distilled-patch16-224')

    
    train_images, train_labels = parse_data(training_data_path)
    test_images, test_labels = parse_data(test_data_path)
    
    
    train_dataset = DICOMDatasetForCLIP(train_images, train_labels, feature_extractor)
    test_dataset = DICOMDatasetForCLIP(test_images, test_labels, feature_extractor)
    
    # sampled_test_dataset = sample_dataset(test_dataset, sample_len=1000, sample_seed=42)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = SimpleBaseline(model)
    model.half()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, eps = 1e-5)  # Only optimize the classifier layer


    
    model.to(device)

    # Training and validation code remains the same
    num_epochs = 1
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=num_epochs, device=device)

if __name__ == '__main__':
    main()


