
from torch.utils.data import DataLoader, random_split, Dataset
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
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, DeiTForImageClassificationWithTeacher
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


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
        # return processed_image.half(), torch.tensor(label, dtype=torch.half) 
        return processed_image, torch.tensor(label, dtype=torch.float) 
class Client:
    def __init__(self, model, train_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
    
    def train(self):
        self.model.train()
        running_loss = 0.0
        for images, labels in tqdm(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device).unsqueeze(1)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(self.train_loader)

def federated_average(models):
    with torch.no_grad():
        # Start by using the model parameters from the first model
        global_model = models[0]
        for key in global_model.state_dict().keys():
            # Average the parameters from each client
            global_model.state_dict()[key].copy_(
                torch.mean(
                    torch.stack([model.state_dict()[key].float() for model in models]), dim=0
                )
            )
    return global_model
class Clip(nn.Module):
    def __init__(self, encoder):
        super(Clip, self).__init__()
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
        print(x.shape)
        x = self.classifier(x)  # Classify the vector
        return torch.sigmoid(x)  # Use sigmoid to output a probability
def local_training(num_clients = 5, batch_size = 512, server_model = None):
    # Setup
    # num_clients = 5
    epochs = 1
    # batch_size = 512
    clients = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/deit-tiny-distilled-patch16-224')
    
    # Load and prepare data
    data_path = "/home/xmw5190/FedMFM/data/RSNA/train_client.json"
    images, labels = parse_data(data_path)
    dataset = DICOMDatasetForCLIP(images, labels, feature_extractor)
    
    # Split data among clients
    client_datasets = random_split(dataset, [len(dataset) // num_clients + (1 if x < len(dataset) % num_clients else 0) for x in range(num_clients)])
    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets]

    # Create clients
    for i in range(num_clients):
        if not server_model:
            model = DeiTForImageClassificationWithTeacher.from_pretrained('facebook/deit-tiny-distilled-patch16-224').to(device)
        # model = Clip(model).half().to(device)
        else:
            model = server_model
        model = Clip(model).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, eps = 1e-8)
        criterion = nn.BCELoss()
        clients.append(Client(model, client_loaders[i], criterion, optimizer, device))
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        models = []
        for client in clients:
            loss = client.train()
            print(f"Client Loss: {loss}")
            models.append(client.model)
        
        # Aggregate models
        global_model = federated_average(models)
        for client in clients:
            client.model.load_state_dict(global_model.state_dict())
    return global_model
if __name__ == '__main__':
    # main()
    local_training()