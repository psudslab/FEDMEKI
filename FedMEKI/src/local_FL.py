#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 23:58:02 2024

@author: xmw5190
"""

# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoFeatureExtractor, DeiTForImageClassificationWithTeacher
from torch.utils.data import DataLoader, random_split, Dataset
import random
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import json
from PIL import Image
from copy import deepcopy
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
from tqdm import tqdm
import wfdb
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Set the seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Parse data function
def parse_data(file_path, modality):
    with open(file_path, 'r') as file:
        data = json.load(file)

    if modality == 'image':
        image_paths, labels = [], []
        for item in data:
            image_path = item['modality_path']
            label = 1 if 'yes' in item['conversations'][1]["value"].strip().lower() else 0
            image_paths.append(image_path)
            labels.append(label)
        return image_paths, labels

    elif modality == 'covid':
        image_paths, labels = [], []
        for item in data:
            image_path = item['modality_path']
            label = 1 if 'yes' in item['conversations'][1]["value"].strip().lower() else 0
            image_paths.append(image_path)
            labels.append(label)
        return image_paths, labels

    elif modality == 'ecg':
        ecg_paths, labels = [], []
        for item in data:
            ecg_path = item['modality_path']
            label = 1 if 'yes' in item['conversations'][1]["value"].strip().lower() else 0
            ecg_paths.append(ecg_path)
            labels.append(label)
        return ecg_paths, labels

    elif modality == 'clinicals':
        clinical_paths, labels = [], []
        for item in data:
            clinical_path = item['modality_path']
            label = 1 if 'yes' in item['conversations'][1]["value"].strip().lower() else 0
            clinical_paths.append(clinical_path)
            labels.append(label)
        return clinical_paths, labels

    else:
        raise ValueError("Invalid modality specified. Choose from 'image', 'covid', 'ecg', or 'clinicals'.")

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, image_processor):
        self.image_paths = image_paths
        self.labels = labels
        self.image_processor = AutoFeatureExtractor.from_pretrained('facebook/deit-tiny-distilled-patch16-224')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        if image_path.lower().endswith('.dcm'):
            # Load DICOM image
            dicom = pydicom.dcmread(image_path)
            image = apply_voi_lut(dicom.pixel_array, dicom)

            if dicom.PhotometricInterpretation == "MONOCHROME1":
                image = np.max(image) - image
            image = Image.fromarray(image).convert("RGB")  # Convert to RGB PIL Image
        elif image_path.lower().endswith('.png'):
            # Load PNG image
            image = Image.open(image_path).convert("RGB")

        processed_image = self.image_processor(image, return_tensors="pt")['pixel_values'].squeeze()
        return processed_image, torch.tensor(label, dtype=torch.float)

class ECGDataset(Dataset):
    def __init__(self, ecg_paths, labels):
        self.ecg_paths = ecg_paths
        self.labels = labels

    def __len__(self):
        return len(self.ecg_paths)

    def __getitem__(self, idx):
        ecg_path = self.ecg_paths[idx]
        label = self.labels[idx]

        record = wfdb.rdsamp(ecg_path)
        signal = record[0]
        signal = torch.tensor(signal, dtype=torch.float).T

        return signal, torch.tensor(label, dtype=torch.float)

class ClinicalDataset(Dataset):
    def __init__(self, clinical_data, labels):
        self.clinical_data = clinical_data
        self.labels = labels

    def __len__(self):
        return len(self.clinical_data)

    def __getitem__(self, idx):
        clinical_features = self.clinical_data[idx]
        label = self.labels[idx]

        clinical_features = torch.load(clinical_features)

        return clinical_features, torch.tensor(label, dtype=torch.float)

class MeanPool2d(nn.Module):
    def forward(self, x):
        return x.mean(dim=1)

class UnifiedModel(nn.Module):
    def __init__(self, image_model=None):
        super(UnifiedModel, self).__init__()
        self.visual_encoder = DeiTForImageClassificationWithTeacher.from_pretrained('facebook/deit-tiny-distilled-patch16-224') if not image_model else image_model

        self.signal_module = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(12),  # BatchNorm to stabilize inputs
            nn.ReLU(),
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(12),  # BatchNorm to stabilize inputs
            nn.ReLU(),
            nn.Linear(1000, 4096)
        )
        
        self.clinical_module = nn.Sequential(
            nn.Linear(48, 512),
            nn.ReLU(),
            TransformerEncoder(TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=512, dropout=0.1), num_layers=1),
            nn.LayerNorm(512),  # LayerNorm for transformer layers
            nn.ReLU(),
            nn.Linear(512, 4096)
        )

        self.fc_heads = nn.ModuleDict({
            'lung_opacity': nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(192, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ),
            'covid_detection': nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(192, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ),
            'ecg_abnormal': nn.Sequential(MeanPool2d(), nn.Linear(4096, 1)),
            'mortality': nn.Sequential(MeanPool2d(), nn.Linear(4096, 1)),
        })

        self.sigmoid = nn.Sigmoid()
        self.current_task = None

    def set_task(self, task):
        self.current_task = task

    def forward(self, image=None, ecg=None, clinicals=None):

        if image is not None:
            features = self.visual_encoder(image, output_hidden_states=True).hidden_states[-1].permute(0, 2, 1)
        if clinicals is not None:
            features = self.clinical_module(clinicals)
        if ecg is not None:
            features = self.signal_module(ecg)
        
        fc_layer = self.fc_heads[self.current_task]
        x = fc_layer(features)
        return x

def evaluate_image_model(model, test_loader, device):
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():  # No need to track gradients for evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(image=images)
           
            predicted = (outputs > 0.5).float()  # Threshold predictions
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    accuracy = correct / total
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    mcc = matthews_corrcoef(all_labels, all_predictions)
    
    print(f'Accuracy on test set: {accuracy * 100:.2f}%')
    print(f'Precision on test set: {precision * 100:.2f}%')
    print(f'Recall on test set: {recall * 100:.2f}%')
    print(f'F1 Score on test set: {f1 * 100:.2f}%')
    
    return accuracy, precision, recall, f1, mcc

def evaluate_ecg_model(model, test_loader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.to(device), labels.to(device).unsqueeze(1)
            outputs = model(ecg=signals)
            predicted = (outputs > 0.5).float()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    accuracy = correct / total
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    print(f'Accuracy on test set: {accuracy * 100:.2f}%')
    print(f'Precision on test set: {precision * 100:.2f}%')
    print(f'Recall on test set: {recall * 100:.2f}%')
    print(f'F1 Score on test set: {f1 * 100:.2f}%')
    
    return accuracy, precision, recall, f1

def evaluate_clinical_model(model, test_loader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for clinicals, labels in test_loader:
            clinicals, labels = clinicals.to(device), labels.to(device).unsqueeze(1)
            outputs = model(clinicals=clinicals)
            predicted = (outputs > 0.5).float()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    accuracy = correct / total
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    print(f'Accuracy on test set: {accuracy * 100:.2f}%')
    print(f'Precision on test set: {precision * 100:.2f}%')
    print(f'Recall on test set: {recall * 100:.2f}%')
    print(f'F1 Score on test set: {f1 * 100:.2f}%')
    
    return accuracy, precision, recall, f1

def federated_average(models):
    with torch.no_grad():
        global_model = deepcopy(models[0])
        for key in global_model.state_dict().keys():
            global_model.state_dict()[key].copy_(
                torch.mean(torch.stack([model.state_dict()[key].float() for model in models]), dim=0)
            )
    return global_model

class FedAvgClient:
    def __init__(self, model, image_loader, covid_loader, ecg_loader, clinical_loader, criterion, device, use_amp=False):
        self.model = model
        self.image_loader = image_loader
        self.covid_loader = covid_loader
        self.ecg_loader = ecg_loader
        self.clinical_loader = clinical_loader
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

    def train(self, global_model, modality, loader):
        self.model.to(self.device)
        self.model.train()
        running_loss = 0.0

        global_params = {name: param for name, param in global_model.named_parameters()}

        for data in tqdm(loader, desc=f"Training on {modality.capitalize()} Data"):
            if modality == "image":
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.model.set_task("lung_opacity")
                outputs = self.model(image=inputs)
            elif modality == "covid":
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.model.set_task("covid_detection")
                outputs = self.model(image=inputs)
            elif modality == "ecg":
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.model.set_task("ecg_abnormal")
                outputs = self.model(ecg=inputs)
            elif modality == "clinicals":
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.model.set_task("mortality")
                outputs = self.model(clinicals=inputs)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    labels = labels.unsqueeze(-1)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                labels = labels.unsqueeze(-1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            running_loss += loss.item()
        self.model.cpu()
        return running_loss / len(loader)
class FedProxClient:
    def __init__(self, model, image_loader, covid_loader, ecg_loader, clinical_loader, criterion, device, mu=1e-6, use_amp=False):
        self.model = model
        self.image_loader = image_loader
        self.covid_loader = covid_loader
        self.ecg_loader = ecg_loader
        self.clinical_loader = clinical_loader
        self.criterion = criterion
        self.device = device
        self.mu = mu  # Proximal term parameter
        self.use_amp = use_amp
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

    def is_concerning_param(self, name, modality):
        concerning_modules = {
            'image': ['visual_encoder', 'lung_opacity'],
            'covid': ['visual_encoder', 'covid_detection'],
            'ecg': ['signal_module', 'ecg_abnormal'],
            'clinicals': ['clinical_module', 'mortality']
        }
        for module in concerning_modules[modality]:
            if module in name:
                return True
        return False

    def train(self, global_model, modality, loader):
        self.model.to(self.device)
        self.model.train()
        running_loss = 0.0

        # Ensure the global model is in evaluation mode to prevent updates
        global_model.eval()

        # Convert global model parameters to a dictionary for quick lookup
        global_params = {name: param for name, param in global_model.named_parameters()}

        # Convert concerning parameters to a dictionary for quick lookup
        concerning_params = {name: param for name, param in self.model.named_parameters() if self.is_concerning_param(name, modality)}

        # Convert keys to sets for set operations
        global_keys = set(global_params.keys())
        concerning_keys = set(concerning_params.keys())

        # Find overlapping and non-overlapping keys
        overlapping_keys = global_keys & concerning_keys
        non_overlapping_global = global_keys - concerning_keys
        non_overlapping_concerning = concerning_keys - global_keys
        # print(overlapping_keys)

        for data in tqdm(loader, desc=f"Training on {modality.capitalize()} Data"):
            if modality == "image":
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.model.set_task("lung_opacity")
                outputs = self.model(image=inputs)
            elif modality == "covid":
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.model.set_task("covid_detection")
                outputs = self.model(image=inputs)
            elif modality == "ecg":
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.model.set_task("ecg_abnormal")
                outputs = self.model(ecg=inputs)
            elif modality == "clinicals":
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.model.set_task("mortality")
                outputs = self.model(clinicals=inputs)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():

                    labels = labels.unsqueeze(-1)
                    loss = self.criterion(outputs, labels)
      

                    # Add proximal term
                    prox_term = 0.0
                    for name, param in concerning_params.items():
                        if name in global_params:
                            global_param = global_params[name].to(self.device)
                            if param.size() == global_param.size():
                                prox_term += (param - global_param).pow(2).sum()
                            else:
                                print(f"Skipping parameter {name} due to size mismatch: local {param.size()} vs global {global_param.size()}")
                    loss += 0.5 * self.mu * prox_term

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                
                labels = labels.unsqueeze(-1)
                loss = self.criterion(outputs, labels)


                # Add proximal term
                prox_term = 0.0
                for name, param in concerning_params.items():
                    if name in global_params:
                        global_param = global_params[name].to(self.device)
                        if param.size() == global_param.size():
                            prox_term += (param - global_param).pow(2).sum()
                        else:
                            print(f"Skipping parameter {name} due to size mismatch: local {param.size()} vs global {global_param.size()}")
                loss += 0.5 * self.mu * prox_term

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            running_loss += loss.item()
        self.model.cpu()
        return running_loss / len(loader)
        
        

def federated_training(image_loader=None, covid_loader=None, ecg_loader=None, clinical_loader=None, server_model=None, classifier=None, num_clients=5, epochs=1, use_server_data=False, federate_learning=True, use_amp=False, device = None, fl_method = 'FedAvg'):
    # Setup
    # Set your path to corresponding json files
    server_rsna_data_path = path_to_server_rsna_data  
    server_covid_data_path = path_to_server_covid_data  
    server_ecg_data_path = path_to_server_ecg_data  
    server_clinical_data_path = path_to_server_clinical_data  
    
    
    normal_batch_size = 64
    clients = []
    if not device:
      device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/deit-tiny-distilled-patch16-224')
    if not server_model:
      global_model = UnifiedModel(
          image_model=DeiTForImageClassificationWithTeacher.from_pretrained('facebook/deit-tiny-distilled-patch16-224')
      ).to(device)    
    else:
      global_model = server_model

    criterion = nn.BCEWithLogitsLoss()
    
    if federate_learning:
        client_image_datasets = random_split(image_loader.dataset, [len(image_loader.dataset) // num_clients + (1 if x < len(image_loader.dataset) % num_clients else 0) for x in range(num_clients)]) if image_loader else [None] * num_clients
        client_covid_datasets = random_split(covid_loader.dataset, [len(covid_loader.dataset) // num_clients + (1 if x < len(covid_loader.dataset) % num_clients else 0) for x in range(num_clients)]) if covid_loader else [None] * num_clients
        client_ecg_datasets = random_split(ecg_loader.dataset, [len(ecg_loader.dataset) // num_clients + (1 if x < len(ecg_loader.dataset) % num_clients else 0) for x in range(num_clients)]) if ecg_loader else [None] * num_clients
        client_clinical_datasets = random_split(clinical_loader.dataset, [len(clinical_loader.dataset) // num_clients + (1 if x < len(clinical_loader.dataset) % num_clients else 0) for x in range(num_clients)]) if clinical_loader else [None] * num_clients

        for i in range(num_clients):
            model = deepcopy(global_model)
            if fl_method == 'FedAvg':
                clients.append(FedAvgClient(
                    model,
                    DataLoader(client_image_datasets[i], batch_size=normal_batch_size, shuffle=True) if image_loader else None,
                    DataLoader(client_covid_datasets[i], batch_size=normal_batch_size, shuffle=True) if covid_loader else None,
                    DataLoader(client_ecg_datasets[i], batch_size=normal_batch_size, shuffle=True) if ecg_loader else None,
                    DataLoader(client_clinical_datasets[i], batch_size=normal_batch_size, shuffle=True) if clinical_loader else None,
                    criterion,
                    device,
                    use_amp=use_amp
                ))
          elif fl_method == 'FedProx':
              clients.append(FedProxClient(
                    model,
                    DataLoader(client_image_datasets[i], batch_size=normal_batch_size, shuffle=True) if image_loader else None,
                    DataLoader(client_covid_datasets[i], batch_size=normal_batch_size, shuffle=True) if covid_loader else None,
                    DataLoader(client_ecg_datasets[i], batch_size=normal_batch_size, shuffle=True) if ecg_loader else None,
                    DataLoader(client_clinical_datasets[i], batch_size=normal_batch_size, shuffle=True) if clinical_loader else None,
                    criterion,
                    device,
                    use_amp=use_amp
                ))
    
    if use_server_data:
        server_image_loader, server_covid_loader, server_ecg_loader, server_clinical_loader = None, None, None, None
        if image_loader:
            server_image_paths, server_image_labels = parse_data(server_rsna_data_path, 'image')
            server_image_dataset = ImageDataset(server_image_paths, server_image_labels, feature_extractor)
            server_image_loader = DataLoader(server_image_dataset, batch_size=normal_batch_size, shuffle=True)
        if covid_loader:
            server_covid_paths, server_covid_labels = parse_data(server_covid_data_path, 'covid')
            server_covid_dataset = ImageDataset(server_covid_paths, server_covid_labels, feature_extractor)
            server_covid_loader = DataLoader(server_covid_dataset, batch_size=normal_batch_size, shuffle=True)
        if ecg_loader:
            server_ecg_paths, server_ecg_labels = parse_data(server_ecg_data_path, 'ecg')
            server_ecg_dataset = ECGDataset(server_ecg_paths, server_ecg_labels)
            server_ecg_loader = DataLoader(server_ecg_dataset, batch_size=normal_batch_size, shuffle=True)
        if clinical_loader:
            server_clinical_paths, server_clinical_labels = parse_data(server_clinical_data_path, 'clinicals')
            server_clinical_dataset = ClinicalDataset(server_clinical_paths, server_clinical_labels)
            server_clinical_loader = DataLoader(server_clinical_dataset, batch_size=normal_batch_size, shuffle=True)

        server_optimizer = optim.Adam(global_model.parameters(), lr=0.0001)
        server_scaler = torch.cuda.amp.GradScaler() if use_amp else None
        if fl_method == 'FedAvg':
            server = FedAvgClient(
                global_model,
                server_image_loader if server_image_loader else None,
                server_covid_loader if server_covid_loader else None,
                server_ecg_loader if server_ecg_loader else None,
                server_clinical_loader if server_clinical_loader else None,
                criterion,
                device,
                use_amp=use_amp
            )
        elif if fl_method == 'FedProx':
            server = FedProxClient(
                    global_model,
                    server_image_loader if server_image_loader else None,
                    server_covid_loader if server_covid_loader else None,
                    server_ecg_loader if server_ecg_loader else None,
                    server_clinical_loader if server_clinical_loader else None,
                    criterion,
                    device,
                    use_amp=use_amp
                )
        if server_model:
            server.model.load_state_dict(server_model.float().state_dict())
    else:
        if fl_method == 'FedAvg':
            server = FedAvgClient(
                global_model,
                None,
                None,
                None,
                None,
                criterion,
                device,
                use_amp=use_amp
            )
        else:
            server = FedProxClient(
                global_model,
                None,
                None,
                None,
                None,
                criterion,
                device,
                use_amp=use_amp
            )
    if federate_learning:
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}")
            
            for modality, loader_attr in [("image", "image_loader"), ("covid", "covid_loader"), ("ecg", "ecg_loader"), ("clinicals", "clinical_loader")]:
                models = []
                # if epoch < 5 or modality == "clinicals":
                if 1:
                    for client in clients:
                        loader = getattr(client, loader_attr)
                        if loader:
                            loss = client.train(deepcopy(server.model), modality, loader)
                            print(f"Client {modality.capitalize()} Loss: {loss}")
                            models.append(client.model)
                    
                    if models:
                        averaged_model = federated_average(models)
                        server.model.load_state_dict(averaged_model.state_dict())
                    
                    for client in clients:
                        client.model.load_state_dict(server.model.state_dict())

            if use_server_data:
                for modality, server_loader in [("image", server_image_loader), ("covid", server_covid_loader), ("ecg", server_ecg_loader), ("clinicals", server_clinical_loader)]:
                    # if epoch < 5 or modality == "clinicals":
                        if server_loader:
                            loss = server.train(deepcopy(server.model), modality, server_loader)
                            print(f"Server {modality.capitalize()} Loss: {loss}")
                        
                        for client in clients:
                            client.model.load_state_dict(server.model.state_dict())

        return global_model
    else:
        if use_server_data:
            for epoch in range(epochs):
                for modality, server_loader in [("image", server_image_loader), ("covid", server_covid_loader), ("ecg", server_ecg_loader), ("clinicals", server_clinical_loader)]:
                    if server_loader:
                        loss = server.train(server.model, modality, server_loader)
                        print(f"Epoch {epoch+1} - Server {modality.capitalize()} Loss: {loss}")
                        for client in clients:
                            client.model.load_state_dict(server.model.state_dict())
            return server.model
        else:
            raise ValueError("If federate_learning is False, use_server_data must be True to perform fine-tuning on server data.")

if __name__ == '__main__':

    # Set your path to corresponding json files
    # Load client data
    client_rsna_data_path = path_to_client_rsna_data
    client_covid_data_path = path_to_client_covid_data
    client_ecg_data_path = path_to_client_ecg_data
    client_clinical_data_path = path_to_client_clinical_data

    # Load test data
    test_rsna_data_path = path_to_test_rsna_data
    test_covid_data_path = path_to_test_covid_data
    test_ecg_data_path = path_to_test_ecg_data
    test_clinical_data_path = path_to_test_clinical_data

    
    # Specify the modalities to be used
    modalities = ['image', 'covid', 'ecg', 'clinicals']
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # Initialize dictionaries to hold the paths and labels
    client_data_paths = {
        'image': client_rsna_data_path,
        'covid': client_covid_data_path,
        'ecg': client_ecg_data_path,
        'clinicals': client_clinical_data_path
    }

    data_loaders = {
        'image': None,
        'covid': None,
        'ecg': None,
        'clinicals': None
    }

    # Parse data for the specified modalities
    for modality in modalities:
        if modality in client_data_paths:
            data_path = client_data_paths[modality]
            if data_path:
                if modality == 'image' or modality == 'covid':
                    image_paths, image_labels = parse_data(data_path, modality)
                    data_loaders[modality] = DataLoader(ImageDataset(image_paths, image_labels, AutoFeatureExtractor.from_pretrained('facebook/deit-tiny-distilled-patch16-224')), batch_size=32, shuffle=True)
                elif modality == 'ecg':
                    ecg_paths, ecg_labels = parse_data(data_path, modality)
                    data_loaders[modality] = DataLoader(ECGDataset(ecg_paths, ecg_labels), batch_size=32, shuffle=True)
                elif modality == 'clinicals':
                    clinical_paths, clinical_labels = parse_data(data_path, modality)
                    data_loaders[modality] = DataLoader(ClinicalDataset(clinical_paths, clinical_labels), batch_size=32, shuffle=True)

    # Federated training
    global_model = federated_training(
        image_loader=data_loaders['image'],
        covid_loader=data_loaders['covid'],
        ecg_loader=data_loaders['ecg'],
        clinical_loader=data_loaders['clinicals'],
        use_server_data=True,
        epochs=10,
        num_clients=5,
        use_amp=True,
        device = device
    )

    # Test data paths
    
    test_data_paths = {
        'image': test_rsna_data_path,
        'covid': test_covid_data_path,
        'ecg': test_ecg_data_path,
        'clinicals': test_clinical_data_path
    }

    test_data_loaders = {
        'image': None,
        'covid': None,
        'ecg': None,
        'clinicals': None
    }

    # Parse test data for the specified modalities
    for modality in modalities:
        if modality in test_data_paths:
            test_data_path = test_data_paths[modality]
            if test_data_path:
                if modality == 'image' or modality == 'covid':
                    test_image_paths, test_image_labels = parse_data(test_data_path, modality)
                    test_data_loaders[modality] = DataLoader(ImageDataset(test_image_paths, test_image_labels, AutoFeatureExtractor.from_pretrained('facebook/deit-tiny-distilled-patch16-224')), batch_size=32, shuffle=False)
                elif modality == 'ecg':
                    test_ecg_paths, test_ecg_labels = parse_data(test_data_path, modality)
                    test_data_loaders[modality] = DataLoader(ECGDataset(test_ecg_paths, test_ecg_labels), batch_size=32, shuffle=False)
                elif modality == 'clinicals':
                    test_clinical_paths, test_clinical_labels = parse_data(test_data_path, modality)
                    test_data_loaders[modality] = DataLoader(ClinicalDataset(test_clinical_paths, test_clinical_labels), batch_size=32, shuffle=False)



    if test_data_loaders['image']:
        print("Evaluating Lung Opacity Image Model:")
        global_model.set_task('lung_opacity')  # Set the task as needed
        evaluate_image_model(global_model, test_data_loaders['image'], device)

    if test_data_loaders['covid']:
        print("Evaluating COVID Detection Model:")
        global_model.set_task('covid_detection')  # Set the task as needed
        evaluate_image_model(global_model, test_data_loaders['covid'], device)

    if test_data_loaders['ecg']:
        print("Evaluating ECG Model:")
        global_model.set_task('ecg_abnormal')  # Set the task as needed
        evaluate_ecg_model(global_model, test_data_loaders['ecg'], device)

    if test_data_loaders['clinicals']:
        print("Evaluating Clinical Model:")
        global_model.set_task('mortality')  # Set the task as needed
        evaluate_clinical_model(global_model, test_data_loaders['clinicals'], device)