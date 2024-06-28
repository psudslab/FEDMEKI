#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 23:01:50 2024

@author: xmw5190
"""

import csv
import json
import random
import os
import uuid
import numpy as np
import torch
from sklearn.model_selection import train_test_split 

seed = 42
random.seed(seed)
np.random.seed(seed)


# Function to generate responses based on classification
def determine_gpt_response(classification):
   if classification == 'Lung Opacity':
       return 'Yes'
   elif classification == 'Normal':
       return 'No'
   else:
       return 'Unavailable'

# Paths to your files and directories

csv_file_path = path_to_stage_2_detailed_class_info
train_image_base_path = path_to_stage_2_train_images
# csv_file_path = "/data/junyu/RSNA/stage_2_detailed_class_info.csv"
# train_image_base_path = '/data/junyu/RSNA/stage_2_train_images/'

# Output JSON file paths
json_train_main = path_to_client_output
json_train_secondary = path_to_server_output
json_test = path_to_test_output
# json_train_main = "/data/xiaochen/FedMFM/preprocessed_jsons/RSNA_client.json"
# json_train_secondary = "/data/xiaochen/FedMFM/preprocessed_jsons/RSNA_server.json"
# json_test = "/data/xiaochen/FedMFM/preprocessed_jsons/RSNA_test.json"

# Read CSV and process data
patient_data = []
with open(csv_file_path, mode='r', newline='') as csvfile:
   csvreader = csv.DictReader(csvfile)
   for row in csvreader:
       image_path = os.path.join(train_image_base_path, f"{row['patientId']}.dcm")
       if os.path.exists(image_path):
           patient_data.append({
               'image': image_path,
               'classification': row['class']
           })

# Shuffle and split the data
random.shuffle(patient_data)  # Shuffle to randomize the distribution
total_images = len(patient_data)
first_split_index = int(total_images * 0.8)  # 90% for training
second_split_index = int(total_images * 0.9)  # Further split the 90% into 90%/10%

# Split the datasets
train_data = patient_data[:first_split_index]
secondary_train_data = patient_data[first_split_index:second_split_index]
test_data = patient_data[second_split_index:]

# Function to save JSON with detailed data format
def save_to_json(data, file_path, is_test=False):
   output_data = []
   for entry in data:
       image_path = entry['image']
       classification = entry['classification']
       if entry['classification'] != "No Lung Opacity / Not Normal":
           if is_test:
               output_data.append({
                   "label": classification,
                   "question": f"Is there any lung opacity in this CT image?",
                   "id": str(uuid.uuid4()),
                   "modality_path": image_path,
                   "src": 'RSNA',
                   'task_type': 'lung opacity classification',
                   'modalities': "image only"
               })
           else:
               gpt_response = determine_gpt_response(classification)
               output_data.append({
                   'modality_path': image_path,
                   'conversations': [
                       {'from': 'question', 'value': "<image>\nAssess this CT image: should it be classified as lung opacity?"},
                       {'from': 'answer', 'value': gpt_response}
                   ],
                   'task_type': 'Lung opacity classification involves using medical imaging techniques to distinguish between normal lung appearances and areas of increased density, which are indicative of various pulmonary conditions.',
                   'src': 'RSNA',
                   'modalities': "Image only"
               })
   with open(file_path, 'w', encoding='utf-8') as f:
       json.dump(output_data, f, indent=4)

# Generate JSON files for each dataset
save_to_json(train_data, json_train_main)
save_to_json(secondary_train_data, json_train_secondary)
save_to_json(test_data, json_test)

print(f"Data split into:\n- Main training set: {len(train_data)} images\n- Secondary training set: {len(secondary_train_data)} images\n- Test set: {len(test_data)} images")
def split_client_data(train_json_path, validation_ratio=1/8, seed=42):
    with open(train_json_path, 'r') as file:
        client_data = json.load(file)

    # Ensure client_data is a list
    if not isinstance(client_data, list):
        raise ValueError("Client data must be a list of samples.")

    # Split client data into train and validation sets
    train_data, validation_data = train_test_split(client_data, test_size=validation_ratio, random_state=seed)

    # Save the new train and validation splits
    client_train_json_path = path_to_client_output
    client_valid_json_path = path_to_valid_output
    # client_train_json_path = '/data/xiaochen/FedMFM/preprocessed_jsons/RSNA_client.json'
    # client_valid_json_path = '/data/xiaochen/FedMFM/preprocessed_jsons/RSNA_valid.json'
    
    with open(client_train_json_path, 'w') as train_file:
        json.dump(train_data, train_file, indent=4)
    
    with open(client_valid_json_path, 'w') as valid_file:
        json.dump(validation_data, valid_file, indent=4)
    

    
    print(f"Client data split into train and validation sets with ratio 1:7 and saved to {client_train_json_path} and {client_valid_json_path}")

# Call the function to split client data
split_client_data(json_train_main)

