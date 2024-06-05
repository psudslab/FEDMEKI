#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 20:55:39 2024

@author: xmw5190
"""



import json
import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split 
seed = 42
random.seed(seed)

def load_ecg_classification_data(csv_file):
    df = pd.read_csv(csv_file, usecols=['scp_codes', 'filename_lr'])
    classification_data = []
    for _, row in df.iterrows():
        scp_codes = eval(row['scp_codes'])  # Convert string representation of dictionary to a dictionary
        filename_lr = row['filename_lr']
        
        # Check if the ECG is normal or abnormal based on 'NORM' field
        if 'NORM' in scp_codes and scp_codes['NORM'] > 50:
            label = "No"
        else:
            label = "Yes"
        
        classification_data.append({
            "filename_lr": filename_lr,
            "label": label
        })
    
    return classification_data

def split_data(data, train_ratio, valid_ratio, test_ratio, seed=42):
    random.seed(seed)
    random.shuffle(data)
    
    total = len(data)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)
    
    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data = data[valid_end:]
    
    return train_data, valid_data, test_data

def save_data(data, filepath):
    with open(filepath, 'w') as outfile:
        json.dump(data, outfile, indent=4)

def transform_ecg_classification_format(data, output_file):
    transformed_data = []

    for item in data:
        filename_lr = item['filename_lr']
        label = item['label']

        # Format the question and answer in the expected conversational style
        question = "Is the given ECG abnormal?"
        formatted_answer = f"Answer: {label}"

        # Create the transformed item based on your expected format
        transformed_item = {
            "modality_path": f"/data/xiaochen/data/physionet.org/files/ptb-xl/1.0.3/{filename_lr}",
            "conversations": [
                {
                    "from": "question",
                    "value": f"<ecg>\n{question}"
                },
                {
                    "from": "answer",
                    "value": formatted_answer
                }
            ],
            "task_type": "The ECG abnormal detection task involves analyzing ECG signals to determine whether they are normal or abnormal, based on specific clinical criteria.",
            "modalities": "ECG only"
        }
        
        transformed_data.append(transformed_item)

    # Write the transformed data to a new JSON file
    with open(output_file, 'w') as outfile:
        json.dump(transformed_data, outfile, indent=4)

# Load the ECG classification data
csv_file = '/data/xiaochen/data/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv'
classification_data = load_ecg_classification_data(csv_file)

# Split the combined data into new train, valid, and test sets
train_ratio = 0.80
valid_ratio = 0.10
test_ratio = 0.10

train_data, valid_data, test_data = split_data(classification_data, train_ratio, valid_ratio, test_ratio)

# Save the new splits into respective JSON files
train_json_path = '/data/xiaochen/FedMFM/preprocessed_jsons/ecg_client.json'
valid_json_path = '/data/xiaochen/FedMFM/preprocessed_jsons/ecg_server.json'
test_json_path = '/data/xiaochen/FedMFM/preprocessed_jsons/ecg_test.json'
combined_json_path = '/data/xiaochen/FedMFM/preprocessed_jsons/ecg_all.json'
transform_ecg_classification_format(train_data, train_json_path)
transform_ecg_classification_format(valid_data, valid_json_path)
transform_ecg_classification_format(test_data, test_json_path)

print("Data split and saved with new ratios: 80:10:10")



 # Load JSON files
with open(train_json_path, 'r') as train_file:
   train_data = json.load(train_file)

with open(valid_json_path, 'r') as valid_file:
   valid_data = json.load(valid_file)

with open(test_json_path, 'r') as test_file:
   test_data = json.load(test_file)

# Assuming all JSON files contain lists
if isinstance(train_data, list) and isinstance(valid_data, list) and isinstance(test_data, list):
   combined_data = train_data + valid_data
else:
   # If the JSON files contain dictionaries, merge them appropriately
   combined_data = {**train_data, **valid_data}

# Save combined JSON
with open(combined_json_path, 'w') as combined_file:
   json.dump(combined_data, combined_file, indent=4)

print(f"Combined JSON saved to {combined_json_path}")

# New section: Split client data into train and validation sets
def split_client_data(train_json_path, validation_ratio=1/8, seed=42):
   with open(train_json_path, 'r') as file:
       client_data = json.load(file)

   # Ensure client_data is a list
   if not isinstance(client_data, list):
       raise ValueError("Client data must be a list of samples.")

   # Split client data into train and validation sets
   train_data, validation_data = train_test_split(client_data, test_size=validation_ratio, random_state=seed)

   # Save the new train and validation splits
   client_train_json_path = '/data/xiaochen/FedMFM/preprocessed_jsons/ecg_client.json'
   client_valid_json_path = '/data/xiaochen/FedMFM/preprocessed_jsons/ecg_valid.json'
  
   with open(client_train_json_path, 'w') as train_file:
       json.dump(train_data, train_file, indent=4)
  
   with open(client_valid_json_path, 'w') as valid_file:
       json.dump(validation_data, valid_file, indent=4)
  
   print(f"Client data split into train and validation sets with ratio 1:7 and saved to {client_train_json_path} and {client_valid_json_path}")

# Call the function to split client data
split_client_data(train_json_path)
