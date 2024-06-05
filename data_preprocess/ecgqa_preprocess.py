#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:27:43 2024

@author: xmw5190
"""

import json
import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split

seed = 42
random.seed(seed)

def load_and_combine_data(folders):
    combined_data = []
    for folder in folders:
        for filename in os.listdir(folder):
            if filename.endswith('.json'):
                filepath = os.path.join(folder, filename)
                with open(filepath, 'r') as infile:
                    data = json.load(infile)
                    combined_data.extend(data)
    return combined_data

def save_data(data, filepath):
    with open(filepath, 'w') as outfile:
        json.dump(data, outfile, indent=4)

def load_ecg_id_to_path_mapping(csv_file):
    df = pd.read_csv(csv_file, usecols=['ecg_id', 'filename_lr'])
    ecg_id_to_path = dict(zip(df['ecg_id'], df['filename_lr']))
    return ecg_id_to_path

def transform_ecgqa_format(data, output_file, ecg_id_to_path):
    transformed_data = []

    for item in data:
        # Only process items where attribute_type is "noise"
        if item.get('attribute_type') == 'noise':
            # Extract relevant data from original format
            question = item.get('question', '')
            answers = item.get('answer', [])
            answer = ', '.join(answers) if answers else "Unknown"
            ecg_ids = item.get('ecg_id', [])

            # Find the modality_path based on ecg_id
            modality_paths = [ecg_id_to_path.get(ecg_id, None) for ecg_id in ecg_ids]
            modality_path = ', '.join([f"/data/xiaochen/data/physionet.org/files/ptb-xl/1.0.3/{path}" for path in modality_paths if path]) if modality_paths else None

            # Format the answer in the expected conversational style
            formatted_answer = f"Answer: {answer}"

            # Create the transformed item based on your expected format
            transformed_item = {
                "modality_path": modality_path,
                "conversations": [
                    {
                        "from": "question",
                        "value": f"<text>\n{question}"
                    },
                    {
                        "from": "answer",
                        "value": formatted_answer
                    }
                ],
                "task_type": "The medical question answering task involves interpreting complex medical queries and accurately providing answers, based on given signals concerning patient's health condition.",
                "modalities": "Text and signal."
            }
            
            transformed_data.append(transformed_item)

    # Write the transformed data to a new JSON file
    with open(output_file, 'w') as outfile:
        json.dump(transformed_data, outfile, indent=4)

# Load the combined data from the original train, valid, and test folders
original_folders = ["/data/xiaochen/data/ecg-qa/ecgqa/ptbxl/template/train/", "/data/xiaochen/data/ecg-qa/ecgqa/ptbxl/template/valid/", "/data/xiaochen/data/ecg-qa/ecgqa/ptbxl/template/test/"]
combined_data = load_and_combine_data(original_folders)

# Filter data to only include items with attribute_type "Noise"
filtered_data = [item for item in combined_data if item.get('attribute_type') == 'noise']

# Load the ecg_id to modality path mapping
csv_file = '/data/xiaochen/data/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv'
ecg_id_to_path = load_ecg_id_to_path_mapping(csv_file)

# Split the combined data into new train, valid, and test sets using train_test_split
train_ratio = 0.80
valid_ratio = 0.10
test_ratio = 0.10

train_data, valid_test_data = train_test_split(filtered_data, test_size=(valid_ratio + test_ratio), random_state=seed)
valid_data, test_data = train_test_split(valid_test_data, test_size=(test_ratio / (valid_ratio + test_ratio)), random_state=seed)

# Save the new splits into respective JSON files
train_json_path = '/data/xiaochen/FedMFM/preprocessed_jsons/ecgqa_client.json'
valid_json_path = '/data/xiaochen/FedMFM/preprocessed_jsons/ecgqa_server.json'
test_json_path = '/data/xiaochen/FedMFM/preprocessed_jsons/ecgqa_test.json'
toy_json_path = '/data/xiaochen/FedMFM/preprocessed_jsons/ecgqa_toy.json'
combined_json_path = '/data/xiaochen/FedMFM/preprocessed_jsons/ecgqa_all.json'

# Transform and save the data
transform_ecgqa_format(train_data, train_json_path, ecg_id_to_path)
transform_ecgqa_format(valid_data, valid_json_path, ecg_id_to_path)
transform_ecgqa_format(test_data, test_json_path, ecg_id_to_path)

# Create a toy dataset (for example, using the first 30 items from train_data)
toy_data = train_data[:30]
transform_ecgqa_format(toy_data, toy_json_path, ecg_id_to_path)

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

def split_client_data(train_json_path, validation_ratio=1/8, seed=42):
    with open(train_json_path, 'r') as file:
        client_data = json.load(file)

    # Ensure client_data is a list
    if not isinstance(client_data, list):
        raise ValueError("Client data must be a list of samples.")

    # Split client data into train and validation sets
    train_data, validation_data = train_test_split(client_data, test_size=validation_ratio, random_state=seed)

    # Save the new train and validation splits
    client_train_json_path = '/data/xiaochen/FedMFM/preprocessed_jsons/ecgqa_client.json'
    client_valid_json_path = '/data/xiaochen/FedMFM/preprocessed_jsons/ecgqa_valid.json'
    
    with open(client_train_json_path, 'w') as train_file:
        json.dump(train_data, train_file, indent=4)
    
    with open(client_valid_json_path, 'w') as valid_file:
        json.dump(validation_data, valid_file, indent=4)
    
    print(f"Client data split into train and validation sets with ratio 1:7 and saved to {client_train_json_path} and {client_valid_json_path}")

# Call the function to split client data
split_client_data(train_json_path)


def split_few_shot_test(test_json_path, few_shot_test_json_path, num_samples=1000, seed=42):
    with open(test_json_path, 'r') as file:
        test_data = json.load(file)

    # Ensure test_data is a list
    if not isinstance(test_data, list):
        raise ValueError("Test data must be a list of samples.")

    # Shuffle the test data
    random.seed(seed)
    random.shuffle(test_data)

    # Take the first num_samples for few shot test
    few_shot_test_data = test_data[:num_samples]

    # Save the few shot test data
    with open(few_shot_test_json_path, 'w') as few_shot_test_file:
        json.dump(few_shot_test_data, few_shot_test_file, indent=4)

    print(f"{num_samples} samples split from test data and saved to {few_shot_test_json_path}")

# Call the function to split few shot test data
few_shot_test_json_path = '/data/xiaochen/FedMFM/preprocessed_jsons/ecgqa_few_shot_test.json'
split_few_shot_test(test_json_path, few_shot_test_json_path)
