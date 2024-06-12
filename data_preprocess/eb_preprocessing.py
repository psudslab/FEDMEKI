#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 20:55:39 2024

@author: xmw5190
"""

import json
import random
import pandas as pd
from sklearn.model_selection import train_test_split

seed = 42
random.seed(seed)

def load_ecg_classification_data(csv_file):
    df = pd.read_csv(csv_file, usecols=['scp_codes', 'filename_lr', 'extra_beats'])
    classification_data = []
    for _, row in df.iterrows():
        filename_lr = row['filename_lr']
        extra_beats = row['extra_beats']
        
        # Determine label based on presence of 'es' in extra_beats column
        if pd.isna(extra_beats) or 'es' not in str(extra_beats).lower():
            label = "No"
        else:
            label = "Yes"
        
        classification_data.append({
            "filename_lr": filename_lr,
            "label": label,
            "extra_beats": extra_beats
        })
    
    return classification_data

def filter_data_for_es(classification_data):
    positive_data = [item for item in classification_data if item['label'] == "Yes"]
    negative_data = [item for item in classification_data if item['label'] == "No"]
    return positive_data[:1000], negative_data[:1000]

def save_data(data, filepath):
    with open(filepath, 'w') as outfile:
        json.dump(data, outfile, indent=4)

def transform_ecg_classification_format(data, output_file, task_type, description):
    transformed_data = []

    for item in data:
        filename_lr = item['filename_lr']
        label = item['label']

        # Format the question and answer in the expected conversational style
        question = "Does the given ECG contain ectopic beats?"
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
            "task_type": task_type,
            "modalities": "ECG only",
            "description": description
        }
        
        transformed_data.append(transformed_item)

    # Write the transformed data to a new JSON file
    with open(output_file, 'w') as outfile:
        json.dump(transformed_data, outfile, indent=4)

# Load the ECG classification data
csv_file = '/data/xiaochen/data/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv'
classification_data = load_ecg_classification_data(csv_file)

# Filter the data for ES-related and non-ES data
positive_data, negative_data = filter_data_for_es(classification_data)

# Combine positive and negative data
combined_data = positive_data + negative_data

# Define the task type and description
task_type = "Ectopic Beats Detection"
description = "The Ectopic Beats Detection task involves analyzing ECG signals to determine whether ectopic beats are present, based on specific clinical criteria."

# Save the combined data to a single JSON file
output_file = '/data/xiaochen/FedMFM/preprocessed_jsons/eb_zero_shot.json'
transform_ecg_classification_format(combined_data, output_file, task_type, description)

print(f"Data saved to {output_file} with 1,000 positive and 1,000 negative samples.")
