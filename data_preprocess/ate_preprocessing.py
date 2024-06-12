#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:27:43 2024

@author: xmw5190
"""

import pandas as pd
import json
import os

# Load the CSV file into a pandas dataframe
input_csv_path = '/data/junyu/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 1 (validate & csv)/valid.csv'
df = pd.read_csv(input_csv_path)

def transform_chexpert_format(df, output_file, image_base_prefix):
    transformed_data = []

    for index, row in df.iterrows():
        # Modify the image path based on the specified prefix
        image_path = row['Path'].replace('CheXpert-v1.0/valid/', image_base_prefix)
        question = "Does this image show evidence of atelectasis?"
        answer = "Yes" if row['Atelectasis'] == 1 else "No"

        # Format the answer in the expected conversational style
        formatted_answer = f"Answer: {answer}"

        # Create the transformed item based on your expected format
        transformed_item = {
            "modality_path": image_path,
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
            "task_type": "The Atelectasis detection task involves interpreting chest X-ray images to accurately determine the presence or absence of atelectasis.",
            "modalities": "Image only"
        }
        
        transformed_data.append(transformed_item)

    # Write the transformed data to a new JSON file
    with open(output_file, 'w') as outfile:
        json.dump(transformed_data, outfile, indent=4)

# Base path prefix for images
image_base_prefix = '/data/junyu/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 1 (validate & csv)/valid/'

# Output JSON path
output_json_path = '/data/xiaochen/FedMFM/preprocessed_jsons/ate_test.json'

# Transform and save the data
transform_chexpert_format(df, output_json_path, image_base_prefix)

print(f"Transformed data saved to {output_json_path}")
