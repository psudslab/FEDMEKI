#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:27:43 2024

@author: xmw5190
"""

import json
import os
import random

seed = 42
random.seed(seed)

def load_data(file_path):
    with open(file_path, 'r') as infile:
        data = json.load(infile)
    return data

def save_data(data, filepath):
    with open(filepath, 'w') as outfile:
        json.dump(data, outfile, indent=4)

def transform_slake_format(data, output_file, image_base_path):
    transformed_data = []

    for item in data:
        if isinstance(item, dict):
            # Extract relevant data from original format
            question = item.get('question', '')
            answer = item.get('answer', 'Unknown')
            img_name = item.get('img_name', '')

            # Create the image path
            image_path = os.path.join(image_base_path, img_name)

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
                "task_type": "The medical Visual Question Answering task involves interpreting complex medical queries based on images and accurately providing answers, often based on specific clinical knowledge and patient case details.",
                "modalities": "Text and image"
            }
            
            transformed_data.append(transformed_item)

    # Write the transformed data to a new JSON file
    with open(output_file, 'w') as outfile:
        json.dump(transformed_data, outfile, indent=4)

def split_few_shot_test(data, num_samples=1000):
    # Shuffle the data
    random.seed(seed)
    random.shuffle(data)

    # Take the first num_samples for few shot test
    few_shot_test_data = data[:num_samples]

    return few_shot_test_data

# Load the data from the JSON file
data_path = path_to_your_slake_test_json
# data_path = "/data/xiaochen/MEDVQA/Slake/Slake1.0/test.json"
data = load_data(data_path)

# Transform the data and create the few-shot test set
data_path = path_to_your_slake_test_json
# image_base_path = "/data/xiaochen/MEDVQA/Slake/Slake1.0/imgs/"
few_shot_test_data = split_few_shot_test(data, num_samples=1000)

# Save the transformed few-shot test data
few_shot_test_json_path = path_to_your_save_json
# few_shot_test_json_path = '/data/xiaochen/FedMFM/preprocessed_jsons/slake_few_shot_test.json'
transform_slake_format(few_shot_test_data, few_shot_test_json_path, image_base_path)

print(f"{len(few_shot_test_data)} samples split from test data and saved to {few_shot_test_json_path}")
