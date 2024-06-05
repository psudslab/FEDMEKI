#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 16:27:43 2024

@author: xmw5190
"""

import json
import random
from sklearn.model_selection import train_test_split 

seed = 42
random.seed(seed)

def transform_medqa_format(input_file, output_file):
    transformed_data = []

    # Open the input file and process each line as an individual JSON object
    with open(input_file, 'r') as infile:
        # c = 0
        for line in infile:
            item = json.loads(line)  # Load each line as a separate JSON object

            # Extract relevant data from original format
            question = item.get('question', '')
            answer = item.get('answer', '')
            answer_idx = item.get('answer_idx', '')
            options = item.get('options', {})

            # Format the question to include options
            options_text = '\n'.join([f"{key}: {value}" for key, value in options.items()])
            formatted_question = f"{question}\nOptions:\n{options_text}"

            # Format the answer in the expected conversational style
            formatted_answer = f"Answer: {answer}" if answer_idx and answer in options.values() else "Classification: Unknown"

            # Create the transformed item based on your expected format
            transformed_item = {
                "modality_path": None,
                "conversations": [
                    {
                        "from": "question",
                        "value": f"<text>\n{formatted_question}"
                    },
                    {
                        "from": "answer",
                        "value": formatted_answer
                    }
                ],
                "task_type": "The medical question answering task involves interpreting complex medical queries and accurately providing answers, often based on specific clinical knowledge and patient case details.",
                "modalities": "Text only"
            }
            
            transformed_data.append(transformed_item)
            # if c==30:
            #     break
    # Write the transformed data to a new JSON file
    with open(output_file, 'w') as outfile:
        json.dump(transformed_data, outfile, indent=4)


# Example usage
transform_medqa_format("/data/xiaochen/data/MedQA/data_clean/questions/US/train.jsonl", '/data/xiaochen/FedMFM/preprocessed_jsons/medqa_train.json')
transform_medqa_format("/data/xiaochen/data/MedQA/data_clean/questions/US/dev.jsonl", '/data/xiaochen/FedMFM/preprocessed_jsons/medqa_server.json')
transform_medqa_format("/data/xiaochen/data/MedQA/data_clean/questions/US/test.jsonl", '/data/xiaochen/FedMFM/preprocessed_jsons/medqa_test.json')

client_json_path = '/data/xiaochen/FedMFM/preprocessed_jsons/medqa_client.json'
server_json_path = '/data/xiaochen/FedMFM/preprocessed_jsons/medqa_server.json'
combined_json_path = '/data/xiaochen/FedMFM/preprocessed_jsons/medqa_all.json'

# Load JSON files
with open(client_json_path, 'r') as client_file:
    client_data = json.load(client_file)

with open(server_json_path, 'r') as server_file:
    server_data = json.load(server_file)

# Assuming both JSON files contain lists
if isinstance(client_data, list) and isinstance(server_data, list):
    combined_data = client_data + server_data
else:
    # If the JSON files contain dictionaries, merge them appropriately
    combined_data = {**client_data, **server_data}

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
    client_train_json_path = '/data/xiaochen/FedMFM/preprocessed_jsons/medqa_client.json'
    client_valid_json_path = '/data/xiaochen/FedMFM/preprocessed_jsons/medqa_valid.json'
    
    with open(client_train_json_path, 'w') as train_file:
        json.dump(train_data, train_file, indent=4)
    
    with open(client_valid_json_path, 'w') as valid_file:
        json.dump(validation_data, valid_file, indent=4)
    
    print(f"Client data split into train and validation sets with ratio 1:7 and saved to {client_train_json_path} and {client_valid_json_path}")

# Call the function to split client data
split_client_data('/data/xiaochen/FedMFM/preprocessed_jsons/medqa_client.json')
