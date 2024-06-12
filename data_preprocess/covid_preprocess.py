import os
import json
import random

# Paths to the directories
positive_dir = "/data/junyu/covid2/COVID-19_Radiography_Dataset/COVID/images/"
negative_dir = "/data/junyu/covid2/COVID-19_Radiography_Dataset/Normal/images/"

# Output directories
output_dir_json = "/data/xiaochen/FedMFM/preprocessed_jsons"
os.makedirs(output_dir_json, exist_ok=True)

# Read the data from the directories
def read_data_from_directory(directory, label):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            data.append((img_path, label))
    return data

# Read the positive and negative data
positive_data = read_data_from_directory(positive_dir, 'positive')
negative_data = read_data_from_directory(negative_dir, 'negative')

# Aggregate all data together
all_data = positive_data + negative_data
print(len(all_data))
# Shuffle the data
random.seed(42)
random.shuffle(all_data)

# Split the data
total = len(all_data)
client_end = int(total * 0.7)
valid_end = client_end + int(total * 0.1)
server_end = valid_end + int(total * 0.1)

client_data = all_data[:client_end]
valid_data = all_data[client_end:valid_end]
server_data = all_data[valid_end:server_end]
test_data = all_data[server_end:]

# Create the transformed data format
def create_transformed_data(data, question, task_type):
    transformed_data = []
    for img_path, label in data:
        formatted_answer = f"Answer: {'Yes' if label == 'positive' else 'No'}"
        transformed_item = {
            "modality_path": img_path,
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
            "task_type": task_type,
            "modalities": "Image only"
        }
        transformed_data.append(transformed_item)
    return transformed_data

# Define the question and task type
question = "Based on this image, is the patient COVID-19 positive?"
task_type = "The COVID-19 prediction task involves analyzing chest images to determine whether a patient is positive for COVID-19."

# Transform the data
client_transformed = create_transformed_data(client_data, question, task_type)
valid_transformed = create_transformed_data(valid_data, question, task_type)
server_transformed = create_transformed_data(server_data, question, task_type)
test_transformed = create_transformed_data(test_data, question, task_type)
print(len(client_transformed))

# Save the transformed data to JSON files
def save_to_json(data, filepath):
    with open(filepath, 'w') as outfile:
        json.dump(data, outfile, indent=4)

#save_to_json(client_transformed, os.path.join(output_dir_json, 'covid_client.json'))
#save_to_json(valid_transformed, os.path.join(output_dir_json, 'covid_valid.json'))
#save_to_json(server_transformed, os.path.join(output_dir_json, 'covid_server.json'))
#save_to_json(test_transformed, os.path.join(output_dir_json, 'covid_test.json'))
#
## Create a toy dataset with 1% of the data
#toy_data_size = max(1, int(total * 0.01))  # Ensure at least one item is in the toy dataset
#toy_data = random.sample(all_data, toy_data_size)
#toy_transformed = create_transformed_data(toy_data, question, task_type)
#save_to_json(toy_transformed, os.path.join(output_dir_json, 'covid_toy.json'))
#
#print("Data split and saved with specified ratios.")
