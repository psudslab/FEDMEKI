#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 17:34:59 2024

@author: xmw5190
"""

import torch
from torch.utils.data import DataLoader,random_split
import torch.optim as optim
from transformers import BartForConditionalGeneration, AdamW
from torcheval.metrics.functional import bleu_score
# from dataset import MedQADataset  # Ensure this is properly imported


import json
from torch.utils.data import Dataset
from transformers import BartTokenizer
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from datasets import load_metric
from transformers import BartModel, BartTokenizer, BartConfig

class BARTWithExternalEmbedding(nn.Module):
    def __init__(self, pretrained_bart_model='facebook/bart-base', external_embedding_layer=None, vocab_size=None):
        super(BARTWithExternalEmbedding, self).__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(pretrained_bart_model,forced_bos_token_id=0)
        
        # Optional: Use external embedding layer if provided
        if external_embedding_layer:
            self.embedding = external_embedding_layer
        else:
            self.embedding = self.bart.get_input_embeddings()

        # No need for a separate lm_head, BartForConditionalGeneration includes it
        # Ensure vocab_size matches if using external embeddings or modifying output layer

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None):
        if decoder_input_ids is None:
            decoder_input_ids = self._shift_right(input_ids)

        inputs_embeds = self.embedding(input_ids)
        outputs = self.bart(inputs_embeds=inputs_embeds, 
                            attention_mask=attention_mask, 
                            decoder_input_ids=decoder_input_ids)
        
        return outputs.logits

    def generate(self, input_ids, **generator_args):
        # This method is now valid as we're using BartForConditionalGeneration
        return self.bart.generate(input_ids, **generator_args)

    def _shift_right(self, input_ids):
        shifted = torch.zeros_like(input_ids)
        shifted[:, 1:] = input_ids[:, :-1]
        shifted[:, 0] = self.bart.config.bos_token_id
        return shifted

        
class MedQADataset(Dataset):
    """ Dataset for loading and processing MedQA data for QA with seq2seq models."""
    def __init__(self, json_file, tokenizer_name='facebook/bart-large', max_length=512):
        """
        Args:
            json_file (string): Path to the json file with MedQA data.
            tokenizer_name (string): Identifier for the tokenizer.
            max_length (int): Maximum sequence length of the input sequences.
        """
        with open(json_file, 'r') as file:
            self.data = json.load(file)
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['conversations'][0]['value']
        answer = item['conversations'][1]['value']
        # print('q:', question)
        # print('a:', answer)
        # Tokenize the question
        question_encoding = self.tokenizer(question, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        
        # Tokenize the answer
        answer_encoding = self.tokenizer(answer, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")

        return {
            'input_ids': question_encoding['input_ids'].squeeze(0),
            'attention_mask': question_encoding['attention_mask'].squeeze(0),
            'labels': answer_encoding['input_ids'].squeeze(0)
        }


def evaluate_model(model, dataloader, device):
    model.eval()
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    all_references = []
    all_candidates = []

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)

        # Generate text using the model
        with torch.no_grad():
            generated_ids = model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2)

        # Decode generated IDs to text
        candidates = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        all_candidates.extend(candidates)  # Flatten the list of candidates

        # Assuming 'labels' in batch are token IDs for the ground truth/reference sentences
        reference_ids = batch['labels']
        references_batch = [tokenizer.decode(ref_ids, skip_special_tokens=True) for ref_ids in reference_ids]
        all_references.extend([[ref] for ref in references_batch])  # List of lists for each reference
        # print(input_ids)

    print(all_candidates[-1])
    print(all_references[-1])
    
    # Calculate BLEU score, ensuring each candidate has corresponding references
    bleu_score_value = bleu_score(all_candidates, all_references, n_gram=2)
    print(f"BLEU score: {bleu_score_value}")

    return bleu_score_value

class Client:
    def __init__(self, model, train_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self):
        self.model.train()
        total_loss = 0
        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
    
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)#.float()
    
            # Ensure correct shapes for outputs and labels
            # print("Outputs Shape:", outputs.shape)  # Expected: [batch_size, seq_length, num_classes]
            # print("Labels Shape:", labels.shape)  # Expected: [batch_size, seq_length]
    

            # # Reshape outputs and labels correctly
            outputs = outputs.view(-1, outputs.size(-1))  # Flatten outputs to [batch_size * seq_length, num_classes]
            labels = labels.view(-1)  # Flatten labels to [batch_size * seq_length]
    
            # print("Outputs Shape:", outputs.shape)  # Expected: [batch_size, seq_length, num_classes]
            # print("Labels Shape:", labels.shape)  # Expected: [batch_size, seq_length]
            # Calculate loss
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
    
        return total_loss / len(self.train_loader)



def average_weights(state_dicts):
    """Averages all the weights from state_dicts and returns a single state_dict."""
    new_state_dict = state_dicts[0].copy()
    for key in new_state_dict.keys():
        new_state_dict[key] = torch.mean(torch.stack([state_dict[key].float() for state_dict in state_dicts]), 0)
    return new_state_dict

def federated_training_with_centralized_data(client_dataset, server_dataset, test_dataset, model_fn, device, num_clients=1, epochs=10):
    # Prepare client and server dataloaders
    client_datasets = random_split(client_dataset, [len(client_dataset) // num_clients + (1 if x < len(client_dataset) % num_clients else 0) for x in range(num_clients)])
    client_loaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in client_datasets]
    server_loader = DataLoader(server_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize clients with the global model
    global_model = model_fn().to(device)
    clients = [Client(model=global_model, train_loader=loader, criterion=torch.nn.CrossEntropyLoss(ignore_index=1), optimizer=optim.Adam(global_model.parameters(), lr=1e-6), device=device) for loader in client_loaders]

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")

        # Train each client
        state_dicts = []
        for client in clients:
            loss = client.train()
            print(f"Client {clients.index(client) + 1} Loss: {loss}")
            state_dicts.append(client.model.state_dict())

        # Aggregate models
        global_state_dict = average_weights(state_dicts)
        global_model.load_state_dict(global_state_dict)

        # Fine-tune on centralized server data
        server_model = Client(model=global_model, train_loader=server_loader, criterion=torch.nn.CrossEntropyLoss(ignore_index=1), optimizer=optim.Adam(global_model.parameters(), lr=1e-6), device=device)
        server_loss = server_model.train()
        print(f"Fine-tuning Loss on Server Data: {server_loss}")

        # Evaluate model
        bleu_score = evaluate_model(server_model.model, test_loader, device)
        print(f"BLEU score on test data: {bleu_score}")

        # Update all client models with the fine-tuned global model
        for client in clients:
            client.model.load_state_dict(server_model.model.state_dict())

from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F

# Assuming MedQADataset is defined and imported correctly
dataset = MedQADataset(json_file="/data/xiaochen/FedMFM/preprocessed_jsons/medqa_server.json")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the model

external_embeddings = BartModel.from_pretrained('facebook/bart-base', forced_bos_token_id=0).get_input_embeddings()
model = BARTWithExternalEmbedding(external_embedding_layer=external_embeddings)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Optimizer
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)


# Training loop
# Example: Assume datasets and model_fn are defined
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_fn = BARTWithExternalEmbedding  # Adjust based on your actual model function

# Load and split datasets here or within the training functions as needed
# client_dataset = MedQADataset(json_file="/data/xiaochen/FedMFM/preprocessed_jsons/medqa_client.json", tokenizer_name='facebook/bart-base')
client_dataset = MedQADataset(json_file="/data/xiaochen/FedMFM/preprocessed_jsons/medqa_server.json", tokenizer_name='facebook/bart-base')
server_dataset = MedQADataset(json_file="/data/xiaochen/FedMFM/preprocessed_jsons/medqa_server.json", tokenizer_name='facebook/bart-base')
test_dataset = MedQADataset(json_file="/data/xiaochen/FedMFM/preprocessed_jsons/medqa_server.json", tokenizer_name='facebook/bart-base')
# test_dataset = MedQADataset(json_file="/data/xiaochen/FedMFM/preprocessed_jsons/medqa_test.json", tokenizer_name='facebook/bart-base')

# Perform training
federated_training_with_centralized_data(client_dataset, server_dataset, test_dataset, model_fn, device)
