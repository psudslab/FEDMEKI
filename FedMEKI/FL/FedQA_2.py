#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 22:06:44 2024

@author: xmw5190
"""

import torch
import torch.nn as nn
from transformers import BartModel, BartConfig, BartTokenizer
from torch.utils.data import Dataset, DataLoader
import json
from torch.utils.data import random_split
import torch.optim as optim

import json
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from tqdm import tqdm
import json
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torcheval.metrics.functional import bleu_score
import math
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from copy import deepcopy
from torch.cuda.amp import GradScaler, autocast
torch.autograd.set_detect_anomaly(True)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


class TransformerQA(nn.Module):
    def __init__(self, external_embedding_layer, d_model=4096, nhead=8, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=2048, max_seq_length=512):
        super(TransformerQA, self).__init__()
        self.d_model = d_model
        self.embedding = external_embedding_layer
        self.vocab_size = external_embedding_layer.num_embeddings
        self.positional_encoding = self.init_positional_encoding(max_seq_length, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, batch_first=True)
        self.output_layer = nn.Linear(d_model, self.vocab_size)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_model)

    def init_positional_encoding(self, max_seq_length, d_model):
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, output_logits=True):
        device = next(self.parameters()).device
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_masks(src, tgt, 2, device)
        # print(src[0])
        # print(src_padding_mask[0])

        batch_size = src.size(0)
        seq_length = src.size(1)
        expanded_positional_encoding = self.positional_encoding[:seq_length, :].expand(batch_size, seq_length, self.d_model)
        src = self.embedding(src) 
        src = src + expanded_positional_encoding
        tgt = self.embedding(tgt) 
        tgt = tgt + expanded_positional_encoding
        
        
        # print("src:",src[0])
 
        
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask)
        # output = self.layer_norm(output)
        # print("opt:",output[0])
        # output = self.output_layer(self.relu(output))
        output = self.output_layer(output)
        # output = torch.clamp(output, min=1e-6)
        # output[:,:,1] *= 0
        # print("opt:",output[0])
        if output_logits:
            return output
        return torch.argmax(output, dim=-1)
    # def generate_square_subsequent_mask(self, sz):
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask
    def generate_square_subsequent_mask(self,sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        # Use half precision by converting the float mask to float16
        # mask = mask.half()  # Converts to half precision
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))#.half()
        return mask
    def create_masks(self, src, tgt, pad_token_id, device):
        # Adjust these mask shapes if necessary for batch first
        tgt_seq_len = tgt.size(1)
        src_seq_len = src.size(1)

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(device)#.half()
        # src_mask = self.generate_square_subsequent_mask(src_seq_len).to(device)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=device)#.half()

        src_padding_mask = (src == pad_token_id).to(device)
        tgt_padding_mask = (tgt == pad_token_id).to(device)

        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
    def decode_sequence(model, input_ids, max_length=50):
        model.eval()
        with torch.no_grad():
            for _ in range(max_length):
                outputs = model(input_ids)
                # Assuming the last dimension of outputs is the logits
                logits = outputs[:, -1, :]  # Get logits for the last token in the sequence
                next_token_id = torch.argmax(logits, dim=-1).unsqueeze(-1)  # Greedy decoding: choose highest probability token
                input_ids = torch.cat([input_ids, 2], dim=1)  # Append to the input sequence
                if next_token_id.item() == eos_token_id:  # Assuming eos_token_id is defined
                    break
        return input_ids
def calculate_filtered_bleu(all_candidates, all_references):
    # Filter out pairs where the candidate is empty
    filtered_candidates = []
    filtered_references = []
    
    for candidate, reference in zip(all_candidates, all_references):
        if len(candidate.strip()) > 0:  # Ensure candidate is not empty
            filtered_candidates.append(candidate)
            filtered_references.append(reference)


    # Calculate BLEU score only for non-empty candidates
    if len(filtered_candidates) > 0:
        return bleu_score(filtered_candidates, filtered_references, n_gram=1)
    else:
        print("No valid candidates to evaluate.")
        return 0.0
def evaluate_model(model, dataloader, device):
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("Henrychur/MMedLM2", trust_remote_code=True)
    all_references = []
    all_candidates = []

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            generated_ids = model(input_ids, labels, output_logits=False)

        candidates = [i.split('</s>')[0].replace("<s>", '').replace("<s>", '') for i in tokenizer.batch_decode(generated_ids, skip_special_tokens=False)]
        all_candidates.extend(candidates)

        reference_ids = batch['labels']
        references_batch = [tokenizer.decode(ref_ids, skip_special_tokens=True) for ref_ids in reference_ids]
        all_references.extend([[ref] for ref in references_batch])

    print(all_candidates[:50])
    print(all_references[:50])
    bleu_score_value = calculate_filtered_bleu(all_candidates, all_references)
    model.cpu()

    return bleu_score_value
class MedQADataset(Dataset):
    """ Dataset for loading and processing MedQA data for QA with seq2seq models."""
    def __init__(self, json_file, tokenizer_name="Henrychur/MMedLM2", max_length=512):
        """
        Args:
            json_file (string): Path to the json file with MedQA data.
            tokenizer_name (string): Identifier for the tokenizer.
            max_length (int): Maximum sequence length of the input sequences.
        """
        with open(json_file, 'r') as file:
            self.data = json.load(file)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
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
        answer_encoding = self.tokenizer(answer.split('Answer:')[-1], truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        return {
            'input_ids': question_encoding['input_ids'].squeeze(0),
            'attention_mask': question_encoding['attention_mask'].squeeze(0),
            'labels': answer_encoding['input_ids'].squeeze(0)
        }
class Client:
    def __init__(self, model, train_loader, criterion, device):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optim.Adam(model.parameters(), lr=1e-5)
        self.device = device

    def train(self):
        self.model.to(self.device)
        self.model.train()
        scaler = GradScaler()
        total_loss = 0
        for batch in tqdm(self.train_loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
    
            self.optimizer.zero_grad()
            with autocast():
                outputs = self.model(input_ids, labels)#.float()
    
                outputs = outputs.view(-1, outputs.size(-1))  # Flatten outputs to [batch_size * seq_length, num_classes]
                labels = labels.view(-1)  # Flatten labels to [batch_size * seq_length]
        
                loss = self.compute_loss(outputs, labels, 2)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.01)
            # self.optimizer.step()
            total_loss += loss.item()
        self.model.cpu()
        return total_loss / len(self.train_loader)
    def compute_loss(self, logits, targets, padding_index):

    # Create a mask by comparing all target tokens to the padding index
        mask = (targets != padding_index)
        # Flatten the logits and targets to fit into F.cross_entropy
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1)
        
        # Use the mask to select non-padding elements in logits and targets
        logits_non_pad = logits_flat[mask_flat]
        targets_non_pad = targets_flat[mask_flat]
        
        # Calculate the loss only on non-padding tokens
        loss = F.cross_entropy(logits_non_pad, targets_non_pad)
        
        return loss


def average_weights(state_dicts):
    """Averages all the weights from state_dicts and returns a single state_dict."""
    new_state_dict = state_dicts[0].copy()
    for key in new_state_dict.keys():
        new_state_dict[key] = torch.mean(torch.stack([state_dict[key].float() for state_dict in state_dicts]), 0)
    return new_state_dict

def federated_training_with_centralized_data(client_dataset, server_dataset, test_dataset, model_fn, device, num_clients=5, epochs=10):
    # Prepare client and server dataloaders
    client_datasets = random_split(client_dataset, [len(client_dataset) // num_clients + (1 if x < len(client_dataset) % num_clients else 0) for x in range(num_clients)])
    # client_loaders = [DataLoader(ds, batch_size=8, shuffle=True) for ds in client_datasets]
    # server_loader = DataLoader(server_dataset, batch_size=8, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    client_loaders = [DataLoader(ds, batch_size=8, shuffle=True, pin_memory=True, num_workers=4) for ds in client_datasets]
    server_loader = DataLoader(server_dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=4)

    
    # Initialize clients with the global model
    # external_embeddings = BartModel.from_pretrained('facebook/bart-base', forced_bos_token_id=0).get_input_embeddings()
    external_embeddings = torch.load("/data/xiaochen/FedMFM/MMedLM2/embedding_layer")
    global_model = model_fn(external_embeddings).float()
    clients = [Client(model = deepcopy(global_model), train_loader=loader, criterion=torch.nn.CrossEntropyLoss(ignore_index = 2), device=device) for loader in client_loaders]
    server_model = Client(model = deepcopy(global_model), train_loader=server_loader, criterion=torch.nn.CrossEntropyLoss(ignore_index = 2), device=device)
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
        server_model.model.load_state_dict(global_state_dict)

        # Fine-tune on centralized server data
        
        # server_loss = server_model.train()
        # print(f"Fine-tuning Loss on Server Data: {server_loss}")

        # Evaluate model
        bleu_score = evaluate_model(server_model.model, test_loader, device)
        print(f"BLEU score on test data: {bleu_score}")

        # Update all client models with the fine-tuned global model
        for client in clients:
            client.model.load_state_dict(server_model.model.state_dict())
        torch.cuda.empty_cache()
    return server_model
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F

# Assuming MedQADataset is defined and imported correctly
dataset = MedQADataset(json_file="/data/xiaochen/FedMFM/preprocessed_jsons/medqa_server.json")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the model

# external_embeddings = BartModel.from_pretrained('facebook/bart-base', forced_bos_token_id=0).get_input_embeddings()
# external_embeddings = torch.load("/data/xiaochen/FedMFM/MMedLM2/embedding_layer")
# model = TransformerQA(external_embedding_layer=external_embeddings)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# Optimizer
# optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)


# Training loop
# Example: Assume datasets and model_fn are defined
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_fn = TransformerQA  # Adjust based on your actual model function

# Load and split datasets here or within the training functions as needed
client_dataset = MedQADataset(json_file="/data/xiaochen/FedMFM/preprocessed_jsons/medqa_client.json", tokenizer_name="Henrychur/MMedLM2")

# client_dataset = MedQADataset(json_file="/data/xiaochen/FedMFM/preprocessed_jsons/medqa_server.json", tokenizer_name="Henrychur/MMedLM2")
server_dataset = MedQADataset(json_file="/data/xiaochen/FedMFM/preprocessed_jsons/medqa_server.json", tokenizer_name="Henrychur/MMedLM2")
# test_dataset = MedQADataset(json_file="/data/xiaochen/FedMFM/preprocessed_jsons/medqa_server.json", tokenizer_name="Henrychur/MMedLM2")
test_dataset = MedQADataset(json_file="/data/xiaochen/FedMFM/preprocessed_jsons/medqa_test.json", tokenizer_name="Henrychur/MMedLM2")

# Perform training
federated_training_with_centralized_data(client_dataset, server_dataset, test_dataset, model_fn, device)