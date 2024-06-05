#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 22:06:44 2024

@author: xmw5190
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import json
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
from copy import deepcopy
from torch.cuda.amp import GradScaler, autocast
from torcheval.metrics.functional import bleu_score
import random
import torch.nn.functional as F



torch.autograd.set_detect_anomaly(True)

# Seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

class CustomGPT2Model(GPT2LMHeadModel):
    def __init__(self, config, external_embeddings):
        super().__init__(config)
        self.external_embeddings = external_embeddings
        self.lm_head = nn.Linear(config.n_embd, self.external_embeddings.weight.size(1), bias=False)
        self.lm_head.weight = self.external_embeddings.weight
        # self.pad_token_id = kwargs.get('pad_token_id', 2)
        
    def forward(self, input_ids=None, past_key_values=None, attention_mask=None, token_type_ids=None, 
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None, use_cache=None, 
                output_attentions=None, output_hidden_states=None, return_dict=None):

        if inputs_embeds is None:
            inputs_embeds = self.external_embeddings(input_ids)

        return super().forward(input_ids=None, inputs_embeds=inputs_embeds, past_key_values=past_key_values, 
                               attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, 
                               head_mask=head_mask, labels=labels, use_cache=use_cache, 
                               output_attentions=output_attentions, output_hidden_states=output_hidden_states, 
                               return_dict=return_dict)
    def generate_answer(self, input_ids, attention_mask, max_new_tokens=10):
        generated_ids = self.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=2,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_k=10,
            pad_token_id=self.config.pad_token_id,
            eos_token_id=self.config.eos_token_id,
            bos_token_id=self.config.bos_token_id
        )
        return generated_ids
class CustomGPT2Config(GPT2Config):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.n_layer = kwargs.get('n_layer', 1)  # Set to one layer
        self.n_embd = kwargs.get('n_embd', 512)  # Custom embedding size
        self.n_head = kwargs.get('n_head', 8)    # Set number of attention heads
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.unk_token_id = tokenizer.unk_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.mask_token_id = tokenizer.mask_token_id

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
        attention_mask = batch['attention_mask'].to(device)

        with torch.no_grad():
            generated_ids =  model.generate_answer(input_ids, attention_mask)

        # candidates = [i.split('</s>')[0].replace("<s>", '').replace("<s>", '') for i in tokenizer.batch_decode(generated_ids, skip_special_tokens=False)]
        candidates = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
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
        with open(json_file, 'r') as file:
            self.data = json.load(file)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['conversations'][0]['value']
        answer = item['conversations'][1]['value']
        question_encoding = self.tokenizer(question, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
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
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
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
                outputs = self.model(input_ids, labels=labels).logits


                # Flatten the logits and labels for the loss computation
                logits = outputs.view(-1, outputs.size(-1))  # [batch_size * seq_length, vocab_size]
                labels = labels.view(-1)  # [batch_size * seq_length]



                loss = self.compute_loss(logits, labels, 2)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            total_loss += loss.item()
        self.model.cpu()
        return total_loss / len(self.train_loader)

    def compute_loss(self, logits, targets, padding_index):
        # print(targets)
        # print(padding_index)
        mask = (targets != padding_index)
        # print(mask)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1)
        logits_non_pad = logits_flat[mask_flat]
        targets_non_pad = targets_flat[mask_flat]

        # print(f"Masked Logits shape: {logits_non_pad.shape}")
        # print(f"Masked Targets shape: {targets_non_pad.shape}")

        assert torch.all(targets_non_pad >= 0) and torch.all(targets_non_pad < logits_non_pad.size(-1)), "Target indices are out of range"

        loss = F.cross_entropy(logits_non_pad, targets_non_pad)
        return loss



def average_weights(state_dicts):
    new_state_dict = state_dicts[0].copy()
    for key in new_state_dict.keys():
        new_state_dict[key] = torch.mean(torch.stack([state_dict[key].float() for state_dict in state_dicts]), 0)
    return new_state_dict

def federated_training_with_centralized_data(client_dataset, server_dataset, test_dataset, model_fn, device, num_clients=5, epochs=10):
    client_datasets = random_split(client_dataset, [len(client_dataset) // num_clients + (1 if x < len(client_dataset) % num_clients else 0) for x in range(num_clients)])
    client_loaders = [DataLoader(ds, batch_size=32, shuffle=True, pin_memory=True, num_workers=4) for ds in client_datasets]
    server_loader = DataLoader(server_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)

    external_embeddings = torch.load("/data/xiaochen/FedMFM/MMedLM2/embedding_layer")
    global_model = model_fn(external_embeddings).float()
    clients = [Client(model=deepcopy(global_model), train_loader=loader, criterion=torch.nn.CrossEntropyLoss(ignore_index=2), device=device) for loader in client_loaders]
    server_model = Client(model=deepcopy(global_model), train_loader=server_loader, criterion=torch.nn.CrossEntropyLoss(ignore_index=2), device=device)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")

        state_dicts = []
        for client in clients:
            loss = client.train()
            print(f"Client {clients.index(client) + 1} Loss: {loss}")
            state_dicts.append(client.model.state_dict())

        global_state_dict = average_weights(state_dicts)
        server_model.model.load_state_dict(global_state_dict)
        server_model.train()

        

        for client in clients:
            client.model.load_state_dict(server_model.model.state_dict())
        torch.cuda.empty_cache()
    bleu_score_value = evaluate_model(server_model.model, test_loader, device)
    print(f"BLEU score on test data: {bleu_score_value}")
    return server_model

dataset = MedQADataset(json_file="/data/xiaochen/FedMFM/preprocessed_jsons/medqa_server.json")
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_fn = lambda embeddings: CustomGPT2Model(CustomGPT2Config(tokenizer = AutoTokenizer.from_pretrained("Henrychur/MMedLM2", trust_remote_code=True), n_layer=1, n_embd=4096, n_head=8), embeddings)
client_dataset = MedQADataset(json_file="/data/xiaochen/FedMFM/preprocessed_jsons/medqa_client.json", tokenizer_name="Henrychur/MMedLM2")
# client_dataset = MedQADataset(json_file="/data/xiaochen/FedMFM/preprocessed_jsons/medqa_server.json", tokenizer_name="Henrychur/MMedLM2")
server_dataset = MedQADataset(json_file="/data/xiaochen/FedMFM/preprocessed_jsons/medqa_server.json", tokenizer_name="Henrychur/MMedLM2")
# test_dataset = MedQADataset(json_file="/data/xiaochen/FedMFM/preprocessed_jsons/medqa_server.json", tokenizer_name="Henrychur/MMedLM2")
test_dataset = MedQADataset(json_file="/data/xiaochen/FedMFM/preprocessed_jsons/medqa_test.json", tokenizer_name="Henrychur/MMedLM2")

federated_training_with_centralized_data(client_dataset, server_dataset, test_dataset, model_fn, device)
