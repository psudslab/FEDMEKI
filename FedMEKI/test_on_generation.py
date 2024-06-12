#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:27:12 2024

@author: xmw5190
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained("/home/xmw5190/FedMFM/LAMM/ckpt/octavius_2d_e4_bs64/", trust_remote_code=True)
model = torch.load("/home/xmw5190/FedMFM/LAMM/ckpt/octavius_2d_e4_bs64/pytorch_model.pt")
tokenizer = AutoTokenizer.from_pretrained("Henrychur/MMedLM2", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("/data/xiaochen/FedMFM/MMedLM2/", trust_remote_code=True)


device = torch.device("cuda:0")
model.to(device)

sentence = 'Question: will a 93 years old patient with severe heart failure and hyperpressure die within 48 hours? Return a probability. Answer:' 
batch = tokenizer(
            sentence,
            return_tensors="pt", 
            add_special_tokens=False
        )
with torch.no_grad():
    generated = model.generate(inputs = batch["input_ids"].to(device), epsilon_cutoff=0.95, temperature=0.1, max_length=200, do_sample=True)
    print('model predict: ',tokenizer.decode(generated[0]))

