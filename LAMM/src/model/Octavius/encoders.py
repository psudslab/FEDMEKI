#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 17:12:20 2024

@author: xmw5190
"""
import torch.nn as nn

class SignalProcessor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SignalProcessor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5
        )
        self.hidden_size = hidden_size

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change from (batch, seq, features) to (batch, features, seq)
        x = self.conv1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # Change back to (batch, seq, features)
        outputs, (hn, cn) = self.lstm(x)
        return outputs  # Return all hidden states


class ClinicalProcessor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(ClinicalProcessor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5
        )
        self.hidden_size = hidden_size

    def forward(self, x):
        outputs, (hn, cn) = self.lstm(x)
        return outputs  # Return all hidden states

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, hidden_size)  # Output size equals hidden size for compatibility

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x