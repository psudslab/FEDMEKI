#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 17:16:23 2024

@author: xmw5190
"""

import numpy as np
import pandas as pd
import sparse

def load_patient_data(npz_path, csv_path, patient_ids):
    """
    Load time series data for specific patients from an NPZ file using a mapping provided by a CSV file.
    
    Parameters:
    - npz_path (str): The file path to the NPZ file containing the time series data.
    - csv_path (str): The file path to the CSV file containing patient IDs and their corresponding indices.
    - patient_ids (list): A list of patient IDs for which to retrieve the time series data.
    
    Returns:
    - dict: A dictionary where keys are patient IDs and values are the corresponding time series data.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Load the NPZ file containing the time series data
    data = sparse.load_npz(npz_path).todense() # Adjust 'arr_0' based on how data is stored in the NPZ file
    print(data.shape)
    # Filter the DataFrame to get only the rows for the specified patient IDs
    filtered_df = df[df['ID'].isin(patient_ids)]
    
    # Initialize a dictionary to store the data
    patient_data = {}
    
    # Iterate through the filtered DataFrame
    for _, row in filtered_df.iterrows():
        index = row['ID']
        series_index = row.name  # Assumes the row index in CSV matches the data index in NPZ
        patient_data[index] = data[series_index]
        
    return patient_data


npz_file_path = "/data/xiaochen/data/physionet.org/files/mimic-eicu-fiddle-feature/1.0.0/FIDDLE_mimic3/features/Shock_4h/X.npz"
csv_file_path = "/data/xiaochen/data/physionet.org/files/mimic-eicu-fiddle-feature/1.0.0/FIDDLE_mimic3/population/Shock_4h.csv"
patient_ids_to_load = [200001, 200010]

data = load_patient_data(npz_file_path, csv_file_path, patient_ids_to_load)
print(data)
