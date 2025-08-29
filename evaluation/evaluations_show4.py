#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Firas Adleh
Date: 2024-09-01
Version: 1.0
Description: Display the apartment generation times in boxplot graphs.
"""

import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import configs

def extract_number(filename):
    # Regular expression to find the number between '_' and the file extension
    match = re.search(r'_(\d+)\.', filename)
    if match:
        return int(match.group(1))
    else:
        return None

def process_folder(folder_path):
    numbers = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            number = extract_number(filename)
            if number is not None:
                numbers.append(number)
    return numbers

def plot_combined_boxplots(data_dict, grid=True, color='skyblue'):
    plt.figure(figsize=(12, 8))
    
    data = []
    labels = []
    
    for folder_name, numbers in data_dict.items():
        if numbers:  # Ensure there are numbers to plot
            # Use the custom label if provided, else use the folder name
            label = folder_name
            data.append(numbers)
            labels.append(label)

    data.append(data[0] + data[1] + data[2] + data[3] + data[4])
    labels.append("500 Combined")

    plt.boxplot(data, labels=labels)
    plt.axvline(x=5.5, color='red', linestyle='--', linewidth=2)
    plt.title("Apartment Generation Time")
    #plt.xlabel('Generation Templates')
    plt.ylabel('Seconds')
    
    if grid:
        plt.grid(True, linestyle='--', alpha=0.7)
    else:
        plt.grid(False)
    
    plt.show()


folder_paths = [
    # graph 3 as G1 and graph1 as G2 and graph2 as G3 to sort graphs according to room count 
    ('./evaluation/scenes/hpc_eval/3/evaluation_scenes/cylinders', "G1"),
    ('./evaluation/scenes/hpc_eval/1/evaluation_scenes/cylinders', "G2"),
    ('./evaluation/scenes/hpc_eval/2/evaluation_scenes/cylinders', "G3"),
    
    ('./evaluation/scenes/hpc_eval/4/evaluation_scenes/cylinders', "G4"),
    ('./evaluation/scenes/hpc_eval/5/evaluation_scenes/cylinders', "G5"),
]



data_dict = {}

for folder in folder_paths:
    folder_name = folder[1] 
    numbers = process_folder(folder[0])
    if numbers:
        data_dict[folder_name] = numbers

if data_dict:
    plot_combined_boxplots(data_dict, 
                           grid=True, 
                           color='green')
else:
    print("No valid numbers were extracted from the files in the provided folders.")






