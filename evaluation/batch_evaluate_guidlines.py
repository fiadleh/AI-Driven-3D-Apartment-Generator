#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Firas Adleh
Date: 2024-09-01
Version: 1.0
Description: Calls other script 'evaluate_guidlines.py' to check a group of 3D apartments in a given folder for anthropometric constraint violations.
"""

import os
import subprocess
import json
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import configs



# Define the folder containing .glb files
#folder_path = configs.constants['app_path'] + 'evaluation_scenes'  # Change this to your folder path
blender_path = configs.constants['blender_path']  # Change this to the path of your Blender executable
blender_script = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "evaluate_guidlines.py"
)  # Path to your Blender evaluation script

# Initialize counters for all object types and violations
total_bed_cases = total_bed_violations = 0
total_chair_cases = total_chair_violations = 0
total_cabinet_cases = total_cabinet_violations = 0
total_coffee_table_cases = total_coffee_table_violations = 0
total_dining_table_cases = total_dining_table_violations = 0
total_other_table_cases = total_other_table_violations = 0
total_nightstand_cases = total_nightstand_violations = 0


def run_blender_on_file(file_path):
    """Runs Blender to evaluate a single .glb file and returns the results."""
    args = [
            configs.constants['blender_path'],
            "-b", "-P", blender_script,
            "--",
            "--", file_path,
            #"--make_col_file",
        ]
    #subprocess.check_call(args)
    #exit()
    result = subprocess.run(
        args, #[blender_path, "--background", "--python", blender_script, "--", file_path]
        capture_output=True,
        text=True
    )
    print(result.stdout)
    
    

    # Parse the result, assuming it's returned as JSON (you'll need to modify the Blender script to output results in JSON)
    try:
        # clean the output
        json_output = '{' + result.stdout.split('{')[1].split('}')[0] + '}'
        result_json = json.loads(json_output)
        return result_json
    except:
        print(f"Error: Failed to decode JSON output from Blender for file {file_path}")
        print('res:',result.stdout)
        #exit()
        return None


# Function to iterate over a folder and process all .glb files
def process_folder(folder_path):
    """Process all .glb files in a folder and return aggregated results."""
    bed_cases = bed_violations = chair_cases = chair_violations = 0
    cabinet_cases = cabinet_violations = coffee_table_cases = coffee_table_violations = 0
    dining_table_cases = dining_table_violations = other_table_cases = other_table_violations = 0
    nightstand_cases = nightstand_violations = 0

    # Iterate over all .glb files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.glb'):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {filename} in {folder_path}")

            # Run Blender on the .glb file and get the results
            result = run_blender_on_file(file_path)

            if result:
                bed_cases += result.get('bed_cases', 0)
                bed_violations += result.get('bed_violations', 0)
                chair_cases += result.get('chair_cases', 0)
                chair_violations += result.get('chair_violations', 0)
                cabinet_cases += result.get('cabinet_cases', 0)
                cabinet_violations += result.get('cabinet_violations', 0)
                coffee_table_cases += result.get('coffee_table_cases', 0)
                coffee_table_violations += result.get('coffee_table_violations', 0)
                dining_table_cases += result.get('dining_table_cases', 0)
                dining_table_violations += result.get('dining_table_violations', 0)
                other_table_cases += result.get('other_table_cases', 0)
                other_table_violations += result.get('other_table_violations', 0)
                nightstand_cases += result.get('nightstand_cases', 0)
                nightstand_violations += result.get('nightstand_violations', 0)

    # Return the aggregated results for this folder
    return {
        "bed_cases": bed_cases,
        "bed_violations": bed_violations,
        "chair_cases": chair_cases,
        "chair_violations": chair_violations,
        "cabinet_cases": cabinet_cases,
        "cabinet_violations": cabinet_violations,
        "coffee_table_cases": coffee_table_cases,
        "coffee_table_violations": coffee_table_violations,
        "dining_table_cases": dining_table_cases,
        "dining_table_violations": dining_table_violations,
        "other_table_cases": other_table_cases,
        "other_table_violations": other_table_violations,
        "nightstand_cases": nightstand_cases,
        "nightstand_violations": nightstand_violations
    }



def plot_results(results1, results2, label1, label2):
    labels = ['Beds', 'Chairs', 'Cabinets', 'Coffee Tables', 'Dining Tables', 'Other Tables', 'Nightstands']

    cases1 = [
        results1['bed_cases'], results1['chair_cases'], results1['cabinet_cases'], 
        results1['coffee_table_cases'], results1['dining_table_cases'], 
        results1['other_table_cases'], results1['nightstand_cases']
    ]

    violations1 = [
        results1['bed_violations'], results1['chair_violations'], results1['cabinet_violations'], 
        results1['coffee_table_violations'], results1['dining_table_violations'], 
        results1['other_table_violations'], results1['nightstand_violations']
    ]

    cases2 = [
        results2['bed_cases'], results2['chair_cases'], results2['cabinet_cases'], 
        results2['coffee_table_cases'], results2['dining_table_cases'], 
        results2['other_table_cases'], results2['nightstand_cases']
    ]

    violations2 = [
        results2['bed_violations'], results2['chair_violations'], results2['cabinet_violations'], 
        results2['coffee_table_violations'], results2['dining_table_violations'], 
        results2['other_table_violations'], results2['nightstand_violations']
    ]

    x = range(len(labels))

    fig, ax = plt.subplots()

    bar_width = 0.15
    spacing = 0.10  # Add extra spacing between the sets of bars
    base_color1 = 'lightblue'  # You can change this to any other base color
    base_color2 = 'cornflowerblue'  # You can change this to any other base color

    base_color3 = 'salmon'  # You can change this to any other base color
    base_color4 = 'brown'  # You can change this to any other base color

    # Adjust positions of bars for Folder 1 and Folder 2
    positions1 = [p - bar_width / 2 - spacing / 2 for p in x]  # Shift Folder 1 bars slightly to the left
    positions2 = [p + bar_width / 2 + spacing / 2 for p in x]  # Shift Folder 2 bars slightly to the right

    # Plot for Folder 1 (opaque)
    ax.bar(positions1, cases1, bar_width, label=f'{label1} - Cases', align='center', color=base_color1, alpha=0.5)
    ax.bar(positions1, violations1, bar_width, label=f'{label1} - Violations', align='edge', color=base_color2, alpha=0.5)

    # Plot for Folder 2 (transparent)
    ax.bar(positions2, cases2, bar_width, label=f'{label2} - Cases', align='center', color=base_color3, alpha=0.5)
    ax.bar(positions2, violations2, bar_width, label=f'{label2} - Violations', align='edge', color=base_color4, alpha=0.5)

    ax.set_xlabel('Anthropometric Constraints')
    ax.set_ylabel('Counts')
    ax.set_title(f'Comparison of Collision Violations - {label1} vs {label2}')
    #ax.set_xticks([p + bar_width / 2 for p in x])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()

if __name__ == "__main__":
    # Define the folders for comparison
    folder1 = configs.constants['app_path'] + 'hpc_eval/100 generated scenes 20 * 5' 
    folder2 = configs.constants['app_path'] + '3d_front_scenes' 

    #folder1 = configs.constants['app_path'] + 'evaluation_scenes' 
    #folder2 = configs.constants['app_path'] + 'test' 
    

    # Process each folder and gather results
    results1 = process_folder(folder1)
    results2 = process_folder(folder2)

    # Plot the comparison
    plot_results(results1, results2, "Generated", "3D-Front")