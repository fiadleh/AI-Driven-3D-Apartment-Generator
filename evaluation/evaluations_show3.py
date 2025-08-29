#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Firas Adleh
Date: 2024-09-01
Version: 1.0
Description: Display infos about of apartments like distribution of apartments and room sizes, total number of rooms per type, in graphs.
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import configs
from functions import get_3d_front_room_sizes_and_total



def get_room_sizes_and_total(data):
    room_sizes = {}
    total_size = 0
    room_count = {}
    bed_living_dining_size = 0

    # Navigate to scene -> room and extract sizes
    for room in data.get('scene', {}).get('room', []):
        room_type = room.get('type')

        if 'bedroom' in room_type.lower() or 'kidsroom' in room_type.lower() or 'nannyroom' in room_type.lower():
            room_type = 'Bedroom'
        if 'bathroom' in room_type.lower():
            room_type = 'Bathroom'
        if 'dining' in room_type.lower():
            room_type = 'Diningroom'
        if 'living' in room_type.lower():
            room_type = 'Livingroom'

        size = room.get('size', 0)
        if 'bedroom' in room_type.lower() or 'living' in room_type.lower() or 'dining' in room_type.lower():
            
            
            if size > 0:
                # Collect sizes for each room type
                if room_type not in room_sizes:
                    room_sizes[room_type] = []
                room_sizes[room_type].append(size)

            # Count rooms per type
            room_count[room_type] = room_count.get(room_type, 0) + 1
            
            bed_living_dining_size += size
            

        total_size += size

    return room_sizes, total_size, room_count, bed_living_dining_size

def iterate_json_files(folder_path):
    all_room_sizes = {}
    all_house_sizes = []
    all_bed_living_dining_sizes = []
    room_type_count = {}

    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(JSON_3d_front_folder_path, file_name)
            room_sizes, total_size, room_count, bed_living_dining_size, real_bed_living_dining_objects_count , bed_living_dining_objects_count, room_objects_count = get_3d_front_room_sizes_and_total(file_path)

            # Append room sizes to the aggregated dictionary
            for room_type, sizes in room_sizes.items():
                if room_type not in all_room_sizes:
                    all_room_sizes[room_type] = []
                all_room_sizes[room_type].extend(sizes)
            
            all_house_sizes.append(total_size)
            all_bed_living_dining_sizes.append(bed_living_dining_size)

            # Update room type count across all files
            for room_type, count in room_count.items():
                room_type_count[room_type] = room_type_count.get(room_type, 0) + count

    return all_room_sizes, all_house_sizes, room_type_count, all_bed_living_dining_sizes




def get_generated_room_sizes_and_total(file_path, CLASS_ROOM):
    with open(file_path, 'r') as file:
        floorplan_data = json.load(file)

    room_sizes = {}
    total_size = 0
    room_count = {}
    bed_living_dining_size = 0

    for room_floor_info in floorplan_data[3]:
        room_type = CLASS_ROOM[room_floor_info.get('color_id')]
        if 'bedroom' in room_type.lower() or 'kidsroom' in room_type.lower() or 'nannyroom' in room_type.lower():
            room_type = 'Bedroom'
        if 'bathroom' in room_type.lower():
            room_type = 'Bathroom'
        if 'dining' in room_type.lower():
            room_type = 'Diningroom'
        if 'living' in room_type.lower():
            room_type = 'Livingroom'

        size = room_floor_info.get('area', 0)
        if 'bedroom' in room_type.lower() or 'living' in room_type.lower() or 'dining' in room_type.lower():
            
            
            if size > 0:
                if room_type not in room_sizes:
                    room_sizes[room_type] = []
                room_sizes[room_type].append(size)

            bed_living_dining_size += size

            # Count rooms per type
            room_count[room_type] = room_count.get(room_type, 0) + 1

        total_size += size

    return room_sizes, total_size, room_count, bed_living_dining_size


def iterate_generated_json_files(folder_path):
    #all_room_sizes = []
    all_room_sizes = {}
    all_house_sizes = []
    all_bed_living_dining_sizes  = []
    room_type_distribution = {}
    room_type_count = {}
    room_count_per_house = []

    ROOM_CLASS = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "balcony": 5, "entrance": 6, "dining room": 7, "study room": 8,
              "storage": 10 , "front door": 15, "unknown": 16, "interior_door": 17}
    CLASS_ROOM = {}
    for x, y in ROOM_CLASS.items():
        CLASS_ROOM[y] = x

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json') and not 'rooms_points' in file_name:
            filename_splits = file_name.split('_')
            filename_splits.pop()
            generated_floorplan_json_path = os.path.join(folder_path, '_'.join(filename_splits)+'.rooms_points.json')
            

            room_sizes, total_size, room_count, bed_living_dining_size = get_generated_room_sizes_and_total(generated_floorplan_json_path, CLASS_ROOM)
            
            for room_type, sizes in room_sizes.items():
                if room_type not in all_room_sizes:
                    all_room_sizes[room_type] = []
                all_room_sizes[room_type].extend(sizes)

            all_house_sizes.append(total_size)
            all_bed_living_dining_sizes.append(bed_living_dining_size)
            
            
            # Update room type count across all files
            for room_type, count in room_count.items():
                room_type_count[room_type] = room_type_count.get(room_type, 0) + count

    return all_room_sizes, all_house_sizes, room_type_count, all_bed_living_dining_sizes#, room_type_distribution

def plot_comparison_two(folder_paths, labels):
    # Iterate and collect data for both folders
    #results = [iterate_json_files(folder_path) for folder_path in folder_paths]
    results = [None, None]
    results[0] = iterate_json_files(folder_paths[0])
    results[1] = iterate_generated_json_files(folder_paths[1])
    #all_room_sizes2, all_house_sizes2, room_type_count2, room_type_distribution2 = iterate_generated_json_files(dataset_path, folder_path)

    all_room_sizes_1, all_house_sizes_1, room_type_count_1, all_bed_living_dining_sizes_1 = results[0]
    all_room_sizes_2, all_house_sizes_2, room_type_count_2, all_bed_living_dining_sizes_2 = results[1]

    # Plot Room Sizes Distribution as Boxplot
    plt.figure(figsize=(14, 8))
    
    room_types = sorted(list(set(all_room_sizes_1.keys()).union(set(all_room_sizes_2.keys()))))
    
    room_sizes_1 = [all_room_sizes_1.get(room_type, []) for room_type in room_types]
    room_sizes_2 = [all_room_sizes_2.get(room_type, []) for room_type in room_types]
    
    positions_1 = range(1, len(room_types) + 1)
    positions_2 = [pos + 0.4 for pos in positions_1]

    plt.boxplot(room_sizes_1, labels=room_types, vert=False, positions=positions_1, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    plt.boxplot(room_sizes_2, vert=False, positions=positions_2, patch_artist=True, boxprops=dict(facecolor="lightgreen"))

    # Create custom legend handles
    blue_patch = mpatches.Patch(color='lightblue', label=labels[0])
    green_patch = mpatches.Patch(color='lightgreen', label=labels[1])

    plt.title('Distribution of Room Areas by Room Type')
    plt.xlabel('Room Area M²')
    plt.ylabel('Room Type')
    plt.yticks([pos + 0.2 for pos in positions_1], room_types)
    plt.legend(handles=[blue_patch, green_patch], loc='upper right')
    plt.grid(True)
    plt.savefig('imgs/Room Areas by Room Type.pdf') 
    #plt.show()

    # Plot House Sizes Distribution
    plt.figure(figsize=(10, 6))
    
    # Determine common bin range for both datasets
    min_size = min(min(all_house_sizes_1), min(all_house_sizes_2))
    max_size = max(max(all_house_sizes_1), max(all_house_sizes_2))
    bins = 30  # Number of bins
    
    plt.hist(all_house_sizes_1, bins=bins, range=(min_size, max_size), color='lightblue', alpha=0.5, label=labels[0])
    plt.hist(all_house_sizes_2, bins=bins, range=(min_size, max_size), color='lightgreen', alpha=0.5, label=labels[1])
    plt.title('Distribution of House Areas')
    plt.xlabel('House Area')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig('imgs/House Areas.pdf')  
    #plt.show()


    # Plot bed_living_dining Sizes Distribution
    plt.figure(figsize=(10, 6))
    
    # Determine common bin range for both datasets
    min_size = min(min(all_bed_living_dining_sizes_1), min(all_bed_living_dining_sizes_2))
    max_size = max(max(all_bed_living_dining_sizes_1), max(all_bed_living_dining_sizes_2))
    bins = 30  # Number of bins
    
    plt.hist(all_bed_living_dining_sizes_1, bins=bins, range=(min_size, max_size), color='lightblue', alpha=0.5, label=labels[0])
    plt.hist(all_bed_living_dining_sizes_2, bins=bins, range=(min_size, max_size), color='lightgreen', alpha=0.5, label=labels[1])
    plt.title('Distribution of bed_living_dining Areas')
    plt.xlabel('bed/living/dining Area')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig('imgs/Bed-Living-Dining Areas.pdf')



    # Plot Total Number of Rooms per Type
    all_room_types = list(set(room_type_count_1.keys()).union(set(room_type_count_2.keys())))
    counts_1 = [room_type_count_1.get(room_type, 0) for room_type in all_room_types]
    counts_2 = [room_type_count_2.get(room_type, 0) for room_type in all_room_types]

    x = range(len(all_room_types))
    width = 0.4

    plt.figure(figsize=(12, 8))
    plt.barh([pos - width/2 for pos in x], counts_1, height=width, color='lightblue', label=labels[0])
    plt.barh([pos + width/2 for pos in x], counts_2, height=width, color='lightgreen', label=labels[1])
    plt.yticks(x, all_room_types)
    plt.title('Total Number of Rooms per Type Across All Files')
    plt.xlabel('Total Number of Rooms')
    plt.ylabel('Room Type')
    plt.legend()
    plt.grid(True)
    plt.savefig('imgs/Total Number of Rooms Areas.pdf') 
    plt.show()


def plot_comparison(folder_paths, labels):
    # Iterate and collect data for all folders
    results = []#[iterate_json_files(folder_path) for folder_path in folder_paths]
    for fpath in folder_paths:
        if 'front' in fpath:
            results.append(iterate_json_files(fpath))
        else:
            results.append(iterate_generated_json_files(fpath))

    
    # Room Sizes Comparison
    plt.figure(figsize=(14, 8))
    room_types = sorted(list(set().union(*[result[0].keys() for result in results])))
    positions = list(range(1, len(room_types) + 1))
    
    patches = []
    for i, (room_sizes, _, _, _) in enumerate(results):
        sizes = [room_sizes.get(room_type, []) for room_type in room_types]
        pos_offset = [pos + 0.4 * i for pos in positions]
        plt.boxplot(sizes, vert=False, positions=pos_offset, patch_artist=True, boxprops=dict(facecolor=f"C{i}"))
        patches.append(mpatches.Patch(color=f"C{i}", label=labels[i]))
    
    plt.title('Distribution of Room Sizes by Room Type')
    plt.xlabel('Room Size')
    plt.ylabel('Room Type')
    plt.yticks([pos + 0.4 * (len(folder_paths) - 1) / 2 for pos in positions], room_types)
    plt.legend(handles=patches, loc='upper right')
    plt.grid(True)
    plt.savefig('imgs/Room Sizes by Room Type.pdf') 

    # House Sizes Comparison
    plt.figure(figsize=(10, 6))
    
    min_size = min(min(result[1]) for result in results)
    max_size = max(max(result[1]) for result in results)
    bins = 30  # Number of bins
    
    for i, (_, house_sizes, _, _) in enumerate(results):
        plt.hist(house_sizes, bins=bins, range=(min_size, max_size), color=f"C{i}", alpha=0.5, label=labels[i])
    
    plt.title('Distribution of House Areas')
    plt.xlabel('House Size')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig('imgs/House Areas.pdf')  



    # bed/living/dining Sizes Comparison
    plt.figure(figsize=(10, 6))
    
    min_size = min(min(result[3]) for result in results)
    max_size = max(max(result[3]) for result in results)
    bins = 30  # Number of bins
    
    for i, (_, _, _, bed_living_dining_sizes) in enumerate(results):
        plt.hist(bed_living_dining_sizes, bins=bins, range=(min_size, max_size), color=f"C{i}", alpha=0.5, label=labels[i])
    
    plt.title('Distribution of bed/living/dining Areas')
    plt.xlabel('House Size')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig('imgs/Bed-Living-Dining Areas.pdf')  



    # Total Number of Rooms per Type Comparison
    all_room_types = sorted(list(set().union(*[result[2].keys() for result in results])))
    plt.figure(figsize=(12, 8))

    x = range(len(all_room_types))
    width = 0.8 / len(folder_paths)  # Adjust width based on number of folders
    
    for i, (_, _, room_type_count, _) in enumerate(results):
        counts = [room_type_count.get(room_type, 0) for room_type in all_room_types]
        plt.barh([pos - width * len(folder_paths) / 2 + width * i for pos in x], counts, height=width, color=f"C{i}", label=labels[i])
    
    plt.yticks(x, all_room_types)
    plt.title('Total Number of Rooms per Type Across All Files')
    plt.xlabel('Total Number of Rooms')
    plt.ylabel('Room Type')
    plt.legend()
    plt.grid(True)
    plt.savefig('imgs/Total Number of Rooms Areas.pdf')  
    plt.show()


JSON_3d_front_folder_path = './InstructScene/dataset/3D-FRONT/3D-FRONT/'


# Room Type Plots
folder_paths = ['./evaluation/scenes/3d_front_scenes/cylinders', './evaluation/scenes/100_generated/ellipses/combined']
labels = ['3D Front', 'Generated']
plot_comparison_two(folder_paths, labels)


folder_paths = ['./evaluation/scenes/100_generated/ellipses/1',
                './evaluation/scenes/100_generated/ellipses/2',
                './evaluation/scenes/100_generated/ellipses/3',
                './evaluation/scenes/100_generated/ellipses/4',
                './evaluation/scenes/100_generated/ellipses/5']
labels = ['1', '2', '3', '4', '5']
plot_comparison(folder_paths, labels)