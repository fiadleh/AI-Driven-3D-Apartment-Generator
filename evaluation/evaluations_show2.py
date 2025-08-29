#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Firas Adleh
Date: 2024-09-01
Version: 1.0
Description: Display the results of apartments usability evaluations in boxplot graphs.
"""

import os
import json
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import configs
from functions import get_3d_front_room_sizes_and_total



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

        size = room_floor_info.get('area', 0)
        
        if size > 0:
            if room_type not in room_sizes:
                room_sizes[room_type] = []
            room_sizes[room_type].append(size)

        total_size += size
        if 'bedroom' in room_type.lower() or 'living' in room_type.lower() or 'dining' in room_type.lower():
            bed_living_dining_size += size

        # Count rooms per type
        room_count[room_type] = room_count.get(room_type, 0) + 1

    return room_sizes, total_size, room_count, bed_living_dining_size


def read_json_files(directory, JSON_3d_front_folder_path, CLASS_ROOM):
    num_islands_list = []
    island_sizes_list = []
    percentage_collisions_with_objects = []
    percentage_collisions_with_walls = []
    percentage_no_collisions = []
    num_elements_in_island_sizes = []
    reachability_list = []
    num_objects = []
    object_per_squared_meter = []

    for filename in os.listdir(directory):
        if filename.endswith('.json') and not filename.endswith('_text.json') and not filename.endswith('rooms_points.json') and not filename.endswith('apartment_configs.json') and not filename.endswith('furniture_floors_map.json'):
            room_sizes, total_size, room_count, bed_living_dining_size, bed_living_dining_objects_count, test_bed_living_dining_objects_count, room_objects_count =  0,0,0,0,0,0,0
            is_generated = False

            # read info file
            if len(filename.split('_')) < 3 and 'front' in directory:
                # 3d front file
                JSON_3d_front_file_path = os.path.join(JSON_3d_front_folder_path, filename)
                room_sizes, total_size, room_count, bed_living_dining_size, bed_living_dining_objects_count, test_bed_living_dining_objects_count, room_objects_count = get_3d_front_room_sizes_and_total(JSON_3d_front_file_path)
            else:
                # generated file
                is_generated = True
                filename_splits = filename.split('_')
                filename_splits.pop()
                generated_floorplan_json_path = os.path.join(directory, '_'.join(filename_splits)+'.rooms_points.json')
                
                if os.path.isfile(generated_floorplan_json_path):
                    room_sizes, total_size, room_count, bed_living_dining_size = get_generated_room_sizes_and_total(generated_floorplan_json_path, CLASS_ROOM)


            # read evaluation file
            filepath = os.path.join(directory, filename)
  
            if (is_generated or bed_living_dining_objects_count > 5) and bed_living_dining_size > -1:#20:
                with open(filepath, 'r') as fp:
                    data = json.load(fp)
                    num_islands = data[0]
                    island_sizes = data[3]
                    check_collisions_result = data[5]
                    collisions = data[6][3]
                    total_objects = data[6][1]
                    all_objects_count = data[6][1]
                    reachable_objects_objects_count =  data[6][2]

                    objects_names = [s for s in data[7] if not s.startswith("BIG_bbox_")] # ignore "BIG_bbox_" objects if there were saved as objects during the evaluation
                    

                    num_islands_list.append(num_islands)

                    currect_object_per_squared_meter = 0
                    if not is_generated:
                        # 3d front
                        num_objects.append(bed_living_dining_objects_count)
                        if bed_living_dining_size > 0:
                            currect_object_per_squared_meter = bed_living_dining_objects_count / bed_living_dining_size
                            object_per_squared_meter.append(currect_object_per_squared_meter)
                        else:
                            print('----  Error: bed_living_dining_size = 0 in file:', filepath)
                    else:
                        # generated
                        num_objects.append(len(objects_names))
                        bed_living_dining_objects_count = len(objects_names)
                        if bed_living_dining_size > 0:
                            currect_object_per_squared_meter = len(objects_names) / bed_living_dining_size
                            object_per_squared_meter.append(currect_object_per_squared_meter)
                        else:
                            print('----  Error: bed_living_dining_size = 0 in file:', filepath)


                    if len(island_sizes) > 1:
                        sorted_island_sizes = sorted(island_sizes)
                        island_sizes_list.extend(sorted_island_sizes[:-1])  # Ignore the largest element (normal zone)
                        if max(sorted_island_sizes[:-1]) > 70:
                            print("island_sizes: ",island_sizes, filename)

                    if check_collisions_result[0] > 0:
                        percentage_collisions_with_objects.append(check_collisions_result[1] / check_collisions_result[0] * 100)
                        percentage_collisions_with_walls.append(check_collisions_result[2] / check_collisions_result[0] * 100)
                        percentage_no_collisions.append(check_collisions_result[3] / check_collisions_result[0] * 100)

                    if all_objects_count > 0:
                        reachability_list.append(reachable_objects_objects_count / all_objects_count * 100)
                    if check_collisions_result[2] / check_collisions_result[0] > 1:
                        print('Error: check_collisions_result[2] / check_collisions_result[0] > 1 : ',filename, check_collisions_result)
                    num_elements_in_island_sizes.append(len(island_sizes))
                    if all_objects_count> 0 and reachable_objects_objects_count / all_objects_count * 100 < 50:
                        print("------- reachability: ",reachable_objects_objects_count / all_objects_count * 100, filename)
                    if check_collisions_result[1] / check_collisions_result[0] * 100 > 90:
                        print("collision with objects: ",check_collisions_result[1] / check_collisions_result[0] * 100, filename)
                    if check_collisions_result[2] / check_collisions_result[0] * 100 > 50:
                        print("collision with walls: ",check_collisions_result[2] / check_collisions_result[0] * 100, filename)

            else:
                print('skip file, bed_living_dining_objects_count:', bed_living_dining_objects_count, ', bed_living_dining_size:', bed_living_dining_size, '|', filename)

    print('\n\n\n\n ===================== ',directory, 'json_files: ', len(num_islands_list))

    return (num_islands_list, island_sizes_list, 
            percentage_collisions_with_objects, 
            percentage_collisions_with_walls, 
            percentage_no_collisions, 
            num_elements_in_island_sizes,
            reachability_list,
            num_objects,
            object_per_squared_meter)



def plot_boxplots(data, folder_names, save_to_png, group_results = False):
    num_islands_data, island_sizes_data, \
    percentage_collisions_with_objects_data, \
    percentage_collisions_with_walls_data, \
    percentage_no_collisions_data, \
    num_elements_in_island_sizes_data, \
    reachability_data, \
    all_objects_count, \
    object_per_squared_meter = data
    


    short_folder_names = folder_names

    # group results of 1+2 and 3+4 for some graphs because these groups have the same results 
    if group_results:
        group_a = ' '.join(folder_names[0].split(' ')[:-1])
        group_b = ' '.join(folder_names[1].split(' ')[:-1])
        if group_a == group_b:
            group_b = ' '.join(folder_names[2].split(' ')[:-1])
        short_folder_names = [group_a, group_b]
        percentage_collisions_with_objects_data = [percentage_collisions_with_objects_data[0], percentage_collisions_with_objects_data[3]]
        percentage_collisions_with_walls_data = [percentage_collisions_with_walls_data[0], percentage_collisions_with_walls_data[3]]
        percentage_no_collisions_data  = [percentage_no_collisions_data[0], percentage_no_collisions_data[3]]
        all_objects_count = [all_objects_count[0], all_objects_count[3]]
        object_per_squared_meter = [object_per_squared_meter[0], object_per_squared_meter[3]]
    
    

    plt.figure()
    plt.boxplot(num_islands_data, labels=folder_names)
    plt.title('Connected Zones Count')
    plt.ylabel('Number of Connected Zones')
    plt.grid(True)
    plt.savefig('imgs/Connected Zones Count.pdf')  

    plt.figure()
    plt.boxplot(island_sizes_data, labels=folder_names)
    plt.title('Dead Zones Areas')
    plt.ylabel('Area (m²)')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('imgs/Dead Zones Areas (log).pdf')  

    plt.figure()
    plt.boxplot(island_sizes_data, labels=folder_names)
    plt.title('Dead Zones Areas')
    plt.ylabel('Area (m²)')
    plt.grid(True)
    plt.savefig('imgs/Dead Zones Areas 1.pdf')  

    plt.figure()
    plt.boxplot(percentage_collisions_with_objects_data, labels=short_folder_names)
    plt.title('Objects Colliding with Others')
    plt.ylabel('Percentage (%)')
    plt.grid(True)
    plt.savefig('imgs/Objects Colliding with Others.pdf')  

    plt.figure()
    plt.boxplot(percentage_collisions_with_walls_data, labels=short_folder_names)
    plt.title('Objects Colliding with Walls')
    plt.ylabel('Percentage (%)')
    plt.grid(True)
    plt.savefig('imgs/Objects Colliding with Walls.pdf')  

    plt.figure()
    last_data = []
    final_percentage_no_collisions_data = []
    final_short_folder_names = []
    skip_next = False
    for i in range(len(percentage_no_collisions_data)):
        if not skip_next:
            if i < len(percentage_no_collisions_data)-1 and percentage_no_collisions_data[i] == percentage_no_collisions_data[i+1]:
                final_percentage_no_collisions_data.append(percentage_no_collisions_data[i])
                final_short_folder_names.append(short_folder_names[i].split(' ')[0])
                skip_next = True
            else:
                final_percentage_no_collisions_data.append(percentage_no_collisions_data[i])
                final_short_folder_names.append(short_folder_names[i])
        else:        
            skip_next = False
            
    if group_results:
        plt.boxplot(percentage_no_collisions_data, labels=short_folder_names)
    else:
        plt.boxplot(final_percentage_no_collisions_data, labels=final_short_folder_names)

    plt.title('Objects with No Collisions')
    plt.ylabel('Percentage (%)')
    plt.grid(True)
    plt.savefig('imgs/Objects with No Collisions.pdf')  


    plt.figure()
    plt.boxplot(reachability_data, labels=folder_names)
    plt.title('Objects Reachability')
    plt.ylabel('Percent of Reachable Objects (%)')
    plt.grid(True)
    plt.savefig('imgs/Objects Reachability.pdf')


    plt.figure()
    plt.boxplot(all_objects_count, labels=short_folder_names)
    plt.title('Number of Objects')
    plt.ylabel('Number of Objects')
    plt.grid(True)
    plt.savefig('imgs/Number of Objects.pdf')

    plt.figure()
    plt.boxplot(object_per_squared_meter, labels=short_folder_names)
    plt.title('Objects Density')
    plt.ylabel('Object/M²')
    plt.grid(True)
    plt.savefig('imgs/Objects Density.pdf')

    if not save_to_png:
        plt.show()



def main(directories, save_to_png, JSON_3d_front_folder_path, group_results = False):

    ROOM_CLASS = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "balcony": 5, "entrance": 6, "dining room": 7, "study room": 8,
              "storage": 10 , "front door": 15, "unknown": 16, "interior_door": 17}
    CLASS_ROOM = {}
    for x, y in ROOM_CLASS.items():
        CLASS_ROOM[y] = x

    all_num_islands = []
    all_island_sizes = []
    all_percentage_collisions_with_objects = []
    all_percentage_collisions_with_walls = []
    all_percentage_no_collisions = []
    all_num_elements_in_island_sizes = []
    all_reachability = []
    all_objects_counts = []
    all_object_per_squared_meter = []
    folder_names = []

    for directory in directories:
        data = read_json_files(directory[0], JSON_3d_front_folder_path, CLASS_ROOM)
        all_num_islands.append(data[0])
        m2_sizes = [x / 4 for x in data[1]]
        all_island_sizes.append(m2_sizes)
        all_percentage_collisions_with_objects.append(data[2])
        all_percentage_collisions_with_walls.append(data[3])
        all_percentage_no_collisions.append(data[4])
        all_num_elements_in_island_sizes.append(data[5])
        all_reachability.append(data[6])
        all_objects_counts.append(data[7])
        all_object_per_squared_meter.append(data[8])
        folder_names.append(os.path.basename(directory[1]))

    plot_boxplots((all_num_islands, all_island_sizes, 
                   all_percentage_collisions_with_objects, 
                   all_percentage_collisions_with_walls, 
                   all_percentage_no_collisions, 
                   all_num_elements_in_island_sizes, 
                   all_reachability,
                   all_objects_counts,
                   all_object_per_squared_meter), folder_names, save_to_png, group_results)


# Lists of directories containing the JSON files

directories_100 = [
    # 100 generated apartments
    ['./evaluation/scenes/100_generated/cylinders/combined', 'Generated C'],
    ['./evaluation/scenes/100_generated/ellipses/combined', 'Generated E'],

    # 100 selected from 3D-Front
    ['./evaluation/scenes/3d_front_scenes/cylinders', '3DFront C'],
    ['./evaluation/scenes/3d_front_scenes/ellipses', '3DFront E'],
]

directories_before_after_adjust = [

    ['./evaluation/scenes/hpc_eval/evaluation_scenes_before_adjust_combined/cylinders', 'Before Adjust C'],
    ['./evaluation/scenes/100_generated/cylinders/combined', 'After Adjust C'],

    ['./evaluation/scenes/hpc_eval/evaluation_scenes_before_adjust_combined/ellipses', 'Before Adjust E'],
    ['./evaluation/scenes/100_generated/ellipses/combined', 'After Adjust E'],

]

directories_each_graph_type = [

    ['./evaluation/scenes/hpc_eval/100 each graph type/3/C', 'G1 C'],  # graph 3 as G1 and graph1 as G2 and graph2 as G3 to sort graphs according to room count 
    ['./evaluation/scenes/hpc_eval/100 each graph type/3/E', 'G1 E'],


    ['./evaluation/scenes/hpc_eval/100 each graph type/1/C', 'G2 C'],
    ['./evaluation/scenes/hpc_eval/100 each graph type/1/E', 'G2 E'],



    ['./evaluation/scenes/hpc_eval/100 each graph type/2/C', 'G3 C'],
    ['./evaluation/scenes/hpc_eval/100 each graph type/2/E', 'G3 E'],

    

    ['./evaluation/scenes/hpc_eval/100 each graph type/4/C', 'G4 C'],
    ['./evaluation/scenes/hpc_eval/100 each graph type/4/E', 'G4 E'],

    ['./evaluation/scenes/hpc_eval/100 each graph type/5/C', 'G5 C'],
    ['./evaluation/scenes/hpc_eval/100 each graph type/5/E', 'G5 E'],
]

directories_instruct_scene = [
    ['./evaluation/scenes/hpc_eval/instruct_scene_room/bedroom/100', 'bedroom'],
    ['./evaluation/scenes/hpc_eval/instruct_scene_room/livingroom/100', 'livingroom'],
    ['./evaluation/scenes/hpc_eval/instruct_scene_room/diningroom/100', 'diningroom'],
]


JSON_3d_front_folder_path = './InstructScene/dataset/3D-FRONT/3D-FRONT/'


main(directories_100, False, JSON_3d_front_folder_path, True)
main(directories_before_after_adjust, False, JSON_3d_front_folder_path, True)
main(directories_each_graph_type, False, JSON_3d_front_folder_path)
main(directories_instruct_scene, False, JSON_3d_front_folder_path)
