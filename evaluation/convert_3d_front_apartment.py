#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Firas Adleh
Date: 2024-09-01
Version: 1.0
Description: Uses Blender API to convert one apartment created by  3D-FRONT-ToolBox 'json2obj.py' to a final one gltf file for each apartment containing all furniture models and meshes.
"""

import argparse
import sys
import time
import bpy
import os


def get_or_create_collection(collection_name, parent_collection):
    # Check if the collection exists
    if collection_name in bpy.data.collections:
        collection = bpy.data.collections[collection_name]
    else:
        # Create a new collection
        collection = bpy.data.collections.new(collection_name)
        parent_collection.children.link(collection)
    return collection

# import a 3D-Front house converted using jspn2obj into a directory
def import_obj_files(directory, parent_collection):
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        # Relative path from the base directory
        rel_path = os.path.relpath(root, directory)
        path_parts = rel_path.split(os.sep)

        # Create or get the main collections for the current sub-directory
        current_collection = parent_collection
        for part in path_parts:
            if part:  # Ensure it's not the base directory
                current_collection = get_or_create_collection(part, current_collection)

        for file in files:
            # Check if the file is an .obj file
            if file.endswith(".obj"):
                obj_path = os.path.join(root, file)
                mtl_path = obj_path.replace('.obj', '.mtl')
                
                # Import the .obj file
                bpy.ops.wm.obj_import(filepath=obj_path, directory=root)
                
                # Move the imported objects to the current collection
                imported_objects = [obj for obj in bpy.context.selected_objects]
                for obj in imported_objects:
                    if file[0].isnumeric() or file.startswith("meshe"):
                        obj.name = 'room_boundary_'+obj.name
                    elif file.startswith("solid") or file.startswith("shadow"):
                        obj.name = 'ignore_'+obj.name
                    elif any(furniture_term in obj.name.lower() for furniture_term in ['bed', 'cabinet', 'shelf','desk', 'table', 'sofa', 'stand','light','lamp','couch','wardrobe','chair','stool']):
                        obj.name = 'object_'+obj.name
                    # Unlink from the default collection
                    #bpy.context.scene.collection.objects.unlink(obj)
                    obj.users_collection[0].objects.unlink(obj)
                    # Link to the current collection
                    current_collection.objects.link(obj)

if __name__ == "__main__":
    print('\n ************************************************************************\n \
              ************************************************************************\n \
              ******************      Convert 3d front         ******************\n \
              ***********************************************************************')
    
    start_i = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--jspn2obj_house_path", type=str, required=True)
    parser.add_argument("--glb_save_path", type=str, required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)


    # Initialize the import with the root collection
    root_collection = bpy.context.scene.collection
    import_obj_files(args.jspn2obj_house_path, root_collection)

    # export to gltf
    bpy.ops.export_scene.gltf(filepath=args.glb_save_path, export_lights=False)


# example terminal call
#/home/fpc/blender-4.1.1-linux-x64/blender -b -P /home/user2024/Documents/repos/ai-driven-3d-apartment-generator/import_3d_front_apartment.py -- --jspn2obj_house_path /home/fpc/Uni/RL/repos/3D-FRONT-ToolBox/out/20365dbe-40b9-4cc9-b0ef-d7d6b89799ab --glb_save_path /home/fpc/Uni/RL/repos/3D-FRONT-ToolBox/out/20365dbe-40b9-4cc9-b0ef-d7d6b89799ab/scene.glb