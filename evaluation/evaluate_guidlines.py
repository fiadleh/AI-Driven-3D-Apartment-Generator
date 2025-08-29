#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Firas Adleh
Date: 2024-09-01
Version: 1.0
Description: Uses Blender API to check one 3D apartments in a given folder for anthropometric constraint violations and return the results as json.
"""

import json
import os
import bpy
import mathutils
import bmesh

import bpy
import os
import mathutils
import sys



# Takes into account the full transformation stack, including any parent transformations and local transformations.
def get_world_location(obj):
    return obj.matrix_world @ obj.location

# Function to check if two objects collide (bounding box based)
def check_collision(obj1, obj2):
    bbox1 = [obj1.matrix_world @ mathutils.Vector(corner) for corner in obj1.bound_box]
    bbox2 = [obj2.matrix_world @ mathutils.Vector(corner) for corner in obj2.bound_box]
    
    min1 = mathutils.Vector(map(min, zip(*bbox1)))
    max1 = mathutils.Vector(map(max, zip(*bbox1)))
    min2 = mathutils.Vector(map(min, zip(*bbox2)))
    max2 = mathutils.Vector(map(max, zip(*bbox2)))

    # Check if bounding boxes intersect
    return all([max1[i] > min2[i] and min1[i] < max2[i] for i in range(3)])

# Helper function to scale an object
def scale_object(obj, scale_vector):
    # Calculate the bounding box dimensions and center
    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    bbox_min = mathutils.Vector(map(min, zip(*bbox_corners)))
    bbox_max = mathutils.Vector(map(max, zip(*bbox_corners)))
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_dimensions = bbox_max - bbox_min


    # Create the new bounding box dimensions with the given offset
    new_dimensions = mathutils.Vector((
        bbox_dimensions.x + scale_vector[0],#)/bbox_info[0],
        bbox_dimensions.y + scale_vector[1],#)/bbox_info[1],
        bbox_dimensions.z + scale_vector[2]#)/bbox_info[2]
    ))
    
    #obj.scale = mathutils.Vector((obj.scale.x * scale_vector[0], 
    #                              obj.scale.y * scale_vector[1], 
    #                              obj.scale.z * scale_vector[2]))
    obj.scale = mathutils.Vector((new_dimensions[0] / bbox_dimensions[0], 
                                  new_dimensions[1] / bbox_dimensions[1], 
                                  new_dimensions[2] / bbox_dimensions[2]))
                                  
    # update the scene to apply the rotation
    bpy.context.view_layer.update()

def set_pivot_to_bbox_center(obj):
    """
    Move the pivot point of an object to the center of its bounding box.

    :param obj: Blender object
    """
    # Deselect all objects
    bpy.ops.object.select_all(action='DESELECT')
    
    # Select the target object
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # Update the view layer to ensure accurate bounding box calculations
    bpy.context.view_layer.update()
    
    # Calculate the bounding box center
    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    min_corner = mathutils.Vector((float('inf'), float('inf'), float('inf')))
    max_corner = mathutils.Vector((float('-inf'), float('-inf'), float('-inf')))
    
    for corner in bbox_corners:
        min_corner.x = min(min_corner.x, corner.x)
        min_corner.y = min(min_corner.y, corner.y)
        min_corner.z = min(min_corner.z, corner.z)
        max_corner.x = max(max_corner.x, corner.x)
        max_corner.y = max(max_corner.y, corner.y)
        max_corner.z = max(max_corner.z, corner.z)
    
    bbox_center = (min_corner + max_corner) / 2
    
    # Calculate the offset from the object's origin to the bounding box center
    offset = bbox_center - obj.location

    # Move the object's origin to the bounding box center
    obj.location = bbox_center
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='BOUNDS')
    
    # Correct the object's location to maintain its original position
    obj.location -= offset

    # Return the bounding box center for reference
    return bbox_center


# Function to check for collision violations based on constraints
def check_constraint(obj, scale_vector, collision_objects, exclude_names=None):
    col_threshold = 0.05
    #violations = 0
    #num_cases = len(constraint_objects)  # Number of objects to check
    # Scale the object
    original_scale = obj.scale.copy()
    scale_object(obj, scale_vector)
    
    # Check for collisions
    for other_obj in collision_objects:
        if other_obj.name == obj.name or exclude_names and any(exclude in other_obj.name for exclude in exclude_names):
            continue
        if check_collision(obj, other_obj):
            overlap = calculate_overlap_percentage(obj, other_obj)
            if overlap[0] > col_threshold and overlap[1] > col_threshold:
                print(f"Violation: {obj.name} collided with {other_obj.name}")
                # Restore original scale
                obj.scale = original_scale
                return 1
        
    # Restore original scale
    obj.scale = original_scale
    return 0

def calculate_mesh_volume(obj):
    """
    Calculate the volume of a given mesh object.

    :param obj: Blender mesh object
    :return: Volume of the mesh
    """
    mesh = obj.to_mesh(preserve_all_data_layers=True, depsgraph=bpy.context.evaluated_depsgraph_get())
    bm = bmesh.new()
    bm.from_mesh(mesh)
    volume = bm.calc_volume()
    bm.free()
    #bpy.data.meshes.remove(mesh)
    
    return volume
    
def calculate_overlap_percentage(obj1, obj2):
    """
    Calculate the percentage of overlapping meshes between two Blender objects.

    :param obj1: First Blender object
    :param obj2: Second Blender object
    :return: Tuple with percentages of overlap relative to each object (overlap_percentage_obj1, overlap_percentage_obj2)
    """
    # Ensure both objects are meshes
    if obj1.type != 'MESH' or obj2.type != 'MESH':
        #raise TypeError("Both objects must be of type 'MESH'")
        return 0,0
    
    # Duplicate the objects to avoid modifying the original objects
    obj1_copy = obj1.copy()
    obj2_copy = obj2.copy()
    obj1_copy.data = obj1_copy.data.copy()
    obj2_copy.data = obj2_copy.data.copy()
    bpy.context.collection.objects.link(obj1_copy)
    bpy.context.collection.objects.link(obj2_copy)
    
    # Calculate volume of the original objects
    obj1_volume = calculate_mesh_volume(obj1_copy)
    obj2_volume = calculate_mesh_volume(obj2_copy)
    
    # Perform boolean intersection operation
    bpy.context.view_layer.objects.active = obj1_copy
    bpy.ops.object.modifier_add(type='BOOLEAN')
    boolean_modifier = obj1_copy.modifiers["Boolean"]
    boolean_modifier.operation = 'INTERSECT'
    boolean_modifier.object = obj2_copy
    bpy.ops.object.modifier_apply(modifier="Boolean")
    
    # Calculate the volume of the intersecting part
    overlap_volume = calculate_mesh_volume(obj1_copy)
    
    # Remove the copies from the scene
    bpy.data.objects.remove(obj1_copy, do_unlink=True)
    bpy.data.objects.remove(obj2_copy, do_unlink=True)
    
    # Calculate percentage overlap
    overlap_percentage_obj1 = (overlap_volume / obj1_volume) * 100 if obj1_volume > 0 else 0
    overlap_percentage_obj2 = (overlap_volume / obj2_volume) * 100 if obj2_volume > 0 else 0
    
    return overlap_percentage_obj1, overlap_percentage_obj2



def process_glb_file(file_path):
    

    # Initialize counters for all object types and violations
    bed_cases = bed_violations = chair_cases = chair_violations = cabinet_cases = cabinet_violations = coffee_table_cases = coffee_table_violations = dining_table_cases = dining_table_violations = other_table_cases = other_table_violations = nightstand_violations = nightstand_cases = 0

    # Clear the scene before importing new objects
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Import the .glb file
    bpy.ops.import_scene.gltf(filepath=file_path)
    
    # Get all objects in the scene
    scene_objects = {obj.name: obj for obj in bpy.context.scene.objects}

    for obj in bpy.context.scene.objects:
        # Check objects as done previously (code omitted for brevity)

        # Example for one check
        if "bed" in obj.name.lower():
            set_pivot_to_bbox_center(obj)
            bed_cases += 1
            objects_without_nightstand = [obj for obj in scene_objects.values() if "nightstand" not in obj.name.lower() and "object_" in obj.name.lower()]
            scale_vector_bed = (0.915, 0.0, 0.915)
            is_collided = check_constraint(obj, scale_vector_bed, objects_without_nightstand)
            bed_violations += is_collided
        
        if ("chair" in obj.name.lower() or 'stool' in obj.name.lower()) and not "dining" in obj.name.lower():
            set_pivot_to_bbox_center(obj)
            chair_cases += 1
            objects_to_check = [o for o in scene_objects.values() if o.name.startswith("object_")]
            scale_vector_chair = (0.762, 0, 0.762)
            is_collided = check_constraint(obj, scale_vector_chair, objects_to_check)
            chair_violations += is_collided

        if 'cabinet' in obj.name.lower() or 'shelf' in obj.name.lower() or 'wardrobe' in obj.name.lower() or 'desk' in obj.name.lower():
            set_pivot_to_bbox_center(obj)
            cabinet_cases += 1
            objects_to_check = [o for o in scene_objects.values() if o.name.startswith("object_")]
            scale_vector_cabinet = (0.61, 0, 0.61)
            is_collided = check_constraint(obj, scale_vector_cabinet, objects_to_check)
            cabinet_violations += is_collided

        if 'dining' in obj.name.lower() and 'table' in obj.name.lower():
            set_pivot_to_bbox_center(obj)
            dining_table_cases += 1
            objects_to_check = [o for o in scene_objects.values() if o.name.startswith("object_") and "chair" not in o.name.lower()]
            scale_vector_dining_table = (0.915, 0, 0.915)
            is_collided = check_constraint(obj, scale_vector_dining_table, objects_to_check)
            dining_table_violations += is_collided

        if 'coffee' in obj.name.lower() and 'table' in obj.name.lower():
            set_pivot_to_bbox_center(obj)
            coffee_table_cases += 1
            objects_to_check = [o for o in scene_objects.values() if o.name.startswith("object_")]
            scale_vector_coffee_table = (0.38, 0, 0.38)
            is_collided = check_constraint(obj, scale_vector_coffee_table, objects_to_check)
            if is_collided > 0:
                coffee_table_violations += is_collided
            #else:
            #    scale_vector_coffee_table = (0.46, 0, 0.46)
            #    is_collided = check_constraint(obj, scale_vector_coffee_table, objects_to_check)
            #    if is_collided == 0:
            #        coffee_table_violations += 1

        if 'table' in obj.name.lower() and not 'coffee' in obj.name.lower() and not 'dining' in obj.name.lower():
            set_pivot_to_bbox_center(obj)
            other_table_cases += 1
            objects_to_check = [o for o in scene_objects.values() if o.name.startswith("object_") and ('chair' in o.name.lower() or 'stool' in o.name.lower())]
            scale_vector_other_table = (0.305, 0, 0.305)
            is_collided = check_constraint(obj, scale_vector_other_table, objects_to_check)
            if is_collided == 0:
                scale_vector_other_table = (1.0, 0, 1.0)
                is_collided = check_constraint(obj, scale_vector_other_table, objects_to_check)
                if is_collided > 0:
                    other_table_violations += 1

        if 'nightstand' in obj.name.lower():
            set_pivot_to_bbox_center(obj)
            nightstand_cases += 1
            objects_to_check = [o for o in scene_objects.values() if o.name.startswith("object_") and 'bed' in o.name.lower()]
            scale_vector_nightstand = (0.60, 0, 0.60)
            is_collided = check_constraint(obj, scale_vector_nightstand, objects_to_check)
            if is_collided == 0:
                nightstand_violations += 1

    # Return results as a dictionary
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

if __name__ == "__main__":
    # Get the .glb file path from the command line arguments
    file_path = sys.argv[-1]
    #print('+++++++++++++++++++++++++++++++++++++++++++++++')
    # Process the file and get the results
    results = process_glb_file(file_path)
    
    # Print results as JSON (to be captured by the main script)
    print(json.dumps(results))