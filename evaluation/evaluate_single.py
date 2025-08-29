#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Firas Adleh
Date: 2024-09-01
Version: 1.0
Description: Uses Blender API to evaluate the usability of a group of 3D apartments (generated or from 3D-Front).
"""

import argparse
import json
import math
import os
import sys
import bpy

from mathutils import Vector
import bmesh
from mathutils.bvhtree import BVHTree
import numpy as np
import time
import mathutils
import argparse
import json
import os
import sys
import bpy
from mathutils import Vector
import bmesh
from mathutils.bvhtree import BVHTree
import numpy as np
import time
import mathutils


def get_bbox_min_max(obj):
    local_bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_x = min(corner.x for corner in local_bbox_corners)
    max_x = max(corner.x for corner in local_bbox_corners)
    min_y = min(corner.y for corner in local_bbox_corners)
    max_y = max(corner.y for corner in local_bbox_corners)
    return min_x, max_x, min_y, max_y

def get_globat_min_max(prefix_to_check):
    # Initialize the min/max bounds with extreme values
    global_min_x = float('inf')
    global_max_x = float('-inf')
    global_min_y = float('inf')
    global_max_y = float('-inf')

    # Iterate over all objects to get the min/max x and y from bounding boxes
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and not obj.name.startswith(prefix_to_check):
            min_x, max_x, min_y, max_y = get_bbox_min_max(obj)
            global_min_x = min(global_min_x, min_x)
            global_max_x = max(global_max_x, max_x)
            global_min_y = min(global_min_y, min_y)
            global_max_y = max(global_max_y, max_y)

    # Inform the user if no mesh objects were found
    if global_min_x == float('inf'):
        print("No mesh objects found in the scene.")
        return

    # Enlarge the area to avoid unreal dead zones
    global_min_x -= 1
    global_max_x += 1
    global_min_y -= 1
    global_max_y += 1

    return global_min_x, global_max_x, global_min_y, global_max_y


def spawn_grid_of_cylinders(prefix_to_check, interval, radius=0.25, height= 1.75):
    global_min_x, global_max_x, global_min_y, global_max_y = get_globat_min_max(prefix_to_check)
    if IS_DEBUG:
        print('file size: ',global_min_x, global_max_x, global_min_y, global_max_y)
    
    # skip very large files + files with problems/bad scale
    if abs(global_min_x - global_max_x) > 40 or  abs(global_min_x - global_max_x) > 40:
        return False
    
    # Generate a grid of spheres within the min/max bounds
    #z = 0.95
    z = height/2 + 0.01 # add one cm to avoid collisions with floor       0.95
    x_values = list(np.arange(int(global_min_x), int(global_max_x) + 1, interval))
    y_values = list(np.arange(int(global_min_y), int(global_max_y) + 1, interval))

    for x in x_values:
        for y in y_values:
            #bpy.ops.mesh.primitive_uv_sphere_add(location=(x, y, z))
            bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=radius, depth=height, location=(x, y, z))
            #bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, z))
            sphere = bpy.context.object
            sphere.name = f'{prefix_to_check}{x}_{y}'
            #sphere.scale = (ball_radius, ball_radius, 0.2)
            #sphere.scale = (ball_radius, ball_radius, ball_radius)
            #print(f"Sphere added at location: ({x}, {y}, {z}) with scale ({ball_radius}, {ball_radius}, {ball_radius})")
    return True

            
def get_bounding_box(obj):
    """Get the bounding box of the object in world coordinates."""
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_corner = Vector((min(corner[0] for corner in bbox_corners),
                         min(corner[1] for corner in bbox_corners),
                         min(corner[2] for corner in bbox_corners)))
    max_corner = Vector((max(corner[0] for corner in bbox_corners),
                         max(corner[1] for corner in bbox_corners),
                         max(corner[2] for corner in bbox_corners)))
    return min_corner, max_corner

def is_colliding(min1, max1, min2, max2):
    """Check if two bounding boxes are colliding."""
    return (min1[0] <= max2[0] and max1[0] >= min2[0] and
            min1[1] <= max2[1] and max1[1] >= min2[1] and
            min1[2] <= max2[2] and max1[2] >= min2[2])

def resolve_collision(obj1, obj2, term, step=0.1, max_tries=10):
    """Resolve collision by moving obj1 away from obj2 only if the name includes the given term."""
    if term in obj1.name:
        return obj1
        #print(f"Deleted colliding ball: {obj1.name}")
        #bpy.data.objects.remove(obj1)
    if term in obj2.name:
        #print(f"Deleted colliding ball: {obj2.name}")
        #bpy.data.objects.remove(obj2)
        return obj2
    

def change_object_color(obj_name, color_id):
    # Define a color mapping based on integers
    color_map = {
        0: (1.0, 0.0, 0.0, 1.0),  # Red
        1: (0.0, 1.0, 0.0, 1.0),  # Green
        2: (0.0, 0.0, 1.0, 1.0),  # Blue
        3: (1.0, 1.0, 0.0, 1.0),  # Yellow
        4: (1.0, 0.5, 0.0, 1.0),  # Orange
        5: (0.5, 0.0, 0.5, 1.0),  # Purple
        6: (0.0, 1.0, 1.0, 1.0),  # Cyan
        7: (1.0, 0.0, 1.0, 1.0),  # Magenta
        8: (0.5, 0.5, 0.5, 1.0),  # Gray
        9: (1.0, 1.0, 1.0, 1.0)   # White
    }
    
    # Ensure the color_id is within the bounds of the color_map
    if color_id not in color_map:
        print(f"Color ID {color_id} is not valid. Using default color (white).")
        color_id = 9  # Default to white if the color_id is not valid

    color = color_map[color_id]
    
    # Check if the object exists
    if obj_name not in bpy.data.objects:
        print(f"Object '{obj_name}' not found.")
        return
    
    obj = bpy.data.objects[obj_name]
    
    # Ensure the object has a material
    if not obj.data.materials:
        mat = bpy.data.materials.new(name=f"{obj_name}_Material")
        obj.data.materials.append(mat)
    else:
        mat = obj.data.materials[0]
    
    # Enable 'Use Nodes' for the material
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    
    # Clear existing nodes
    for node in nodes:
        nodes.remove(node)
    
    # Create new Principled BSDF node
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = 0, 0
    
    # Set the base color of the BSDF node
    bsdf.inputs['Base Color'].default_value = color
    
    # Create Material Output node
    material_output = nodes.new(type='ShaderNodeOutputMaterial')
    material_output.location = 200, 0
    
    # Link the BSDF node to the Material Output node
    mat.node_tree.links.new(bsdf.outputs['BSDF'], material_output.inputs['Surface'])

    #print(f"Changed color of '{obj_name}' to {color}.")

        


def delete_colliding_balls1(prefix_to_check):
    # Collect all non-ball mesh objects
    non_ball_meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH' and not obj.name.startswith(prefix_to_check)]

    # Check each ball for collisions
    for ball in [obj for obj in bpy.data.objects if obj.name.startswith(prefix_to_check)]:
        ball_min_x, ball_max_x, ball_min_y, ball_max_y = get_bbox_min_max(ball)

        for mesh in non_ball_meshes:
            mesh_min_x, mesh_max_x, mesh_min_y, mesh_max_y = get_bbox_min_max(mesh)

            # Check for collision using bounding box overlap
            if (ball_min_x < mesh_max_x and ball_max_x > mesh_min_x and
                ball_min_y < mesh_max_y and ball_max_y > mesh_min_y):
                print(f"Deleted colliding ball: {ball.name}")
                bpy.data.objects.remove(ball)
                
                break
    
def check_collision_bbox(obj1, obj2):
    #print('check_collision_bbox : ', obj1.name, obj2.name)
    """Check if two objects collide based on their bounding boxes."""
    # Get the bounding boxes of both objects
    bbox1 = [obj1.matrix_world @ Vector(corner) for corner in obj1.bound_box]
    bbox2 = [obj2.matrix_world @ Vector(corner) for corner in obj2.bound_box]

    # Check if the bounding boxes intersect
    return (
        min([v.x for v in bbox1]) <= max([v.x for v in bbox2]) and
        max([v.x for v in bbox1]) >= min([v.x for v in bbox2]) and
        min([v.y for v in bbox1]) <= max([v.y for v in bbox2]) and
        max([v.y for v in bbox1]) >= min([v.y for v in bbox2]) and
        min([v.z for v in bbox1]) <= max([v.z for v in bbox2]) and
        max([v.z for v in bbox1]) >= min([v.z for v in bbox2])
    )

def draw_bounding_box(obj):
    """Create a wireframe cube representing the bounding box of an object."""
    # Get the object's bounding box vertices
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]

    # Create a new mesh for the bounding box
    mesh = bpy.data.meshes.new(f"BBOX_{obj.name}_bbox_mesh")
    bbox_obj = bpy.data.objects.new(f"{obj.name}_bbox", mesh)

    # Link the object to the current collection
    bpy.context.collection.objects.link(bbox_obj)

    # Create the vertices and edges for the bounding box
    vertices = [(v.x, v.y, v.z) for v in bbox]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom edges
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top edges
        (0, 4), (1, 5), (2, 6), (3, 7)   # Side edges
    ]

    # Create the mesh
    mesh.from_pydata(vertices, edges, [])
    mesh.update()

    # Set the bounding box object to wireframe mode
    bbox_obj.display_type = 'WIRE'


def create_bvhtree_from_object(obj):
    """Create a BVH tree from the given object."""
    # Get the mesh data of the object
    mesh = obj.to_mesh()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.transform(obj.matrix_world)  # Transform mesh to world space
    
    # Create BVH tree
    bvh = BVHTree.FromBMesh(bm)
    
    # Clean up
    bmesh.ops.delete(bm, geom=bm.faces)
    bm.free()
    #bpy.data.meshes.remove(mesh)
    
    return bvh

def check_collision_mesh(obj1, obj2, debug = False):
    if obj1.type != 'MESH' or obj2.type != 'MESH':
        return False
    """Check if two objects collide based on their actual meshes."""
    bvh1 = create_bvhtree_from_object(obj1)
    bvh2 = create_bvhtree_from_object(obj2)
    
    if bvh1 and bvh2:
        overlap = bvh1.overlap(bvh2)
        if debug:
            print('debug overlap: ', overlap)
        return len(overlap) > 0
    return False


def check_collision(obj, colliders):
    for collider in colliders:
        if check_collision_bbox(obj, collider) and check_collision_mesh(obj, collider):
            return True
    return False

def find_collision_free_position(obj, offsets, colliders, degrees_per_step = -1):
    original_location = obj.location.copy()
    
    directions = [
        mathutils.Vector((1, 0, 0)),
        mathutils.Vector((-1, 0, 0)),
        mathutils.Vector((0, 1, 0)),
        mathutils.Vector((0, -1, 0)),
        mathutils.Vector((1, 1, 0)),
        mathutils.Vector((1, -1, 0)),
        mathutils.Vector((-1, 1, 0)),
        mathutils.Vector((-1, -1, 0))
    ]

    if degrees_per_step > 0:
        degrees_per_step = abs(degrees_per_step) % 360
        # Calculate the total number of steps
        steps = int(360 / degrees_per_step)
    else:
        steps = 1

    for step in range(steps):
        if degrees_per_step > 0:
            # Calculate the rotation in radians
            angle = math.radians(degrees_per_step)
            
            # Rotate the object around the Z-axis
            obj.rotation_euler[2] += angle

            
            # Optionally update the scene to apply the rotation
            bpy.context.view_layer.update()

        for offset in offsets:
            for direction in directions:
                new_location = original_location + direction * offset
                obj.location = new_location
                bpy.context.view_layer.update()  # Update the scene to get accurate collision detection
                
                if not check_collision(obj, colliders):
                    #print('collision solved : ', obj.name, original_location, new_location)
                    return new_location
                #else:
                    #print('collision : ', obj.name, original_location, new_location)
            
    obj.location = original_location  # Reset to original location if no collision-free position is found
    return original_location



def delete_colliding_objects(prefix, degree_per_rotation = -1):
    """delete colliding objects with a given prefix."""
    objects_to_check = [obj for obj in bpy.data.objects if obj.name.startswith(prefix)]
    other_objs = [obj for obj in bpy.data.objects if not obj.name.startswith('room_volume_') and not obj.name.startswith('door_') and not obj.name.startswith('cube_best_') and not obj.name.startswith('cube_center_')]
    collided_objects = set()

    for obj in objects_to_check:
        for other_obj in other_objs: #bpy.data.objects:
            if obj == other_obj:
                continue
            if check_collision_bbox(obj, other_obj) and check_collision_mesh(obj, other_obj):
                #print('################ collision: ', obj.name, other_obj.name, "###############")
                old_pos = obj.location.copy()
                new_pos = find_collision_free_position(obj, [0.05, 0.10, 0.20], other_objs, degree_per_rotation)
                if old_pos == new_pos:
                    if IS_DEBUG:
                        print('deleted because of collision : ', obj.name, other_obj.name)
                    # add object to collided
                    collided_objects.add(obj)

                    # delete the collided ball
                    bpy.data.objects.remove(obj)
                    other_objs.remove(obj)
                    break

    #for obj in collided_objects:
    #    draw_bounding_box(obj)

def make_grid_array(prefix_to_check, interval):

    global_min_x, global_max_x, global_min_y, global_max_y = get_globat_min_max(prefix_to_check)


    x_values = list(np.arange(int(global_min_x), int(global_max_x) + 1, interval))
    y_values = list(np.arange(int(global_min_y), int(global_max_y) + 1, interval))
    rows, cols = (len(x_values), len(y_values))
    final_array = [[0 for _ in range(cols)] for _ in range(rows)]
    ids_to_names_array = [[0 for _ in range(cols)] for _ in range(rows)]
    #print(final_array)
    #print(x_values)
    #print(y_values)
    bpy.context.view_layer.update()  # Update the scene to get accurate collision detection
    for i_x, x in enumerate(x_values):
        for i_y, y in enumerate(y_values):
            cylinder_name = f'{prefix_to_check}{x}_{y}'
            ids_to_names_array[i_x][i_y] = cylinder_name
            if cylinder_name in bpy.data.objects:
                final_array[i_x][i_y] = 1
                #print(sphere_name, ' found')
            #else:
                #print(sphere_name, ' not found')
                #final_array[i_x][i_y] = 3
                
    #print(final_array)
    return final_array, ids_to_names_array            


def check_objects_between(obj1, obj2, steps=5):
    # Get the world coordinates of the two objects
    loc1 = obj1.location
    loc2 = obj2.location
    
    # Calculate the direction vector from obj1 to obj2
    direction = loc2 - loc1
    distance = direction.length
    
    # Normalize the direction vector
    direction.normalize()
    
    # Get the current Depsgraph
    depsgraph = bpy.context.evaluated_depsgraph_get()
    
    # Raycast from obj1 to obj2 in steps
    for step in range(steps):
        factor = step / steps
        start_point = loc1.lerp(loc2, factor)
        
        # Perform a raycast at each step
        result, location, normal, index, hit_object, matrix = bpy.context.scene.ray_cast(
            depsgraph, start_point, direction, distance = distance * (1 - factor)
        )
        
        # Check if the raycast hit an object that is not obj1 or obj2
        if result and hit_object != obj1 and hit_object != obj2:
            return True, hit_object
    
    # If no objects are detected between obj1 and obj2
    return False, None

def check_surrounding_objects(obj, grid_prefix, num_rays=8, max_distance=10.0):
    # Get the current Depsgraph
    depsgraph = bpy.context.evaluated_depsgraph_get()
    
    # Get the world location of the object
    origin = obj.location
    
    # Calculate the directions for the rays
    directions = []
    for i in range(num_rays):
        angle = (2 * math.pi / num_rays) * i
        x = math.cos(angle)
        y = math.sin(angle)
        direction = mathutils.Vector((x, y, 0)).normalized()
        directions.append(direction)
    
    # Cast rays in all directions
    for direction in directions:
        result, location, normal, index, hit_object, matrix = bpy.context.scene.ray_cast(
            depsgraph, origin, direction, distance=max_distance
        )
        
        # Check if the raycast hit an object that is not the original object
        if result and hit_object != obj:# and grid_prefix in hit_object.name:
            return True, hit_object
    
    # If no objects are detected within the specified distance
    return False, None

def is_object_inside_bboxes(target_obj, objects_list):
    # Get the target object's location in world coordinates
    target_location = target_obj.location
    
    # Iterate over each object in the list
    for obj in objects_list:
        # Get the bounding box coordinates in world space
        bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
        
        # Find the minimum and maximum coordinates of the bounding box
        min_corner = mathutils.Vector((
            min(corner.x for corner in bbox_corners),
            min(corner.y for corner in bbox_corners),
            min(corner.z for corner in bbox_corners)
        ))
        max_corner = mathutils.Vector((
            max(corner.x for corner in bbox_corners),
            max(corner.y for corner in bbox_corners),
            max(corner.z for corner in bbox_corners)
        ))
        
        # Check if the target object's location is within this bounding box
        if (min_corner.x <= target_location.x <= max_corner.x and
            min_corner.y <= target_location.y <= max_corner.y and
            min_corner.z <= target_location.z <= max_corner.z):
            return True, obj
    
    # If the target object is not inside any of the bounding boxes
    return False, None



def flood_fill(names_array, grid_prefix, grid, x, y, island_id):
    #print('flood_fill : ', x, y)
    # Base cases for flood fill
    if x < 0 or x >= len(grid) or y < 0 or y >= len(grid[0]):
        return 0
    if grid[x][y] != 1:
        return 0
    
    # Mark the current cell as visited (by setting it to the island_id)
    grid[x][y] = island_id
    
    # Initialize island size counter
    size = 1
    
    current_cell_name = names_array[x][y]
    # Recursively call flood fill for all adjacent cells and accumulate the island size
    #if grid[x + 1][y] == 1:
    #    # check if objects are blocking the space between two cells
    #    cell_name = f'{prefix}{x + 1}_{y}'
    #    if cell_name in bpy.data.objects:
    #        is_blocked, blocking_object = check_objects_between(bpy.data.objects[base_name], bpy.data.objects[cell_name])
    #        if not is_blocked:
    #            size += flood_fill(prefix, grid, x + 1, y, island_id)  # Down
    blocking_objects = []
    neighbors_checked = 0
    for offset in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),       (x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1), (x - 1, y - 1)]: # Down, Up, Right, Left, ---- (Down Right), (Down Left), (Up Right), (Up Left)
        if offset[0] >= 0 and offset[0] < len(grid) and offset[1] >= 0 and offset[1] < len(grid[0]) and grid[offset[0]][offset[1]] == 1:
            # check if objects are blocking the space between two cells
            new_cell_name =  names_array[offset[0]][offset[1]] #f'{prefix}{offset[0]}_{offset[1]}'
            if new_cell_name in bpy.data.objects:
                neighbors_checked += 1
                is_blocked, blocking_object = check_objects_between(bpy.data.objects[current_cell_name], bpy.data.objects[new_cell_name])
                if blocking_object != None:
                    blocking_objects.append(blocking_object)
                if not is_blocked or grid_prefix in blocking_object.name:
                    #print(x, y,' -> rec flood_fill: ', offset)
                    size += flood_fill(names_array, grid_prefix, grid, offset[0], offset[1], island_id)  # Up
            #else:
                #print(new_cell_name, ' not found !!!!!!!!!!!!')

    

    if neighbors_checked == 0:
        # check if inside another object
        is_inside, container_object = is_object_inside_bboxes(bpy.data.objects[current_cell_name], [obj for obj in bpy.data.objects if grid_prefix not in obj.name and not obj.name.startswith('room_volume_') and not obj.name.startswith('door_') and not obj.name.startswith('cube_best_') and not obj.name.startswith('cube_center_')])
        #check_surrounding_objects(bpy.data.objects[current_cell_name], grid_prefix, num_rays=32, max_distance=20.0)
        if is_inside:
            if IS_DEBUG:
                print('object is inside another object', current_cell_name, container_object.name)   
            blocking_objects.append(container_object)
            blocking_objects.append(container_object)
        else:
            if IS_DEBUG:
                print('object is not inside another object ', current_cell_name)   

    unique_blocking_objects = set(blocking_objects)

    # check if the same blocking object from all directions -> currect object is inside the blocking_object
    # if object is inside then it must be deleted
    if len(blocking_objects) > 1 and len(unique_blocking_objects) == 1:
        if IS_DEBUG:
            print('object deleted because of raytracing: ', current_cell_name, blocking_objects, unique_blocking_objects)
        grid[x][y] = 0
        bpy.data.objects.remove(bpy.data.objects[current_cell_name], do_unlink=True)
        size = 0
    else:
        if IS_DEBUG:
            print('object stays: ', current_cell_name, neighbors_checked)   

    if IS_DEBUG:
        print(current_cell_name, len(blocking_objects), len(unique_blocking_objects))
    #size += flood_fill(grid, x - 1, y, island_id)  # Up
    #size += flood_fill(grid, x, y + 1, island_id)  # Right
    #size += flood_fill(grid, x, y - 1, island_id)  # Left
    
    return size

def count_islands(names_array, grid_prefix, grid):
    """ Search for connected areas of objects with prefix and return the results"""
    if not grid:
        return 0
    
    num_islands = 0
    rows = len(grid)
    cols = len(grid[0])
    
    total_land = sum(row.count(1) for row in grid)
    total_cells = rows * cols
    island_sizes = []
    
    island_id = 2  # Start island IDs from 2 since 0 and 1 are already used
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1:
                
                # Use flood fill to mark the entire island and get its size
                size = flood_fill(names_array, grid_prefix, grid, i, j, island_id)
                if size > 0:
                    island_sizes.append(size)
                    island_id += 1
                    # We found an unvisited land cell, start a new island
                    num_islands += 1
    
    if IS_DEBUG:
        # Print the grid with island IDs
        print("Grid with island IDs:")
        for row in grid:
            print(row)
        
        # Calculate and print the percentage of each island
        print("\nIsland percentages:")
        for idx, size in enumerate(island_sizes):
            island_land_percentage = (size / total_land) * 100
            island_total_percentage = (size / total_cells) * 100
            print(f"Island {idx + 1} (ID {idx + 2}):")
            print(f"  Percentage of total land: {island_land_percentage:.2f}%")
            print(f"  Percentage of total area (land + water): {island_total_percentage:.2f}%")
    
    return num_islands, grid, island_sizes


def colorize_islands(names_array, id_to_island_array):
    ids_to_names_array = 0
    for x in range(len(names_array)):
        for y in range(len(names_array[x])):
            if names_array[x][y] in bpy.data.objects:
                change_object_color(names_array[x][y], id_to_island_array[x][y])
            
def import_glb(glb_file_path):
    """
    Import a .glb file into the Blender scene at a specified 3D position and scale.

    :param glb_file_path: Path to the .glb file
    """
    # Ensure Blender's context is correct and set the position of the 3D cursor
    bpy.ops.object.select_all(action='DESELECT')
    
    print('importing: ' , glb_file_path)
    # Import the .glb file
    bpy.ops.import_scene.gltf(filepath=glb_file_path)

    # Get all imported objects (assuming they are the only ones selected)
    imported_objects = bpy.context.selected_objects

    # Create a new collection for the imported objects
    #collection_name = "ImportedCollection"
    #imported_collection = bpy.data.collections.new(name=collection_name)
    #bpy.context.scene.collection.children.link(imported_collection)


    return imported_objects

'''
start_time = time.time()
# Prefix to check
prefix_to_check = "Ball_"

# Example usage:
spawn_grid_of_balls(prefix_to_check, 0.50, 0.25)  # Generate grid of balls with specific interval and ball radius 254 seconds


# Execute the visualization
visualize_colliding_objects(prefix_to_check)
pos_to_land_array, pos_to_names_array = make_grid_array(prefix_to_check, 0.5)

# increase recursion limit to handle bigger scenes
sys.setrecursionlimit(5000)
num_islands, grid_with_IDs, island_sizes = count_islands(pos_to_land_array)
print("Number of islands:", num_islands)
colorize_islands(pos_to_names_array, grid_with_IDs)
print("--- %s seconds ---" % (time.time() - start_time))
'''

def check_collisions(prefix1, prefix2, col_threshold1 = 0, col_threshold2 = 0):
    # Collect objects based on prefixes but ignore lights and unknown objects
    group1 = [obj for obj in bpy.data.objects if prefix1 in obj.name and not 'None' in obj.name and not 'shadow' in obj.name and not 'Lighting' in obj.name and not 'solid' in obj.name and not 'glass' in obj.name]
    group2 = [obj for obj in bpy.data.objects if prefix2 in obj.name and not 'None' in obj.name and not 'shadow' in obj.name and not 'Lighting' in obj.name and not 'solid' in obj.name and not 'glass' in obj.name]

    # Initialize counters
    group1_count = len(group1)
    group1_collides_within = 0
    group1_collides_with_group2 = 0
    group1_no_collision = 0

    # Check collisions within group1
    for i, obj in enumerate(group1):
        collides_within = False
        collides_with_group2 = False
        #calculate_overlap_percentage(obj1, obj2)
        # Check collisions with other objects in group1
        for other_obj in group1[i+1:]:
            if check_collision_bbox(obj, other_obj):# and check_collision_mesh(obj, other_obj):
                overlap = calculate_overlap_percentage(obj, other_obj)
                if overlap[0] > col_threshold1 and overlap[1] > col_threshold2:
                    if IS_DEBUG:
                        print('collision object: ',obj.name, other_obj.name)
                        #check_collision_mesh(obj, other_obj, True)
                        print(overlap)
                    collides_within = True
                    break

        # Check collisions with objects in group2
        for other_obj in group2:
            if check_collision_bbox(obj, other_obj):# and check_collision_mesh(obj, other_obj):
                overlap = calculate_overlap_percentage(obj, other_obj)
                if overlap[0] > col_threshold1 and overlap[1] > col_threshold2:
                    if IS_DEBUG:
                        print('collision walls: ',obj.name, other_obj.name)
                        #check_collision_mesh(obj, other_obj, True)
                        #print(calculate_overlap_percentage(obj, other_obj))
                        print(overlap)
                    collides_with_group2 = True
                    break

        # Update counters
        if collides_within:
            group1_collides_within += 1
        if collides_with_group2:
            group1_collides_with_group2 += 1
        if not collides_within and not collides_with_group2:
            group1_no_collision += 1

    return (group1_count, group1_collides_within, group1_collides_with_group2, group1_no_collision)



def scale_and_check_collisions(obj, scale_offset, target_prefix):

    set_pivot_to_bbox_center(obj)

    # Store the original scale
    #original_scale = obj.scale.copy()
    temp_obj = create_bigger_bbox_object(obj, scale_offset)
    #set_pivot_to_bbox_center(temp_obj)
    # Select the target object
    #obj.select_set(True)
    #bpy.context.view_layer.objects.active = obj
    
    # Update the view layer to ensure accurate bounding box calculations
    bpy.context.view_layer.update()
    #draw_bounding_box(obj)
    # Scale the object by the given offset in meters
    #obj.scale.x += scale_offset[0]
    #obj.scale.y += scale_offset[1]
    #obj.scale.z += scale_offset[2]

    # Update the view layer to ensure accurate bounding box calculations
    bpy.context.view_layer.update()

    # Check for collisions with objects that have the target prefix
    collision_found = False
    for other_obj in bpy.data.objects:
        if target_prefix in other_obj.name:
            if check_collision_bbox(temp_obj, other_obj):
                collision_found = True
                break

    # Reset the object's scale to its original value
    #obj.scale = original_scale
    #if collision_found:
    bpy.data.objects.remove(temp_obj, do_unlink=True)

    return collision_found



def reachability_check(source_prefix, target_prefix, scale_offset_cm):
    # ignore lights and unknown objects
    source_objects = [obj for obj in bpy.data.objects if source_prefix in obj.name and not 'None' in obj.name and not 'shadow' in obj.name and not 'pendant' in obj.name and not 'Lighting' in obj.name and not 'ceiling' in obj.name and not 'solid' in obj.name and not 'glass' in obj.name]

    total_objects = len(source_objects)
    reachable_objects = []
    unreachable_objects = []
    for obj in source_objects:
        if scale_and_check_collisions(obj, scale_offset_cm, target_prefix):
            #collision_count += 1
            reachable_objects.append(obj.name)
        else:
            unreachable_objects.append(obj.name)
            if IS_DEBUG:
                print('unreachable object: ', obj.name)

    return source_objects, reachable_objects, unreachable_objects

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
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_corner = Vector((float('inf'), float('inf'), float('inf')))
    max_corner = Vector((float('-inf'), float('-inf'), float('-inf')))
    
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


def get_bbox_dimensions(obj):
    """
    Calculate and return the bounding box dimensions of a given object.

    :param obj: Blender object
    :return: Tuple with bounding box dimensions (width, height, depth) and center (center_x, center_y, center_z)
    """
    # Ensure the object is of type 'MESH'
    if obj.type != 'MESH':
        raise TypeError("Object must be of type 'MESH'")
    
    # Update the view layer to get the latest bounding box information
    bpy.context.view_layer.update()
    
    # Get the bounding box corners in world coordinates
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    
    # Initialize min and max coordinates
    min_corner = Vector((float('inf'), float('inf'), float('inf')))
    max_corner = Vector((float('-inf'), float('-inf'), float('-inf')))
    
    # Find the bounding box min and max corners
    for corner in bbox_corners:
        min_corner.x = min(min_corner.x, corner.x)
        min_corner.y = min(min_corner.y, corner.y)
        min_corner.z = min(min_corner.z, corner.z)
        max_corner.x = max(max_corner.x, corner.x)
        max_corner.y = max(max_corner.y, corner.y)
        max_corner.z = max(max_corner.z, corner.z)
    
    # Calculate the dimensions
    width = max_corner.x - min_corner.x
    height = max_corner.y - min_corner.y
    depth = max_corner.z - min_corner.z
    
    # Calculate the center of the bounding box
    center_x = (max_corner.x + min_corner.x) / 2
    center_y = (max_corner.y + min_corner.y) / 2
    center_z = (max_corner.z + min_corner.z) / 2
    
    return (width, height, depth, center_x, center_y, center_z)


def create_bigger_bbox_object(original_obj, offset_m):
    # Calculate the bounding box dimensions and center
    bbox_corners = [original_obj.matrix_world @ Vector(corner) for corner in original_obj.bound_box]
    bbox_min = Vector(map(min, zip(*bbox_corners)))
    bbox_max = Vector(map(max, zip(*bbox_corners)))
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_dimensions = bbox_max - bbox_min

    bbox_info = get_bbox_dimensions(original_obj)
    #bbox_dimensions = Vector((bbox_info[0], bbox_info[1], bbox_info[2]))

    # Create the new bounding box dimensions with the given offset
    new_dimensions = Vector((
        bbox_dimensions.x + offset_m[0],#)/bbox_info[0],
        bbox_dimensions.y + offset_m[1],#)/bbox_info[1],
        bbox_dimensions.z + offset_m[2]#)/bbox_info[2]
    ))
    #print(original_obj.name)
    #print(bbox_dimensions.x , offset_m[0] ,bbox_info[0] ,'  ----->', new_dimensions[0])
    #print(bbox_dimensions.y , offset_m[1] ,bbox_info[1] ,'  ----->', new_dimensions[1])
    #print(bbox_dimensions.z , offset_m[2] ,bbox_info[2] ,'  ----->', new_dimensions[2])


    # Create a new mesh for the bounding box
    mesh = bpy.data.meshes.new(name=f"BIG_bbox_{original_obj.name}")
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=original_obj.location)
    #bm = bpy.context.object

    #bm = bpy.ops.mesh.primitive_cube_add
    #bm(size=1.0, location=original_obj.location) #original_obj.location

    # Scale the new mesh to match the new dimensions
    obj = bpy.context.object
    obj.scale = new_dimensions

    obj.name = f"BIG_bbox_{original_obj.name}"

    # Link the new object to the collection
    #bpy.context.collection.objects.link(obj)
    

    return obj


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # Delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # Delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # Delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # Delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


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



def create_extruded_ellipse(name, major_axis, minor_axis, height, location=(0, 0, 0)):
    # Create a new mesh and object
    mesh = bpy.data.meshes.new(name + "_Mesh")
    ellipse_object = bpy.data.objects.new(name, mesh)
    
    # Set location
    ellipse_object.location = location
    
    # Link the object to the scene
    bpy.context.collection.objects.link(ellipse_object)
    
    # Define the number of vertices
    num_vertices = 64
    #num_vertices = 32
    
    # Create the vertices for the ellipse on the XY plane
    verts = []
    for i in range(num_vertices):
        angle = 2 * math.pi * i / num_vertices
        x = major_axis/2 * math.cos(angle)
        y = minor_axis/2 * math.sin(angle)
        verts.append((x, y, -height/2))
    
    # Duplicate the vertices for the top face
    verts_top = [(x, y, height/2) for (x, y, z) in verts]
    verts.extend(verts_top)
    
    # Create faces for the side of the ellipse
    faces = []
    for i in range(num_vertices):
        next_i = (i + 1) % num_vertices
        faces.append([i, next_i, next_i + num_vertices, i + num_vertices])
    
    # Create faces for the bottom and top faces of the ellipse
    bottom_face = [i for i in range(num_vertices)]
    top_face = [i + num_vertices for i in range(num_vertices)]
    faces.append(bottom_face)
    faces.append(top_face)
    
    # Add the vertices and faces to the mesh
    mesh.from_pydata(verts, [], faces)
    
    # Update the mesh with new data
    mesh.update()

    #set_pivot_to_bbox_center(ellipse_object)
    # Set the origin to the center of the object's geometry
    #bpy.context.view_layer.objects.active = ellipse_object
    #bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    
    # Re-center the object at the desired location
    #ellipse_object.location = mathutils.Vector(location)



def spawn_grid_of_ellipses(prefix_to_check, interval, major_axis, minor_axis, height):
    global_min_x, global_max_x, global_min_y, global_max_y = get_globat_min_max(prefix_to_check)
    if IS_DEBUG:
        print('file size: ',global_min_x, global_max_x, global_min_y, global_max_y)
    
    # skip very large files + files with problems/bad scale
    if abs(global_min_x - global_max_x) > 40 or  abs(global_min_x - global_max_x) > 40:
        return False
    
    # Generate a grid of spheres within the min/max bounds
    z = height/2 + 0.01 # add one cm to avoid collisions with floor       0.95
    x_values = list(np.arange(int(global_min_x), int(global_max_x) + 1, interval))
    y_values = list(np.arange(int(global_min_y), int(global_max_y) + 1, interval))

    for x in x_values:
        for y in y_values:
            #bpy.ops.mesh.primitive_uv_sphere_add(location=(x, y, z))
            #bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=radius, depth=1.80, location=(x, y, z))
            create_extruded_ellipse(f'{prefix_to_check}{x}_{y}', major_axis=major_axis, minor_axis=minor_axis, height=height, location=(x, y, z))

            #bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, z))
            sphere = bpy.context.object
            sphere.name = f'{prefix_to_check}{x}_{y}'
            #sphere.scale = (ball_radius, ball_radius, 0.2)
            #sphere.scale = (ball_radius, ball_radius, ball_radius)
            #print(f"Sphere added at location: ({x}, {y}, {z}) with scale ({ball_radius}, {ball_radius}, {ball_radius})")
    return True



def delete_objects_with_prefixes(prefixes):
    """
    Deletes all objects in the current Blender scene whose names start with any of the given prefixes.

    :param prefixes: List of prefixes to check against object names.
    """
    # Get the current scene
    scene = bpy.context.scene
    
    # Loop through all objects in the scene
    for obj in scene.objects:
        # Check if the object's name starts with any of the given prefixes
        if any(obj.name.startswith(prefix) for prefix in prefixes):
            # Delete the object
            bpy.data.objects.remove(obj, do_unlink=True)




def get_3d_front_room_sizes_and_total(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    room_sizes = {}
    room_objects_count = {}
    total_size = 0
    room_count = {}
    bed_living_dining_size = 0
    bed_living_dining_objects_count = 0
    real_bed_living_dining_objects_count = 0
    real_bed_living_dining_objects_names = []
    all_bed_living_dining_objects_refs = []


    
    
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
        
        # only track rooms with area 
        if size > 0:           
            if 'bedroom' in room_type.lower() or 'living' in room_type.lower() or 'dining' in room_type.lower():
                current_room_objects_count = 0
                for child in room.get('children'):
                    for furniture_obj in data.get('furniture'):
                        if furniture_obj.get('title') != None and furniture_obj.get('uid') == child.get('ref') and any(furniture_term in furniture_obj.get('title').lower() for furniture_term in ['bed', 'cabinet', 'shelf','desk', 'table', 'sofa', 'stand', 'lamp','couch','wardrobe','chair','stool']):
                             current_room_objects_count += 1
                             break 

                # only track rooms with at least one furniture object
                if current_room_objects_count > 0:         
                    if room_type not in room_sizes:
                        room_sizes[room_type] = []
                    room_sizes[room_type].append(size)

                    room_children = room.get('children')
                    room_furniture_children_count = 0
                    for child in room.get('children'):
                        if 'furniture' in child.get('instanceid'): #and (furniture_obj.get('instanceid') == None or 'mesh' not in furniture_obj.get('instanceid'))
                            bed_living_dining_objects_count += 1
                            room_furniture_children_count += 1
                            all_bed_living_dining_objects_refs.append(child.get('ref'))
                    bed_living_dining_size += size

                    if room_type not in room_objects_count:
                        room_objects_count[room_type] = []
                    room_objects_count[room_type].append(room_furniture_children_count)

                    # Count rooms per type
                    room_count[room_type] = room_count.get(room_type, 0) + 1

        total_size += size

        
    
    if bed_living_dining_size == 0:
        print('----- bad sizes = 0 : ',file_path )
    ''' '''
    #if bed_living_dining_objects_count == 16:# and bed_living_dining_size < 18:
    #    print(len(data.get('furniture'))) # instanceid
    #any(furniture_term in furniture_obj.get('title').lower() for furniture_term in ['bed', 'cabinet', 'shelf','desk', 'table', 'sofa', 'stand','light','lamp','couch','wardrobe','chair','stool'])
    for furniture_obj in data.get('furniture'): #not 'None' in obj.name and not 'shadow' in obj.name and not 'pendant' in obj.name and not 'Lighting' in obj.name and not 'ceiling' in obj.name and not 'solid' in obj.name and not 'glass' in obj.name
        #not any(furniture_term in furniture_obj.get('title').lower() for furniture_term in ['shadow', 'pendant', 'Lighting','ceiling', 'solid', 'glass']) 
        if furniture_obj.get('title') != None and furniture_obj.get('uid') in all_bed_living_dining_objects_refs and any(furniture_term in furniture_obj.get('title').lower() for furniture_term in ['bed', 'cabinet', 'shelf','desk', 'table', 'sofa', 'stand', 'lamp','couch','wardrobe','chair','stool']):
            real_bed_living_dining_objects_count += 1
            real_bed_living_dining_objects_names.append(furniture_obj.get('title'))
    #print('real_bed_living_dining_objects_count', real_bed_living_dining_objects_count)
    #print('bed_living_dining_objects_count', bed_living_dining_objects_count)
    #print('bed_living_dining_size', bed_living_dining_size)
    #exit()
    if real_bed_living_dining_objects_count == 0:
        print('----- Error: bad real_bed_living_dining_objects_count = 0 : ',file_path )

    if (real_bed_living_dining_objects_count > 0 and real_bed_living_dining_objects_count/bed_living_dining_size > 0.8):
        print('----- !!!!: obj per meter = ',round(real_bed_living_dining_objects_count/bed_living_dining_size, 2) , real_bed_living_dining_objects_names,file_path )

    return room_sizes, total_size, room_count, bed_living_dining_size, real_bed_living_dining_objects_count , bed_living_dining_objects_count, room_objects_count




if __name__ == "__main__":
    #print('\n ************************************************************************\n \
    #         ************************************************************************\n \
    #          ******************      calculate dead_zone  / out of boundary / reachability      ******************\n \
    #          ***********************************************************************')
    print('******************      calculate dead_zone  / out of boundary / reachability         ******************')
    
    start_i = time.time()
    IS_DEBUG = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb_filepath", type=str, required=True)
    parser.add_argument("--body_type", type=str, required=False)
    parser.add_argument("--make_col_file", dest='make_col_file', action='store_true', required=False)
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    #print(args)
    #exit()

    body_type = args.body_type #'ellipses' #'ellipses' , 'cylinders'
    #print(body_type)
    #exit()
    output_filepath = args.glb_filepath.replace('.glb','.json')
    if not os.path.isfile(output_filepath):
        
        # check if area in 3d front json is 0, if so skip this file
        
        if 'front' in output_filepath:
            dataset_3d_front_path = '/home/user2024/Documents/repos/ai-driven-3d-apartment-generator/InstructScene/dataset/3D-FRONT/3D-FRONT'
            file_path = os.path.join(dataset_3d_front_path, os.path.basename(output_filepath))
            room_sizes, total_size, room_count, bed_living_dining_size, real_bed_living_dining_objects_count , bed_living_dining_objects_count, room_objects_count = get_3d_front_room_sizes_and_total(file_path)
        

            if bed_living_dining_size == 0:
                print('SKIP !!!!!!, bed_living_dining_size == 0 ', output_filepath)
                exit()
            elif bed_living_dining_objects_count == 0:
                print('SKIP !!!!!!, bed_living_dining_objects_count == 0 ', output_filepath)
                exit()
            else:
                print('json info :', room_sizes, total_size, room_count, bed_living_dining_size)

        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        reset_scene()

        import_glb(args.glb_filepath)

        # delelte helping objects
        delete_objects_with_prefixes(["room_volume","door_", "cube_best_", "cube_center_"])

        sys.stdout = original_stdout
        
        
        start_time = time.time()
        # Prefix to check
        prefix_to_check = "shape_"
        prefix1 = "object_"
        
        all_objects_names = [obj.name for obj in bpy.data.objects if prefix1 in obj.name and not 'None' in obj.name and not 'shadow' in obj.name and not 'pendant' in obj.name and not 'Lighting' in obj.name and not 'ceiling' in obj.name and not 'solid' in obj.name and not 'glass' in obj.name]
        
        
        # Generate grid of balls with specific interval and ball radius 254 seconds
        ''' '''
        if body_type == 'cylinders':
            print("spawn grid of cylinders")
            res = spawn_grid_of_cylinders(prefix_to_check, 0.50, 0.24, 1.75) #0.25
        elif body_type == 'ellipses':
            print("spawn grid of ellipses")
            res = spawn_grid_of_ellipses(prefix_to_check, 0.50, 0.48, 0.285, 1.75)

        if body_type in ['cylinders', 'ellipses'] and not res:
            print(' ===========  skip very large files !  ===============\n\n')
            exit()
        
        
    
        # Execute the visualization
        if body_type == 'cylinders':
            delete_colliding_objects(prefix_to_check) # without rotations -> for cylinders
        elif body_type == 'ellipses':
            delete_colliding_objects(prefix_to_check, 15) # with rotations -> for eclipses
        
        num_islands, grid_with_IDs, island_sizes = 0, [], []
        total_objects, reachable_objects, unreachable_objects = [], [], []
        if body_type in ['cylinders', 'ellipses']:
            pos_to_land_array, pos_to_names_array = make_grid_array(prefix_to_check, 0.5)

            # increase recursion limit to handle bigger scenes
            sys.setrecursionlimit(5000)
            num_islands, grid_with_IDs, island_sizes = count_islands(pos_to_names_array, prefix_to_check, pos_to_land_array)
            print("Number of islands:", num_islands)
            colorize_islands(pos_to_names_array, grid_with_IDs)
            print("--- %s seconds ---" % (time.time() - start_time))

            reachability_distance = 0.40
            total_objects, reachable_objects, unreachable_objects = reachability_check(prefix1, prefix_to_check, (reachability_distance*2, reachability_distance*2, reachability_distance*2))
            print(f"Total number of objects with prefix '{prefix1}': {len(total_objects)}")
            print(f"Number of objects with prefix '{prefix1}' that collided with objects having prefix '{prefix_to_check}': {len(reachable_objects)}")


        prefix1 = "object_"
        prefix2 = "room_boundary_"
        #check_collisions_result = check_collisions(prefix1, prefix2, 0.5)
        check_collisions_result = check_collisions(prefix1, prefix2, 5.0, 5.0)

        print(f"Number of objects in group1: {check_collisions_result[0]}")
        print(f"Number of objects in group1 collides with other objects : {check_collisions_result[1]}")
        print(f"Number of objects in group1 collides with walls: {check_collisions_result[2]}")
        print(f"Number of objects in group1 that don't collide with any objects or walls: {check_collisions_result[3]}")
        
        
        ''' '''
        if args.make_col_file:
            # export scene if there are unreachble objects for debug
            #if len(reachable_objects) < len(total_objects):
            # redirect output to log file
            logfile = 'eval_single.log'
            open(logfile, 'a').close()
            old = os.dup(sys.stdout.fileno())
            sys.stdout.flush()
            os.close(sys.stdout.fileno())
            fd = os.open(logfile, os.O_WRONLY)

            bpy.ops.export_scene.gltf(filepath=args.glb_filepath.replace('.glb', '_col.glb'), export_lights=True)

            # disable output redirection
            os.close(fd)
            os.dup(old)
            os.close(old)
        
        #check_collisions_result = (all_objects_number, objects_number_colliding_with_others, objects_number_colliding_with_walls, objects_number_with_no_collisions)
        
        #[obj for obj in bpy.data.objects if source_prefix in obj.name
        # save  to a json file0
        with open(output_filepath, 'w') as fp:
            json.dump([num_islands, grid_with_IDs, "island_sizes" ,island_sizes, "collisions", check_collisions_result, ["reachability", len(total_objects), len(reachable_objects), unreachable_objects], all_objects_names], fp)
    else:
        print('file already found: ', output_filepath)


    #/home/fpc/blender-4.1.1-linux-x64/blender -b -P /home/user2024/Documents/repos/ai-driven-3d-apartment-generator/calculate_dead_zone.py -- --glb_filepath /home/user2024/Documents/repos/ai-driven-3d-apartment-generator/3d_front_scenes/d4372f50-5d15-4f80-a9c9-0159f4010215.glb

    #/home/fpc/blender-4.1.1-linux-x64/blender -b -P /home/user2024/Documents/repos/ai-driven-3d-apartment-generator/calculate_dead_zone.py -- --glb_filepath /home/user2024/Documents/repos/ai-driven-3d-apartment-generator/out/adjust.glb

