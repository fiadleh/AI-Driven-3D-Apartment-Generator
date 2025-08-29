import bpy
import os
import json

def load_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def import_obj(obj_path):
    bpy.ops.import_scene.obj(filepath=obj_path)

def apply_material(obj, mtl_path, texture_path):
    mat = bpy.data.materials.new(name="Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    
    # Load the MTL file and apply settings
    # Note: Blender does not directly support MTL file import, so you'll need to handle it manually if needed
    
    # Apply texture
    tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
    tex_image.image = bpy.data.images.load(texture_path)
    mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])
    
    # Assign material to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

def load_furniture_model(furniture_info, furniture_dir):
    obj_id = furniture_info['jid']
    obj_dir = os.path.join(furniture_dir, obj_id)
    obj_path = os.path.join(obj_dir, 'normalized_model.obj')
    mtl_path = os.path.join(obj_dir, 'model.mtl')
    texture_path = os.path.join(obj_dir, 'texture.png')
    
    if os.path.exists(obj_path):
        import_obj(obj_path)
        imported_obj = bpy.context.selected_objects[0]
        apply_material(imported_obj, mtl_path, texture_path)

def import_house_layout(json_filepath, furniture_dir):
    data = load_json(json_filepath)
    for furniture in data['furniture']:
        if furniture['valid']:
            load_furniture_model(furniture, furniture_dir)

def export_scene(output_filepath):
    bpy.ops.export_scene.gltf(filepath=output_filepath, export_format='GLB')

# Example usage
json_filepath = './dataset/3D-FRONT/3D-FRONT/0a28563b-19e4-4d67-8130-9a87a547daf6.json'  # Replace with the actual path to your JSON file
furniture_dir = './dataset/3D-FRONT/3D-FUTURE-model'  # Replace with the actual path to your furniture model directory
output_filepath = './house.glb'  # Desired output GLB file path

# Clear existing mesh objects
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

import_house_layout(json_filepath, furniture_dir)
export_scene(output_filepath)
