#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Firas Adleh
Date: 2024-09-01
Version: 1.0
Description: Uses Blender API to validate a folder of 3D apartment files (used to find corrupted files after transfareing files via FTP from the HPC server).
"""

import os
import struct
import subprocess
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import configs


def find_glb_files(directory):
    """Recursively finds all .glb files in the given directory."""
    glb_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.glb'):
                glb_files.append(os.path.join(root, file))
    return glb_files


def is_valid_glb(filepath, blender_executable):
    """Use a subprocess to call Blender to check if the GLB file is valid."""
    try:
        # Create a temporary Blender Python script that attempts to import the .glb file
        temp_script = """
import bpy
import sys

try:
    bpy.ops.wm.read_factory_settings(use_empty=True)  # Reset the Blender scene
    bpy.ops.import_scene.gltf(filepath=r'{filepath}')  # Attempt to import the .glb file
    sys.exit(0)  # Exit with code 0 if successful
except Exception as e:
    sys.exit(1)  # Exit with code 1 if there is an error
""".format(filepath=filepath)

        script_path = os.path.join(os.path.dirname(filepath), "temp_import_script.py")

        # Write the script to a file
        with open(script_path, 'w') as f:
            f.write(temp_script)

        # Call Blender using the subprocess module
        result = subprocess.run(
            [blender_executable, '--background', '--python', script_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Check the exit code from the subprocess
        if result.returncode == 0:
            return True  # The file is valid
        else:
            print(f"Failed to load {filepath}. Blender returned an error.")
            print(result.stderr.decode('utf-8'))  # Print the error details from Blender
            return False

    except Exception as e:
        print(f"Error checking {filepath}: {e}")
        return False
    finally:
        # Clean up the temporary script
        if os.path.exists(script_path):
            os.remove(script_path)

def test_glb_files(directory, blender_executable_path):
    """Test the validity of .glb files and print those that are corrupt."""
    glb_files = find_glb_files(directory)
    corrupt_files = []
    
    for glb_file in glb_files:
        if is_valid_glb(glb_file, blender_executable_path):
            print(f"Valid GLB file: {glb_file}")
        else:
            print(f"Corrupt GLB file: {glb_file}")
            corrupt_files.append(glb_file)
    
    # Summary of corrupt files
    if corrupt_files:
        print("\nThe following .glb files are corrupt:")
        for file in corrupt_files:
            print(file)
    else:
        print("\nAll .glb files are valid!")

# Example usage
directory_path = "/home/user2024/Documents/repos/ai-driven-3d-apartment-generator/hpc_eval/"  # Replace with the folder you want to search
blender_executable_path = configs.constants['blender_path']  # Replace with the path to Blender executable

test_glb_files(directory_path, blender_executable_path)