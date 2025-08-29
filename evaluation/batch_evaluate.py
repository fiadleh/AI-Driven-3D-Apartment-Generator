#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Firas Adleh
Date: 2024-09-01
Version: 1.0
Description: Calls other script 'evaluate_single.py' to evaluate the usability of a group of 3D apartments (generated or from 3D-Front).
"""

import os
import subprocess
import sys
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import configs
#execluded/c
glb_directory  = configs.constants['app_path'] + "3d_front_scenes/"#"/home/user2024/Documents/repos/ai-driven-3d-apartment-generator/3d_front_scenes/"
#glb_directory  = configs.constants['app_path'] + "evaluation_scenes/"
#glb_directory  = "/home/user2024/Documents/repos/ai-driven-3d-apartment-generator/evaluation_scenes_before_adjust"
directory = os.fsencode(glb_directory)
#/home/fpc/blender-4.1.1-linux-x64/blender -b -P /home/user2024/Documents/repos/ai-driven-3d-apartment-generator/calculate_dead_zone.py -- --glb_filepath /home/user2024/Documents/repos/ai-driven-3d-apartment-generator/3d_front_scenes/d4372f50-5d15-4f80-a9c9-0159f4010215.glb

#/home/fpc/blender-4.1.1-linux-x64/blender -b -P /home/user2024/Documents/repos/ai-driven-3d-apartment-generator/calculate_dead_zone.py -- --glb_filepath /home/user2024/Documents/repos/ai-driven-3d-apartment-generator/out/adjust.glb

all_glb_paths= os.listdir(directory)
for i, file in enumerate(all_glb_paths):
    filename = os.fsdecode(file)    
    if filename.endswith("glb") and not filename.endswith('_col.glb'):
        BLENDER_SCRIPT_PATH = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "evaluate_single.py"
        )

        jspn2obj_scene_directory = os.path.join(
            glb_directory,
            filename
        )


        args = [
            configs.constants['blender_path'],
            "-b", "-P", BLENDER_SCRIPT_PATH,
            "--",
            "--glb_filepath", jspn2obj_scene_directory,
            "--body_type", 'cylinders', # 'ellipses' , 'cylinders' , ''
            #"--make_col_file",
        ]
        #print(args)
        #exit()
        print(i,'/',len(all_glb_paths)," ++++++++++++++++++ ", filename, flush=True)


        #with open(os.devnull, 'w') as fnull:
        #    subprocess.run(args, stdout=fnull, stderr=fnull)

        start_time = time.time()
        #with open(os.devnull, 'w') as fnull:
        #    subprocess.run(args, stdout=fnull, stderr=fnull)
        print("--- %s seconds ---" % (round(time.time() - start_time)))
        
        subprocess.check_call(args)


