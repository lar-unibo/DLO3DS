#!/usr/bin/env python3

import os, sys
from tqdm import tqdm
import subprocess
import cv2
from termcolor import cprint
import json 

class CameraBlender(object):

    def __init__(self, main_folder, obj_file, camera_matrix, camera_distort, camera_height, camera_width):

        ######
        cprint("CameraBlender class...", "white")

        self.blender_rendering_folder = os.path.join(main_folder, "utilities/blender_rendering")
        self.coco_folder = os.path.join(self.blender_rendering_folder, "coco_data")
        self.image_path = os.path.join(self.coco_folder, "images/000000.png")
        self.json_path  = os.path.join(self.blender_rendering_folder, "image.json")

        self.obj_file = obj_file
        dlo_id = int(self.obj_file.split("_")[1])

        self.obj_path = os.path.join(self.blender_rendering_folder, os.path.join("data_models/dlo_" + str(dlo_id), self.obj_file + ".obj"))

        print("obj_path: ", self.obj_path)

        # CAMERA PARAMS
        self.camera_matrix = camera_matrix
        self.distort_vec = camera_distort
        self.camera_height = camera_height
        self.camera_width = camera_width
        cprint("OK!", "white")


    def callback(self, pose):

        self.updateJSON(pose)
        
        with tqdm(unit='B', unit_scale=False, miniters=1, desc="Running BlenderProc... ") as t:          
            p = subprocess.Popen(["blenderproc", 
                                "run", 
                                os.path.join(self.blender_rendering_folder, "img_from_pose.py")],
                                env=dict(os.environ, PYTHONPATH=""),
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.STDOUT)
            for _ in p.stdout:
                t.update()
                sys.stdout.flush()
            p.stdout.close()
        print("Running BlenderProc... Done!", sep=' ', end='\n', flush=True)

        return cv2.imread(self.image_path, cv2.IMREAD_COLOR)


    def updateJSON(self, pose):

        if isinstance(pose, list):
            json_pose = pose      
        else:
            cprint("pose should be a list!", "red")     

        data = {"camera_matrix": list(self.camera_matrix.flatten()), 
                "distort_vec": list(self.distort_vec),
                "camera_height": int(self.camera_height),
                "camera_width": int(self.camera_width),
                "pose": list(json_pose),
                "obj_path": str(self.obj_path)}

        with open(self.json_path, 'w') as f:
            json.dump(data, f)
