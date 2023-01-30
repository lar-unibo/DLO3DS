#! /home/lar/miniconda3/envs/ariadneplus/bin/python

import blenderproc as bproc
import numpy as np
from scipy.spatial.transform import Rotation
import os, json 

class Rendering():

    def __init__(self):     
        self.coco_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "coco_data")
        self.json_path  = os.path.join(os.path.dirname(os.path.realpath(__file__)), "image.json")
    
    def readJSON(self):
        with open(self.json_path) as config_file:
                config = json.load(config_file)

        self.camera_matrix = np.array(config["camera_matrix"]).reshape(3,3)
        self.distort_vec = np.array(config["distort_vec"])
        self.camera_height = config["camera_height"]
        self.camera_width = config["camera_width"]
        self.pose = config["pose"]
        self.obj_path = config["obj_path"]

    def run_blenderproc(self):

        self.readJSON()

        bproc.init()

        # Load object
        obj = bproc.loader.load_obj(self.obj_path)[0]
        obj.set_cp("cp_category_id", 1)
        obj.set_cp("cp_physics", False)

        # camera
        bproc.camera.set_intrinsics_from_K_matrix(self.camera_matrix, self.camera_width, self.camera_height)

        # OpenCV -> OpenGL
        T = bproc.math.change_source_coordinate_frame_of_transformation_matrix(self.matrixFromPose(self.pose), ["X", "-Y", "-Z"])
        bproc.camera.add_camera_pose(T) # tmat is a 4x4 numpy array

        # Create a point light next to it
        light = bproc.types.Light()
        light.set_location([0, 2, 3])
        light.set_energy(300)

        # Create a point light next to it
        light2 = bproc.types.Light()
        light2.set_location([2, 0, 3])
        light2.set_energy(300)

        # Create a point light next to it
        light3 = bproc.types.Light()
        light3.set_location([0, -2, 3])
        light3.set_energy(300)

        # Render the scene
        data = bproc.renderer.render()
        seg_data = bproc.renderer.render_segmap(map_by=["instance", "class", "name"])
        bproc.writer.write_coco_annotations(self.coco_folder,
                                            instance_segmaps=seg_data["instance_segmaps"],
                                            instance_attribute_maps=seg_data["instance_attribute_maps"],
                                            colors=data["colors"],
                                            color_file_format="PNG",
                                            append_to_existing_output=False)
    
    def matrixFromPose(self, pose):
        T = np.eye(4)
        T[:3,:3] = Rotation.from_quat([pose[3], pose[4], pose[5], pose[6]]).as_matrix()
        T[0,3] = pose[0]
        T[1,3] = pose[1]
        T[2,3] = pose[2]
        return T



if __name__ == "__main__":
    r = Rendering()
    r.run_blenderproc()