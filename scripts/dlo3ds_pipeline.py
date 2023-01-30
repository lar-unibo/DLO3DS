#! /home/alessio/miniconda3/envs/wh/bin/python

import numpy as np
import os, copy, json
import cv2
from termcolor import cprint 

# dlo3ds
from dlo3ds_core.reconstruction import SplineReconstruction
from dlo3ds_core.estimate3D import Estimate3D
from dlo3ds_core.optimization import OptimizeCamera
from dlo3ds_core.filtering import Lowess
from dlo3ds_core.camera_blender import CameraBlender

# FASTDLO
from fastdlo_core.fastdlo import FASTDLO

# panda
from panda_driver.panda_handler import PandaHandler

# ros
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion 
import sensor_msgs.msg
from cv_bridge import CvBridge
import rospkg


class DLO3DS(object):

    def __init__(self, iter_test = None):

        rospack = rospkg.RosPack()
        self.main_fodler = rospack.get_path("dlo3ds")
        self.script_path = os.path.dirname(os.path.realpath(__file__))
        self.iter_test = iter_test

        cprint("SIMULATION *IS* ACTIVE", "white")

        self.obj_name = str(rospy.get_param('dlo3ds/obj_file')).split(".")[0]  
        cprint("Object name: {}".format(self.obj_name), "white")

        # FRAMES PARAMS
        self.camera_frame = str(rospy.get_param('dlo3ds/camera_frame'))
        self.world_frame = str(rospy.get_param('dlo3ds/world_frame'))

        # CAMERA PARAMS
        self.camera_matrix = [float(value) for value in rospy.get_param('dlo3ds/camera_matrix').split(", ")]
        self.camera_matrix = np.array(self.camera_matrix).reshape(3,3)
        self.distort_vec = np.array([float(value) for value in rospy.get_param('dlo3ds/distort_vec').split(", ")])
        self.camera_height = int(rospy.get_param('dlo3ds/camera_height'))
        self.camera_width = int(rospy.get_param('dlo3ds/camera_width'))

        # OPTIMIZATION PARAMS
        self.opt_delta_z = float(rospy.get_param('dlo3ds/opt_delta_z')) 
        self.opt_offset = int(rospy.get_param('dlo3ds/opt_offset')) 
        self.opt_min_z = float(rospy.get_param('dlo3ds/opt_min_z')) 

        # DLO PROC PARAMS
        self.dlo_proc = FASTDLO(main_folder=self.script_path, img_w=self.camera_width, img_h=self.camera_height)
        #print("dlo_proc class ok!")

        cprint("Waiting for CameraBlender class...", "white")
        self.camera_srv = CameraBlender(self.main_fodler, self.obj_name, self.camera_matrix, self.distort_vec, self.camera_height, self.camera_width)
        cprint("CameraBlender OK!", "green")

        # DLO3DS PARAMS
        self.overlap_scans = float(rospy.get_param('dlo3ds/overlap_scans')) 
        self.num_points = int(rospy.get_param('dlo3ds/num_points'))
        self.num_views = int(rospy.get_param('dlo3ds/num_views'))

        self.output_path = os.path.join(self.main_fodler, os.path.join(str(rospy.get_param('dlo3ds/output_path')), self.obj_name + "_ov_" + str(self.overlap_scans)))
        os.makedirs(self.output_path, exist_ok=True)

        # PANDA PARAMS
        self.robot = PandaHandler(ee_frame=self.camera_frame, simulation=True)
        self.robot.set_vel_acc(1, 1)
        self.robot.homing(open_gripper_cmd=True)


        # TRIANGULATION CLASS
        self.estimator = Estimate3D(camera_matrix = self.camera_matrix,
                                    camera_dist = self.distort_vec,
                                    camera_height = self.camera_height,
                                    camera_width = self.camera_width,
                                    num_points = self.num_points)

        # SPLINE RECONSTRUCTION CLASS
        self.reconstruction = SplineReconstruction(camera_matrix=self.camera_matrix,
                                                    distort_vec=self.distort_vec,
                                                    camera_width=self.camera_width,
                                                    camera_height=self.camera_height,
                                                    num_points=self.num_points,
                                                    overlap_scans=self.overlap_scans)

        # LOWESS CLASS
        self.lowess = Lowess()

        # GENERAL VARIABLES
        self.bridge = CvBridge()
        self.estimated_points = {}

        
        #### SAVE PARAMS 
        data = {"camera_matrix": list(self.camera_matrix.flatten()), 
                "distort_vec": list(self.distort_vec),
                "camera_height": int(self.camera_height),
                "camera_width": int(self.camera_width),
                "opt_delta_z": float(self.opt_delta_z),
                "opt_offset": int(self.opt_offset),
                "opt_min_z": float(self.opt_min_z),
                "overlap_scans": float(self.overlap_scans)}

        data_params_path = os.path.join(self.output_path, "params.json")
        with open(data_params_path, 'w') as f:
            json.dump(data, f)

        


    ###############################################
    # WHEN USING RTDLO CLASS                      #
    ###############################################    
    def get_splines(self, img, debug=False):
        splines, mask = self.dlo_proc.run(img) 

        if debug:
            cv2.imshow("mask", mask)
            cv2.waitKey(0)
        
        print("obtained {0} spline(s)".format(len(splines)))
        return splines
   

    def get_samples(self, poses):

        imgs=[]
        for p in poses:
            rv, img = self.get_sample(p)
            if rv == False: 
                return False, None
            imgs.append(img)

        return rv, imgs


    def get_sample(self, pose):
        
        if isinstance(pose, list):
            goal_pose = Pose(position = Point(x=pose[0], y=pose[1], z=pose[2]), orientation = Quaternion(x=pose[3], y=pose[4], z=pose[5], w=pose[6]))      
        else:
            goal_pose = pose

        # move to next pose
        rv = self.robot.go_to_pose_goal_error_recovery(goal_pose, self.camera_frame)
        
        if rv == True:
            img = self.camera_srv.callback([ goal_pose.position.x, goal_pose.position.y, goal_pose.position.z,
                                    goal_pose.orientation.x, goal_pose.orientation.y, goal_pose.orientation.z, goal_pose.orientation.w])

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.camera_width, self.camera_height))        
            return True, img
        else:
            return False, None


    def filter_points(self, data_dict):
        data_dict = self.reconstruction.orthFilteringOpt(data_dict)
        return data_dict

    def select_target_spline(self, splines, point_ref_discriminator):
        spline, _ = self.reconstruction.splinesDiscriminator(point_ref_discriminator, splines)       # select spline to follow
        new_points = self.reconstruction.evaluateSpline(spline)
        return spline, new_points

    def save_txt(self, points_3d, error_values, iter):
        points_3d_list_opt_np = np.array(points_3d).reshape(-1,3)
        error_norm_np = np.array(error_values).reshape(-1,1)
        np_to_save = np.hstack((points_3d_list_opt_np, error_norm_np))
        np.savetxt(os.path.join(self.output_path, "points_" + str(iter) + ".txt"), np_to_save.reshape(-1, 4), fmt="%f")

    def normalize_quat_pose(self, pose):
        quat = np.array(pose[3:])
        quat = quat / np.linalg.norm(quat)
        return Pose(position=Point(x=pose[0], y=pose[1], z=pose[2]), orientation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]))



    def evaluate_3D_points(self, data_dict):

        # 3D ESTIMATION
        points3d = self.estimator.estimate3D(data_dict)

        # convert points3d to list
        list_points =[(p[0][0], p[1][0], p[2][0]) for p in points3d]

        ### PROJECT 2D TO CALCULATE REPROJECTION ERROR
        error_dict = self.estimator.computeReprojectionErrorVector(points3d, data_dict)

        error_values = list(error_dict.values())
        mean_error = np.mean(error_values)
        std_error = np.std(error_values)
        max_error = np.max(error_values)
        
        if mean_error < 1: ccolor = "green"
        elif mean_error < 2: ccolor = "yellow"
        else: ccolor = "red"
        cprint("Reprojection Error: mean {0:.3f}, std {1:.3f}, max {2:.3f}".format(mean_error, std_error, max_error), ccolor)

        if ccolor == "red": return False, list_points, error_values
        return True, list_points, error_values
    

        
    def convert_points_camera_frame(self, list_points, camera_pose):
        T0 = self.reconstruction.getMatrixFromVec(camera_pose)

        list_points_cam = []
        for p in list_points:
            p_world = np.array([p[0], p[1], p[2], 1]).reshape(4,-1)
            p_cam = np.matmul(np.linalg.inv(T0), p_world).reshape(-1)
            list_points_cam.append((p_cam[0], p_cam[1], p_cam[2]))
        return list_points_cam

    def optimize_baseline_and_z(self, x0, camera_pose):
        ###############################################
        # RUN OPTIMIZATION                            #
        ###############################################    
        list_points_cam_0 = self.convert_points_camera_frame(x0, camera_pose)

        opt = OptimizeCamera(camera_matrix=self.camera_matrix, 
                            distort_vec=self.distort_vec, 
                            camera_height=self.camera_height, 
                            camera_width=self.camera_width, 
                            num_points=self.num_points,
                            z_min=self.opt_min_z,
                            offset=self.opt_offset,
                            delta_z=self.opt_delta_z)


        rv, result = opt.run_optimization_with_timeout(list_points_cam_0)

        if rv == False:
            result = [0.025, 0.0]
            print("optimization fails! using default parameters... baseline {}, delta_z {}".format(result[0], result[1]))


        pose_mid = copy.deepcopy(camera_pose)
        pose_mid[2] += result[1]
        print("POSE mid :", [round(v, 4) for v in pose_mid])

        flag_2_views = False
        if self.num_views == 2:
            flag_2_views = True

        n_additional_samples = self.num_views - 1 if flag_2_views is False else self.num_views
        step = result[0] / (n_additional_samples // 2)
        poses = []
        
        
        for i in range(n_additional_samples//2): 
            pose_tmp = copy.deepcopy(pose_mid)
            relative_disp = step * (n_additional_samples // 2 - i)
            pose_disp = np.matmul(self.reconstruction.getMatrixFromVec(pose_tmp), np.array([-relative_disp, 0, 0, 0]).T)
            pose_tmp[0] -= pose_disp[0]
            pose_tmp[1] -= pose_disp[1]
            pose_tmp[2] -= pose_disp[2]
            poses.append(pose_tmp)

        if not flag_2_views: poses.append(pose_mid)

        for i in range(1, n_additional_samples//2 + 1):
            pose_tmp = copy.deepcopy(pose_mid)
            relative_disp = step * i
            pose_disp = np.matmul(self.reconstruction.getMatrixFromVec(pose_tmp), np.array([-relative_disp, 0, 0, 0]).T)
            pose_tmp[0] += pose_disp[0]
            pose_tmp[1] += pose_disp[1]
            pose_tmp[2] += pose_disp[2]
            poses.append(pose_tmp)

        return poses


    def local_estimatation(self, points_ref, poses, iter, save_debug = True):


        ###############################################
        # GET SAMPLES                                 #
        ###############################################  
        rv, list_imgs = self.get_samples(poses)
        if rv == False:
            return False, None, None
        ###############################################
        # GET SPLINE                                  #
        ############################################### 
        list_splines = [self.get_splines(img) for img in list_imgs]



        ###############################################
        # SELECT SPLINE                               #
        ############################################### 
        list_spline_target = []
        list_points = [] 
        for splines in list_splines:
            if points_ref is None:
                points_ref = self.reconstruction.chooseReferenceSplineInit(splines)
        
            spline, points_ref = self.select_target_spline(splines, points_ref)
            list_spline_target.append(spline)
            list_points.append(points_ref)

        ###############################################
        # DATA DICT                                   #
        ############################################### 
        data_dict = {}
        for i, _ in enumerate(list_imgs):
            # store data 
            small_dict = {}
            small_dict["camera_pose"] = self.reconstruction.getMatrixFromVec(poses[i])
            small_dict["points"] = list_points[i]
            small_dict["spline"] = list_spline_target[i]
            small_dict["img"] = list_imgs[i]
            data_dict[i] = small_dict

        ###############################################
        # FILTERING                                   #
        ############################################### 
        data_dict = self.filter_points(data_dict)

       
        ###############################################
        # EVALUATE 3D POINTS                          #
        ###############################################        
        rv, list_points, error_values = self.evaluate_3D_points(data_dict)
        if rv == False:
            return False, None, None
            
        self.estimated_points[iter] = copy.deepcopy(list_points)

        if save_debug:
            self.save_debug_images(list_points, data_dict, iter, save_all = True)

        return True, list_points, error_values



    def save_debug_images(self, points_3d, data_dict, iter, save_all=False):

        for key, data in data_dict.items():
            canvas = data["img"].copy()
            point2d = self.reconstruction.project2DSimple(data["camera_pose"], points_3d)
            for p in point2d:
                i, j = [round(p[1]), round(p[0])]
                if i < self.camera_height and i >= 0 and j < self.camera_width and j >= 0:
                    cv2.circle(canvas, tuple([int(j), int(i)]), 3, (0, 0, 255), -1)


            for p in data["points_f"]:
                cv2.circle(canvas, tuple([int(p[0]), int(p[1])]), 3, (255, 0), -1)

            cv2.imwrite(os.path.join(self.output_path, "img_" + str(iter) + "_K_" + str(key) + ".png"), canvas)
            if save_all:
                cv2.imwrite(os.path.join(self.output_path, "img_" + str(iter) + "_K_" + str(key) + "_raw.png"), data["img"])
                np.savetxt(os.path.join(self.output_path, "img_" + str(iter) + "_K_" + str(key) + "_raw.txt"), np.array(data["camera_pose"]).reshape(1,-1), fmt="%f")

    
    def initialization(self, pose_init, spline_idx = 0):

        # camera alignment
        rv, img = self.get_sample(pose_init)
        if rv == False: return False, None, None
        splines = self.get_splines(img)

        print("*** num splines retrieved: ", len(splines))

        pose_centering = self.reconstruction.centering(splines[spline_idx], pose_init)
        print("POSE CENTERING INIT: ", [round(v, 4) for v in pose_centering])  

        # choose ref spline
        rv, img = self.get_sample(pose_init)
        if rv == False: return False, None, None

        splines = self.get_splines(img)

        points_ref = self.reconstruction.chooseReferenceSplineInit(splines)
        poses = self.compute_fixed_baseline_poses(pose_centering)
        return True, points_ref, poses

    def compute_fixed_baseline_poses(self, pose_centering):
        pose_1 = copy.deepcopy(pose_centering)
        pose_2 = copy.deepcopy(pose_1)
        pose_2_disp = np.matmul(self.reconstruction.getMatrixFromVec(pose_2), np.array([-0.01, 0, 0, 0]).T)
        pose_2[0] += pose_2_disp[0]
        pose_2[1] += pose_2_disp[1]
        pose_2[2] += pose_2_disp[2]

        pose_3 = copy.deepcopy(pose_2)
        pose_3_disp = np.matmul(self.reconstruction.getMatrixFromVec(pose_3), np.array([-0.01, 0, 0, 0]).T)
        pose_3[0] += pose_3_disp[0]
        pose_3[1] += pose_3_disp[1]
        pose_3[2] += pose_3_disp[2]

        poses = [pose_1, pose_2, pose_3]
        return poses

    def move_forward(self, points3d, camera_pose, debug = False):
        
        pose = self.reconstruction.forward(points3d, camera_pose)
        print("FORWARD POSE: ", [round(v, 4) for v in pose])

        if isinstance(pose, list):
            goal_pose = Pose(position = Point(x=pose[0], y=pose[1], z=pose[2]), orientation = Quaternion(x=pose[3], y=pose[4], z=pose[5], w=pose[6]))      
        else:
            goal_pose = pose

        rv = self.robot.go_to_pose_goal_error_recovery(goal_pose, self.camera_frame)

        if debug:
            img = rospy.wait_for_message("usb_cam/image_raw", sensor_msgs.msg.Image)
            canvas = cv2.cvtColor(self.bridge.imgmsg_to_cv2(img, desired_encoding='passthrough'), cv2.COLOR_BGR2RGB)
            points2d = self.reconstruction.project2DSimple(self.reconstruction.getMatrixFromVec(pose), points3d)
            for px, py in points2d:
                cv2.circle(canvas, tuple([int(px), int(py)]), 5, (0, 255, 255), -1)
            cv2.imshow("move_forward", canvas)
            cv2.waitKey(0)

        return rv, pose

    def move_recovery(self, points3d, camera_pose, debug = False):
        pose = self.reconstruction.recovery(points3d, camera_pose, overlap=0.75)
        print("RECOVERY POSE: ", [round(v, 4) for v in pose])

        if isinstance(pose, list):
            goal_pose = Pose(position = Point(x=pose[0], y=pose[1], z=pose[2]), orientation = Quaternion(x=pose[3], y=pose[4], z=pose[5], w=pose[6]))      
        else:
            goal_pose = pose

        rv = self.robot.go_to_pose_goal_error_recovery(goal_pose, self.camera_frame)

        return rv, pose

    def choose_ref_spline(self, points3d, camera_pose, iter, debug = False):
        ###############################################
        # CHOOSING REF SPLINE                         #
        ###############################################

        # choose ref spline
        rv, img = self.get_sample(camera_pose)
        splines = self.get_splines(img)

        if len(splines) < 1:
            cprint("ZERO splines returned, trying again...", "red")
            rv, img = self.get_sample(camera_pose)
            splines = self.get_splines(img)

        if rv == False:
            return False, None, None

        spline = self.reconstruction.selectSplineNext(points3d, splines, camera_pose)
        points_ref = self.reconstruction.evaluateSpline(spline)
        
        # show selected spline
        if debug:
            canvas = img.copy()
            for sp in splines:
                sp_points = self.reconstruction.evaluateSpline(sp)
                for px, py in sp_points:
                    cv2.circle(canvas, tuple([int(px), int(py)]), 3, (0, 0, 255), -1)

            spline_points = self.reconstruction.evaluateSpline(spline)
            for px, py in spline_points:
                cv2.circle(canvas, tuple([int(px), int(py)]), 5, (0, 255, 255), -1)

            points2d = self.reconstruction.project2DSimple(self.reconstruction.getMatrixFromVec(camera_pose), points3d)
            for px, py in points2d:
                cv2.circle(canvas, tuple([int(px), int(py)]), 5, (255, 0, 0), -1)

            cv2.imshow("choose_ref_spline", canvas)
            cv2.waitKey(0)
        
        # **centering**
        pose_centering = self.reconstruction.centering(spline, camera_pose)

        # re-evaluate splines for updating point_ref
        rv, img = self.get_sample(camera_pose)
        splines = self.get_splines(img)

        if rv == False:
            return False, None, None

        spline = self.reconstruction.selectSplineNext(points3d, splines, pose_centering)
        points_ref = self.reconstruction.evaluateSpline(spline)

        if debug:
            canvas = img.copy()
            #for sp in splines:
            #    sp_points = self.reconstruction.evaluateSpline(sp)
            #    for px, py in sp_points:
            #        cv2.circle(canvas, tuple([int(px), int(py)]), 3, (0, 0, 255), -1)

            spline_points = self.reconstruction.evaluateSpline(spline)
            for px, py in spline_points:
                cv2.circle(canvas, tuple([int(px), int(py)]), 5, (0, 255, 255), -1)

            points2d = self.reconstruction.project2DSimple(self.reconstruction.getMatrixFromVec(pose_centering), points3d)
            for px, py in points2d:
                cv2.circle(canvas, tuple([int(px), int(py)]), 5, (255, 0, 0), -1)

            cv2.imshow("choose_ref_spline", canvas)
            cv2.waitKey(0)
        

        #cv2.imwrite(os.path.join(self.output_path, "img_" + str(iter) + "_C.png"), img)
        return True, points_ref, pose_centering

    def loop(self, points_3d, camera_pose, iter = 0):
        cprint("--> FORWARD", "yellow")      
        #######################################
        # FORWARD
        print("CAMERA POSE: ", [round(v, 4) for v in camera_pose])
        print("seleted range: ", len(points_3d) - int(len(points_3d)*self.overlap_scans), -1)
        points_3d_input = points_3d[len(points_3d) - int(len(points_3d)*self.overlap_scans):-1]
        
        rv, forward_pose = self.move_forward(points_3d_input, camera_pose)
        if rv == False:
            return False, None, None, None

        rv, points_ref, camera_pose = self.choose_ref_spline(points_3d_input, forward_pose, iter)
        if rv == False:
            return False, None, None, None
        #######################################
        poses = self.optimize_baseline_and_z(points_3d_input, camera_pose)
        camera_pose_new = poses[int((len(poses)-1)/2)]

        cprint("\n--> ESTIMATION OPTIMIZED", "yellow")
        rv, points_3d_list_opt, error_values = self.local_estimatation(points_ref, poses = poses, iter = iter)

        if rv == False:
            cprint("trying recovery motion...", "red")
            rv, recovery_pose = self.move_recovery(points_3d_input, camera_pose)
            if rv == False:
                return False, None, None, None

            rv, points_ref, camera_pose = self.choose_ref_spline(points_3d_input, recovery_pose, iter)
            if rv == False:
                return False, None, None, None
            poses = self.optimize_baseline_and_z(points_3d_input, camera_pose)
            camera_pose_new = poses[int((len(poses)-1)/2)]

            cprint("\n--> ESTIMATION OPTIMIZED [RECOVERY]", "yellow")
            rv, points_3d_list_opt, error_values = self.local_estimatation(points_ref, poses = poses, iter = iter)
            if rv == False:
                return False, None, None, None

        return True, copy.deepcopy(points_3d_list_opt), copy.deepcopy(camera_pose_new), copy.deepcopy(error_values)


    def run(self, pose_init, num_iterations=10, check_limit_workspace=True):
        self.pose_init = self.normalize_quat_pose(pose_init)

        rv, points_ref, poses = self.initialization(self.pose_init)
        if rv == False:
            return

        cprint("\n--> ESTIMATION FIXED BASELINE", "yellow")
        rv, points_3d_list, _ = self.local_estimatation(points_ref, poses = poses, iter = 0)
        print(points_3d_list)
        if rv == False:
            return
        camera_pose = poses[0]

        cprint("\n--> ESTIMATION WITH OPTIMIZED PARAMETERS", "yellow")
        poses = self.optimize_baseline_and_z(points_3d_list, camera_pose)
        rv, points_3d_list_opt, error_values = self.local_estimatation(points_ref, poses = poses, iter = 0)
        if rv == False:
            return

        camera_pose = poses[int((len(poses)-1)/2)]

        self.save_txt(points_3d_list_opt, error_values, iter=0)

        for i in range(1, num_iterations):
            cprint("ITERATION: {}".format(i), "red")
            #input("PRESS ENTER TO CONTINUE")
            rv, points_3d_list_opt, camera_pose, error_values = self.loop(points_3d = points_3d_list_opt, camera_pose = camera_pose, iter=i)
            if rv == False:
                return

            self.save_txt(points_3d_list_opt, error_values, iter=i)
   
            # check to verify if dlo is finished
            if (camera_pose[0]**2 + camera_pose[1]**2)**0.5 > 0.75 and check_limit_workspace:
                print("workspace limit!")
                return


if __name__ == '__main__':

    rospy.init_node('dlo3ds')


    for dlo_id in [5]:
        obj_name = f"dlo_{dlo_id}_r_35.obj"
        rospy.set_param('dlo3ds/obj_file', obj_name)
        r = DLO3DS()

        pose_init = [0.15, 0.30, 0.30, 0.707, -0.707, 0.0, 0.0]
        r.run(pose_init)

        del r