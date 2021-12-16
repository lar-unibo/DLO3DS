#!/usr/bin/env python  
import math
import numpy as np
import cv2 
#from irmp import RayTracing

####################################################################################

class Estimate3D(object):

    def __init__(self, camera_matrix, camera_dist, camera_height, camera_width, num_points):

        self.camera_matrix = camera_matrix
        self.camera_dist = camera_dist
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.num_points = num_points

    ###############################################
    # RAYTRACING CALL                             #
    ###############################################     
    def estimate3D(self, data_dict, min_num_samples = 2):

        if len(list(data_dict.keys())) >= min_num_samples:

            points3d = RayTracing.computeFromData(data_dict, self.camera_matrix, num_points=self.num_points)
            return points3d
        
        return None, None               


    ###############################################
    # PROJECT BACK 2D                             #
    ###############################################           
    def project2D(self, points_3d, data_dict):

        if points_3d is None and data_dict is None:
            return None

        ex_ = 0
        ey_ = 0
        counter = 0
        for _, values in data_dict.items():
                        
            T = np.linalg.inv(values["camera_pose"])
            tvec = np.array(T[0:3, 3])
            rvec, _ = cv2.Rodrigues(T[:3,:3])
            point2d = cv2.projectPoints(np.array(points_3d), rvec, tvec, self.camera_matrix, self.camera_dist)
            
            good_points2d = point2d[0].squeeze()
            # reprojection error
            points2d = np.array(good_points2d).T
            points2d = list(zip(points2d[0], points2d[1]))

            ex, ey = self.reprojectionError(values["points_f"], points2d)

            ex_ += ex
            ey_ += ey
            counter += 1

        ex_ /= counter
        ey_ /= counter
        return np.sqrt(ex_**2 + ey_**2)
    

    def computeReprojectionErrorVector(self, points_3d, data_dict):
        error_dict = {i: 0 for i in range(len(points_3d))}
        num_imgs = len(data_dict.keys())

        for _, values in data_dict.items():     
            T = np.linalg.inv(values["camera_pose"])
            tvec = np.array(T[0:3, 3])
            rvec, _ = cv2.Rodrigues(T[:3,:3])
            point2d = cv2.projectPoints(np.array(points_3d), rvec, tvec, self.camera_matrix, self.camera_dist)
            points_projected = point2d[0].squeeze()

            for i, _ in enumerate(values["points_f"]):
                diff = np.subtract(values["points_f"][i], points_projected[i])

                v = (diff[0]**2 + diff[1]**2)**0.5
                error_dict[i] += v
        
        for key, _ in error_dict.items():
            error_dict[key] = error_dict[key] / num_imgs
        return error_dict


    ##### REPROJECTION ERROR
    def reprojectionError(self, spline_points, projected_points):
        spline_points = np.array(spline_points)
        projected_points = np.array(projected_points)

        #print(len(spline_points), len(projected_points))
        assert(len(spline_points) == len(projected_points))

        error_list_x = []
        error_list_y = []
        for i, _ in enumerate(spline_points):
            diff = np.subtract(spline_points[i], projected_points[i])
            error_list_x.append(diff[0])
            error_list_y.append(diff[1])

        np_error_x = np.array(error_list_x).reshape(len(error_list_x), 1)
        np_error_y = np.array(error_list_y).reshape(len(error_list_y), 1)

        error_x = math.sqrt(np.matmul(np_error_x.T, np_error_x))
        error_y = math.sqrt(np.matmul(np_error_y.T, np_error_y))
        return error_x, error_y


    ##### PROJECT 2D
    def project2DSimple(self, camera_pose, points_3d):
        T = np.linalg.inv(camera_pose)
        tvec =np.array(T[0:3, 3])
        rvec, _ = cv2.Rodrigues(T[:3,:3])

        point2d = cv2.projectPoints(np.array(points_3d), rvec, tvec, self.camera_matrix, self.camera_dist)
        
        point2d = point2d[0].squeeze()
        points_ref = []
        for p in point2d:
            i, j = [round(p[1]), round(p[0])]
            if i < self.camera_height and i >= 0 and j < self.camera_width and j >= 0:
                points_ref.append([j,i])
        return np.array(points_ref)
    
    


####################################################################################

class RayTracing():

    @staticmethod
    def lineLineIntersection(rays):
        '''
        Computes 3D intersection of rays

        Parameters
        ------------
        rays: list of tuples of np.arrays
                A list of tuples of np.arrays. tuple(0) ray center, tuple(1) ray direction

        Returns
        ------------
        np.array
                Vector 3x1 containing the 3D coordinates

        '''

        lines = []
        points = []
        for r in rays:
            l = np.array([r[1][0], r[1][1], r[1][2]]).reshape(3, 1)
            l = l / np.linalg.norm(l)
            lines.append(l)

            p = np.array([r[0][0], r[0][1], r[0][2]]).reshape(3, 1)
            points.append(p)


        v_l = np.zeros((3, 3))
        v_r = np.zeros((3, 1))
        for index in range(0, len(lines)):
            l = lines[index]
            p = points[index]
            v = np.eye(3) - np.matmul(l, l.T)
            v_l = v_l + v
            v_r = v_r + np.matmul(v, p)

        x = np.matmul(np.linalg.pinv(v_l), v_r)
        return x


    @staticmethod
    def compute3DRay(point_2d, camera_matrix_inv, camera_pose):
        '''
        Computes 3D Ray originated from the camera origin frame and passing by point_2d

        Parameters
        ------------
        point_2d: list
                A list containing the 2D pixel coordinate of the target point where the ray should pass
        camera_matrix_inv: np.array(3,3)
                A matrix defined as the inverse of the camera intrinsics matrix K
        # camera_pose: np.array(4,4)
                A matrix containing the pose of the camera wrt world frame

        Returns
        ------------
        tuples of np.array
            Two tuples of np.array defining the computed line parameters (line center and line direction)
        '''

        point_2d = np.array([point_2d[0], point_2d[1], 1.0]).reshape(3, 1)
        ray = np.matmul(camera_matrix_inv, point_2d)
        ray = ray / np.linalg.norm(ray)
        ray = ray.reshape(3)

        ray_v = np.array([ray[0], ray[1], ray[2], 1])
        ray_v = np.matmul(camera_pose,ray_v)

        ray_v2 = np.array([ray_v[0], ray_v[1], ray_v[2]]).reshape(3,1)
        center = np.array([camera_pose[0,3], camera_pose[1,3], camera_pose[2,3]]).reshape(3,1)

        ray_dir = ray_v2 - center
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        line = (np.array([center[0], center[1], center[2]]), np.array([ray_dir[0], ray_dir[1], ray_dir[2]]))
        return line


    @staticmethod
    def compute(points_dict, camera_matrix, camera_poses):      
        '''
        Computes 3D coordinate of the set of points defined in points_dict

        Parameters
        -----------
        points_dict: dict 
            A dictionary made of group of points. Every key define a group. Goal: get 3D coordinate of the set of 2D points for every key
        camera_matrix: np.array(3,3) 
            intrinsics parameter matrix
        camera_poses: list of np.array(4,4)
            List of matrices containing the poses of the camera wrt world frame, one matrix for each 2D point of the set

        Returns
        output: list of np.array 
            A list containing the 3D coordinates for each point (key) in points_dict
        '''
        output = []
        camera_matrix_inv = np.linalg.inv(camera_matrix)
        for _, points in points_dict.items():
            lines = [RayTracing.compute3DRay(points[i], camera_matrix_inv, camera_poses[i]) for i in range(len(points))]     
            x = RayTracing.lineLineIntersection(lines)
            output.append(x)
        
        return output

    @staticmethod
    def computeFromData(data_dict, camera_matrix, num_points):      
        

        camera_matrix_inv = np.linalg.inv(camera_matrix)

        lines_dict = {i: [] for i in range(num_points)}
        for key, val in data_dict.items():
            for i, p in enumerate(val["points_f"]):
                line = RayTracing.compute3DRay(p, camera_matrix_inv, val["camera_pose"])
                lines_dict[i].append(line)

        output = []
        for key, lines in lines_dict.items():
            x = RayTracing.lineLineIntersection(lines)
            output.append(x)
        return output



    @staticmethod
    def computeSinglePoint(points2d, camera_matrix, camera_poses):      
        '''
        Computes 3D coordinate for a list of 2d points in different views

        Parameters
        -----------
        points2d: list 
            A list of tuples. Each tuple is a 2d point in image space. 
        camera_matrix: np.array(3,3) 
            intrinsics parameter matrix
        camera_poses: list of np.array(4,4)
            List of matrices containing the poses of the camera wrt world frame, one matrix for each 2D point of the set

        Returns
        output: np.array 
            Array of computed 3D coordinates
        '''

        lines = [RayTracing.compute3DRay(points2d[i], np.linalg.inv(camera_matrix), camera_poses[i]) for i in range(len(points2d))]     
        return RayTracing.lineLineIntersection(lines)