import numpy as np
import copy
import glob, os
from termcolor import cprint
from scipy.interpolate import splprep, splev
from skmisc.loess import loess

import rospy
import rospkg


class Filtering:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return "Point: {}, {}, {}".format(self.x, self.y, self.z)
        
    @ staticmethod    
    def distance(p1, p2):
        return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2) ** 0.5

    @ staticmethod    
    def distance_xy(p1, p2):
        return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

    @ staticmethod
    def minDistancePoint(target, list_points):
        min_dist = np.inf
        min_idx = 0
        for i, p in enumerate(list_points):
            dist = Filtering.distance(target, p)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        
        point_selected = list_points[min_idx]
        list_points.pop(min_idx)
        
        return point_selected, min_dist

    @ staticmethod
    def minDistancePointXY(target, list_points):
        min_dist = np.inf
        min_idx = 0
        for i, p in enumerate(list_points):
            dist = Filtering.distance_xy(target, p)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        
        point_selected = list_points[min_idx]
        list_points.pop(min_idx)

        return point_selected, min_dist

    @ staticmethod
    def minDistancePointXYW(target, list_points):
        target_xyz = target[0]
        min_dist = np.inf
        min_idx = 0
        for i, p in enumerate(list_points):
            dist = Filtering.distance_xy(target[0], p[0])
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        
        point_selected = list_points[min_idx]
        list_points.pop(min_idx)

        return point_selected, min_dist


    @ staticmethod
    def filtering(list_points, th=5):
        data_points_2 = [Filtering(p[0], p[1], p[2]) for p in list_points]

        ordered_list = [data_points_2[0]]
        tmp = copy.deepcopy(data_points_2)
        tmp.pop(0)

        old_dist = 0.05
        while len(tmp) > 0:
            new_point, new_dist = Filtering.minDistancePointXY(ordered_list[-1], tmp)
            if new_dist < old_dist:
                ordered_list.append(new_point)
            else:
                break

        xp = np.array([p.x for p in ordered_list])
        yp = np.array([p.y for p in ordered_list])
        zp = np.array([p.z for p in ordered_list]) 

        return xp, yp, zp

    @ staticmethod
    def filteringW(list_points):
        data_points_2 = [[Filtering(p[0], p[1], p[2]), p[3]] for p in list_points]

        ordered_list = [data_points_2[0]]
        tmp = copy.deepcopy(data_points_2)
        tmp.pop(0)

        old_dist = 0.05
        while len(tmp) > 0:
            new_point, new_dist = Filtering.minDistancePointXYW(ordered_list[-1], tmp)
            if new_dist < old_dist:
                ordered_list.append(new_point)
            else:
                break

        xp = np.array([p[0].x for p in ordered_list])
        yp = np.array([p[0].y for p in ordered_list])
        zp = np.array([p[0].z for p in ordered_list]) 
        weights = np.array([p[1] for p in ordered_list]) 
        return xp, yp, zp, weights

    def listoToListDistance(target, query):  
        result = []
        for t in target:
            diffs = [((t[0] - q[0])**2 + (t[1] - q[1])**2 + (t[2] - q[2])**2)**0.5 for q in query]
            result.append(np.min(diffs))

        return result

    @staticmethod
    def computeSpline(points, num_points = 10, k = 3, s = 0):      
        points = np.array(points).squeeze()
        tck, u = splprep(points.T, u=None, k=k, s=s, per=0)
        u_new = np.linspace(u.min(), u.max(), num_points)
        x_, y_, z_ = splev(u_new, tck, der=0)
        return list(zip(x_, y_, z_)) 


class Lowess(object):

    def __init__(self, simulation=True, obj_file=None, data_estimated_path=None):

        #
        rospack = rospkg.RosPack()
        self.main_fodler = rospack.get_path("dlo3ds")
        blender_rendering_folder = os.path.join(self.main_fodler, "utilities/blender_rendering")

        self.simulation = simulation

        if obj_file is None:
            obj_file = rospy.get_param("dlo3ds/obj_file")     

        print("obj file: ", obj_file)
        if self.simulation:
            dlo_id = int(obj_file.split("_")[1])
            self.gt_data = os.path.join(os.path.join(blender_rendering_folder, "data_models/dlo_" + str(dlo_id)), "dlo_" + str(dlo_id) + ".txt")
            print("gt data: ", self.gt_data)
            
        if data_estimated_path is None:
            self.output_txt_3d_data = os.path.join(self.main_fodler, os.path.join(str(rospy.get_param('dlo3ds/output_path')), obj_file))
        else:
            self.output_txt_3d_data = os.path.join(self.main_fodler, data_estimated_path)

        print("DATA PATH: ", self.output_txt_3d_data)


    def computeLowess(self, points, num_scans):
        import statsmodels.api as sm

        xf, yf, zf, weights = Filtering.filteringW(points)

        parameter = np.arange(len(zf))
        frac_ratio = 1/float(num_scans)

        lowess = sm.nonparametric.lowess(zf, parameter, frac=frac_ratio)
        lowess_result = list(zip(*lowess))[1]
        return xf, yf, lowess_result 

    def computeLowessScikit(self, points, num_scans, weights=False):
        
        if weights:
            print("using weights for LOWESS")
            xf, yf, zf, weights = Filtering.filteringW(points)
            weights_inv = np.subtract(np.max(weights), np.array(weights).reshape(-1))
        else:
            print("NOT using weights for LOWESS")
            xf, yf, zf = Filtering.filtering(points)
            weights_inv = np.ones(xf.shape)

        parameter = np.arange(len(zf))
        frac_ratio = 1/float(num_scans)

        l = loess(parameter, zf, weights=weights_inv, span=frac_ratio)
        l.fit()
        pred = l.predict(parameter, stderror=True)
        conf = pred.confidence()

        lowess_result = pred.values
        return xf, yf, lowess_result 

    def loadData(self):
        datas = sorted(glob.glob(os.path.join(self.output_txt_3d_data, "points_*.txt")))
        
        collection_txt_data = [np.loadtxt(d) for d in datas]
        num_scans = len(collection_txt_data)

        points3d_raw = []
        for data in collection_txt_data:
            points3d_raw.extend(data)

        return points3d_raw, num_scans

    def filterHighErrorPoints(self, points):
        points_f = []
        for p in points:
            if p[3] > 2.0:
                print("xxxxx ", p)
                continue
            
            points_f.append(p)

        return points_f
