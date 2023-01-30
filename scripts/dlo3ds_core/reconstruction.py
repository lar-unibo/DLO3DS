import math, copy, cv2
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Pose 
from termcolor import cprint

import matplotlib.pyplot as plt



####################################################################################

class SplineReconstruction(object):

    def __init__(self, camera_matrix, distort_vec, camera_width, camera_height, num_points, overlap_scans):
        self.camera_matrix = camera_matrix
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.num_points = num_points
        self.distort_vec = distort_vec
        self.overlap_scans = overlap_scans

    def changeRange(self, value, r1, r2):
        delta1 = r1[1] - r1[0]
        delta2 = r2[1] - r2[0]
        return (delta2 * (value - r1[0]) / delta1)


    ########################################################
    # REFERENCE SPLINE
    ########################################################

    def chooseReferenceSplineInit(self, splines_set):
        row_values = np.linspace(0, self.camera_height, self.num_points)
        points_ref_init = [[self.camera_width // 2, xx] for xx in row_values]
        spline_ref, _ = self.splinesDiscriminator(points_ref_init, splines_set)
        points_ref = self.evaluateSpline(spline_ref)
        return points_ref


    def splinesDiscriminator(self, points_ref, splines, splines_evs = False):      
        
        if len(splines) == 0:
            cprint("NO splines available!", "red")

        slopes = []
        for spline in splines:
            if splines_evs is False: 
                spline_ev = self.evaluateSpline(spline)
            else:
                spline_ev = spline 

            d = [points_ref[i][0] - spline_ev[i][0] for i, _ in enumerate(spline_ev)]
            model = np.polyfit(np.arange(len(d)), d, 1)           
            slopes.append(model[0])

        if len(slopes) > 0:
            idx = np.argmin(slopes)
        else:
            idx = 0
        return splines[idx], idx


    ###############################################
    # FILTERING                                   #
    ###############################################     
    



    def coarseFiltering1(self, data_dict, debug = False):
        
        distances_list = [np.sum(np.sqrt( np.sum( np.diff( np.array(values["points"]), axis=0)**2, axis=1))) for _, values in data_dict.items() ]
        tot_distance_ref = np.min(distances_list)
        idx = np.argmin(distances_list)
        u_ref = np.linspace(0, 1, self.num_points)
        spline_points_ref = self.evaluateSpline(data_dict[idx]["spline"], u=u_ref)
        if data_dict[idx]["reverse"]:
            spline_points_ref = np.flip(spline_points_ref, axis=0)

        # we need to match each point_ref with a spline_point...
        for cc, values in data_dict.items():
            #print("cc ", cc)
            spline_points = self.evaluateSpline(values["spline"], num_points=self.num_points)
            if values["reverse"]:
                spline_points = np.flip(spline_points, axis=0)

            tot_spline_distance = np.sum(np.sqrt( np.sum( np.diff(spline_points, axis=0)**2, axis=1)))
            #print("tot spline distance and ref: ", tot_spline_distance, tot_distance_ref)

            u_new = [1 - self.changeRange(u, [0, tot_spline_distance], [0, tot_distance_ref]) for u in np.linspace(0, 1, self.num_points)]
            spline_points_new = self.evaluateSpline(values["spline"], u=u_new)
            tot_spline_distance = np.sum(np.sqrt( np.sum( np.diff(spline_points_new, axis=0)**2, axis=1)))    
            #print("tot spline distance after: ", tot_spline_distance)

            ############################
            u_eval = [1 - u for u in np.linspace(0.0, 1.0, 1000)]
            points_to_eval = self.evaluateSpline(values["spline"], u=u_eval)
            xs, ys = zip(*points_to_eval)
            xs = np.array(xs)
            ys = np.array(ys)

            # closest point to ref spline y coordinate
            #index = (np.fabs(np.asarray(points_to_eval[:,1]) - spline_points_ref[0][1])).argmin() # what if multiple points ???
            #difference = np.subtract(points_to_eval[index], spline_points_new[0])
            #distance = np.sqrt(np.sum(np.square(difference)))  

            # check number of roots
            roots = self.findRootsApprox(xs, ys - spline_points_ref[0][1])
            #print("# roots: ", len(roots))
            if len(roots) == 0:
                #print("roots are zeroo, trying other endpoint!")
                roots = self.findRootsApprox(xs, ys - spline_points_ref[-1][1])
                #print("# roots: ", len(roots))

            if len(roots) == 0:
                #print("roots still zero, trying extrapolating...")
                u_eval = [1 - u for u in np.linspace(-0.05, 1.05, 1000)]
                points_to_eval = self.evaluateSpline(values["spline"], u=u_eval)
                xs, ys = zip(*points_to_eval)
                xs = np.array(xs)
                ys = np.array(ys)       
                roots = self.findRootsApprox(xs, ys - spline_points_ref[0][1])
                #print("# roots: ", len(roots))

            if len(roots) == 1:
                #print("roots is 1")
                index = self.idMinimumDistancePointToList((roots[0], spline_points_ref[0][1]), points_to_eval)
                difference = np.subtract(points_to_eval[index], spline_points_new[0])
                distance = np.sqrt(np.sum(np.square(difference)))
            else:
                #print("roots is not 1")
                #print("trying other endpoint")
                roots = self.findRootsApprox(xs, ys - spline_points_ref[-1][1])
                if len(roots) == 1:
                    index = self.idMinimumDistancePointToList((roots[0], spline_points_ref[-1][1]), points_to_eval)
                    difference = np.subtract(points_to_eval[index], spline_points_new[-1])
                    distance = np.sqrt(np.sum(np.square(difference)))   
                else:
                    print("error!!!! xxxxxxxxxxxx")
                    print("roots:", roots)
                    print("y values: ", spline_points_ref[0][1], spline_points_ref[-1][1])

                    canvas = values["img"].copy()
                    for px, py in points_to_eval:
                        cv2.circle(canvas, tuple([int(px), int(py)]), 3, (0, 0, 255), -1)
                    for px, py in spline_points_ref:
                        cv2.circle(canvas, tuple([int(px), int(py)]), 5, (0, 255, 255), -1)


                    #cv2.imshow("img", canvas)
                    #cv2.waitKey(0)

                    index = self.idMinimumDistancePointToList((np.mean(roots), spline_points_ref[-1][1]), points_to_eval)
                    difference = np.subtract(points_to_eval[index], spline_points_new[-1])
                    distance = np.sqrt(np.sum(np.square(difference)))                 

            u_delta = np.sign(difference[1]) * (distance / tot_spline_distance)
            #print("distance and u delta: ", distance, u_delta)

            #####################################
                            
            u_new2 = [u + u_delta for u in u_new]
            spline_points_new2 = self.evaluateSpline(values["spline"], u=u_new2)

            data_dict[cc]["points_i"] = copy.deepcopy(spline_points_new2)

            if debug: 
                canvas = values["img"].copy()
                cv2.circle(canvas, tuple([int(spline_points_ref[0][0]), int(spline_points_ref[0][1])]), 7, (255, 255, 0), -1)
                cv2.circle(canvas, tuple([int(spline_points_ref[-1][0]), int(spline_points_ref[-1][1])]), 7, (255, 255, 0), -1)

                for px, py in spline_points:
                    cv2.circle(canvas, tuple([int(px), int(py)]), 2, (255, 0, 0), -1)

                for px, py in spline_points_ref:
                    cv2.circle(canvas, tuple([int(px), int(py)]), 3, (0, 255, 0), -1)


                for px, py in spline_points_new:
                    cv2.circle(canvas, tuple([int(px), int(py)]), 3, (0, 255, 255), -1)

                for px, py in spline_points_new2:
                    cv2.circle(canvas, tuple([int(px), int(py)]), 3, (0, 0, 255), -1)

                cv2.imshow("canvas" + str(cc), canvas)

        if debug: cv2.waitKey(0)

        return data_dict, copy.deepcopy(spline_points_ref)




    def refineFiltering1(self, data_dict, points_ref, debug = False):
        
        for cc, values in data_dict.items():
            us = np.linspace(-0.05, 1.05, 1000)
            spline_points = self.evaluateSpline(values["spline"], u=us)       
            if values["reverse"]:
               spline_points = np.flip(spline_points, axis=0)

            #print("cc ", cc, spline_points[0], spline_points[-1])

            xs,ys = zip(*spline_points)
            xs = np.array(xs)
            ys = np.array(ys)
           
            spline_points_new = []
            old_u_value = 0
            for t, point_ref in enumerate(points_ref):
                roots = self.findRootsApprox(xs, ys - point_ref[1])
                #print("roots: ", roots)
 

                if len(roots) > 0:
                    u_value_coarse = us[self.idMinimumDistancePointToList(values["points_i"][t], spline_points)]

                    # parameter u associated at each root
                    u_values = []
                    u_values_diff = []
                    for r in roots:
                        u_value = us[self.idMinimumDistancePointToList((r, point_ref[1]), spline_points)]
                        u_values.append(u_value)
                        u_values_diff.append(np.abs(u_value - u_value_coarse))

                    if t > 0:
                        gaps_vs_old = np.array([np.abs(np.abs(u - old_u_value) - old_u_gap) for u in u_values])
                        #print("gaps_vs_old: ", gaps_vs_old)
                        #print("u values: ", u_values, u_values_diff)
                        #print("u_value_coarse ", u_value_coarse)

                        for _ in range(len(roots)): 
                            idx = np.argmin(gaps_vs_old)
                            if u_values[idx] <= old_u_value:
                                gaps_vs_old[idx] = 1000
                            else:
                                break
                        
                        #print("u chosen: ", u_values[idx])
                        r_selected = roots[idx]
                        old_u_gap = u_values[idx] - old_u_value
                        old_u_value = u_values[idx]

                    else:
                        r_selected = roots[np.argmin(u_values_diff)]
                        old_u_value = u_values[np.argmin(u_values_diff)]
                        old_u_gap = u_values[np.argmin(u_values_diff)] - u_value_coarse
                        #print("selection: ", old_u_value, old_u_gap)

                    new_point = [r_selected, point_ref[1]]
                    spline_points_new.append(new_point)

                    canvas = values["img"].copy()

                    '''
                    for r in roots:
                        cv2.circle(canvas, tuple([int(r), int(point_ref[1])]), 5, (0, 255, 255), -1)
                    cv2.circle(canvas, tuple([int(new_point[0]), int(new_point[1])]), 3, (0, 0, 255), -1)
                    cv2.circle(canvas, tuple([int(values["points_i"][t][0]), int(values["points_i"][t][1])]), 3, (0, 255, 0), -1)
                    cv2.imshow("canvas" + str(cc), canvas)
                    cv2.waitKey(0)
                    '''
                    
                    
                else:
                    # alternative approach, useless with findRootsApprox
                    print("roots are zero....trying alternative approach")
                    spline_points_np = np.array(spline_points)      
                    p1 = np.array([0, point_ref[1]])
                    p2 = np.array([self.camera_width, point_ref[1]])
                    dev = np.absolute(np.cross(p2-p1,spline_points_np-p1)/np.linalg.norm(p2-p1))

                    spline_points_dev = np.hstack([spline_points_np, dev.reshape(-1,1)])
                    spline_points_dev_f = spline_points_dev[np.argmin(spline_points_dev[:,-1])].reshape(-1, 3)[0]
                    spline_points_new.append([spline_points_dev_f[0], spline_points_dev_f[1]])

                    u_value = us[self.idMinimumDistancePointToList(spline_points_new[-1], spline_points)]
                    u_value_coarse = us[self.idMinimumDistancePointToList(values["points_i"][t], spline_points)]
                    old_u_value = u_value
                    old_u_gap = u_value - u_value_coarse

            data_dict[cc]["points_f"] = copy.deepcopy(spline_points_new)

            if debug:
                canvas = values["img"].copy()

                for px, py in values["points_i"]:
                    cv2.circle(canvas, tuple([int(px), int(py)]), 5, (0, 255, 255), -1)

                for px, py in spline_points_new:
                    cv2.circle(canvas, tuple([int(px), int(py)]), 3, (0, 255, 0), -1)

                cv2.imshow("canvas" + str(cc), canvas)
        if debug: cv2.waitKey(0)

        return data_dict



    def distance2D(self, point1, point2):
        return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

    def idMinimumDistancePointToList(self, point, list_points):
        return np.argmin([self.distance2D(point, p) for p in list_points])


    def findRoots(self, x,y):
        # https://stackoverflow.com/questions/46909373/how-to-find-the-exact-intersection-of-a-curve-as-np-array-with-y-0/46911822#46911822
        s = np.abs(np.diff(np.sign(y))).astype(bool)
        return x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1)


    def findRootsApprox(self, x, y):
        total_roots = []
        roots = self.findRoots(x,y)
        for r in roots:
            total_roots.append(r)

        #print("standard roots: ", total_roots)
        for i in range(1, 3):
            roots1 = self.findRoots(x,y+i)
            if len(total_roots) == 0:
                total_roots.extend(roots1)
            else:
                to_append = []
                for r in roots1:
                    if np.min([np.abs(t-r) for t in total_roots]) > 10 * i:
                        to_append.append(r)
                total_roots.extend(to_append)

            roots1 = self.findRoots(x,y-i)
            if len(total_roots) == 0:
                total_roots.extend(roots1)
            else:
                to_append = []
                for r in roots1:
                    if np.min([np.abs(t-r) for t in total_roots]) > 10 * i:
                        to_append.append(r)
                total_roots.extend(to_append)

            #print("total: ", len(total_roots))

        return total_roots


    def getDirectionSpline(self, data_dict):
        for cc, values in data_dict.items():
            spline_points = self.evaluateSpline(values["spline"], num_points=self.num_points)
            
            if spline_points[0][1] < spline_points[-1][1]:
                data_dict[cc]["reverse"] = True
            else:
                data_dict[cc]["reverse"] = False

            #print("cc ", cc, data_dict[cc]["reverse"])
     
        return data_dict    


    def orthFilteringOpt(self, data_dict):

        data_dict = self.getDirectionSpline(data_dict)

        data_dict, spline_points_ref = self.coarseFiltering1(data_dict)       
        
        data_dict = self.refineFiltering1(data_dict, spline_points_ref)

        return data_dict


    def computeSpline(self, points, num_points = 10, k = 3, s = 0, debug = False):      
        points = np.array(points).squeeze()
        tck, u = splprep(points.T, u=None, k=k, s=s, per=0)
        u_new = np.linspace(u.min(), u.max(), num_points)
        x_, y_, z_ = splev(u_new, tck, der=0)
        return list(zip(x_, y_, z_)) 

    def evaluateSpline(self, spline, num_points = 10, u=None) :
        # Evaluate spline points from spline parameters
        t = spline[0]
        c = [spline[1],spline[2]]
        k = int(spline[3])
        tck = (t,c,k)

        if u is None:
            u_new = np.linspace(0, 1, num_points)
        else:
            u_new = u 

        x_new, y_new = splev(u_new, tck, der=0)

        new_points = list(zip(x_new, y_new))      
        return np.array(new_points)





    ########################################################
    # CENTERING                         
    ########################################################

    def convexHull(self, points):
        from scipy.spatial import ConvexHull, Delaunay
        # https://stackoverflow.com/questions/31562534/scipy-centroid-of-convex-hull
        T = Delaunay(points).simplices
        n = T.shape[0]
        W = np.zeros(n)
        C = 0
        for m in range(n):
            sp = points[T[m, :], :]
            W[m] = ConvexHull(sp).volume
            C += W[m] * np.mean(sp, axis=0)
        
        center = C / np.sum(W)
        return center


    def orientationFromSpline(self, spline, current_pose, debug = False):

        if isinstance(current_pose, Pose):
            vec = [ current_pose.position.x, current_pose.position.y, current_pose.position.z, 
                    current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w]
        else:
            vec = current_pose

        angle = - self.getSplineAngle(spline[2], spline[1])
        if debug: print('orientationFromSpline: Angle {}'.format(angle))
        
        quat_init = Rotation.from_quat([vec[3], vec[4], vec[5], vec[6]])
        quat_rot = Rotation.from_euler('z', angle, degrees=False) 
        quat_mul = quat_init * quat_rot
        return quat_mul.as_quat()

    def getSplineAngle(self, x, y, debug = False) :
        covmat = np.cov([x,y])

        eig_values, eig_vecs = np.linalg.eig(covmat)
        largest_index = np.argmax(eig_values)

        dir = - eig_vecs[:, largest_index]  
        dir = dir / np.linalg.norm(dir)
        dir_camera = np.array([1, 0])

        angle = math.atan2( dir_camera[0]*dir[1] - dir_camera[1]*dir[0], dir_camera[0]*dir[0] + dir_camera[1]*dir[1])
        if math.fabs(angle) > math.pi/2:
            angle = - (math.pi - angle)
        
        if debug: print("eig: {}, angle: {}".format(eig_vecs[:, largest_index], angle))
        return angle

    def getXYZ(self, px, py, depth, camera_matrix, debug = False):

        fx = camera_matrix[0,0]
        fy = camera_matrix[1,1] 
        cx = camera_matrix[0,2]
        cy = camera_matrix[1,2]

        x = depth * (px - cx) / fx
        y = depth * (py - cy) / fy
        z = depth

        if debug: print("Pixel values: {}{}, XYZ Values: {}{}{}".format(px, py, x, y, z))
        return x, y, z


    def getMatrixFromVec(self, v):   

        if isinstance(v, Pose):
            vec = [v.position.x, v.position.y, v.position.z, v.orientation.x, v.orientation.y, v.orientation.z, v.orientation.w]
        else:
            vec = v

        T = np.eye(4)
        T[0, 3] = vec[0]
        T[1, 3] = vec[1]
        T[2, 3] = vec[2]
        T[:3,:3] = Rotation.from_quat([vec[3], vec[4], vec[5], vec[6]]).as_matrix()
        return T

        
    def centering(self, spline, current_pose):

        if isinstance(current_pose, Pose):
            vec = [ current_pose.position.x, current_pose.position.y, current_pose.position.z, 
                    current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w]
        else:
            vec = current_pose

        # orientation
        quat_new = self.orientationFromSpline(spline, vec)

        # position
        T = self.getMatrixFromVec(current_pose)
        
        spline_points = self.evaluateSpline(spline) 
        try:
            new_center = self.convexHull(spline_points)
        except:
            new_center = spline_points[len(spline_points)//2]

        middle_point = self.getXYZ(new_center[0], new_center[1], vec[2], self.camera_matrix ) 
        v = np.array([middle_point[0], middle_point[1], middle_point[2], 1]).T
        middle_point = np.matmul(T,v)

        new_pose = [middle_point[0], middle_point[1], vec[2], quat_new[0], quat_new[1], quat_new[2], quat_new[3]]

        return new_pose



    ########################################################
    # FORWARD STEP
    ########################################################
    def forward(self, points3d, camera_pose):

        # control of the overlap
        new_pose = copy.deepcopy(camera_pose)
        last_point = copy.deepcopy(points3d[-1])
        #print("LAST POINT: ", list(last_point))

        new_pose[0] += (last_point[0] - camera_pose[0]) * (2 - self.overlap_scans / 0.5)
        new_pose[1] += (last_point[1] - camera_pose[1]) * (2 - self.overlap_scans / 0.5)

        # rotation
        quat_new = self.orientationForward(points3d, camera_pose)
        new_pose[3] = quat_new[0]
        new_pose[4] = quat_new[1]
        new_pose[5] = quat_new[2]
        new_pose[6] = quat_new[3]
        
        return new_pose 

    ########################################################
    # RECOVERY STEP
    ########################################################
    def recovery(self, points3d, camera_pose, overlap=0.5):

        # control of the overlap
        new_pose = copy.deepcopy(camera_pose)
        last_point = copy.deepcopy(points3d[-1])
        #print("LAST POINT: ", list(last_point))

        new_pose[0] += (last_point[0] - camera_pose[0]) * (2 - overlap / 0.5)
        new_pose[1] += (last_point[1] - camera_pose[1]) * (2 - overlap / 0.5)

        # rotation
        quat_new = self.orientationForward(points3d, camera_pose)
        new_pose[3] = quat_new[0]
        new_pose[4] = quat_new[1]
        new_pose[5] = quat_new[2]
        new_pose[6] = quat_new[3]
        
        return new_pose 


    def project2DSimple(self, camera_pose, points_3d):
        T = np.linalg.inv(camera_pose)
        tvec =np.array(T[0:3, 3])
        rvec, _ = cv2.Rodrigues(T[:3,:3])

        point2d = cv2.projectPoints(np.array(points_3d), rvec, tvec, self.camera_matrix, self.distort_vec)
        point2d = point2d[0].squeeze()
        points_ref = []
        for p in point2d:
            i, j = [round(p[1]), round(p[0])]
            if i < self.camera_height and i >= 0 and j < self.camera_width and j >= 0:
                points_ref.append([j,i])

        return np.array(points_ref)
    

    def orientationForward(self, points3d, camera_pose, spline = None, debug = False):

        Tc = self.getMatrixFromVec(camera_pose)
        points2d = self.project2DSimple(Tc, points3d)    

        if len(points2d) > 0:
            x,y = zip(*points2d)   

            if spline is None:
                if debug: print("spline is none")
                #print(y[-1], x[-1], y[0], x[0])
                dir = np.array([y[-1] - y[0], x[-1] - x[0]])
                dir = dir / np.linalg.norm(dir)

            else:
                
                bottom_point_idx = np.argmin([self.camera_height - yy for yy in y])
                p = points2d[bottom_point_idx]

                if debug: print("spline is not none!")
                spoints = list(zip(spline.cx, spline.cy))

                dist0 = ((spoints[0][0] - p[0])**2 + (spoints[0][1] - p[1])**2)**0.5
                dist1 = ((spoints[-1][0] - p[0])**2 + (spoints[-1][1] - p[1])**2)**0.5
                if debug: print("distances: ", dist0, dist1)

                if dist0 < dist1:
                    new_points = [spoints[i] for i in range(len(spoints)//4-1, -1, -1)]
                else:
                    new_points = [spoints[i] for i in range(3*len(spoints)//4-1, len(spoints))]

                x = [p[0] for p in new_points]
                y = [p[1] for p in new_points]
               
                covmat = np.cov([y,x])
                eig_values, eig_vecs = np.linalg.eig(covmat)
                largest_index = np.argmax(eig_values)

                dir = - eig_vecs[:, largest_index] 
                dir = dir / np.linalg.norm(dir)


            dir_camera = np.array([-1, 0])
            angle = math.atan2( dir_camera[0]*dir[1] - dir_camera[1]*dir[0], dir_camera[0]*dir[0] + dir_camera[1]*dir[1])
            if debug: print("orientationForward -- ANGLE: {}, dir {}".format(angle, dir))
 
        else:
            print("len points2d is zero, keeping angle to 0!")    
            angle = 0        


        quat_init = Rotation.from_quat([camera_pose[3], camera_pose[4], camera_pose[5], camera_pose[6]])
        quat_rot = Rotation.from_euler('z', - angle, degrees=False) 
        quat_mul = quat_init * quat_rot
        return quat_mul.as_quat()



    ################################
    # NEXT SPLINE
    ##################################
    '''
    def selectSplineNext(self, points_3d, splines, camera_pose, debug = False):
        
        #### FROM 3D TO 2D
        Tc = self.getMatrixFromVec(camera_pose)
        points_ref = self.project2DSimple(Tc, points_3d)

        num_points_ref = len(points_ref) # number of points in the image 
        print("num points ref: ", num_points_ref)

        points = np.array(points_ref)
        distances = np.sqrt( np.sum( np.diff(points, axis=0)**2, axis=1)) # point to point distance array
        tot_distance = np.sum(distances)

        # we need to match each point_ref with a spline_point...
        mean_distances = []
        for cc, spline in enumerate(splines):

            spline_points = self.evaluateSpline(spline, num_points=num_points_ref)
            tot_spline_distance = np.sum(np.sqrt( np.sum( np.diff(spline_points, axis=0)**2, axis=1)))
            print("tot_spline_distace : ", tot_spline_distance)
            u_array = np.linspace(0, 1, num_points_ref)
            u_new = []
            for u in u_array:
                u_new.append(1 - self.changeRange(u, [0, tot_spline_distance], [0, tot_distance])) # 1 because we want to reverse it!

            spline_points = self.evaluateSpline(spline, u=u_new)

            difference = np.subtract(spline_points, np.flip(points_ref, axis=0))
            distances = np.sqrt(np.sum(np.square(difference), axis=1))
            
            mean_distance = np.mean(distances)
            mean_distances.append(mean_distance)

        print("mean distances: ", mean_distances)

        return splines[np.argmin(mean_distances)]
    '''

    def selectSplineNext(self, points_3d, splines_set, camera_pose):
        #### FROM 3D TO 2D
        Tc = self.getMatrixFromVec(camera_pose)
        points_ref = self.project2DSimple(Tc, points_3d)

        spline_ref, _ = self.splinesTrackingDiscriminator(points_ref, splines_set)
        return spline_ref


    def splinesTrackingDiscriminator(self, points_ref, splines, positive = True, debug = False):      
        
        if len(splines) == 0:
            cprint("NO splines available!", "red")

   
        dev_values = []
        mean_values = []

        for spline in splines:
            u_eval = [1 - u for u in np.linspace(-0.05, 1.05, 1000)]
            spline_points = self.evaluateSpline(spline, u=u_eval)
            
            d = []
            for pref in points_ref:
                index = self.idMinimumDistancePointToList(pref, spline_points)
                difference = np.subtract(spline_points[index], pref)
                distance = np.sqrt(np.sum(np.square(difference)))  
                d.append(distance)    
            
            mean, std = np.mean(d), np.std(d)
            dev_values.append(std)
            mean_values.append(mean)

        # first step: filter high std_dev
        std_dev_threshold = min(dev_values) + 10 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        stats = list(zip(mean_values, dev_values))
        candidates = [[i, s] for i, s in enumerate(stats) if s[1] < std_dev_threshold]

        if debug:
            print("stats:", stats)
            print("candidates:", candidates)

        idx = 0

        # second step: choose spline with lower distance
        if positive:
            means = [s[1][0] for s in candidates if s[1][0] >= 0]

            if means:
                idx = candidates[np.argmin(means)][0]
            else:
                #print("splinesDiscriminator error: means is empty! trying positive=False")
                positive = False
                    
        if positive is False:
            means = [s[1][0] for s in candidates if s[1][0] < 0]
            if means:  
                idx = candidates[np.argmax(means)][0]            
            else:
                print("splinesDiscriminator error: means is empty!")
        
        if debug: print("idx: {}, means: {}".format(idx, means[idx]))

        return splines[idx], idx