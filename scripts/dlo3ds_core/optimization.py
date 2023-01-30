import numpy as np
from termcolor import cprint
from scipy.optimize import minimize, LinearConstraint


# x[0] baseline
# x[1] altezza camera
ret = (0, 0)

class OptimizeCamera():

    def __init__(self, camera_matrix, distort_vec, camera_height, camera_width, num_points, delta_z, z_min, offset):

        self.camera_matrix = camera_matrix
        self.distort_vec = distort_vec
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.num_points = int(num_points) 

        self.z_min_distance = z_min
        self.z_max_distance = z_min + 0.15
        self.offset = offset
        self.z_min = - delta_z
        self.z_max = delta_z

        # initial condition
        self.x0 = np.array([0.05, 0.0])

        # Bounds
        self.bounds = [(0.0, np.inf), (self.z_min, self.z_max)]

        print(
        '''\nOptimize Camera parameters
        \tbounds: {}
        \txo: {}
        \toffset: {}
        \tz_min_distance: {}'''.format(self.bounds, self.x0, self.offset, self.z_min_distance))

        #################


    
    @staticmethod
    def cost_function(x, point_3d_list, camera_matrix) :
        '''
        x_0 = baseline
        x_1 = z camera
        '''
        res = 0
        for point_3d in point_3d_list:
            res += (point_3d[2] + x[1])**2 / (x[0] * camera_matrix[0,0])

        return res/len(point_3d_list)


    def get_linear_constraints(self, point_3d_list, camera_matrix, camera_width, offset) :
        
        coeff_con = []
        inf_con = []
        sup_con = []


        for point_3d in point_3d_list:
            p_x = point_3d[0] * camera_matrix[0,0] / point_3d[2] + camera_matrix[0,2]

            coeff_con.append([0, 1])
            inf_con.append(self.z_min_distance - point_3d[2])
            sup_con.append(self.z_max_distance - point_3d[2])

            # + baseline
            coeff_con.append([ + camera_matrix[0,0], p_x + offset])
            inf_con.append(- p_x * point_3d[2] + offset * point_3d[2])
            sup_con.append(np.inf)

            coeff_con.append([ + camera_matrix[0,0], p_x - camera_width + offset])
            inf_con.append(-np.inf)
            sup_con.append((- p_x + camera_width - offset) * point_3d[2])

            # - baseline
            coeff_con.append([ - camera_matrix[0,0], p_x + offset])
            inf_con.append(- p_x * point_3d[2] + offset * point_3d[2])
            sup_con.append(np.inf)

            coeff_con.append([ - camera_matrix[0,0], p_x - camera_width + offset])
            inf_con.append(-np.inf)
            sup_con.append((- p_x + camera_width - offset) * point_3d[2])


        coeff_con = np.array(coeff_con, dtype=np.float32)
        inf_con = np.array(inf_con, dtype=np.float32)
        sup_con = np.array(sup_con, dtype=np.float32)

        return LinearConstraint(A = coeff_con, lb = inf_con, ub = sup_con)


    def run_optimization(self, point_3d_list):

        self.point_3d_list = np.array(point_3d_list).reshape(len(point_3d_list), 3)

        # Constraints
        self.linear_constraint = self.get_linear_constraints(self.point_3d_list, self.camera_matrix, self.camera_width, self.offset)
    
        cprint("\n## Start optmiziation ##", "yellow")
        res = minimize(fun = OptimizeCamera.cost_function, 
                        x0 = self.x0, 
                        args = (self.point_3d_list, self.camera_matrix),
                        bounds = self.bounds, 
                        constraints = self.linear_constraint,
                        method="trust-constr",
                        options = {'disp': False, 'maxiter': 1e8}
                        )

        print("method: \t{}".format(res.method))
        print("message: \t{}".format(res.message))
        print("success: \t{}".format(res.success))
        print("baseline: \t{0:.4f}".format(res.x[0]))
        print("delta_z: \t{0:.4f}".format(res.x[1]))
        cprint("## End optmiziation ##\n", "yellow")
        return res.x


    def run_optimization_rv(self, point_3d_list, queue):

        self.point_3d_list = np.array(point_3d_list).reshape(len(point_3d_list), 3)

        # Constraints
        self.linear_constraint = self.get_linear_constraints(self.point_3d_list, self.camera_matrix, self.camera_width, self.offset)
    
        cprint("\n## Start optmiziation ##", "yellow")
        res = minimize(fun = OptimizeCamera.cost_function, 
                        x0 = self.x0, 
                        args = (self.point_3d_list, self.camera_matrix),
                        bounds = self.bounds, 
                        constraints = self.linear_constraint,
                        method="trust-constr",
                        options = {'disp': False, 'maxiter': 1e8}
                        )

        print("method: \t{}".format(res.method))
        print("message: \t{}".format(res.message))
        print("success: \t{}".format(res.success))
        print("baseline: \t{0:.4f}".format(res.x[0]))
        print("delta_z: \t{0:.4f}".format(res.x[1]))
        cprint("## End optmiziation ##\n", "yellow")

        ret = queue.get()
        ret = (res.x[0], res.x[1])
        queue.put(ret)


    def run_optimization_with_timeout(self, point_3d_list, timer=5):

        import multiprocessing

        queue = multiprocessing.Queue()
        queue.put(ret)
        p = multiprocessing.Process(target=self.run_optimization_rv, args=(point_3d_list, queue))
        p.start()

        # Wait for 10 seconds or until process finishes
        p.join(timer)

        if p.is_alive(): # If thread is still active
            cprint("still running, killing...", "red")
            p.kill()
            p.join()
            return False, None
        
        return True, queue.get()
        





if __name__ == "__main__":
    opt = OptimizeCamera()
    opt.run_optimization()

