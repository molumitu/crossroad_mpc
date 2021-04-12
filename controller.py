from controller_params import MPC_Param
import numpy as np
from scipy.optimize import minimize
from Utils import CROSSROAD_SIZE, horizon, Nc
from predict_surroundings import veh_predict, double_circle_transfer
import mpc_cpp
import time

class MPControl():
    def __init__(self, ref, mpc_params:MPC_Param):

        self.ref = ref
        self.routes_num = ref.routes_num - 1 #减去红灯路线
        self.bounds = mpc_params.bounds
        self.red_bounds = mpc_params.red_bounds
        self.Q = mpc_params.Q
        self.R = mpc_params.R
        self.P = mpc_params.P


        self.tem_action_array = np.zeros((self.routes_num, Nc * 2))

    def step(self, ego_list, n_ego_vehicles_list, traffic_light,time_list, nit_list):
        # ego_list [v_x, v_y, r, x, y, phi, steer_current, a_x_current]
        # n_ego_vehicles_list [x, y, v, a, acc, route]
        ref_best_index = 0

        # 0：left 1:straight 2:right
        # vehicles_array : N*horizon*4   N=8

        n = len(n_ego_vehicles_list)
        vehicles_array = np.zeros((n,horizon,4))
        for i, veh in enumerate(n_ego_vehicles_list):
            vehicles_array[i] = veh_predict(veh, horizon)
        vehicles_xy_array = vehicles_array[:,:,:2].copy()    #用来计算cost function
        vehicles_xy_array_front, vehicles_xy_array_rear = double_circle_transfer(vehicles_array, n) #用来计算安全距离

        safe_dist_front = 2.8
        safe_dist_rear = 2.8

        multi_future_ref_tuple_list = self.ref.multi_future_ref_points(ego_list[3], ego_list[4], horizon)
        future_ref_array = np.array(multi_future_ref_tuple_list[0])

            
        if ((self.ref.routeID in ('dl', 'du') and ego_list[4] < -CROSSROAD_SIZE/2 and (traffic_light == 2)) or\
            (self.ref.routeID in ('ur', 'ud') and ego_list[4] > CROSSROAD_SIZE/2 and (traffic_light == 2)) or\
            (self.ref.routeID in ('lu', 'lr') and ego_list[3] < -CROSSROAD_SIZE/2 and (traffic_light == 0)) or\
            (self.ref.routeID in ('rd', 'rl') and ego_list[3] > CROSSROAD_SIZE/2 and (traffic_light == 0))):
            ##### 红灯模式########################################
            mpc_signal = 4
            future_ref_array = np.array(multi_future_ref_tuple_list[-1])
            red_ineq_cons = {'type': 'ineq',
                'fun' : lambda u: mpc_cpp.red_mpc_constraints_wrapper(u, ego_list, vehicles_xy_array_front, vehicles_xy_array_rear, safe_dist_front, safe_dist_rear),
                'jac': lambda u: mpc_cpp.red_mpc_constraints_jac_wrapper(u, ego_list, vehicles_xy_array_front, vehicles_xy_array_rear, safe_dist_front, safe_dist_rear)
            }
            red_ineq_cons_alpha = {'type': 'ineq',
                'fun' : lambda u: mpc_cpp.red_mpc_alpha_constraints_wrapper(u, ego_list),
                'jac': lambda u: mpc_cpp.red_mpc_alpha_constraints_jac_wrapper(u, ego_list)
            }
            start = time.perf_counter_ns()
            results = minimize(
                    lambda u: mpc_cpp.red_mpc_wrapper(u, ego_list, vehicles_xy_array, future_ref_array, self.Q, self.R, self.P),
                    jac = True,
                    x0 = [0] * Nc,
                    method = 'SLSQP',
                    bounds = self.red_bounds,
                    constraints = [red_ineq_cons],
                    options={'disp': False,
                            'maxiter': 50,
                            'ftol' : 1e-4} 
                    )
            end = time.perf_counter_ns()
            time_list.append(end-start)
            nit_list.append(results.nit)
            result_a_x = results.x
            if not results.success:
                print('red fail!')
                result_a_x[0] = -6
            mpc_action = np.stack((np.zeros(Nc), np.array(result_a_x)), axis = 1).flatten()
            

        #轨迹优选模式
        else:
            ineq_cons = {'type': 'ineq',
            'fun' : lambda u: mpc_cpp.mpc_constraints(u, ego_list, vehicles_xy_array_front, vehicles_xy_array_rear, safe_dist_front, safe_dist_rear),
            'jac': lambda u: mpc_cpp.mpc_constraints_wrapper(u, ego_list, vehicles_xy_array_front, vehicles_xy_array_rear,safe_dist_front, safe_dist_rear)
            }
            ineq_cons_alpha = {'type': 'ineq',
            'fun' : lambda u: mpc_cpp.mpc_alpha_constraints(u, ego_list),
            'jac': lambda u: mpc_cpp.mpc_alpha_constraints_wrapper(u, ego_list)
            }


            result_list = []
            result_fun_list = []
            result_index_list = []
            for i in range(self.routes_num-1):
                future_ref_array = np.array(multi_future_ref_tuple_list[i])
                start = time.perf_counter_ns()
                results = minimize(
                                    lambda u: mpc_cpp.mpc_wrapper(u, ego_list, vehicles_xy_array, future_ref_array, self.Q, self.R, self.P),
                                    jac = True,
                                    x0 = self.tem_action_array[i,:].flatten(),
                                    method = 'SLSQP',
                                    bounds = self.bounds,
                                    constraints = [ineq_cons],
                                    options={'disp': False,
                                            'maxiter': 50,
                                            'ftol' : 1e-4} 
                                    )
                end = time.perf_counter_ns()
                time_list.append(end-start)
                nit_list.append(results.nit)
                if results.success:
                    result_list.append(results)
                    result_fun_list.append(results.fun)
                    result_index_list.append(i)
                    self.tem_action_array[i,:] = np.concatenate((results.x[2:],results.x[-2:]),axis =0)
                    #print(f'results.fun[{i}]',results.fun)
                #     else:
                #         pass
                #         #print(f'[{i}] fail')
                # except ValueError:
                #     valueError_index_list.append(i)
                #     #print('ValueError')

            if not result_list:
                #print('fail')
                mpc_action = [0.] * Nc * 2
                mpc_action[0] = 0
                mpc_action[1] = -6
                future_ref_array = np.array(multi_future_ref_tuple_list[0])
                mpc_signal = 5 #fail
            else:
                min_index = np.argmin(result_fun_list)
                ref_best_index = result_index_list[min_index]
                selected_result =  result_list[min_index]
                mpc_action = selected_result.x
                future_ref_array = np.array(multi_future_ref_tuple_list[ref_best_index])
                mpc_signal = ref_best_index

        if (abs(ego_list[3]) > 25 or abs(ego_list[4]) > 25) and ego_list[6] == 0:
            mpc_action[0] = 0
        return mpc_action[:2], future_ref_array, mpc_signal


