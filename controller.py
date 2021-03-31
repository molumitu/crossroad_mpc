from controller_params import MPC_Param
from scipy.optimize.zeros import VALUEERR
import numpy as np
from scipy.optimize import minimize
from Env_utils import CROSSROAD_SIZE, horizon
from predict_surroundings import veh_predict, double_circle_transfer
import mpc_cpp
import time

class MPControl():
    def __init__(self, ref, mpc_params:MPC_Param):

        self.ref = ref
        self.routes_num = ref.routes_num
        self.bounds = mpc_params.bounds
        self.red_bounds = mpc_params.red_bounds
        # self.result_array = np.zeros((step_length,10+horizon*5 + 1))

        self.Q = mpc_params.Q
        self.R = mpc_params.R
        self.P = mpc_params.P


        self.tem_action_array = np.zeros((self.routes_num, horizon * 2))

    def step(self, ego_list, n_ego_vehicles_list, traffic_light):
        # ego_list [v_x, v_y, r, x, y, phi, steer_current, a_x_current]
        # n_ego_vehicles_list [x, y, v, a, route]
        ref_best_index = 0

        # 0：left 1:straight 2:right
        # vehicles_array : N*horizon*4   N=8
        n = len(n_ego_vehicles_list)
        vehicles_array = np.zeros((n,horizon,4))
        for i, veh in enumerate(n_ego_vehicles_list):
            vehicles_array[i] = veh_predict(veh, horizon)
        vehicles_xy_array = vehicles_array[:,:,:2].copy()    #用来计算cost function
        vehicles_xy_array_front, vehicles_xy_array_rear = double_circle_transfer(vehicles_array, n) #用来计算安全距离
        #########对直行车加不同的dist
        if abs(ego_list[3]) > 25 or abs(ego_list[4]) > 25:
            safe_dist = 2.5 + ego_list[0] / 6
        else:
            safe_dist = 2.5 + 1.5 * ego_list[0] / 6


        ineq_cons = {'type': 'ineq',
            'fun' : lambda u: mpc_cpp.mpc_constraints(u, ego_list, vehicles_xy_array_front, vehicles_xy_array_rear, safe_dist),
            'jac': lambda u: mpc_cpp.mpc_constraints_wrapper(u, ego_list, vehicles_xy_array_front, vehicles_xy_array_rear, safe_dist)
            }

        # # # only static obstacle##################################################################
        # x = np.ones((1,horizon)) * -22
        # y = np.ones((1,horizon)) * -1.5
        # safe_dist = 5.
        # vehicles_xy_array_static = np.stack((x,y),axis = 2)
        # ineq_cons_2 = {'type': 'ineq',
        #     'fun' : lambda u: mpc_cpp.mpc_constraints(u, ego_list, vehicles_xy_array_static, safe_dist)}
        ############################################################################################

        ineq_cons_alpha = {'type': 'ineq',
            'fun' : lambda u: mpc_cpp.mpc_alpha_constraints(u, ego_list),
            'jac': lambda u: mpc_cpp.mpc_alpha_constraints_wrapper(u, ego_list)
            }
        multi_future_ref_tuple_list = self.ref.multi_future_ref_points(ego_list[3], ego_list[4], horizon)
        future_ref_array = np.array(multi_future_ref_tuple_list[0])

            
        if (((self.ref.routeID in ('dl', 'du') and ego_list[4] < -CROSSROAD_SIZE/2 ) or\
            (self.ref.routeID in ('ur', 'ud') and ego_list[4] > CROSSROAD_SIZE/2 ))and (traffic_light != 0)) or\
            (((self.ref.routeID in ('lu', 'lr') and ego_list[3] < -CROSSROAD_SIZE/2 ) or\
            (self.ref.routeID in ('rd', 'rl') and ego_list[3] > CROSSROAD_SIZE/2 ))and (traffic_light != 2)):
            ##### 红灯模式########################################
            mpc_signal = 5
            future_ref_array = np.array(multi_future_ref_tuple_list[-1])
            red_ineq_cons = {'type': 'ineq',
                'fun' : lambda u: mpc_cpp.red_mpc_constraints_wrapper(u, ego_list, vehicles_xy_array_front, vehicles_xy_array_rear, safe_dist + 2),
                'jac': lambda u: mpc_cpp.red_mpc_constraints_jac_wrapper(u, ego_list, vehicles_xy_array_front, vehicles_xy_array_rear, safe_dist + 2)
            }

            red_ineq_cons_alpha = {'type': 'ineq',
                'fun' : lambda u: mpc_cpp.red_mpc_alpha_constraints_wrapper(u, ego_list),
                'jac': lambda u: mpc_cpp.red_mpc_alpha_constraints_jac_wrapper(u, ego_list)
            }
            results = minimize(
                    lambda u: mpc_cpp.red_mpc_wrapper(u, ego_list, vehicles_xy_array, future_ref_array, self.Q, self.R, self.P),
                    jac = True,
                    x0 = [0] * horizon,
                    method = 'SLSQP',
                    bounds = self.red_bounds,
                    constraints = [red_ineq_cons, red_ineq_cons_alpha],
                    options={'disp': False,
                            'maxiter': 100,
                            'ftol' : 1e-4} 
                    )
            result_a_x = results.x
            if not results.success:
                result_a_x[0] = -6
            mpc_action = np.stack((np.zeros(horizon), np.array(result_a_x)), axis = 1).flatten()
            
        else:
            #print('现在是绿灯')
            # if ego_list[0] < 1:
            #     mpc_action = [0, (2 - ego_list[0])**2/15] * horizon
            #     mpc_signal = 4
            # else:
            #print('自车状态',ego_list)
            result_list = []
            result_index_list = []
            valueError_list = []
            for i in range(self.routes_num ):
                future_ref_array = np.array(multi_future_ref_tuple_list[i])
                try:  
                    start = time.time()
                    results = minimize(
                                        lambda u: mpc_cpp.mpc_wrapper(u, ego_list, vehicles_xy_array, future_ref_array, self.Q, self.R, self.P),
                                        jac = True,
                                        x0 = self.tem_action_array[i,:].flatten(),
                                        method = 'SLSQP',
                                        bounds = self.bounds,
                                        constraints = [ineq_cons, ineq_cons_alpha],
                                        #constraints = ineq_cons_alpha,
                                        #constraints = ineq_cons,
                                        options={'disp': False,
                                                'maxiter': 100,
                                                'ftol' : 1e-4} 
                                        )
                    end = time.time()
                    #print("time:", end - start)
                    if results.success:
                        result_list.append(results)
                        result_index_list.append(i)
                        self.tem_action_array[i,:] = np.concatenate((results.x[2:],results.x[-2:]),axis =0)
                        #print(f'results.fun[{i}]',results.fun)
                    else:
                        pass
                        #print(f'[{i}] fail')
                except ValueError:
                    valueError_list.append(i)
                    #print('ValueError')

            if not result_list and not valueError_list:
                print('fail')
                mpc_action = [0.] * horizon * 2
                mpc_action[0] = 0
                mpc_action[1] = -6
                future_ref_array = np.array(multi_future_ref_tuple_list[0])
            elif valueError_list:
                mpc_action = self.tem_action_array[valueError_list[0],:]
                future_ref_array = np.array(multi_future_ref_tuple_list[valueError_list[0]])
            else:
                min_index = np.argmin([result.fun for result in result_list])
                mpc_action = result_list[min_index].x
                #print("choosed ref index :",result_list[min_index])
                ref_best_index = result_index_list[min_index]
                mpc_signal = ref_best_index           
                future_ref_array = np.array(multi_future_ref_tuple_list[ref_best_index])




        if (abs(ego_list[3]) > 28 or abs(ego_list[4]) > 28) and ego_list[6] == 0:
            mpc_action[0] = 0
        return mpc_action[:2]


