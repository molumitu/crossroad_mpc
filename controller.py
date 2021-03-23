from scipy.optimize.zeros import VALUEERR
import numpy as np
from scipy.optimize import minimize
from Env_utils import horizon
from predict_surroundings import route_to_task, veh_predict
import mpc_cpp
import time

class MPControl():
    def __init__(self, ref):
        self.ref = ref
        self.routes_num = 3
        self.bounds = [(-0.28, 0.28), (-6.1, 2.8)] * horizon
        self.red_bounds = [(-6.1, 0)] * horizon
        # self.result_array = np.zeros((step_length,10+horizon*5 + 1))

        self.Q = np.array([10, 10, 0., 0., 0])
        self.R = np.array([0.5, 0.1])
        #self.P = np.array([0.5*2*10])
        self.P = np.array([0])
        self.tem_action_array = np.zeros((self.routes_num, horizon * 2))

    def step(self, ego_list, n_ego_vehicles_list, traffic_light):
        # ego_list [v_x, v_y, r, x, y, phi, steer_current, a_x_current]
        # n_ego_vehicles_list [x, y, v, a, route]
        ref_best_index = 0

        # 0：left 1:straight 2:right
        # vehicles_array : N*horizon*4   N=8
        n = len(n_ego_vehicles_list)      # 给python function 用的
        vehicles_array = np.zeros((n,horizon,4))
        for i, veh in enumerate(n_ego_vehicles_list):
            task = route_to_task(veh)
            vehicles_array[i] = veh_predict(veh, horizon)
        vehicles_xy_array = vehicles_array[:,:,:2].copy()
        safe_dist = 5
        ineq_cons = {'type': 'ineq',
            'fun' : lambda u: mpc_cpp.mpc_constraints(u, ego_list, vehicles_xy_array, safe_dist),
            'jac': lambda u: mpc_cpp.mpc_constraints_wrapper(u, ego_list, vehicles_xy_array, safe_dist)
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

        #current_ref_point, future_ref_tuple_list = self.ref.future_ref_points(ego_list[3], ego_list[4], horizon)
        multi_future_ref_tuple_list = self.ref.multi_future_ref_points(ego_list[3], ego_list[4], horizon)
        future_ref_array = np.array(multi_future_ref_tuple_list[0])

            
            
        if (traffic_light == 0) or ego_list[4] > -25:
            print('现在是绿灯')
            if ego_list[0] < 0.2:
                mpc_action = [0, (2 - ego_list[0])**2/12] * horizon
                mpc_signal = 4
            else:
                print('自车状态',ego_list)
                result_list = []
                result_index_list = []
                valueError_list = []
                for i in range(self.routes_num -1 ):
                    future_ref_array = np.array(multi_future_ref_tuple_list[i])
                    print(future_ref_array[0,:])
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
                                                    'maxiter': 1000,
                                                    'ftol' : 1e-4} 
                                            )
                        end = time.time()
                        #print("time:", end - start)
                        if results.success:
                            result_list.append(results)
                            result_index_list.append(i)
                            self.tem_action_array[i,:] = np.concatenate((results.x[2:],results.x[-2:]),axis =0)
                            print(f'results.fun[{i}]',results.fun)
                        else:
                            print(f'[{i}] fail')
                    except ValueError:
                        valueError_list.append(i)
                        #print('ValueError')

                if not result_list and not valueError_list:
                    print('fail')
                    mpc_action = [0.] * horizon * 2
                    if ego_list[0] > 1:
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
        else:
            mpc_signal = 5
            future_ref_array = np.array(multi_future_ref_tuple_list[3])

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
                    constraints = [red_ineq_cons_alpha],
                    options={'disp': False,
                            'maxiter': 1000,
                            'ftol' : 1e-4} 
                    )
            result_a_x = results.x
            mpc_action = np.stack((np.zeros(horizon), np.array(result_a_x)), axis = 1).flatten()
        return mpc_action[:2]


