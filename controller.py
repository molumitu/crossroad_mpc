from numpy.ma.core import concatenate
from controller_params import MPC_Param
import numpy as np
from scipy.optimize import minimize
from Utils import CROSSROAD_SIZE, horizon, Nc, egoID_to_num_dict, init_ego_num, ego_dynamics_keys, other_info_keys
from predict_surroundings import veh_predict, double_circle_transfer, penalty_vehicles_selector
import mpc_cpp
import time
from mpc_cpp import state_trans_LPF

class MPControl():
    def __init__(self, ref, mpc_params:MPC_Param, egoID:str):

        self.egoID = egoID
        self.ref = ref
        self.routes_num = ref.routes_num - 1 #减去红灯路线
        self.bounds = mpc_params.bounds
        self.red_bounds = mpc_params.red_bounds
        self.Q = np.array(mpc_params.Q)
        self.R = np.array(mpc_params.R)
        self.P = np.array(mpc_params.P) 
        self.safe_dist_front = mpc_params.safe_dist_front
        self.safe_dist_rear = mpc_params.safe_dist_rear
        self.red_safe_dist_front = mpc_params.red_safe_dist_front
        self.red_safe_dist_rear = mpc_params.red_safe_dist_rear
        self.route_weight_list = np.array(mpc_params.route_weight_list)
        self.tem_action_array = np.zeros((self.routes_num, Nc * 2))
        self.current_route_num = 0

    def step(self, ego_list, surround_vehicles_list, surround_vehicles_list_without_egos, each_ego_egoID_list, traffic_light,ego_future_position,time_list, nit_list):
        # ego_list [v_x, v_y, r, x, y, phi, steer_current, a_x_current]
        # surround_vehicles_list [x, y, v, a, acc, route]
        ref_best_index = self.current_route_num
        # 0：left 1:straight 2:right
        # vehicles_array : N*horizon*4   N=8

        n0 = len(surround_vehicles_list)

        surround_vehicles_need_penalty = penalty_vehicles_selector(traffic_light, surround_vehicles_list)#用来计算cost function
        n1 = len(surround_vehicles_need_penalty)
        vehicles_array = np.zeros((n1,horizon,4))
        for i, veh in enumerate(surround_vehicles_need_penalty):
            vehicles_array[i] = veh_predict(veh, horizon)
        vehicles_xy_array = vehicles_array[:,:,:2].copy()  

        ##-----------constraints without egos----------------------------------
        n2 = len(surround_vehicles_list_without_egos)
        vehicles_array_without_egos = np.zeros((n2,horizon,4))
        for i, veh in enumerate(surround_vehicles_list_without_egos):
            vehicles_array_without_egos[i] = veh_predict(veh, horizon) #用来计算constraint
        vehicles_xy_array_without_egos = vehicles_array_without_egos[:,:,:2].copy()    
        vehicles_xy_array_front, vehicles_xy_array_rear = double_circle_transfer(vehicles_array_without_egos, n2) #用来计算安全距离
        ##-----------constraints add egos----------------------------------
        for ego, future_position in ego_future_position.items():
            if ego != self.egoID and ego in egoID_to_num_dict.keys() and ego in each_ego_egoID_list:
                vehicles_xy_array_front = np.concatenate((vehicles_xy_array_front, future_position[0]))
                vehicles_xy_array_rear = np.concatenate((vehicles_xy_array_front, future_position[1]))
        


        multi_future_ref_tuple_list = self.ref.multi_future_ref_points(ego_list[3], ego_list[4], horizon)
        future_ref_array = np.array(multi_future_ref_tuple_list[0])

            
        if ((self.ref.routeID in ('dl', 'du') and ego_list[4] < -CROSSROAD_SIZE/2 and (traffic_light == 2 or  traffic_light == 1 or traffic_light == 3)) or\
            (self.ref.routeID in ('ur', 'ud') and ego_list[4] > CROSSROAD_SIZE/2 and (traffic_light == 2 or traffic_light == 1 or traffic_light == 3)) or\
            (self.ref.routeID in ('lu', 'lr') and ego_list[3] < -CROSSROAD_SIZE/2 and (traffic_light == 0 or traffic_light == 3 or traffic_light == 1)) or\
            (self.ref.routeID in ('rd', 'rl') and ego_list[3] > CROSSROAD_SIZE/2 and (traffic_light == 0 or traffic_light == 3 or traffic_light == 1))):
            ##### 红灯模式########################################
            mpc_signal = 4
            
            future_ref_array = np.array(multi_future_ref_tuple_list[-1])
            red_ineq_cons = {'type': 'ineq',
                'fun' : lambda u: mpc_cpp.red_mpc_constraints_wrapper(u, ego_list, vehicles_xy_array_front, vehicles_xy_array_rear, self.red_safe_dist_front, self.red_safe_dist_rear),
                'jac': lambda u: mpc_cpp.red_mpc_constraints_jac_wrapper(u, ego_list, vehicles_xy_array_front, vehicles_xy_array_rear, self.red_safe_dist_front, self.red_safe_dist_rear)
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
                #print('red fail!')
                result_a_x[0] = -8
            mpc_action = np.stack((np.zeros(Nc), np.array(result_a_x)), axis = 1).flatten()
            self.current_route_num = 0 #相当于保持不变

        #轨迹优选模式
        else:
            fix_route = False
            if (abs(ego_list[3]) > 20 or abs(ego_list[4]) > 20):
                P = self.P/10
            else:
                P = self.P
            if (abs(ego_list[3]) > 24 or abs(ego_list[4]) > 24):
                fix_route = True
                routes_list = [self.current_route_num]
                self.route_weight_list[self.current_route_num] = 0.7
            else:
                routes_list = [i for i in range(self.routes_num)]

            ineq_cons = {'type': 'ineq',
            'fun' : lambda u: mpc_cpp.mpc_constraints(u, ego_list, vehicles_xy_array_front, vehicles_xy_array_rear, self.safe_dist_front, self.safe_dist_rear),
            'jac': lambda u: mpc_cpp.mpc_constraints_wrapper(u, ego_list, vehicles_xy_array_front, vehicles_xy_array_rear,self.safe_dist_front, self.safe_dist_rear)
            }
            ineq_cons_alpha = {'type': 'ineq',
            'fun' : lambda u: mpc_cpp.mpc_alpha_constraints(u, ego_list),
            'jac': lambda u: mpc_cpp.mpc_alpha_constraints_wrapper(u, ego_list)
            }


            result_list = []
            result_fun_list = []
            result_index_list = []

            self.route_weight_list[ref_best_index] = 0.7

            for i in [0]:
                future_ref_array = np.array(multi_future_ref_tuple_list[i])
                start = time.perf_counter_ns()
                results = minimize(
                                    lambda u: mpc_cpp.mpc_wrapper(u, ego_list, vehicles_xy_array, future_ref_array, self.Q, self.R, P),
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
                    result_fun_list.append(results.fun * self.route_weight_list[i])
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
                mpc_action = [0., -8] * Nc
                future_ref_array = np.array(multi_future_ref_tuple_list[0])
                mpc_signal = 5
            else:
                min_index = np.argmin(result_fun_list)
                ref_best_index = result_index_list[min_index]
                selected_result =  result_list[min_index]
                mpc_action = selected_result.x
                future_ref_array = np.array(multi_future_ref_tuple_list[ref_best_index])
                mpc_signal = ref_best_index
        self.current_route_num = ref_best_index    
        if (abs(ego_list[3]) > 26 or abs(ego_list[4]) > 26) and ego_list[6] == 0:
            mpc_action[0] = 0
        future_position_array = self.cal_future_position(ego_list, mpc_action)

        return mpc_action, future_ref_array, ref_best_index, future_position_array

    def cal_future_position(self, ego_list:list, mpc_action:np.ndarray):
        ego_future_position_array = np.zeros((1, horizon, 4))
        state = ego_list.copy()
        for i in range(horizon):
            action = mpc_action[2*i: 2*(i+1)]
            state, params = state_trans_LPF(state, action)
            ego_future_position_array[0, i, 0] = state[3]
            ego_future_position_array[0, i, 1] = state[4]
            ego_future_position_array[0, i, 2] = state[0]
            ego_future_position_array[0, i, 3] = state[5]
        future_position_front, future_position_rear = double_circle_transfer(ego_future_position_array, 1)
        return future_position_front, future_position_rear



