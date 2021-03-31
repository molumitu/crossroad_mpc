        
from ipopt_solver import create_solver_with_cons
from ipopt_params import IPOPT_Param
from predict_surroundings import route_to_task, veh_predict
from Utils import horizon, STEP_TIME
import time
import numpy as np

class MPControl_ipopt:
    def __init__(self, ref, ipopt_params:IPOPT_Param):
        self.ref = ref
        self.routes_num = ipopt_params.routes_num

        # self.result_array = np.zeros((step_length,10+horizon*5 + 1))

        self.Q = ipopt_params.Q
        self.R = ipopt_params.R
        #self.P = np.array([0.5*2*10])
        self.P = ipopt_params.P

        self.tem_action_array = np.zeros((self.routes_num, horizon * 2))

        self.lbx = ipopt_params.lbx
        self.ubx = ipopt_params.ubx
        self.red_lbx = ipopt_params.red_lbx
        self.red_ubx = ipopt_params.red_ubx
        self.solver = create_solver_with_cons(horizon, STEP_TIME, n_vehicles=0)



    def step_(self, ego_list, n_ego_vehicles_list, traffic_light):
        n = len(n_ego_vehicles_list)

        self.solver = create_solver_with_cons(horizon, STEP_TIME, n_vehicles= n)

        vehicles_array = np.zeros((n,horizon,4))
        for i, veh in enumerate(n_ego_vehicles_list):
            task = route_to_task(veh)
            vehicles_array[i] = veh_predict(veh, horizon)
        vehicles_xy_array = vehicles_array[:,:,:2].copy()

        multi_future_ref_tuple_list = self.ref.multi_future_ref_points(ego_list[3], ego_list[4], horizon)

        future_ref_array = np.array(multi_future_ref_tuple_list[0])
        future_x = list(future_ref_array[:,0])
        future_y = list(future_ref_array[:,1])
        future_phi = list(future_ref_array[:,2])
        vehs_x = list(vehicles_xy_array[:,:,0].flatten())
        vehs_y = list(vehicles_xy_array[:,:,1].flatten())


        params = list(ego_list) + future_x + future_y + future_phi + vehs_x + vehs_y

        lbg = np.zeros((1+4+n)*horizon)
        ubg = np.ones((1+4+n)*horizon) * float('inf')

        sol=self.solver(x0=self.tem_action_array[0,:],lbx=self.lbx,ubx=self.ubx, lbg=lbg, ubg=ubg, p=params)

        print(sol)

        mpc_action = sol['x']
    
        mpc_action_array = np.array(mpc_action).squeeze()

        return mpc_action_array[:2]


    def step(self, ego_list, n_ego_vehicles_list, traffic_light):
        # ego_list [v_x, v_y, r, x, y, phi, steer_current, a_x_current]
        # n_ego_vehicles_list [x, y, v, a, route]
        ref_best_index = 0
        # 0：left 1:straight 2:right
        # vehicles_array : N*horizon*4   N=8
        n = len(n_ego_vehicles_list)
        self.solver = create_solver_with_cons(horizon, STEP_TIME, n_vehicles= n)
        vehicles_array = np.zeros((n,horizon,4))
        for i, veh in enumerate(n_ego_vehicles_list):
            task = route_to_task(veh)
            vehicles_array[i] = veh_predict(veh, horizon)
        vehicles_xy_array = vehicles_array[:,:,:2].copy()
        safe_dist = 4 + 2 * ego_list[0] / 6


        # # # only static obstacle##################################################################
        # x = np.ones((1,horizon)) * -22
        # y = np.ones((1,horizon)) * -1.5
        # safe_dist = 5.
        # vehicles_xy_array_static = np.stack((x,y),axis = 2)
        # ineq_cons_2 = {'type': 'ineq',
        #     'fun' : lambda u: mpc_cpp.mpc_constraints(u, ego_list, vehicles_xy_array_static, safe_dist)}
        ############################################################################################

        #current_ref_point, future_ref_tuple_list = self.ref.future_ref_points(ego_list[3], ego_list[4], horizon)
        multi_future_ref_tuple_list = self.ref.multi_future_ref_points(ego_list[3], ego_list[4], horizon)
        future_ref_array = np.array(multi_future_ref_tuple_list[0])

            
            
        if (traffic_light == 0) or ego_list[4] > -25:
            print('现在是绿灯')
            if ego_list[0] < 0.1:
                mpc_action = [0, (2 - ego_list[0])**2/15] * horizon
                mpc_action_array = np.array(mpc_action).squeeze()
                mpc_signal = 4
            else:

                result_list = []
                result_index_list = []
                for i in range(self.routes_num):
                    future_ref_array = np.array(multi_future_ref_tuple_list[i])
                    future_x = list(future_ref_array[:,0])
                    future_y = list(future_ref_array[:,1])
                    future_phi = list(future_ref_array[:,2])
                    vehs_x = list(vehicles_xy_array[:,:,0].flatten())
                    vehs_y = list(vehicles_xy_array[:,:,1].flatten())
                    params = list(ego_list) + future_x + future_y + future_phi + vehs_x + vehs_y

                    lbg = np.zeros((1+4+n)*horizon)
                    ubg = np.ones((1+4+n)*horizon) * float('inf')

                    sol=self.solver(x0=self.tem_action_array[i,:],lbx=self.lbx,ubx=self.ubx, lbg=lbg, ubg=ubg, p=params)

                    result_list.append(sol)
                    result_index_list.append(i)

                if not result_list:
                    print('fail')
                    mpc_action = [0.] * horizon * 2
                    mpc_action[0] = 0
                    mpc_action[1] = -6
                    future_ref_array = np.array(multi_future_ref_tuple_list[ref_best_index])
                else:
                    min_index = np.argmin([sol['f'] for sol in result_list])
                    
                    #print("choosed ref index :",result_list[min_index])
                    ref_best_index = result_index_list[min_index]
                    mpc_signal = ref_best_index           
                    future_ref_array = np.array(multi_future_ref_tuple_list[ref_best_index])

                    mpc_action = result_list[min_index]['x']
                    mpc_action_array = np.array(mpc_action).squeeze()
                    self.tem_action_array[ref_best_index,:] = np.concatenate((mpc_action_array[2:],mpc_action_array[-2:]),axis =0)
        else:
            ##### 红灯模式########################################
            mpc_signal = 5
            future_ref_array = np.array(multi_future_ref_tuple_list[-1])
            future_x = list(future_ref_array[:,0])
            future_y = list(future_ref_array[:,1])
            future_phi = list(future_ref_array[:,2])
            vehs_x = list(vehicles_xy_array[:,:,0].flatten())
            vehs_y = list(vehicles_xy_array[:,:,1].flatten())
            params = list(ego_list) + future_x + future_y + future_phi + vehs_x + vehs_y

            lbg = np.zeros((1+4+n)*horizon)
            ubg = np.ones((1+4+n)*horizon) * float('inf')

            sol=self.solver(x0=-6 * np.ones((1, horizon*2)),lbx=self.red_lbx,ubx=self.red_ubx, lbg=lbg, ubg=ubg, p=params)
            if not all(np.array(sol['g']) > -1e-4):
                print('fail')
            mpc_action = sol['x']
            mpc_action_array = np.array(mpc_action).squeeze()
            mpc_action_array[0] = 0
            print(sol['f'])
            print(sol['x'])
            

        
        # if np.abs(ego_list[3]) > 26 or np.abs(ego_list[4]) > 26:
        #     mpc_action[0] = 0
        return mpc_action_array[:2]