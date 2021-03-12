from scipy.optimize.zeros import VALUEERR
from Reference import ReferencePath
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import minimize
from Env_new import Crossroad
from Env_utils import L, STEP_TIME,W, deal_with_phi
from MPControl import ModelPredictiveControl
from mpc_to_matlab import mpc_cost_function, mpc_constraints

import mpc_cpp



def route_to_task(veh):
    if veh[4] == ('1o', '4i') or veh[4] == ('2o', '1i') :
        task = 0   #左转
    elif veh[4] == ('4o', '1i') or veh[4] == ('3o', '4i'):
        task = 2  #右转
    else:
        task = 1  #直行
    return task

def veh_predict(veh, horizon):
    veh_x, veh_y, veh_v, veh_phi, veh_task = veh
    veh_phi = deal_with_phi(veh_phi)
    veh_phi_rad = veh_phi * np.pi / 180.
    veh_array = np.zeros((horizon,4))

    veh_x_delta = veh_v * STEP_TIME * np.cos(veh_phi_rad)
    veh_y_delta = veh_v * STEP_TIME * np.sin(veh_phi_rad)

    rise = np.array([i+1 for i in range(horizon)])
    ones = np.ones(horizon)
    veh_x_array = veh_x * ones + rise * veh_x_delta
    veh_y_array = veh_y * ones + rise * veh_y_delta
    veh_v_array = veh_v * ones

    if veh_task  == 0:
        veh_phi_rad_delta = np.where(-25 < veh_x < 25, (veh_v / 26.875) * STEP_TIME, 0)
    elif veh_task == 2:
        veh_phi_rad_delta = np.where(-25 < veh_y < 25, -(veh_v / 19.375) * STEP_TIME, 0)
    else:
        veh_phi_rad_delta = 0

    veh_phi_array = (veh_phi * ones + rise * veh_phi_rad_delta) * 180 / np.pi
    veh_array[:,0] = veh_x_array
    veh_array[:,1] = veh_y_array
    veh_array[:,2] = veh_v_array
    veh_array[:,3] = veh_phi_array
    return veh_array

def set_ego_init_state(ref):
    random_index = 120

    x, y, phi = ref.indexs2points(random_index)
    steer = 0.
    a_x = 0.
    v = 8. 
    if ref.task == 'left':
        routeID = 'dl'
    elif ref.task == 'straight':
        routeID = 'du'
    else:
        ref.task == 'right'
        routeID = 'dr'
    return dict(ego=dict(v_x=v,
                            v_y=0,
                            r=0,
                            x=x,
                            y=y,
                            phi=phi,
                            steer = steer,
                            a_x = a_x,
                            l=L,
                            w=W,
                            routeID=routeID,
                            ))    # 这里指出了自车的名字叫ego, 这里也可以加多车

def run_mpc():
    step_length = 120
    horizon = 20

    task = 'left'
    ref = ReferencePath(task, ref_index=0)

    ref_best_index = 0
    init_ego_state = set_ego_init_state(ref)

    env = Crossroad(init_ego_state = init_ego_state)
    obs = env.obs    # 自车的状态list， 周车信息的recarray 包含x,y,v,phi

    #mpc = ModelPredictiveControl(obs, horizon, ref, task = 'left')
    bounds = [(-0.31, 0.31), (-7.1, 3.1)] * horizon
    u_init = np.zeros((horizon, 2))


    routes_num = 3
    tem_action_array = np.zeros((routes_num, horizon * 2))
    result_array = np.zeros((step_length,10+horizon*5))

    Q = np.array([10., 10., 0.01, 0., 0])
    R = np.array([5., 2.])

    for name_index in range(step_length):

        ego_list = obs[0] # a list [v_x, v_y, r, x, y, phi, steer_current, a_x_current]


        n_ego_vehicles_list = env.traffic.n_ego_vehicles_list['ego']
        if n_ego_vehicles_list is None:
            ineq_cons = ()
        else:
            # 0：left 1:straight 2:right
            # vehicles_array : N*horizon*4   N=8
            n = len(n_ego_vehicles_list)      # 给python function 用的
            vehicles_array = np.zeros((n,horizon,4))
            for i, veh in enumerate(n_ego_vehicles_list):
                task = route_to_task(veh)
                vehicles_array[i] = veh_predict(veh, horizon)
            vehicles_xy_array = vehicles_array[:,:,:2].copy()
            safe_dist = 5.
            # ineq_cons = {'type': 'ineq',
            #     'fun' : lambda u: mpc_constraints(u, ego_list, vehicles_xy_array, n, horizon, STEP_TIME, safe_dist)} # python_function
            ineq_cons = {'type': 'ineq',
                'fun' : lambda u: mpc_cpp.mpc_constraints(u, ego_list, vehicles_xy_array, safe_dist)}

        # only static obstacle
        x = np.ones((1,horizon)) * -3.3
        y = np.ones((1,horizon)) * -9.2

        vehicles_xy_array_static = np.stack((x,y),axis = 2)
        # ineq_cons_2 = {'type': 'ineq',
        #     'fun' : lambda u: mpc_cpp.mpc_constraints(u, ego_list, vehicles_xy_array_static, safe_dist)}
        ineq_cons_2 = ()

        #current_ref_point, future_ref_tuple_list = ref.future_ref_points(ego_list[3], ego_list[4], horizon)
        multi_future_ref_tuple_list = ref.multi_future_ref_points(ego_list[3], ego_list[4], horizon)



        result_list = []
        result_index_list = []
        valueError_list = []
        for i in range(routes_num):
            future_ref_array = np.array(multi_future_ref_tuple_list[i])
            try:
                results = minimize(
                                    # lambda u: mpc_cost_function(u, ego_list, future_ref_list, horizon, STEP_TIME, Q, R), # python_function
                                    lambda u: mpc_cpp.mpc_cost_function(u, ego_list, future_ref_array, Q, R),
                                    x0 = tem_action_array[i,:].flatten(),
                                    method = 'SLSQP',
                                    bounds = bounds,
                                    constraints = ineq_cons,
                                    tol=1e-6,
                                    #options={'disp': True} 
                                    )
                if results.success:
                    result_list.append(results)
                    result_index_list.append(i)
                    tem_action_array[i,:] = np.concatenate((results.x[2:],results.x[-2:]),axis =0)
                    print(f'results.fun[{i}]',results.fun)
                else:
                    print(f'[{i}] fail')
            except ValueError:    #感觉是求解器内部的bug，探索到了控制量边界就会ValueError
                valueError_list.append(i)
                print('ValueError')
                # tem_action[:,0] = np.clip(tem_action[:,0], -0.2, 0.2)
                # tem_action[:,1] = np.clip(tem_action[:,1], -7, 3)                   
                # mpc_action = tem_action_array[i,:]

        if not result_list and not valueError_list:
            print('fail')
            # import sys
            # sys.exit()
            mpc_action = [0.] * horizon * 2
            if obs[0][0] > 2:
                mpc_action[0] = 0.
                mpc_action[1] = -6.
            future_ref_array = np.array(multi_future_ref_tuple_list[0])
        elif valueError_list:
            mpc_action = tem_action_array[valueError_list[0],:]
            future_ref_array = np.array(multi_future_ref_tuple_list[valueError_list[0]])
        else:
            min_index = np.argmin([result.fun for result in result_list])
            mpc_action = result_list[min_index].x
            ref_best_index = result_index_list[min_index]
            future_ref_array = np.array(multi_future_ref_tuple_list[ref_best_index])
            
        


        # if not results.success:
        #     print('fail')
        #     # import sys
        #     # sys.exit()
        #     mpc_action = [0.] * horizon * 2
        #     if obs[0][0] > 1:
        #         mpc_action[0] = 0.1
        #         mpc_action[1] = -6.


        obs, reward, done, info = env.step(mpc_action[:2])
        #obs, reward, done, info = env.step(np.array([steer_action[name_index], a_x_action[name_index]]))

        result_array[name_index,0] = mpc_action[0]     # steer
        result_array[name_index,1] = mpc_action[1]     # a_x 
        result_array[name_index,2:10] = obs[0]          # v_x, v_y, r, x, y, phi, steer, a_x

        result_array[name_index,10:10+horizon*1] = future_ref_array[:,0]               # ref_x
        result_array[name_index,10+horizon*1:10+horizon*2] = future_ref_array[:,1]     # ref_y
        result_array[name_index,10+horizon*2:10+horizon*3] = future_ref_array[:,2]     # ref_phi

        result_array[name_index,10+horizon*3:10+horizon*4] = mpc_action[slice(0,horizon*2,2)]  # steer_tem
        result_array[name_index,10+horizon*4:10+horizon*5] = mpc_action[slice(1,horizon*2,2)]  # a_x_tem


    record_result = result_array
    import datetime
    current_time = datetime.datetime.now()
    np.savetxt(f'result/record_result{current_time:%Y_%m_%d_%H_%M_%S}.csv', record_result, delimiter = ',')




if __name__ == '__main__':
    run_mpc()


