from solver import create_solver
from scipy.optimize.zeros import VALUEERR
from Reference import ReferencePath
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.optimize import minimize
from Env_new import Crossroad
from Env_utils import L, STEP_TIME,W, deal_with_phi
from mpc_to_matlab import mpc_cost_function  # 只是把python类变成了函数，一样慢


from solver import create_solver
from solver_with_constraints import create_solver_with_cons


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

    x, y, phi = ref.indexs2points(random_index, path_index = 0)
    steer = 0.
    a_x = 0.
    v = 6. 
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
    tol_start = time.perf_counter_ns()
    step_length = 150
    horizon = 20

    task = 'left'
    ref = ReferencePath(task)
    init_ego_state = set_ego_init_state(ref)

    env = Crossroad(init_ego_state = init_ego_state)
    obs = env.obs    # 自车的状态list， 周车信息的recarray 包含x,y,v,phi

    routes_num = 1
    tem_action_array = np.zeros((routes_num, horizon * 2))
    result_array = np.zeros((step_length,10+horizon*5))

    start = time.perf_counter_ns()
    solver = create_solver_with_cons(horizon, STEP_TIME, n_vehicles=n)
    print('solver startup', time.perf_counter_ns() - start)
    umin = np.array([-0.26, -6.1])
    umax = np.array([ 0.26, 2.8])
    lbx = np.hstack([np.kron(np.ones(horizon), umin)])
    ubx = np.hstack([np.kron(np.ones(horizon), umax)])
    time_list = []

    for name_index in range(step_length):

        ego_list = obs[0] # a list [v_x, v_y, r, x, y, phi, steer_current, a_x_current]

        multi_future_ref_tuple_list = ref.multi_future_ref_points(ego_list[3], ego_list[4], horizon)
        future_ref_array = np.array(multi_future_ref_tuple_list[0])
        future_x = list(future_ref_array[:,0])
        future_y = list(future_ref_array[:,1])
        future_phi = list(future_ref_array[:,2])


        params = list(ego_list) + future_x + future_y + future_phi

        start = time.perf_counter_ns()
        sol=solver(x0=tem_action_array,lbx=lbx,ubx=ubx, p=params)
        end = time.perf_counter_ns()
        time_list.append((end - start)/1e6)
        mpc_action = sol['x']
    
        mpc_action_array = np.array(mpc_action).squeeze()


        obs, reward, done, info = env.step(mpc_action_array[:2])
        tem_action_array = np.concatenate((mpc_action_array[2:],mpc_action_array[-2:]),axis =0)
        # tem_action_array = [*mpc_action_array[2:], *mpc_action_array[-2:]]

        result_array[name_index,0] = mpc_action_array[0]     # steer
        result_array[name_index,1] = mpc_action_array[1]     # a_x 
        result_array[name_index,2:10] = obs[0]          # v_x, v_y, r, x, y, phi, steer, a_x

        result_array[name_index,10:10+horizon*1] = future_ref_array[:,0]               # ref_x
        result_array[name_index,10+horizon*1:10+horizon*2] = future_ref_array[:,1]     # ref_y
        result_array[name_index,10+horizon*2:10+horizon*3] = future_ref_array[:,2]     # ref_phi

        result_array[name_index,10+horizon*3:10+horizon*4] = mpc_action_array[slice(0,horizon*2,2)]  # steer_tem
        result_array[name_index,10+horizon*4:10+horizon*5] = mpc_action_array[slice(1,horizon*2,2)]  # a_x_tem


    record_result = result_array
    import datetime
    current_time = datetime.datetime.now()
    np.savetxt(f'compare_solver_result/ipopt_result{current_time:%Y_%m_%d_%H_%M_%S}.csv', record_result, delimiter = ',')
    np.savetxt(f'compare_solver_result/ipopt_time{current_time:%Y_%m_%d_%H_%M_%S}.csv', time_list)
    tol_end = time.perf_counter_ns()
    print('tol_time:', (tol_end - tol_start)/1e6)



if __name__ == '__main__':
    run_mpc()


