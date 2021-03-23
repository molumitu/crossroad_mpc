from controller import MPControl
from scipy.optimize.zeros import VALUEERR
from Reference import ReferencePath
import numpy as np

from scipy.optimize import minimize
from Env_new import Env
from Env_utils import L, STEP_TIME,W, deal_with_phi, horizon
from predict_surroundings import route_to_task, veh_predict
import mpc_cpp
import time
from init_ego_state import generate_ego_init_state



def run():
    step_length = 600
    init_ego_state = {}
    init_ego_ref = {}
    init_ego_state['ego'], init_ego_ref['ego'] = generate_ego_init_state('dl', 4500)
    init_ego_state['ego1'], init_ego_ref['ego1'] = generate_ego_init_state('dl', 120)
    env = Env(init_ego_state)
    start = time.perf_counter_ns()
    obs = env.obs    # 自车的状态list， 周车信息的recarray 包含x,y,v,phi
    mpc = MPControl(init_ego_ref['ego'])
    mpc1 = MPControl(init_ego_ref['ego1'])
    n_ego_vehicles_list = {}
    mpc_action = {}
    for name_index in range(step_length):
        n_ego_vehicles_list['ego'] = env.traffic.n_ego_vehicles_list['ego']
        n_ego_vehicles_list['ego1'] = env.traffic.n_ego_vehicles_list['ego1']
        mpc_action['ego'] = mpc.step(obs[0]['ego'], n_ego_vehicles_list['ego'], env.traffic_light)
        mpc_action['ego1'] = mpc1.step(obs[0]['ego1'], n_ego_vehicles_list['ego1'], env.traffic_light)
        obs, reward, done, info = env.step(mpc_action)
        print('traffic light:',env.traffic_light)

    #obs, reward, done, info = env.step(np.array([steer_action[name_index], a_x_action[name_index]])) 复盘用

    # self.result_array[name_index,0] = mpc_action[0]     # steer
    # self.result_array[name_index,1] = mpc_action[1]     # a_x 
    # self.result_array[name_index,2:10] = obs[0]          # v_x, v_y, r, x, y, phi, steer, a_x

    # self.result_array[name_index,10:10+horizon*1] = future_ref_array[:,0]               # ref_x
    # self.result_array[name_index,10+horizon*1:10+horizon*2] = future_ref_array[:,1]     # ref_y
    # self.result_array[name_index,10+horizon*2:10+horizon*3] = future_ref_array[:,2]     # ref_phi

    # self.result_array[name_index,10+horizon*3:10+horizon*4] = mpc_action[slice(0,horizon*2,2)]  # steer_tem
    # self.result_array[name_index,10+horizon*4:10+horizon*5] = mpc_action[slice(1,horizon*2,2)]  # a_x_tem
    # self.result_array[name_index,10+horizon*5] = mpc_signal #记录从哪个控制器获得的控制量
            
            

    end = time.perf_counter_ns()
    print('tol_time:', (end - start)/1e9)
    # record_result = result_array
    # import datetime
    # current_time = datetime.datetime.now()
    # np.savetxt(f'result/record_result{current_time:%Y_%m_%d_%H_%M_%S}.csv', record_result, delimiter = ',')

if __name__ == '__main__':
    run()


