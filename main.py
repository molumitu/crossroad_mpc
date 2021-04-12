from controller import MPControl
from controller_params import MPC_Param
from Env import Env
from Utils import horizon, Nc
import time
from init_ego_state import generate_ego_init_state
import numpy as np
import pandas as pd
import json
egoID_to_num_dict = {'ego1':1, 'ego2':2, 'ego3':3 ,'ego4':4,'ego5':5, 'ego6':6, 'ego7':7 ,'ego8':8,'ego9':9, 'ego10':10, 'ego11':11 ,'ego12':12,'ego13':13,'ego14':14,'ego15':15,'ego16':16,'ego17':17,'ego18':18}
init_ego_num = 16 


def run():
    time_list = []
    nit_list = []
    step_length = 500  #step_time是0.1s
    result_array = np.zeros((init_ego_num, step_length,10+horizon*5 + 1))


    # ego_ID_dict = {'ego1':2200, 'ego2':2200, 'ego3':2200 ,'ego4':2200,'ego5':2200, 'ego6':2200, 'ego7':2200 ,'ego8':2200, 'ego9':2200, 'ego10':2200, 'ego11':2200 ,'ego12':2200}
    # ego_route_dict = {'ego1':'dl', 'ego2':'du', 'ego3':'dr' ,'ego4':'rd','ego5':'rl', 'ego6':'ru', 'ego7':'ur' ,'ego8':'ud','ego9':'ul', 'ego10':'lu', 'ego11':'lr' ,'ego12':'ld'}
    
    ego_ID_dict = {'ego1':2200, 'ego2':2200,'ego3':2200 ,'ego4':2200,'ego5':2200, 'ego6':2200,'ego7':2200 ,'ego8':2200, 'ego9':3200, 'ego10':3200,'ego11':4700 }
    ego_route_dict = {'ego1':'dl', 'ego2':'du','ego3':'ur' ,'ego4':'ud','ego5':'dl', 'ego6':'du','ego7':'ur' ,'ego8':'ud', 'ego9':'dl', 'ego10':'du','ego11':'ur'}
    
    ego_ID_dict = {'ego1':2200, 'ego4':2200,'ego5':5700, 'ego8':4700}
    ego_route_dict = {'ego1':'dl', 'ego4':'ud','ego5':'dl', 'ego8':'ud'}

    ego_ID_dict = {'ego1':4200, 'ego4':4200,'ego5':1700, 'ego8':1700}
    ego_route_dict = {'ego1':'dl', 'ego4':'ud','ego5':'dl', 'ego8':'ud'}

    ego_ID_dict = {'ego1':2200}
    ego_route_dict = {'ego1':'dl'}
    ego_ID_keys = ego_ID_dict.keys()

    init_ego_state = {}
    init_ego_ref = {}
    mpc_controller = {}

    mpc_params = MPC_Param()   #这里暂时均采用默认参数

    for egoID, index in ego_ID_dict.items():
        init_ego_state[egoID], init_ego_ref[egoID] = generate_ego_init_state(ego_route_dict[egoID], index)
        mpc_controller[egoID] = MPControl(init_ego_ref[egoID], mpc_params)
    env = Env(init_ego_state)
    obs = env.obs
    n_ego_vehicles_list = {}
    mpc_action = {}

    for step in range(step_length):
        for egoID in ego_ID_keys:
            n_ego_vehicles_list[egoID] = env.traffic.each_ego_vehicles_list[egoID]  #[x, y, v, a, acc, route]
            mpc_action[egoID],future_ref_array, mpc_signal = mpc_controller[egoID].step(obs[egoID], n_ego_vehicles_list[egoID], env.traffic_light,time_list, nit_list)
            ego_index = egoID_to_num_dict[egoID] - 1
            result_array[ego_index,step,0] = mpc_action[egoID][0]     # steer
            result_array[ego_index,step,1] = mpc_action[egoID][1]     # a_x 
            result_array[ego_index,step,2:10] = obs[egoID]          # v_x, v_y, r, x, y, phi, steer,
            result_array[ego_index,step,10:10+horizon*1] = future_ref_array[:,0]               # ref_x
            result_array[ego_index,step,10+horizon*1:10+horizon*2] = future_ref_array[:,1]     # ref_y
            result_array[ego_index,step,10+horizon*2:10+horizon*3] = future_ref_array[:,2]     # ref
            result_array[ego_index,step,10+horizon*3:10+horizon*4] = mpc_action[egoID][slice(0,horizon*2,2)]  # steer_tem
            result_array[ego_index,step,10+horizon*4:10+horizon*5] = mpc_action[egoID][slice(1,horizon*2,2)]  # a_x_tem
            result_array[ego_index,step,10+horizon*5] = mpc_signal #记录从哪个mpc及状态获取的控制量
        obs, reward, done, info = env.step(mpc_action)
        env.render()
        ego_ID_keys = env.n_ego_dict.keys()

    # np.savetxt(f'record_multi_ego_result/time_mul_9_ge.csv', time_list, delimiter = ',')
    # np.savetxt(f'record_multi_ego_result/nit_mul_9_ge.csv', nit_list, delimiter = ',')
    # import datetime
    # current_time = datetime.datetime.now()
    # writer = pd.ExcelWriter(f'record_multi_ego_result/{current_time:%Y_%m_%d_%H_%M_%S}.xlsx', engine='xlsxwriter')
    # for i in range(init_ego_num):
    #     df = pd.DataFrame(result_array[i,:,:])
    #     df.to_excel(writer, sheet_name='ego%d' %(i+1), header = False, index = False)
    # writer.save()


def run_back():
    # ego_ID_dict = {'ego1':2200, 'ego2':2200, 'ego3':2200 ,'ego4':2200,'ego5':2200, 'ego6':2200, 'ego7':2200 ,'ego8':2200,'ego9':2200, 'ego10':2200, 'ego11':2200 ,'ego12':2200}
    # ego_route_dict = {'ego1':'dl', 'ego2':'du', 'ego3':'dr' ,'ego4':'rd','ego5':'rl', 'ego6':'ru', 'ego7':'ur' ,'ego8':'ud','ego9':'ul', 'ego10':'lu', 'ego11':'lr' ,'ego12':'ld'}
    ego_ID_dict = {'ego1':2200, 'ego2':2200,'ego3':2200 ,'ego4':2200,'ego5':2200, 'ego6':2200,'ego7':2200 ,'ego8':2200, 'ego9':3200, 'ego10':3200,'ego11':4700 ,'ego12':4700}
    ego_route_dict = {'ego1':'dl', 'ego2':'du','ego3':'ur' ,'ego4':'ud','ego5':'dl', 'ego6':'du','ego7':'ur' ,'ego8':'ud', 'ego9':'dl', 'ego10':'du','ego11':'ur' ,'ego12':'ud'}
    # ego_ID_dict = {'ego1':2200, 'ego2':2000,'ego3':2200 ,'ego4':2000}
    # ego_route_dict = {'ego1':'dl', 'ego2':'du','ego3':'ur' ,'ego4':'ud'}
    ego_ID_keys = ego_ID_dict.keys()
    action_recorded = {}
    reader = pd.ExcelFile(f'record_multi_ego_result/2021_04_09_20_10_12.xlsx')
    for egoID in ego_ID_keys:
        action_recorded[egoID] = np.array(pd.read_excel(reader,egoID, header = None, index = None))[:,:2]


    step_length = 1000  #step_time是0.1s

    init_ego_state = {}
    init_ego_ref = {}


    for egoID, index in ego_ID_dict.items():
        init_ego_state[egoID], init_ego_ref[egoID] = generate_ego_init_state(ego_route_dict[egoID], index)
    env = Env(init_ego_state)
    obs = env.obs    # 自车的状态list， 周车信息的recarray 包含x,y,v,phi
    n_ego_vehicles_list = {}
    mpc_action = {}

    import time
    fps = 10
    dt = 1 / fps
    for step in range(step_length):
        start = time.process_time_ns()
        for egoID in ego_ID_keys:
            n_ego_vehicles_list[egoID] = env.traffic.each_ego_vehicles_list[egoID]  #[x, y, v, a, acc, route]
            ego_index = egoID_to_num_dict[egoID]
            mpc_action[egoID]  = action_recorded[egoID][step,:]
        obs, reward, done, info = env.step(mpc_action)
        ego_ID_keys = env.n_ego_dict.keys()
        end = time.process_time_ns()
        elapsed = (end - start) / 1e9
        assert elapsed < dt
        time.sleep(dt - elapsed)

if __name__ == '__main__':
    run()


