from controller import MPControl
from controller_params import MPC_Param
from Env import Env
import time
from init_ego_state import generate_ego_init_state
import numpy as np


egoID_to_num_dict = {'ego1':1, 'ego2':2, 'ego3':3 ,'ego4':4,'ego5':5, 'ego6':6, 'ego7':7 ,'ego8':8,'ego9':9, 'ego10':10, 'ego11':11 ,'ego12':12}

def run():
    # a_x = np.loadtxt('record_action/record_action_a_x2021_04_01_20_40_48.csv', delimiter = ',')
    # steer = np.loadtxt('record_action/record_action_steer2021_04_01_20_40_48.csv', delimiter = ',')
    step_length = 1000  #step_time是0.1s
    record_action_steer = np.zeros((step_length, 12))
    record_action_a_x = np.zeros((step_length, 12))
    # ego_ID_dict = {'ego1':1200}
    # ego_route_dict = {'ego1':'ud'}
    # ego_ID_dict = {'ego1':120, 'ego2':1800,'ego3':4900, 'ego4':7200 , 'ego5':8500, 'ego6':10000, 'ego7':12200, 'ego8':14200, \
    #                'ego11':120, 'ego12':1800,'ego13':4900, 'ego14':7200 , 'ego15':8500, 'ego16':10000, 'ego17':12200, 'ego18':14200, \
    #                 'ego21':120, 'ego22':1800,'ego23':4900, 'ego24':7200 , 'ego25':8500, 'ego26':10000, 'ego27':12200, 'ego28':14200,}
    # ego_route_dict =  {'ego1':'dl', 'ego2':'dl','ego3':'dl', 'ego4':'dl' , 'ego5':'dl', 'ego6':'dl', 'ego7':'dl', 'ego8':'dl', \
    #                'ego11':'du', 'ego12':'du','ego13':'du', 'ego14':'du' , 'ego15':'du', 'ego16':'du', 'ego17':'du', 'ego18':'du', \
    #                 'ego21':'dr', 'ego22':'dr','ego23':'dr', 'ego24':'dr' , 'ego25':'dr', 'ego26':'dr', 'ego27':'dr', 'ego28':'dr',}
    


    ego_ID_dict = {'ego1':1200, 'ego2':1200, 'ego3':1200 ,'ego4':1200,'ego5':1200, 'ego6':1200, 'ego7':1200 ,'ego8':1200,'ego9':1200, 'ego10':1200, 'ego11':1200 ,'ego12':1200}
    ego_route_dict = {'ego1':'dl', 'ego2':'du', 'ego3':'dr' ,'ego4':'rd','ego5':'rl', 'ego6':'ru', 'ego7':'ur' ,'ego8':'ud','ego9':'ul', 'ego10':'lu', 'ego11':'lr' ,'ego12':'ld'}
    ego_ID_keys = ego_ID_dict.keys()

    init_ego_state = {}
    init_ego_ref = {}
    mpc_controller = {}

    mpc_params = MPC_Param()   #这里暂时均采用默认参数

    for egoID, index in ego_ID_dict.items():
        init_ego_state[egoID], init_ego_ref[egoID] = generate_ego_init_state(ego_route_dict[egoID], index)
        mpc_controller[egoID] = MPControl(init_ego_ref[egoID], mpc_params)
    env = Env(init_ego_state)
    obs = env.obs    # 自车的状态list， 周车信息的recarray 包含x,y,v,phi
    n_ego_vehicles_list = {}
    mpc_action = {}

    for step in range(step_length):
        for egoID in ego_ID_keys:
            n_ego_vehicles_list[egoID] = env.traffic.each_ego_vehicles_list[egoID]  #[x, y, v, a, route]
            mpc_action[egoID] = mpc_controller[egoID].step(obs[egoID], n_ego_vehicles_list[egoID], env.traffic_light)
            ego_index = egoID_to_num_dict[egoID]
            record_action_steer[step,ego_index-1] = mpc_action[egoID][0]
            record_action_a_x[step,ego_index-1] = mpc_action[egoID][1]


        obs, reward, done, info = env.step(mpc_action)
        ego_ID_keys = env.n_ego_dict.keys()

    # obs, reward, done, info = env.step(np.array([steer_action[name_index], a_x_action[name_index]])) 复盘用
    # self.result_array[name_index,0] = mpc_action[0]     # steer
    # self.result_array[name_index,1] = mpc_action[1]     # a_x 
    # self.result_array[name_index,2:10] = obs          # v_x, v_y, r, x, y, phi, steer, a_x

    # self.result_array[name_index,10:10+horizon*1] = future_ref_array[:,0]               # ref_x
    # self.result_array[name_index,10+horizon*1:10+horizon*2] = future_ref_array[:,1]     # ref_y
    # self.result_array[name_index,10+horizon*2:10+horizon*3] = future_ref_array[:,2]     # ref_phi

    # self.result_array[name_index,10+horizon*3:10+horizon*4] = mpc_action[slice(0,horizon*2,2)]  # steer_tem
    # self.result_array[name_index,10+horizon*4:10+horizon*5] = mpc_action[slice(1,horizon*2,2)]  # a_x_tem
    # self.result_array[name_index,10+horizon*5] = mpc_signal #记录从哪个控制器获得的控制量
            
    # record_result = result_array
    import datetime
    current_time = datetime.datetime.now()
    # np.savetxt(f'result/record_result{current_time:%Y_%m_%d_%H_%M_%S}.csv', record_result, delimiter = ',')
    np.savetxt(f'record_action/record_action_steer{current_time:%Y_%m_%d_%H_%M_%S}.csv', record_action_steer, delimiter = ',')
    np.savetxt(f'record_action/record_action_a_x{current_time:%Y_%m_%d_%H_%M_%S}.csv', record_action_a_x, delimiter = ',')


def run_back():
    a_x_recorded = np.loadtxt('record_action/record_action_a_x2021_04_01_21_01_43.csv', delimiter = ',')
    steer_recorded = np.loadtxt('record_action/record_action_steer2021_04_01_21_01_43.csv', delimiter = ',')
    step_length = 1000  #step_time是0.1s

    
    ego_ID_dict = {'ego1':1200, 'ego2':1200, 'ego3':1200 ,'ego4':1200,'ego5':1200, 'ego6':1200, 'ego7':1200 ,'ego8':1200,'ego9':1200, 'ego10':1200, 'ego11':1200 ,'ego12':1200}
    ego_route_dict = {'ego1':'dl', 'ego2':'du', 'ego3':'dr' ,'ego4':'rd','ego5':'rl', 'ego6':'ru', 'ego7':'ur' ,'ego8':'ud','ego9':'ul', 'ego10':'lu', 'ego11':'lr' ,'ego12':'ld'}
    ego_ID_keys = ego_ID_dict.keys()

    init_ego_state = {}
    init_ego_ref = {}
    mpc_controller = {}

    mpc_params = MPC_Param()   #这里暂时均采用默认参数

    for egoID, index in ego_ID_dict.items():
        init_ego_state[egoID], init_ego_ref[egoID] = generate_ego_init_state(ego_route_dict[egoID], index)
        # mpc_controller[egoID] = MPControl(init_ego_ref[egoID], mpc_params)
    env = Env(init_ego_state)
    obs = env.obs    # 自车的状态list， 周车信息的recarray 包含x,y,v,phi
    n_ego_vehicles_list = {}
    mpc_action = {}

    import time
    fps = 10 * 3
    dt = 1 / fps
    for step in range(step_length):
        start = time.process_time_ns()
        for egoID in ego_ID_keys:
            n_ego_vehicles_list[egoID] = env.traffic.each_ego_vehicles_list[egoID]  #[x, y, v, a, route]
            ego_index = egoID_to_num_dict[egoID]
            mpc_action[egoID] = np.zeros((2,))
            mpc_action[egoID][0] = steer_recorded[step,ego_index-1]
            mpc_action[egoID][1] = a_x_recorded[step,ego_index-1]
        obs, reward, done, info = env.step(mpc_action)
        ego_ID_keys = env.n_ego_dict.keys()
        end = time.process_time_ns()
        elapsed = (end - start) / 1e9
        assert elapsed < dt
        time.sleep(dt - elapsed)

if __name__ == '__main__':
    run_back()


