from typing import Dict
from controller import MPControl
from controller_params import MPC_Param, Param_Set
from Env import Env
from Utils import horizon, Nc, egoID_to_num_dict, init_ego_num, ego_dynamics_keys, other_info_keys
import time
from init_ego_state import generate_ego_init_state
import numpy as np
import pandas as pd
import json
import pickle
import os
import shutil


def run(init_sim_time, test_name, step_length, random_position_array, random_v, is_control):
    ##----------------指定结果保存路径-------------------------


    data_path = R'C:\Users\zgj_t\Desktop\crossroad_mpc\visualization'
    save_test_path = os.path.join(data_path, test_name)
    if not os.path.exists(save_test_path):
        os.makedirs(save_test_path, exist_ok=True)  # exist_ok = True for multi-processing
    import datetime
    current_time = datetime.datetime.now()
    time_name = f'{current_time:%Y_%m_%d_%H_%M_%S}_{init_sim_time}'
    save_data_path = os.path.join(save_test_path, time_name)
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)
    
    ego_ID_dict = {'ego1':random_position_array[0], 'ego2':random_position_array[1], 'ego3':random_position_array[2] ,'ego4':random_position_array[3],
                    'ego5':random_position_array[4], 'ego6':random_position_array[5], 'ego7':random_position_array[6] ,'ego8':random_position_array[7],
                    'ego9':random_position_array[8], 'ego10':random_position_array[9], 'ego11':random_position_array[10] ,'ego12':random_position_array[11]}
    ego_route_dict = {'ego1':'dl', 'ego2':'du', 'ego3':'dr' ,'ego4':'rd','ego5':'rl', 'ego6':'ru', 
                      'ego7':'ur' ,'ego8':'ud','ego9':'ul', 'ego10':'lu', 'ego11':'lr' ,'ego12':'ld'}
    ego_ID_dict_not_control = {'ego1_not_control':random_position_array[0], 'ego2_not_control':random_position_array[1], 
                                'ego3_not_control':random_position_array[2] ,'ego4_not_control':random_position_array[3],
                                'ego5_control':random_position_array[4], 'ego6_not_control':random_position_array[5],
                                'ego7_not_control':random_position_array[6] ,'ego8_not_control':random_position_array[7],
                                'ego9_not_control':random_position_array[8], 'ego10_not_control':random_position_array[9],
                                'ego11_not_control':random_position_array[10] ,'ego12_not_control':random_position_array[11]}
    ego_route_dict_not_control = {'ego1_not_control':'dl', 'ego2_not_control':'du', 'ego3_not_control':'dr' ,'ego4_not_control':'rd',
                                'ego5_control':'rl', 'ego6_not_control':'ru', 'ego7_not_control':'ur' ,'ego8_not_control':'ud',
                                'ego9_not_control':'ul', 'ego10_not_control':'lu', 'ego11_not_control':'lr' ,'ego12_not_control':'ld'}

    # ego_ID_dict = {'ego4':random_position_array[3],
    #                 'ego5':random_position_array[4], 'ego6':random_position_array[5],
    #                 'ego10':random_position_array[9], 'ego11':random_position_array[10] ,'ego12':random_position_array[11]}
    # ego_route_dict = {'ego4':'rd','ego5':'rl', 'ego6':'ru', 
    #                   'ego10':'lu', 'ego11':'lr' ,'ego12':'ld'}
    # ego_ID_dict_not_control = {'ego4_not_control':random_position_array[3],
    #                             'ego5_control':random_position_array[4], 'ego6_not_control':random_position_array[5],
    #                             'ego10_not_control':random_position_array[9],
    #                             'ego11_not_control':random_position_array[10] ,'ego12_not_control':random_position_array[11]}
    # ego_route_dict_not_control = {'ego4_not_control':'rd',
    #                             'ego5_control':'rl', 'ego6_not_control':'ru',
    #                             'ego10_not_control':'lu', 'ego11_not_control':'lr' ,'ego12_not_control':'ld'}

    # 单车
    # ego_ID_dict = {'ego1':random_position_array[0]}
    # ego_route_dict = {'ego1':'dl'}
    # ego_ID_dict_not_control = {'ego1_not_control':random_position_array[0]}
    # ego_route_dict_not_control = {'ego1_not_control':'dl'}
    # 上下两车
    # ego_ID_dict = {'ego1':random_position_array[0], 'ego8':random_position_array[7]}
    # ego_route_dict = {'ego1':'dl','ego8':'ud'}
    # ego_ID_dict_not_control = {'ego1_not_control':random_position_array[0], 'ego8_not_control':random_position_array[7]}
    # ego_route_dict_not_control = {'ego1_not_control':'dl','ego8_not_control':'ud'}
    if is_control:
        ego_ID_dict_not_control = {}
        ego_route_dict_not_control = {}
    else:
        ego_ID_dict = {}
        ego_route_dict = {}

    time_list = []
    nit_list = []
    
    result_array = np.zeros((init_ego_num, step_length,10+horizon*5 + 1))
    ego_pickle = []
    others_pickle = []
    traffic_light_pickle = []

    ego_ID_keys = list(ego_ID_dict.keys())

    init_ego_state_not_control = {}
    init_ego_state = {}
    init_ego_ref = {}
    init_ego_task = {}
    mpc_controller = {}
    params_selector = Param_Set()

    for egoID, index in ego_ID_dict_not_control.items():
        init_ego_state_not_control[egoID], _ , _= generate_ego_init_state(ego_route_dict_not_control[egoID], index, random_v)

    for egoID, index in ego_ID_dict.items():
        init_ego_state[egoID], init_ego_ref[egoID], init_ego_task[egoID] = generate_ego_init_state(ego_route_dict[egoID], index, random_v)
        mpc_params = params_selector.select_param_by_task(init_ego_task[egoID])
        mpc_controller[egoID] = MPControl(init_ego_ref[egoID], mpc_params, egoID)
    env = Env(init_ego_state, init_ego_state_not_control, init_sim_time, save_data_path)
    obs = env.obs
    each_ego_vehicles_list:Dict[str, list] = {}
    each_ego_vehicles_list_without_egos:Dict[str, list] = {}
    each_ego_egoID_list:Dict[str, list] = {}
    mpc_action = {}
    ref_best_index = {}
    future_ref_array = {}
    future_position_array = {}
    for step in range(step_length):
        ego_future_position = future_position_array.copy()
        for egoID in ego_ID_keys:
            each_ego_vehicles_list[egoID] = env.traffic.each_ego_vehicles_list[egoID]  #[x, y, v, a, acc, route]
            each_ego_vehicles_list_without_egos[egoID] = env.traffic.each_ego_vehicles_list_without_egos[egoID]
            each_ego_egoID_list[egoID] = env.traffic.each_ego_egoID_list[egoID]
            mpc_action[egoID], future_ref_array[egoID], ref_best_index[egoID], future_position_array[egoID] = mpc_controller[egoID].step(obs[egoID], each_ego_vehicles_list[egoID], each_ego_vehicles_list_without_egos[egoID], each_ego_egoID_list[egoID], env.traffic_light, ego_future_position, time_list, nit_list)
            ego_index = egoID_to_num_dict[egoID] - 1
            result_array[ego_index,step,0] = mpc_action[egoID][0]     # steer
            result_array[ego_index,step,1] = mpc_action[egoID][1]     # a_x 
            result_array[ego_index,step,2:10] = obs[egoID]          # v_x, v_y, r, x, y, phi, steer,
            result_array[ego_index,step,10:10+horizon*1] = future_ref_array[egoID][:,0]               # ref_x
            result_array[ego_index,step,10+horizon*1:10+horizon*2] = future_ref_array[egoID][:,1]     # ref_y
            result_array[ego_index,step,10+horizon*2:10+horizon*3] = future_ref_array[egoID][:,2]     # ref
            result_array[ego_index,step,10+horizon*3:10+horizon*3+Nc] = mpc_action[egoID][slice(0,horizon,2)]  # steer_tem
            result_array[ego_index,step,10+horizon*3+Nc:10+horizon*4] = mpc_action[egoID][-2] * int(horizon - Nc)
            result_array[ego_index,step,10+horizon*4:10+horizon*4+Nc] = mpc_action[egoID][slice(1,horizon,2)]  # a_x_tem
            result_array[ego_index,step,10+horizon*4+Nc:10+horizon*5] = mpc_action[egoID][-1] * int(horizon - Nc)
            result_array[ego_index,step,10+horizon*5] = ref_best_index[egoID] #记录从哪个mpc及状态获取的控制量
        obs, reward, done, info = env.step(mpc_action)
        # print(env.traffic_light)

        obs_copy = obs.copy()
        ego_info = {}
        for egoID in obs_copy:
            ego_info[egoID] = {'dynamics':list(obs_copy[egoID]) + [ref_best_index[egoID]], 
                                'ref':future_ref_array[egoID]}
        ego_pickle.append(ego_info)


        all_vehicle_copy = env.traffic.all_vehicles.copy()
        others_pickle.append({k:v for k,v in all_vehicle_copy.items() if k not in [*ego_ID_keys, 'collector']})
        
        traffic_light_pickle.append(env.traffic_light)
        ego_ID_keys = env.n_ego_dict.keys()


    env.traffic.close()
    with open(os.path.join(save_data_path, 'ego.pickle'), 'wb+') as f:
        pickle.dump(ego_pickle, f)
    with open(os.path.join(save_data_path, 'others.pickle'), 'wb+') as f:        
        pickle.dump(others_pickle, f)
    with open(os.path.join(save_data_path, 'traffic_light.pickle'), 'wb+') as f:        
        pickle.dump(traffic_light_pickle, f)



    ##----------------------------------read travel_time----------------------
    import xml.etree.ElementTree as ET
    save_path = os.path.join(save_data_path, f'output_info_{init_sim_time}.xml')
    tree = ET.parse(save_path)
    root = tree.getroot()
    list_egos = [veh for veh in root if veh.attrib['id'] in {*ego_ID_dict.keys(), *ego_ID_dict_not_control.keys()}]
    travel_time = {}
    for veh in list_egos:
        travel_time[veh.attrib['id']] = veh.attrib['duration']
    # shutil.move(sumo_info_file, save_data_path)
    print('Begin generate video!')
    from GenVideo import Gen_Video
    Gen_Video(test_name, time_name, step_length)

    # save result
    np.savetxt(f'record_multi_ego_result/time_mul_54_{test_name}_{init_sim_time}_ge.csv', time_list, delimiter = ',')
    np.savetxt(f'record_multi_ego_result/nit_mul_54_{test_name}_{init_sim_time}_ge.csv', nit_list, delimiter = ',')
    writer = pd.ExcelWriter(f'record_single_ego_result/{current_time:%Y_%m_%d_%H_%M_%S}.xlsx', engine='xlsxwriter')
    for i in range(init_ego_num):
        df = pd.DataFrame(result_array[i,:,:])
        df.to_excel(writer, sheet_name='ego%d' %(i+1), header = False, index = False)
    writer.save()
    return (travel_time, env.travel_time)

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


def run_test(test_name, step_length, random_position, random_v_array, is_control):

    data_path = R'C:\Users\zgj_t\Desktop\crossroad_mpc\visualization'
    # travel_time_tuple = run(20, test_name, step_length, random_position[:,0], random_v_array[0], is_control)
    test_samples = 24
    from multiprocessing import Pool
    with Pool(8) as p:
         travel_time_tuple = p.starmap(run, [(i*5, test_name, step_length, random_position[:,i], random_v_array[i], is_control) for i in range(test_samples)])
    print(travel_time_tuple)
    travel_time = [time[0] for time in travel_time_tuple]
    env_travel_time = [time[1] for time in travel_time_tuple]
    print(env_travel_time)
    with open(os.path.join(data_path,test_name,'travel_time_list.json'), 'w+') as f:        
        json.dump(travel_time, f)
    with open(os.path.join(data_path,test_name,'env_travel_time_list.json'), 'w+') as f:        
        json.dump(env_travel_time, f)

    parameters_selector = Param_Set()
    mpc_parameters = parameters_selector.to_dict()
    hyper_parameters = {
        'min_gap':5.0,
        'vehsPerHour':400.0,
        'T_steer' : 0.3,
        'T_a_x' : 0.1,
        'detection_radius' : 50,
        'pre_traffic_time': 50,
        'init_random_list': random_position.tolist(), 
        'init_v_list': random_v_array.tolist()
                }

    with open(os.path.join(data_path,test_name,'mpc_parameters.json'), 'w+') as f:        
        json.dump(mpc_parameters, f)
    with open(os.path.join(data_path,test_name,'hyper_parameters.json'), 'w+') as f:        
        json.dump(hyper_parameters, f)


if __name__ == '__main__':
    step_length = 1200
    # random_position = np.random.randint(3000,7000, (12, 24))
    # random_v_array = 3 + np.random.rand(24) * 3
    # run_test('travel_time_test', step_length, random_position, random_v_array, is_control = True)
    # run_test('travel_time_test_not_control', step_length, random_position, random_v_array, is_control = False)

    # random_position = np.random.randint(4000,7000, (12, 24))
    # random_v_array = 3 + np.random.rand(24) * 3
    # run_test('Sin_left_ego_Test5_5_3', step_length, random_position, random_v_array, is_control = True)
    # run_test('Sin_left_ego_Test5_5_3_not_control', step_length, random_position, random_v_array,  is_control = False)
    random_position = np.random.randint(4000,7000, (12, 24))
    random_v_array = 3 + np.random.rand(24) * 3

    run_test('Base_Mul_ego_Test5_11_1', step_length, random_position, random_v_array, is_control = True)
    run_test('Base_Mul_ego_Test5_11_1_not_control', step_length, random_position, random_v_array,  is_control = False)









