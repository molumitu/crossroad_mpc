from Env import Env
import time
from init_ego_state import generate_ego_init_state
from ipopt_params import IPOPT_Param
from ipopt_controller import MPControl_ipopt


def run():

    step_length = 6000  #step_time是0.1s
    ego_ID_dict = {'ego1':120, 'ego2':1800, 'ego3':8000, 'ego4':12200, 'ego5':7200, 'ego6':10000}
    #ego_ID_dict = { 'ego2':1800}
    ego_ID_keys = ego_ID_dict.keys()

    init_ego_state = {}
    init_ego_ref = {}
    ipopt_controller = {}

    ipopt_params = IPOPT_Param()   #这里暂时均采用默认参数

    for egoID, index in ego_ID_dict.items():
        init_ego_state[egoID], init_ego_ref[egoID] = generate_ego_init_state('dl', index)
        ipopt_controller[egoID] = MPControl_ipopt(init_ego_ref[egoID], ipopt_params)
    start = time.perf_counter_ns()
    env = Env(init_ego_state)
    obs = env.obs    # 自车的状态list， 周车信息的recarray 包含x,y,v,phi
    n_ego_vehicles_list = {}
    mpc_action = {}



    for name_index in range(step_length):
        start_step = time.perf_counter_ns()    
        
         
        for egoID in ego_ID_keys:
            n_ego_vehicles_list[egoID] = env.traffic.each_ego_vehicles_list[egoID]
            mpc_action[egoID] = ipopt_controller[egoID].step(obs[0][egoID], n_ego_vehicles_list[egoID], env.traffic_light)

        obs, reward, done, info = env.step(mpc_action)
        ego_ID_keys = env.n_ego_dict.keys()
        end_step = time.perf_counter_ns()
        #print('step_time:', (end_step - start_step)/1e9)
        #print('traffic light:',env.traffic_light)


    # obs, reward, done, info = env.step(np.array([steer_action[name_index], a_x_action[name_index]])) 复盘用
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


