from typing import Dict
import warnings
import numpy as np
from numpy.lib import utils
from traffic import Traffic
from mpc_cpp import state_trans_LPF
from Utils import CROSSROAD_SIZE, extension, ego_route_dict_not_control, L
import traci
warnings.filterwarnings("ignore")

class Env:
    def __init__(self, init_ego_state, init_ego_state_not_control, init_sim_time, save_data_path):
        self.init_sim_time = init_sim_time
        self.init_state = init_ego_state
        self.traffic = Traffic(save_data_path, init_sim_time)
        self.traffic.init_traffic(init_ego_state, init_ego_state_not_control, init_sim_time)
        self.traffic_light = self.traffic.traffic_light
        
        self.n_ego_dict:Dict[str, Ego] = {}
        for egoID in init_ego_state.keys():
            self.n_ego_dict[egoID] = Ego(init_ego_state[egoID])

        self.n_ego_dict_not_control:Dict[str, Ego] = {}
        for egoID_not_control in init_ego_state_not_control.keys():
            self.n_ego_dict_not_control[egoID_not_control] = Ego(init_ego_state_not_control[egoID_not_control])

        self.obs =  self._get_obs()#自车dict{[egoID]:ego_list}
        self.reward = 0
        self.reward_info = None
        self.done_type = 'not_done_yet'
        self.done = False
        self.info = None 
        self._get_info()

        self.travel_time = {}

    def _get_obs(self):
        ego_obs_dict = {}
        for egoID, ego in self.n_ego_dict.items():
            ego_obs_dict[egoID] = np.array(list(ego.ego_dynamics.values()))
        return ego_obs_dict

    def _get_reward(self):
        self.reward = 0
        self.reward_info = None
        return self.reward, self.reward_info

    def _get_done(self): 
        self.done = False
        return self.done

    def _get_info(self): 
        info_dict = {}
        for egoID, ego in self.n_ego_dict.items():
            info_dict[egoID] = np.array(list(ego.ego_dynamics.values()))
        return info_dict

    def _update_n_ego_dict(self):
        n_ego_dict_copy = self.n_ego_dict.copy()
        for egoID, ego in n_ego_dict_copy.items():
            ego_x = ego.ego_dynamics['x']
            ego_y = ego.ego_dynamics['y']
            ego_route = ego.routeID
            if judge_done(ego_x, ego_y, ego_route):
                self.travel_time[egoID] = self.traffic.sim_time - self.init_sim_time
                del self.n_ego_dict[egoID]
            # if abs(ego_x) > 97 or abs(ego_y) > 97:
            #     self.travel_time[egoID] = self.traffic.sim_time - self.init_sim_time
            #     del self.n_ego_dict[egoID]

    def _update_n_ego_dict_not_control(self):
        n_ego_dict_not_control_copy = self.n_ego_dict_not_control.copy()
        for egoID_not_control, ego_not_control in n_ego_dict_not_control_copy.items():
            ego_not_control_x, ego_not_control_y = self.traffic.all_vehicles[egoID_not_control][traci.constants.VAR_POSITION]
            ego_not_control_route = ego_route_dict_not_control[egoID_not_control]
            if judge_done(ego_not_control_x, ego_not_control_y, ego_not_control_route):
                self.travel_time[egoID_not_control] = self.traffic.sim_time - self.init_sim_time
                del self.n_ego_dict_not_control[egoID_not_control]

    def step(self, action):
        self._update_n_ego_dict()
        self._update_n_ego_dict_not_control()
        ego_dynamics = {}
        for egoID, ego in self.n_ego_dict.items():
            ego_dynamics[egoID] = ego._update_ego_dynamics(action[egoID][:2])
        
        self.traffic.sync_ego_vehicles(ego_dynamics)
        self.traffic.sim_step()
        self.traffic.get_vehicles_for_each_ego(self.n_ego_dict.keys())
        self.traffic_light = self.traffic.traffic_light

        self.obs = self._get_obs()
        reward, self.reward_info = self._get_reward()
        done = self._get_done()
        info = self._get_info()

        return self.obs, reward, done, info

    def render(self):
        pass

class Ego:
    ego_dynamics_keys = ('v_x', 'v_y', 'r', 'x', 'y', 'phi', 'steer', 'a_x')
    def __init__(self, init_ego_state:Dict[str,float]):
        self.ego_dynamics = {key:init_ego_state[key] for key in self.ego_dynamics_keys}
        self.routeID = init_ego_state['routeID']

    def _update_ego_dynamics(self, action):  # update 
        state = np.array(list(self.ego_dynamics.values()))
        action = np.array(action)
        next_ego_state, next_ego_param = state_trans_LPF(state, action)
        self.ego_dynamics = dict(zip(self.ego_dynamics_keys, next_ego_state))
        return self.ego_dynamics

def judge_done(x, y, routeID):
    if routeID in ['ul', 'dl', 'rl'] and x < -CROSSROAD_SIZE/2 - extension:
        return True
    elif routeID in ['ur', 'dr', 'lr'] and x > CROSSROAD_SIZE/2 + extension:
        return True
    elif routeID in ['ud', 'ld', 'rd'] and y < -CROSSROAD_SIZE/2 - extension:
        return True
    elif routeID in ['lu', 'du', 'ru'] and y > CROSSROAD_SIZE/2 + extension:
        return True
    else:
        return False



    
if __name__ == '__main__':
    pass