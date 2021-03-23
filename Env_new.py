import warnings
import numpy as np
from Env_utils import shift_coordination, rotate_coordination, rotate_and_shift_coordination, deal_with_phi, \
    L, W, CROSSROAD_SIZE, LANE_WIDTH, LANE_NUMBER, judge_feasible,  STEP_TIME
from traffic import Traffic
from mpc_cpp import state_trans_LPF
warnings.filterwarnings("ignore")


class Env():
    def __init__(self,init_ego_state):
        self.n_ego_keys = init_ego_state.keys()        
        self.init_state = init_ego_state
        self.traffic = Traffic()
        self.traffic.init_traffic(self.init_state)
        self.traffic.sim_step()   #### 存疑，是否有必要step一步
        self.traffic_light = self.traffic.traffic_light  #@property


        self.ego = Ego(init_ego_state['ego'])
        self.ego1 = Ego(init_ego_state['ego1'])
        # self.all_vehicles = self.traffic.n_ego_vehicles['ego']
        self.ego.update_relevant_vehicles(self.traffic.n_ego_vehicles['ego']) 
        self.ego1.update_relevant_vehicles(self.traffic.n_ego_vehicles['ego1'])
        self.all_info = None #
        self._get_all_info()

        self.obs =  self._get_obs()#自车和周围所有车(ego_list, recarray)
        self.action = None

        self.done_type = 'not_done_yet'
        self.done = None
        self.reward = 0
        self.reward_info = None


    # def _get_ego_dynamics(self, next_ego_state):  # update 
    #     next = dict(v_x=next_ego_state[0],
    #                v_y=next_ego_state[1],
    #                r=next_ego_state[2],
    #                x=next_ego_state[3],
    #                y=next_ego_state[4],
    #                phi=next_ego_state[5],
    #                steer=next_ego_state[6],
    #                a_x=next_ego_state[7],                   
    #                l=L,
    #                w=W,
    #                )
    #     self.ego_dynamics = next  # 完成自车动力学参数的更新
    #     return next

    def _get_all_info(self): 
        # all_info = dict(all_vehicles=self.all_vehicles,
        #                 ego_dynamics=self.ego_dynamics,
        #                 traffic_light=self.traffic_light)
        all_info = None
        return all_info

    def _get_obs(self):
        # ego_vector = self._construct_ego_vector_short() 
        # vehs_recarray = self._construct_veh_recarray()
        ego_vector = {}
        vehs_recarray = {}
        ego_vector['ego'] = self.ego._construct_ego_vector_short() 
        vehs_recarray['ego'] = self.ego._construct_veh_recarray()
        ego_vector['ego1'] = self.ego1._construct_ego_vector_short() 
        vehs_recarray['ego1'] = self.ego1._construct_veh_recarray()
        return ego_vector, vehs_recarray

    def compute_reward(self):
        reward = 0
        reward_info = None
        return reward, reward_info

    def step(self, action):
        self.action = action
        next_ego_state = {}
        ego_dynamics = {}
        next_ego_state['ego'] = self.ego._get_next_ego_state(self.action['ego'])
        ego_dynamics['ego'] = self.ego._get_ego_dynamics(next_ego_state['ego'])
        next_ego_state['ego1'] = self.ego1._get_next_ego_state(self.action['ego1'])
        ego_dynamics['ego1'] = self.ego1._get_ego_dynamics(next_ego_state['ego1'])
        self.traffic.set_own_car(ego_dynamics)
        # self.traffic.set_own_car(dict(ego=ego_dynamics['ego']))
        # self.traffic.set_own_car(dict(ego1=ego_dynamics['ego1']))

        self.traffic.sim_step()
        self.traffic_light = self.traffic.traffic_light
        self.n_ego_keys = self.traffic.n_ego_dict.keys()
        self.ego.update_relevant_vehicles(self.traffic.n_ego_vehicles['ego'])
        self.ego1.update_relevant_vehicles(self.traffic.n_ego_vehicles['ego1'])
        self.obs = self._get_obs()


        done = None
        reward, self.reward_info = self.compute_reward()
        all_info = self._get_all_info()
        return self.obs, reward, done, all_info
        # all_info 作为一个容器，之后可以往里面塞信息

class Ego():
    def __init__(self, init_ego_state):
        self.init_state = init_ego_state
        ego_dynamics = self._get_ego_dynamics([self.init_state['v_x'],
                                        self.init_state['v_y'],
                                        self.init_state['r'],
                                        self.init_state['x'],
                                        self.init_state['y'],
                                        self.init_state['phi'],
                                        self.init_state['steer'],
                                        self.init_state['a_x']],
                                        )
    def update_relevant_vehicles(self, relevant_vehicles):
        self.relevant_vehicles = relevant_vehicles
        # self.traffic.n_ego_vehicles['ego']

    
    def _get_ego_dynamics(self, next_ego_state):  # update 
        next = dict(v_x=next_ego_state[0],
                   v_y=next_ego_state[1],
                   r=next_ego_state[2],
                   x=next_ego_state[3],
                   y=next_ego_state[4],
                   phi=next_ego_state[5],
                   steer=next_ego_state[6],
                   a_x=next_ego_state[7],                   
                   l=L,
                   w=W,
                   )
        self.ego_dynamics = next  # 完成自车动力学参数的更新
        return next

    def _get_next_ego_state(self, trans_action):
        current_v_x = self.ego_dynamics['v_x']
        current_v_y = self.ego_dynamics['v_y']
        current_r = self.ego_dynamics['r']
        current_x = self.ego_dynamics['x']
        current_y = self.ego_dynamics['y']
        current_phi = self.ego_dynamics['phi']
        current_steer = self.ego_dynamics['steer']
        current_a_x = self.ego_dynamics['a_x']        
        steer, a_x = trans_action
        state = np.array([current_v_x, current_v_y, current_r, current_x, current_y, current_phi, current_steer, current_a_x])
        action = np.array([steer, a_x])
        next_ego_state, next_ego_param = state_trans_LPF(state, action)
        return next_ego_state  #type是个列表
    
    def _construct_ego_vector_short(self):
        ego_v_x = self.ego_dynamics['v_x']
        ego_v_y = self.ego_dynamics['v_y']
        ego_r = self.ego_dynamics['r']
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        ego_phi = self.ego_dynamics['phi']
        ego_steer = self.ego_dynamics['steer']
        ego_a_x = self.ego_dynamics['a_x']        
        ego_list = [ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi, ego_steer, ego_a_x]
        return np.array(ego_list)

    def _construct_veh_recarray(self):
        vehicle_info = np.dtype([
            ("veh_ID", "U16"),
            ("x", np.float),
            ("y", np.float),
            ("v", np.float),
            ("phi", np.float),
            ("l", np.float),
            ("w", np.float),
            ("route", "U3", (2,)),
        ])
        vehs_tuple_list = [tuple(info.values()) for info in self.relevant_vehicles]
        vehs_recarray = np.array(vehs_tuple_list, dtype=vehicle_info).view(np.recarray)
        return vehs_recarray

    
if __name__ == '__main__':
    pass