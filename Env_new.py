import warnings
import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.utils import seeding
from vehicle import VehicleDynamics
from Reference import ReferencePath
from Env_utils import shift_coordination, rotate_coordination, rotate_and_shift_coordination, deal_with_phi, \
    L, W, CROSSROAD_SIZE, LANE_WIDTH, LANE_NUMBER, judge_feasible, MODE2TASK, VEHICLE_MODE_DICT, VEH_NUM, STEP_TIME
from traffic import Traffic
warnings.filterwarnings("ignore")


class Crossroad():
    def __init__(self,init_ego_state,
                 **kwargs):

        self.init_state = init_ego_state
        self.traffic = Traffic()
        self.traffic.init_traffic(self.init_state)
        self.traffic.sim_step()   #### 存疑，是否有必要step一步
        self.v_light = self.traffic.v_light

        self.dynamics = VehicleDynamics()
        #self.ref_path = ref

        ego_dynamics = self._get_ego_dynamics([self.init_state['ego']['v_x'],
                                               self.init_state['ego']['v_y'],
                                               self.init_state['ego']['r'],
                                               self.init_state['ego']['x'],
                                               self.init_state['ego']['y'],
                                               self.init_state['ego']['phi']],
                                              [0,
                                               0,
                                               self.dynamics.vehicle_params['miu'],
                                               self.dynamics.vehicle_params['miu']]  #alpha_f, alpha_r, miu_f, miu_r
                                              )
        self.all_vehicles = self.traffic.n_ego_vehicles['ego']  # 多车的话，在这里要给出多个ID
        self.ego_dynamics = ego_dynamics

        self.all_info = None #打包了自车信息和周车信息
        self._get_all_info()

        self.obs =  self._get_obs()#自车和周围所有车(ego_list, recarray)
        self.action = None
        self.action_number = 2

        self.done_type = 'not_done_yet'
        self.done = None
        self.reward = 0
        self.reward_info = None

        self.ego_info_dim = 6


    def _get_ego_dynamics(self, next_ego_state, next_ego_params):  # update 
        out = dict(v_x=next_ego_state[0],
                   v_y=next_ego_state[1],
                   r=next_ego_state[2],
                   x=next_ego_state[3],
                   y=next_ego_state[4],
                   phi=next_ego_state[5],
                   l=L,
                   w=W,
                   alpha_f=next_ego_params[0],
                   alpha_r=next_ego_params[1],
                   miu_f=next_ego_params[2],
                   miu_r=next_ego_params[3],)
        miu_f, miu_r = out['miu_f'], out['miu_r']
        F_zf, F_zr = self.dynamics.vehicle_params['F_zf'], self.dynamics.vehicle_params['F_zr']
        C_f, C_r = self.dynamics.vehicle_params['C_f'], self.dynamics.vehicle_params['C_r']
        alpha_f_bound, alpha_r_bound = 3 * miu_f * F_zf / C_f, 3 * miu_r * F_zr / C_r
        r_bound = miu_r * self.dynamics.vehicle_params['g'] / (abs(out['v_x'])+1e-8)

        l, w, x, y, phi = out['l'], out['w'], out['x'], out['y'], out['phi']

        def cal_corner_point_of_ego_car():
            x0, y0, a0 = rotate_and_shift_coordination(l / 2, w / 2, 0, -x, -y, -phi)
            x1, y1, a1 = rotate_and_shift_coordination(l / 2, -w / 2, 0, -x, -y, -phi)
            x2, y2, a2 = rotate_and_shift_coordination(-l / 2, w / 2, 0, -x, -y, -phi)
            x3, y3, a3 = rotate_and_shift_coordination(-l / 2, -w / 2, 0, -x, -y, -phi)
            return (x0, y0), (x1, y1), (x2, y2), (x3, y3)
        Corner_point = cal_corner_point_of_ego_car()
        out.update(dict(alpha_f_bound=alpha_f_bound,
                        alpha_r_bound=alpha_r_bound,
                        r_bound=r_bound,
                        Corner_point=Corner_point))
        self.ego_dynamics = out  # 完成自车动力学参数的更新
        return out

    def _get_all_info(self):  # used to update info, must be called every timestep before _get_obs
        all_info = dict(all_vehicles=self.all_vehicles,
                        ego_dynamics=self.ego_dynamics,
                        v_light=self.v_light)
        return all_info

    def _get_obs(self):
        ego_vector = self._construct_ego_vector_short() 
        vehs_recarray = self._construct_veh_recarray()   # 将self.all_vehicle 的数据转成recarray的形式
        return ego_vector, vehs_recarray

    def _construct_ego_vector_short(self):
        ego_v_x = self.ego_dynamics['v_x']
        ego_v_y = self.ego_dynamics['v_y']
        ego_r = self.ego_dynamics['r']
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        ego_phi = self.ego_dynamics['phi']
        ego_feature = [ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi]
        self.ego_info_dim = 6
        return np.array(ego_feature, dtype=np.float32)


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
        vehs_tuple_list = [tuple(info.values()) for info in self.all_vehicles]
        vehs_recarray = np.array(vehs_tuple_list, dtype=vehicle_info).view(np.recarray)
        return vehs_recarray

    def compute_reward(self):
        reward = 0
        reward_info = None
        return reward, reward_info

    def _get_next_ego_state(self, trans_action):
        current_v_x = self.ego_dynamics['v_x']
        current_v_y = self.ego_dynamics['v_y']
        current_r = self.ego_dynamics['r']
        current_x = self.ego_dynamics['x']
        current_y = self.ego_dynamics['y']
        current_phi = self.ego_dynamics['phi']
        steer, a_x = trans_action
        state = np.array([current_v_x, current_v_y, current_r, current_x, current_y, current_phi], dtype=np.float32)
        action = np.array([steer, a_x], dtype=np.float32)
        next_ego_state, next_ego_params = self.dynamics.prediction(state, action, STEP_TIME)
        #next_ego_state, next_ego_params = next_ego_state[0],  next_ego_params[0]
        next_ego_state[0] = next_ego_state[0] if next_ego_state[0] >= 0 else 0.  # 保证第一个参数v_x要大于等于0
        next_ego_state[-1] = deal_with_phi(next_ego_state[-1])  # 对phi再进行一次scale
        return next_ego_state, next_ego_params


    def step(self, action):
        self.action = action

        next_ego_state, next_ego_params = self._get_next_ego_state(self.action)
        ego_dynamics = self._get_ego_dynamics(next_ego_state, next_ego_params)
        self.traffic.set_own_car(dict(ego=ego_dynamics))

        self.traffic.sim_step()
        self.all_vehicles = self.traffic.n_ego_vehicles['ego']
        self.obs = self._get_obs()

        done = None
        reward, self.reward_info = self.compute_reward()
        all_info = self._get_all_info()

        return self.obs, reward, done, all_info
        # all_info 作为一个容器，之后可以往里面塞信息


if __name__ == '__main__':
    pass