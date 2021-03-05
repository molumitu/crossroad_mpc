#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

'''
def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = gym.spaces.Dict(dict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        space = gym.spaces.Box(low, high, dtype=np.float32)
    else:
        raise NotImplementedError(type(observation), observation)

    return space
'''

class Crossroad():
    def __init__(self,
                 training_task, 
                 display=False,
                 **kwargs):
        self.dynamics = VehicleDynamics()
        self.interested_vehs = None
        self.training_task = training_task
        self.ref_path = ReferencePath(self.training_task, **kwargs)
        self.detected_vehicles = None
        self.all_vehicles = None
        self.ego_dynamics = None
        self.num_future_data = 0
        self.init_state = {}
        self.action_number = 2
        self.exp_v = 8.
        self.ego_l, self.ego_w = L, W
        # self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_number,), dtype=np.float32)

        # self.seed()
        self.v_light = None
        self.init_state = self._reset_init_state()
        if not display:
            self.traffic = Traffic(init_n_ego_dict=self.init_state)
            self.reset()
            #action = self.action_space.sample()
            #observation, _reward, done, _info = self.step(action)
            #self._set_observation_space(observation)
            #plt.ion()
        self.obs = None
        self.action = None
        self.veh_mode_dict = VEHICLE_MODE_DICT[self.training_task]
        self.veh_num = VEH_NUM[self.training_task]

        self.done_type = 'not_done_yet'
        self.reward_info = None
        self.ego_info_dim = None
        self.per_tracking_info_dim = None
        self.per_veh_info_dim = None

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def reset(self, **kwargs):  # kwargs include three keys
        self.ref_path = ReferencePath(self.training_task, **kwargs)
        self.init_state = self._reset_init_state()   #再度重新init
        self.traffic.init_traffic(self.init_state)
        self.traffic.sim_step() #为啥要走一步
        ego_dynamics = self._get_ego_dynamics([self.init_state['ego']['v_x'],
                                               self.init_state['ego']['v_y'],
                                               self.init_state['ego']['r'],
                                               self.init_state['ego']['x'],
                                               self.init_state['ego']['y'],
                                               self.init_state['ego']['phi']],
                                              [0,
                                               0,
                                               self.dynamics.vehicle_params['miu'],
                                               self.dynamics.vehicle_params['miu']]
                                              )
        self._get_all_info(ego_dynamics)
        self.obs = self._get_obs()
        self.action = None
        self.reward_info = None
        self.done_type = 'not_done_yet'
        return self.obs

    def close(self):
        del self.traffic

    def step(self, action):
        self.action = self._action_transformation_for_end2end(action)
        reward, self.reward_info = self.compute_reward(self.obs, self.action)
        ##### 这里reward没用
        next_ego_state, next_ego_params = self._get_next_ego_state(self.action)
        ego_dynamics = self._get_ego_dynamics(next_ego_state, next_ego_params)
        self.traffic.set_own_car(dict(ego=ego_dynamics))
        self.traffic.sim_step()
        all_info = self._get_all_info(ego_dynamics)
        self.obs = self._get_obs()
        self.done_type, done = self._judge_done()
        self.reward_info.update({'final_rew': reward})
        # if done:
        #     print(self.done_type)
        all_info.update({'reward_info': self.reward_info, 'ref_index': self.ref_path.ref_index})
        return self.obs, reward, done, all_info
    '''
    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space
    '''
    def _get_ego_dynamics(self, next_ego_state, next_ego_params):
        out = dict(v_x=next_ego_state[0],
                   v_y=next_ego_state[1],
                   r=next_ego_state[2],
                   x=next_ego_state[3],
                   y=next_ego_state[4],
                   phi=next_ego_state[5],
                   l=self.ego_l,
                   w=self.ego_w,
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

        return out

    def _get_all_info(self, ego_dynamics):  # used to update info, must be called every timestep before _get_obs
        # to fetch info
        self.all_vehicles = self.traffic.n_ego_vehicles['ego']  # 多车的话，在这里要给出多个ID
        self.ego_dynamics = ego_dynamics  # coordination 2
        self.v_light = self.traffic.v_light

        # all_vehicles
        # dict(x=x, y=y, v=v, phi=a, l=length,
        #      w=width, route=route)

        all_info = dict(all_vehicles=self.all_vehicles,
                        ego_dynamics=self.ego_dynamics,
                        v_light=self.v_light)
        return all_info

    def _judge_done(self):
        """
        :return:
         1: bad done: collision
         2: bad done: break_road_constrain
         3: good done: task succeed
         4: not done
        """
        if self.traffic.collision_flag:
            return 'collision', 1
        if self._break_road_constrain():
            return 'break_road_constrain', 2
        # elif self._deviate_too_much():
        #     return 'deviate_too_much', 1
        # elif self._break_stability():
        #     return 'break_stability', 1
        elif self._break_red_light():
            return 'break_red_light', 1
        elif self._is_achieve_goal():
            return 'good_done', 3
        else:
            return 'not_done_yet', 0

    def _deviate_too_much(self):
        delta_y, delta_phi, delta_v = self.obs[self.ego_info_dim:self.ego_info_dim+3]
        return True if abs(delta_y) > 15 else False

    def _break_road_constrain(self):
        results = list(map(lambda x: judge_feasible(*x, self.training_task), self.ego_dynamics['Corner_point']))
        return not all(results)

    def _break_stability(self):
        alpha_f, alpha_r, miu_f, miu_r = self.ego_dynamics['alpha_f'], self.ego_dynamics['alpha_r'], \
                                         self.ego_dynamics['miu_f'], self.ego_dynamics['miu_r']
        alpha_f_bound, alpha_r_bound = self.ego_dynamics['alpha_f_bound'], self.ego_dynamics['alpha_r_bound']
        r_bound = self.ego_dynamics['r_bound']
        # if -alpha_f_bound < alpha_f < alpha_f_bound \
        #         and -alpha_r_bound < alpha_r < alpha_r_bound and \
        #         -r_bound < self.ego_dynamics['r'] < r_bound:
        if -r_bound < self.ego_dynamics['r'] < r_bound:
            return False
        else:
            return True

    def _break_red_light(self):
        return True if self.v_light != 0 and self.ego_dynamics['y'] > -CROSSROAD_SIZE/2 and self.training_task != 'right' else False

    def _is_achieve_goal(self):
        x = self.ego_dynamics['x']
        y = self.ego_dynamics['y']
        if self.training_task == 'left':
            return True if x < -CROSSROAD_SIZE/2 - 10 and 0 < y < LANE_NUMBER*LANE_WIDTH else False
        elif self.training_task == 'right':
            return True if x > CROSSROAD_SIZE/2 + 10 and -LANE_NUMBER*LANE_WIDTH < y < 0 else False
        else:
            assert self.training_task == 'straight'
            return True if y > CROSSROAD_SIZE/2 + 10 and 0 < x < LANE_NUMBER*LANE_WIDTH else False

    def _action_transformation_for_end2end(self, action):  # [-1, 1]
        steer_norm, a_x_norm = action[0], action[1]
        scaled_steer = steer_norm
        scaled_a_x = a_x_norm
        # scaled_steer = 0.35 * steer_norm
        # scaled_a_x = 5.*a_x_norm - 2
        # 这里可以写紧急制动程序，但是写在这里不直观
        # if self.v_light != 0 and self.ego_dynamics['y'] < -18 and self.training_task != 'right':
        #     scaled_steer = 0.
        #     scaled_a_x = -3.

        scaled_action = np.array([scaled_steer, scaled_a_x], dtype=np.float32)
        return scaled_action

    def _get_next_ego_state(self, trans_action):
        current_v_x = self.ego_dynamics['v_x']
        current_v_y = self.ego_dynamics['v_y']
        current_r = self.ego_dynamics['r']
        current_x = self.ego_dynamics['x']
        current_y = self.ego_dynamics['y']
        current_phi = self.ego_dynamics['phi']
        steer, a_x = trans_action
        state = np.array([[current_v_x, current_v_y, current_r, current_x, current_y, current_phi]], dtype=np.float32)
        action = np.array([[steer, a_x]], dtype=np.float32)
        next_ego_state, next_ego_params = self.dynamics.prediction(state, action, STEP_TIME)
        next_ego_state, next_ego_params = next_ego_state[0],  next_ego_params[0]
        next_ego_state[0] = next_ego_state[0] if next_ego_state[0] >= 0 else 0.
        next_ego_state[-1] = deal_with_phi(next_ego_state[-1])
        return next_ego_state, next_ego_params

    def _get_obs(self, exit_='D'):
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        ego_phi = self.ego_dynamics['phi']
        ego_v_x = self.ego_dynamics['v_x']
        vehs_vector = self._construct_veh_vector_short(exit_)
        ego_vector = self._construct_ego_vector_short()
        tracking_error = self.ref_path.tracking_error_vector(np.array([ego_x], dtype=np.float32),
                                                             np.array([ego_y], dtype=np.float32),
                                                             np.array([ego_phi], dtype=np.float32),
                                                             np.array([ego_v_x], dtype=np.float32),
                                                             self.num_future_data)[0]
        self.per_tracking_info_dim = 3

        vector = np.concatenate((ego_vector, tracking_error, vehs_vector), axis=0)
        vector = self.convert_vehs_to_rela(vector)

        return vector

    def convert_vehs_to_rela(self, obs_abso):  # 由绝对值计算相对值
        ego_infos, tracking_infos, veh_infos = obs_abso[:self.ego_info_dim], \
                                               obs_abso[self.ego_info_dim:self.ego_info_dim + self.per_tracking_info_dim * (
                                                         1)], \
                                               obs_abso[self.ego_info_dim + self.per_tracking_info_dim * (
                                                           self.num_future_data + 1):]
        ego_vx, ego_vy, ego_r, ego_x, ego_y, ego_phi = ego_infos
        ego = np.array([ego_x, ego_y, 0, 0]*int(len(veh_infos)/self.per_veh_info_dim), dtype=np.float32)
        vehs_rela = veh_infos - ego
        out = np.concatenate((ego_infos, tracking_infos, vehs_rela), axis=0)
        return out

    def convert_vehs_to_abso(self, obs_rela):  # 由相对值计算绝对值
        ego_infos, tracking_infos, veh_rela = obs_rela[:self.ego_info_dim], \
                                               obs_rela[self.ego_info_dim:self.ego_info_dim + self.per_tracking_info_dim * (
                                                       self.num_future_data + 1)], \
                                               obs_rela[self.ego_info_dim + self.per_tracking_info_dim * (
                                                       self.num_future_data + 1):]
        ego_vx, ego_vy, ego_r, ego_x, ego_y, ego_phi = ego_infos
        ego = np.array([ego_x, ego_y, 0, 0]*int(len(veh_rela)/self.per_veh_info_dim), dtype=np.float32)
        vehs_abso = veh_rela + ego
        out = np.concatenate((ego_infos, tracking_infos, vehs_abso), axis=0)
        return out

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

    def _construct_veh_vector_short(self, exit_='D'):
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        v_light = self.v_light
        vehs_vector = []

        name_settings = dict(D=dict(do='1o', di='1i', ro='2o', ri='2i', uo='3o', ui='3i', lo='4o', li='4i'),
                             R=dict(do='2o', di='2i', ro='3o', ri='3i', uo='4o', ui='4i', lo='1o', li='1i'),
                             U=dict(do='3o', di='3i', ro='4o', ri='4i', uo='1o', ui='1i', lo='2o', li='2i'),
                             L=dict(do='4o', di='4i', ro='1o', ri='1i', uo='2o', ui='2i', lo='3o', li='3i'))

        name_setting = name_settings[exit_]

        def filter_interested_vehicles(vs, task):
            dl, du, dr, rd, rl, ru, ur, ud, ul, lu, lr, ld = [], [], [], [], [], [], [], [], [], [], [], []
            for v in vs:
                route_list = v['route']
                start = route_list[0]
                end = route_list[1]
                if start == name_setting['do'] and end == name_setting['li']:
                    dl.append(v)
                elif start == name_setting['do'] and end == name_setting['ui']:
                    du.append(v)
                elif start == name_setting['do'] and end == name_setting['ri']:
                    dr.append(v)

                elif start == name_setting['ro'] and end == name_setting['di']:
                    rd.append(v)
                elif start == name_setting['ro'] and end == name_setting['li']:
                    rl.append(v)
                elif start == name_setting['ro'] and end == name_setting['ui']:
                    ru.append(v)

                elif start == name_setting['uo'] and end == name_setting['ri']:
                    ur.append(v)
                elif start == name_setting['uo'] and end == name_setting['di']:
                    ud.append(v)
                elif start == name_setting['uo'] and end == name_setting['li']:
                    ul.append(v)

                elif start == name_setting['lo'] and end == name_setting['ui']:
                    lu.append(v)
                elif start == name_setting['lo'] and end == name_setting['ri']:
                    lr.append(v)
                elif start == name_setting['lo'] and end == name_setting['di']:
                    ld.append(v)
            if v_light != 0 and ego_y < -CROSSROAD_SIZE/2:
                dl.append(dict(x=LANE_WIDTH/2, y=-CROSSROAD_SIZE/2, v=0., phi=90, l=5, w=2.5, route=None))
                dl.append(dict(x=LANE_WIDTH/2, y=-CROSSROAD_SIZE/2+2.5, v=0., phi=90, l=5, w=2.5, route=None))
                du.append(dict(x=LANE_WIDTH*1.5, y=-CROSSROAD_SIZE/2, v=0., phi=90, l=5, w=2.5, route=None))
                du.append(dict(x=LANE_WIDTH*1.5, y=-CROSSROAD_SIZE/2+2.5, v=0., phi=90, l=5, w=2.5, route=None))

            # fetch veh in range
            dl = list(filter(lambda v: v['x'] > -CROSSROAD_SIZE/2-10 and v['y'] > ego_y-2, dl))  # interest of left straight
            du = list(filter(lambda v: ego_y-2 < v['y'] < CROSSROAD_SIZE/2+10 and v['x'] < ego_x+5, du))  # interest of left straight

            dr = list(filter(lambda v: v['x'] < CROSSROAD_SIZE/2+10 and v['y'] > ego_y, dr))  # interest of right

            rd = rd  # not interest in case of traffic light
            rl = rl  # not interest in case of traffic light
            ru = list(filter(lambda v: v['x'] < CROSSROAD_SIZE/2+10 and v['y'] < CROSSROAD_SIZE/2+10, ru))  # interest of straight

            ur_straight = list(filter(lambda v: v['x'] < ego_x + 7 and ego_y < v['y'] < CROSSROAD_SIZE/2+10, ur))  # interest of straight
            ur_right = list(filter(lambda v: v['x'] < CROSSROAD_SIZE/2+10 and v['y'] < CROSSROAD_SIZE/2, ur))  # interest of right
            ud = list(filter(lambda v: max(ego_y-2, -CROSSROAD_SIZE/2) < v['y'] < CROSSROAD_SIZE/2 and ego_x > v['x'], ud))  # interest of left
            ul = list(filter(lambda v: -CROSSROAD_SIZE/2-10 < v['x'] < ego_x and v['y'] < CROSSROAD_SIZE/2, ul))  # interest of left

            lu = lu  # not interest in case of traffic light
            lr = list(filter(lambda v: -CROSSROAD_SIZE/2-10 < v['x'] < CROSSROAD_SIZE/2+10, lr))  # interest of right
            ld = ld  # not interest in case of traffic light

            # sort
            dl = sorted(dl, key=lambda v: (v['y'], -v['x']))
            du = sorted(du, key=lambda v: v['y'])
            dr = sorted(dr, key=lambda v: (v['y'], v['x']))

            ru = sorted(ru, key=lambda v: (-v['x'], v['y']), reverse=True)

            ur_straight = sorted(ur_straight, key=lambda v: v['y'])
            ur_right = sorted(ur_right, key=lambda v: (-v['y'], v['x']), reverse=True)

            ud = sorted(ud, key=lambda v: v['y'])
            ul = sorted(ul, key=lambda v: (-v['y'], -v['x']), reverse=True)

            lr = sorted(lr, key=lambda v: -v['x'])

            # slice or fill to some number
            def slice_or_fill(sorted_list, fill_value, num):
                if len(sorted_list) >= num:
                    return sorted_list[:num]
                else:
                    while len(sorted_list) < num:
                        sorted_list.append(fill_value)
                    return sorted_list

            fill_value_for_dl = dict(x=LANE_WIDTH/2, y=-(CROSSROAD_SIZE/2+30), v=0, phi=90, w=2.5, l=5, route=('1o', '4i'))
            fill_value_for_du = dict(x=LANE_WIDTH*1.5, y=-(CROSSROAD_SIZE/2+30), v=0, phi=90, w=2.5, l=5, route=('1o', '3i'))
            fill_value_for_dr = dict(x=LANE_WIDTH*(LANE_NUMBER-0.5), y=-(CROSSROAD_SIZE/2+30), v=0, phi=90, w=2.5, l=5, route=('1o', '2i'))

            fill_value_for_ru = dict(x=(CROSSROAD_SIZE/2+15), y=LANE_WIDTH*(LANE_NUMBER-0.5), v=0, phi=180, w=2.5, l=5, route=('2o', '3i'))

            fill_value_for_ur_straight = dict(x=-LANE_WIDTH/2, y=(CROSSROAD_SIZE/2+20), v=0, phi=-90, w=2.5, l=5, route=('3o', '2i'))
            fill_value_for_ur_right = dict(x=-LANE_WIDTH/2, y=(CROSSROAD_SIZE/2+20), v=0, phi=-90, w=2.5, l=5, route=('3o', '2i'))

            fill_value_for_ud = dict(x=-LANE_WIDTH*1.5, y=(CROSSROAD_SIZE/2+20), v=0, phi=-90, w=2.5, l=5, route=('3o', '1i'))
            fill_value_for_ul = dict(x=-LANE_WIDTH*(LANE_NUMBER-0.5), y=(CROSSROAD_SIZE/2+20), v=0, phi=-90, w=2.5, l=5, route=('3o', '4i'))

            fill_value_for_lr = dict(x=-(CROSSROAD_SIZE/2+20), y=-LANE_WIDTH*1.5, v=0, phi=0, w=2.5, l=5, route=('4o', '2i'))

            tmp = dict()
            if task == 'left':
                tmp['dl'] = slice_or_fill(dl, fill_value_for_dl, VEHICLE_MODE_DICT['left']['dl'])
                tmp['du'] = slice_or_fill(du, fill_value_for_du, VEHICLE_MODE_DICT['left']['du'])
                tmp['ud'] = slice_or_fill(ud, fill_value_for_ud, VEHICLE_MODE_DICT['left']['ud'])
                tmp['ul'] = slice_or_fill(ul, fill_value_for_ul, VEHICLE_MODE_DICT['left']['ul'])
            elif task == 'straight':
                tmp['dl'] = slice_or_fill(dl, fill_value_for_dl, VEHICLE_MODE_DICT['straight']['dl'])
                tmp['du'] = slice_or_fill(du, fill_value_for_du, VEHICLE_MODE_DICT['straight']['du'])
                tmp['ud'] = slice_or_fill(ud, fill_value_for_ud, VEHICLE_MODE_DICT['straight']['ud'])
                tmp['ru'] = slice_or_fill(ru, fill_value_for_ru, VEHICLE_MODE_DICT['straight']['ru'])
                tmp['ur'] = slice_or_fill(ur_straight, fill_value_for_ur_straight, VEHICLE_MODE_DICT['straight']['ur'])
            elif task == 'right':
                tmp['dr'] = slice_or_fill(dr, fill_value_for_dr, VEHICLE_MODE_DICT['right']['dr'])
                tmp['ur'] = slice_or_fill(ur_right, fill_value_for_ur_right, VEHICLE_MODE_DICT['right']['ur'])
                tmp['lr'] = slice_or_fill(lr, fill_value_for_lr, VEHICLE_MODE_DICT['right']['lr'])

            return tmp

        list_of_interested_veh_dict = []
        self.interested_vehs = filter_interested_vehicles(self.all_vehicles, self.training_task)
        for part in list(self.interested_vehs.values()):
            list_of_interested_veh_dict.extend(part)

        for veh in list_of_interested_veh_dict:
            veh_x, veh_y, veh_v, veh_phi = veh['x'], veh['y'], veh['v'], veh['phi']
            vehs_vector.extend([veh_x, veh_y, veh_v, veh_phi])
        self.per_veh_info_dim = 4
        return np.array(vehs_vector, dtype=np.float32)

    '''
    def recover_orig_position_fn(self, transformed_x, transformed_y, x, y, d):  # x, y, d are used to transform
        # coordination
        transformed_x, transformed_y, _ = rotate_coordination(transformed_x, transformed_y, 0, -d)
        orig_x, orig_y = shift_coordination(transformed_x, transformed_y, -x, -y)
        return orig_x, orig_y
    '''

    def _reset_init_state(self):
        if self.training_task == 'left':
            random_index = 300
        elif self.training_task == 'straight':
            random_index = int(np.random.random()*(1200+500)) + 700
        else:
            random_index = int(np.random.random()*(420+500)) + 700

        x, y, phi = self.ref_path.indexs2points(random_index)
        v = 7 
        #v = 8 * np.random.random()
        if self.training_task == 'left':
            routeID = 'dl'
        elif self.training_task == 'straight':
            routeID = 'du'
        else:
            assert self.training_task == 'right'
            routeID = 'dr'
        return dict(ego=dict(v_x=v,
                             v_y=0,
                             r=0,
                             x=x,
                             y=y,
                             phi=phi,
                             l=self.ego_l,
                             w=self.ego_w,
                             routeID=routeID,
                             ))

    def compute_reward(self, obs, action):
        obs = self.convert_vehs_to_abso(obs)
        ego_infos, tracking_infos, veh_infos = obs[:self.ego_info_dim], obs[self.ego_info_dim:self.ego_info_dim + self.per_tracking_info_dim * (self.num_future_data+1)], \
                                               obs[self.ego_info_dim + self.per_tracking_info_dim * (self.num_future_data+1):]
        steers, a_xs = action[0], action[1]

        # rewards related to action
        punish_steer = -np.square(steers)
        punish_a_x = -np.square(a_xs)

        # rewards related to ego stability
        punish_yaw_rate = -np.square(ego_infos[2])

        # rewards related to tracking error
        # devi_y = np.square(delta_s)
        devi_y = -np.square(tracking_infos[0])
        devi_phi = -np.array(np.square(tracking_infos[1] * np.pi / 180.), dtype=np.float32)
        devi_v = -np.square(tracking_infos[2])
        delta_s = np.array(tracking_infos[0])
        delta_phi = np.array(tracking_infos[1] * np.pi / 180., dtype=np.float32)
        delta_v = np.array(tracking_infos[2])

        veh2veh4training = np.array(0.)
        veh2veh4real = np.array(0.)
        ego_lws = (L - W) / 2.
        ego_front_points = np.array(ego_infos[3] + ego_lws * np.cos(ego_infos[5] * np.pi / 180.), dtype=np.float32), \
                           np.array(ego_infos[4] + ego_lws * np.sin(ego_infos[5] * np.pi / 180.), dtype=np.float32)
        ego_rear_points = np.array(ego_infos[3] - ego_lws * np.cos(ego_infos[5] * np.pi / 180.), dtype=np.float32), \
                          np.array(ego_infos[4] - ego_lws * np.sin(ego_infos[5] * np.pi / 180.), dtype=np.float32)

        for veh_index in range(int(len(veh_infos) / self.per_veh_info_dim)):
            vehs = veh_infos[veh_index * self.per_veh_info_dim:(veh_index + 1) * self.per_veh_info_dim]
            veh_lws = (L - W) / 2.
            veh_front_points = np.array(vehs[0] + veh_lws * np.cos(vehs[3] * np.pi / 180.), dtype=np.float32), \
                               np.array(vehs[1] + veh_lws * np.sin(vehs[3] * np.pi / 180.), dtype=np.float32)
            veh_rear_points = np.array(vehs[0] - veh_lws * np.cos(vehs[3] * np.pi / 180.), dtype=np.float32), \
                              np.array(vehs[1] - veh_lws * np.sin(vehs[3] * np.pi / 180.), dtype=np.float32)
            for ego_point in [ego_front_points, ego_rear_points]:
                for veh_point in [veh_front_points, veh_rear_points]:
                    veh2veh_dist = np.sqrt(np.square(ego_point[0] - veh_point[0]) + np.square(ego_point[1] - veh_point[1]))
                    veh2veh4training += np.square(veh2veh_dist) if veh2veh_dist - 3.5 < 0 else 0
                    veh2veh4real += np.square(veh2veh_dist) if veh2veh_dist - 2.5 < 0 else 0

        veh2road4training = np.array(0.)
        veh2road4real = np.array(0.)
        if self.training_task == 'left':
            for ego_point in [ego_front_points, ego_rear_points]:
                veh2road4training += np.where(np.logical_and(ego_point[1] < -CROSSROAD_SIZE / 2, ego_point[0] < 1),
                                              np.square(ego_point[0] - 1), 0.)
                veh2road4training += np.where(
                    np.logical_and(ego_point[1] < -CROSSROAD_SIZE / 2, LANE_WIDTH - ego_point[0] < 1),
                    np.square(LANE_WIDTH - ego_point[0] - 1), 0.)
                veh2road4training += np.where(
                    np.logical_and(ego_point[0] < 0, LANE_WIDTH * LANE_NUMBER - ego_point[1] < 1),
                    np.square(LANE_WIDTH * LANE_NUMBER - ego_point[1] - 1), 0.)
                veh2road4training += np.where(np.logical_and(ego_point[0] < -CROSSROAD_SIZE / 2, ego_point[1] - 0 < 1),
                                              np.square(ego_point[1] - 0 - 1), 0.)

                veh2road4real += np.where(np.logical_and(ego_point[1] < -CROSSROAD_SIZE / 2, ego_point[0] < 1),
                                          np.square(ego_point[0] - 1), 0.)
                veh2road4real += np.where(
                    np.logical_and(ego_point[1] < -CROSSROAD_SIZE / 2, LANE_WIDTH - ego_point[0] < 1),
                    np.square(LANE_WIDTH - ego_point[0] - 1), 0.)
                veh2road4real += np.where(
                    np.logical_and(ego_point[0] < -CROSSROAD_SIZE / 2, LANE_WIDTH * LANE_NUMBER - ego_point[1] < 1),
                    np.square(LANE_WIDTH * LANE_NUMBER - ego_point[1] - 1), 0.)
                veh2road4real += np.where(np.logical_and(ego_point[0] < -CROSSROAD_SIZE / 2, ego_point[1] - 0 < 1),
                                          np.square(ego_point[1] - 0 - 1), 0.)
        elif self.training_task == 'straight':
            for ego_point in [ego_front_points, ego_rear_points]:
                veh2road4training += np.where(np.logical_and(ego_point[1] < -CROSSROAD_SIZE / 2, ego_point[0]-LANE_WIDTH < 1),
                                              np.square(ego_point[0]-LANE_WIDTH - 1), 0.)
                veh2road4training += np.where(
                    np.logical_and(ego_point[1] < -CROSSROAD_SIZE / 2, 2 * LANE_WIDTH - ego_point[0] < 1),
                    np.square(2 * LANE_WIDTH - ego_point[0] - 1), 0.)
                veh2road4training += np.where(
                    np.logical_and(ego_point[1] > CROSSROAD_SIZE / 2, LANE_WIDTH * LANE_NUMBER - ego_point[0] < 1),
                    np.square(LANE_WIDTH * LANE_NUMBER - ego_point[0] - 1), 0.)
                veh2road4training += np.where(np.logical_and(ego_point[1] > CROSSROAD_SIZE / 2, ego_point[0] - 0 < 1),
                                              np.square(ego_point[0] - 0 - 1), 0.)

                veh2road4real += np.where(np.logical_and(ego_point[1] < -CROSSROAD_SIZE / 2, ego_point[0]-LANE_WIDTH < 1),
                                          np.square(ego_point[0]-LANE_WIDTH - 1), 0.)
                veh2road4real += np.where(
                    np.logical_and(ego_point[1] < -CROSSROAD_SIZE / 2, 2 * LANE_WIDTH - ego_point[0] < 1),
                    np.square(2 * LANE_WIDTH - ego_point[0] - 1), 0.)
                veh2road4real += np.where(
                    np.logical_and(ego_point[1] > CROSSROAD_SIZE / 2, LANE_WIDTH * LANE_NUMBER - ego_point[0] < 1),
                    np.square(LANE_WIDTH * LANE_NUMBER - ego_point[0] - 1), 0.)
                veh2road4real += np.where(np.logical_and(ego_point[1] > CROSSROAD_SIZE / 2, ego_point[0] - 0 < 1),
                                          np.square(ego_point[0] - 0 - 1), 0.)
        else:
            assert self.training_task == 'right'
            for ego_point in [ego_front_points, ego_rear_points]:
                veh2road4training += np.where(
                    np.logical_and(ego_point[1] < -CROSSROAD_SIZE / 2, ego_point[0] - 2 * LANE_WIDTH < 1),
                    np.square(ego_point[0] - 2 * LANE_WIDTH - 1), 0.)
                veh2road4training += np.where(
                    np.logical_and(ego_point[1] < -CROSSROAD_SIZE / 2, LANE_NUMBER * LANE_WIDTH - ego_point[0] < 1),
                    np.square(LANE_NUMBER * LANE_WIDTH - ego_point[0] - 1), 0.)
                veh2road4training += np.where(np.logical_and(ego_point[0] > CROSSROAD_SIZE / 2, 0 - ego_point[1] < 1),
                                              np.square(0 - ego_point[1] - 1), 0.)
                veh2road4training += np.where(
                    np.logical_and(ego_point[0] > CROSSROAD_SIZE / 2, ego_point[1] - (-LANE_WIDTH * LANE_NUMBER) < 1),
                    np.square(ego_point[1] - (-LANE_WIDTH * LANE_NUMBER) - 1), 0.)

                veh2road4real += np.where(
                    np.logical_and(ego_point[1] < -CROSSROAD_SIZE / 2, ego_point[0] - 2 * LANE_WIDTH < 1),
                    np.square(ego_point[0] - 2 * LANE_WIDTH - 1), 0.)
                veh2road4real += np.where(
                    np.logical_and(ego_point[1] < -CROSSROAD_SIZE / 2, LANE_NUMBER * LANE_WIDTH - ego_point[0] < 1),
                    np.square(LANE_NUMBER * LANE_WIDTH - ego_point[0] - 1), 0.)
                veh2road4real += np.where(np.logical_and(ego_point[0] > CROSSROAD_SIZE / 2, 0 - ego_point[1] < 1),
                                          np.square(0 - ego_point[1] - 1), 0.)
                veh2road4real += np.where(
                    np.logical_and(ego_point[0] > CROSSROAD_SIZE / 2, ego_point[1] - (-LANE_WIDTH * LANE_NUMBER) < 1),
                    np.square(ego_point[1] - (-LANE_WIDTH * LANE_NUMBER) - 1), 0.)

        # reward = 0.05 * devi_v + 0.8 * devi_y + 30 * devi_phi + 0.02 * punish_yaw_rate + \
        #          5 * punish_steer + 0.05 * punish_a_x
        reward = 0
        reward_dict = dict( punish_steer=punish_steer,
                            punish_a_x=punish_a_x,
                            punish_yaw_rate=punish_yaw_rate,
                            delta_v = delta_v,
                            delta_s = delta_s,
                            delta_phi = delta_phi,
                            devi_v=devi_v,
                            devi_y=devi_y,
                            devi_phi=devi_phi,
                            ############## scaled 的数据比较存疑 ##########
                            scaled_punish_steer=5 * punish_steer,
                            scaled_punish_a_x=0.05 * punish_a_x,
                            scaled_punish_yaw_rate=0.02 * punish_yaw_rate,
                            scaled_devi_v=0.05 * devi_v,
                            scaled_devi_y=0.8 * devi_y,
                            scaled_devi_phi=30 * devi_phi,
                            #############################################
                            veh2veh4training=veh2veh4training,
                            veh2road4training=veh2road4training,
                            veh2veh4real=veh2veh4real,
                            veh2road4real=veh2road4real,
                            )

        return reward, reward_dict


    # def render(self, name_index):
    #     # plot basic map
    #     extension = 40
    #     light_line_width = 2
    #     lane_edge_width = 1.5
    #     dotted_line_style = '--'
    #     solid_line_style = '-'

    #     fig, ax = plt.subplots()
    #     ax.set_title("Signal Intersection")
    #     ax.set_xlim([-CROSSROAD_SIZE / 2 - extension, CROSSROAD_SIZE / 2 + extension])
    #     ax.set_ylim([-CROSSROAD_SIZE / 2 - extension, CROSSROAD_SIZE / 2 + extension])
    #     ax.axis("equal")
    #     ax.axis('off')

    #     # ---------rectangle edge------------ 
    #     # ax.add_patch(plt.Rectangle((-CROSSROAD_SIZE / 2 - extension, -CROSSROAD_SIZE / 2 - extension),
    #     #                             CROSSROAD_SIZE + 2 * extension, CROSSROAD_SIZE + 2 * extension, edgecolor='black',
    #     #                             facecolor='none'))

    #     # ----------arrow--------------------
    #     ax.arrow(LANE_WIDTH/2, -CROSSROAD_SIZE / 2-10, 0, 5, color='gray')
    #     ax.arrow(LANE_WIDTH/2, -CROSSROAD_SIZE / 2-10+5, -0.5, 0, color='gray', head_width=1)
    #     ax.arrow(LANE_WIDTH*1.5, -CROSSROAD_SIZE / 2-10, 0, 5, color='gray', head_width=1)
    #     ax.arrow(LANE_WIDTH*2.5, -CROSSROAD_SIZE / 2 - 10, 0, 5, color='gray')
    #     ax.arrow(LANE_WIDTH*2.5, -CROSSROAD_SIZE / 2 - 10+5, 0.5, 0, color='gray', head_width=1)

    #     # ----------horizon--------------
    #     ax.plot([-CROSSROAD_SIZE / 2 - extension, -CROSSROAD_SIZE / 2], [0, 0], color='orange', linewidth = lane_edge_width)
    #     ax.plot([CROSSROAD_SIZE / 2 + extension, CROSSROAD_SIZE / 2], [0, 0], color='orange', linewidth = lane_edge_width)

    #     #
    #     for i in range(1, LANE_NUMBER + 1):
    #         linestyle = dotted_line_style if i < LANE_NUMBER else solid_line_style
    #         ax.plot([-CROSSROAD_SIZE / 2 - extension, -CROSSROAD_SIZE / 2], [i * LANE_WIDTH, i * LANE_WIDTH],
    #                     linestyle=linestyle, color='black', linewidth = lane_edge_width)
    #         ax.plot([CROSSROAD_SIZE / 2 + extension, CROSSROAD_SIZE / 2], [i * LANE_WIDTH, i * LANE_WIDTH],
    #                     linestyle=linestyle, color='black', linewidth = lane_edge_width)
    #         ax.plot([-CROSSROAD_SIZE / 2 - extension, -CROSSROAD_SIZE / 2], [-i * LANE_WIDTH, -i * LANE_WIDTH],
    #                     linestyle=linestyle, color='black', linewidth = lane_edge_width)
    #         ax.plot([CROSSROAD_SIZE / 2 + extension, CROSSROAD_SIZE / 2], [-i * LANE_WIDTH, -i * LANE_WIDTH],
    #                     linestyle=linestyle, color='black', linewidth = lane_edge_width)

    #     # ----------vertical----------------
    #     ax.plot([0, 0], [-CROSSROAD_SIZE / 2 - extension, -CROSSROAD_SIZE / 2], color='orange', linewidth = lane_edge_width)
    #     ax.plot([0, 0], [CROSSROAD_SIZE / 2 + extension, CROSSROAD_SIZE / 2], color='orange', linewidth = lane_edge_width)

    #     #
    #     for i in range(1, LANE_NUMBER + 1):
    #         linestyle = dotted_line_style if i < LANE_NUMBER else solid_line_style
    #         ax.plot([i * LANE_WIDTH, i * LANE_WIDTH], [-CROSSROAD_SIZE / 2 - extension, -CROSSROAD_SIZE / 2],
    #                     linestyle=linestyle, color='black', linewidth = lane_edge_width)
    #         ax.plot([i * LANE_WIDTH, i * LANE_WIDTH], [CROSSROAD_SIZE / 2 + extension, CROSSROAD_SIZE / 2],
    #                     linestyle=linestyle, color='black', linewidth = lane_edge_width)
    #         ax.plot([-i * LANE_WIDTH, -i * LANE_WIDTH], [-CROSSROAD_SIZE / 2 - extension, -CROSSROAD_SIZE / 2],
    #                     linestyle=linestyle, color='black', linewidth = lane_edge_width)
    #         ax.plot([-i * LANE_WIDTH, -i * LANE_WIDTH], [CROSSROAD_SIZE / 2 + extension, CROSSROAD_SIZE / 2],
    #                     linestyle=linestyle, color='black', linewidth = lane_edge_width)

    #     # # ----------stop line--------------
    #     # ax.plot([0, 2 * LANE_WIDTH], [-CROSSROAD_SIZE / 2, -CROSSROAD_SIZE / 2],
    #     #          color='black')
    #     # ax.plot([-2 * LANE_WIDTH, 0], [CROSSROAD_SIZE / 2, CROSSROAD_SIZE / 2],
    #     #          color='black')
    #     # ax.plot([-CROSSROAD_SIZE / 2, -CROSSROAD_SIZE / 2], [0, -2 * LANE_WIDTH],
    #     #          color='black')
    #     # ax.plot([CROSSROAD_SIZE / 2, CROSSROAD_SIZE / 2], [2 * LANE_WIDTH, 0],
    #     #          color='black')

    #     v_light = self.v_light
    #     if v_light == 0:
    #         v_color, h_color = 'green', 'red'
    #     elif v_light == 1:
    #         v_color, h_color = 'orange', 'red'
    #     elif v_light == 2:
    #         v_color, h_color = 'red', 'green'
    #     else:
    #         v_color, h_color = 'red', 'orange'

    #     # top vertical
    #     ax.plot([0, (LANE_NUMBER-1)*LANE_WIDTH], [-CROSSROAD_SIZE / 2, -CROSSROAD_SIZE / 2],
    #                 color=v_color, linewidth=light_line_width)
    #     ax.plot([(LANE_NUMBER-1)*LANE_WIDTH, LANE_NUMBER * LANE_WIDTH], [-CROSSROAD_SIZE / 2, -CROSSROAD_SIZE / 2],
    #                 color='green', linewidth=light_line_width)

    #     # down vertical
    #     ax.plot([-(LANE_NUMBER-1)*LANE_WIDTH, 0], [CROSSROAD_SIZE / 2, CROSSROAD_SIZE / 2],
    #                 color=v_color, linewidth=light_line_width)
    #     ax.plot([-LANE_NUMBER * LANE_WIDTH, -(LANE_NUMBER-1)*LANE_WIDTH], [CROSSROAD_SIZE / 2, CROSSROAD_SIZE / 2],
    #             color='green', linewidth=light_line_width)
                
    #     # left horizon
    #     ax.plot([-CROSSROAD_SIZE / 2, -CROSSROAD_SIZE / 2], [0, -(LANE_NUMBER-1)*LANE_WIDTH],
    #                 color=h_color, linewidth=light_line_width)
    #     ax.plot([-CROSSROAD_SIZE / 2, -CROSSROAD_SIZE / 2], [-(LANE_NUMBER-1)*LANE_WIDTH, -LANE_NUMBER * LANE_WIDTH],
    #                 color='green', linewidth=light_line_width)
    #     # right horizon
    #     ax.plot([CROSSROAD_SIZE / 2, CROSSROAD_SIZE / 2], [(LANE_NUMBER-1)*LANE_WIDTH, 0],
    #                 color=h_color, linewidth=light_line_width)
    #     ax.plot([CROSSROAD_SIZE / 2, CROSSROAD_SIZE / 2], [LANE_NUMBER * LANE_WIDTH, (LANE_NUMBER-1)*LANE_WIDTH],
    #                 color='green', linewidth=light_line_width)

    #     # ----------connection--------------
    #     # ax.plot([LANE_NUMBER * LANE_WIDTH, CROSSROAD_SIZE / 2], [-CROSSROAD_SIZE / 2, -LANE_NUMBER * LANE_WIDTH],
    #     #             color='black', linewidth = lane_edge_width)
    #     # ax.plot([LANE_NUMBER * LANE_WIDTH, CROSSROAD_SIZE / 2], [CROSSROAD_SIZE / 2, LANE_NUMBER * LANE_WIDTH],
    #     #             color='black', linewidth = lane_edge_width)
    #     # ax.plot([-LANE_NUMBER * LANE_WIDTH, -CROSSROAD_SIZE / 2], [-CROSSROAD_SIZE / 2, -LANE_NUMBER * LANE_WIDTH],
    #     #             color='black', linewidth = lane_edge_width)
    #     # ax.plot([-LANE_NUMBER * LANE_WIDTH, -CROSSROAD_SIZE / 2], [CROSSROAD_SIZE / 2, LANE_NUMBER * LANE_WIDTH],
    #     #             color='black', linewidth = lane_edge_width)

    #     from matplotlib.patches import Arc
    #     Radius = CROSSROAD_SIZE - 2 * LANE_NUMBER * LANE_WIDTH
    #     quarter_circle_1 = Arc((25,-25),Radius,Radius,90,0,90, linewidth = lane_edge_width)
    #     quarter_circle_2 = Arc((25,25),Radius,Radius,180,0,90, linewidth = lane_edge_width)
    #     quarter_circle_3 = Arc((-25,25),Radius,Radius,270,0,90, linewidth = lane_edge_width)
    #     quarter_circle_4 = Arc((-25,-25),Radius,Radius,360,0,90, linewidth = lane_edge_width)
    #     ax.add_patch(quarter_circle_1)
    #     ax.add_patch(quarter_circle_2)
    #     ax.add_patch(quarter_circle_3)
    #     ax.add_patch(quarter_circle_4)

    #     def is_in_plot_area(x, y, tolerance=5):
    #         if -CROSSROAD_SIZE / 2 - extension + tolerance < x < CROSSROAD_SIZE / 2 + extension - tolerance and \
    #                 -CROSSROAD_SIZE / 2 - extension + tolerance < y < CROSSROAD_SIZE / 2 + extension - tolerance:
    #             return True
    #         else:
    #             return False

    #     def draw_rotate_rec(x, y, a, l, w, color, linestyle='-'):
    #         RU_x, RU_y, _ = rotate_coordination(l / 2, w / 2, 0, -a)
    #         RD_x, RD_y, _ = rotate_coordination(l / 2, -w / 2, 0, -a)
    #         LU_x, LU_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
    #         LD_x, LD_y, _ = rotate_coordination(-l / 2, -w / 2, 0, -a)
    #         ax.plot([RU_x + x, RD_x + x], [RU_y + y, RD_y + y], color=color, linestyle=linestyle)
    #         ax.plot([RU_x + x, LU_x + x], [RU_y + y, LU_y + y], color=color, linestyle=linestyle)
    #         ax.plot([LD_x + x, RD_x + x], [LD_y + y, RD_y + y], color=color, linestyle=linestyle)
    #         ax.plot([LD_x + x, LU_x + x], [LD_y + y, LU_y + y], color=color, linestyle=linestyle)

    #     def plot_phi_line(x, y, phi, color):
    #         line_length = 5
    #         x_forw, y_forw = x + line_length * cos(phi*pi/180.),\
    #                             y + line_length * sin(phi*pi/180.)
    #         plt.plot([x, x_forw], [y, y_forw], color=color, linewidth=0.5)

    #     # plot cars
    #     for veh in self.all_vehicles:
    #         veh_x = veh['x']
    #         veh_y = veh['y']
    #         veh_phi = veh['phi']
    #         veh_l = veh['l']
    #         veh_w = veh['w']
    #         if is_in_plot_area(veh_x, veh_y):
    #             plot_phi_line(veh_x, veh_y, veh_phi, 'black')
    #             draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, 'black')

    #     # plot_interested vehs
    #     for mode, num in self.veh_mode_dict.items():
    #         for i in range(num):
    #             veh = self.interested_vehs[mode][i]
    #             veh_x = veh['x']
    #             veh_y = veh['y']
    #             veh_phi = veh['phi']
    #             veh_l = veh['l']
    #             veh_w = veh['w']
    #             task2color = {'left': 'b', 'straight': 'c', 'right': 'm'}

    #             if is_in_plot_area(veh_x, veh_y):
    #                 plot_phi_line(veh_x, veh_y, veh_phi, 'black')
    #                 task = MODE2TASK[mode]
    #                 color = task2color[task]
    #                 draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, color, linestyle=':')


    #     ego_v_x = self.ego_dynamics['v_x']
    #     ego_v_y = self.ego_dynamics['v_y']
    #     ego_r = self.ego_dynamics['r']
    #     ego_x = self.ego_dynamics['x']
    #     ego_y = self.ego_dynamics['y']
    #     ego_phi = self.ego_dynamics['phi']
    #     ego_l = self.ego_dynamics['l']
    #     ego_w = self.ego_dynamics['w']
    #     ego_alpha_f = self.ego_dynamics['alpha_f']
    #     ego_alpha_r = self.ego_dynamics['alpha_r']
    #     alpha_f_bound = self.ego_dynamics['alpha_f_bound']
    #     alpha_r_bound = self.ego_dynamics['alpha_r_bound']
    #     r_bound = self.ego_dynamics['r_bound']

    #     plot_phi_line(ego_x, ego_y, ego_phi, 'red')
    #     draw_rotate_rec(ego_x, ego_y, ego_phi, ego_l, ego_w, 'red')

    #     # plot future data
    #     tracking_info = self.obs[self.ego_info_dim:self.ego_info_dim + self.per_tracking_info_dim * (self.num_future_data+1)]
    #     future_path = tracking_info[self.per_tracking_info_dim:]
    #     for i in range(self.num_future_data):
    #         delta_x, delta_y, delta_phi = future_path[i*self.per_tracking_info_dim:
    #                                                                 (i+1)*self.per_tracking_info_dim]
    #         path_x, path_y, path_phi = ego_x+delta_x, ego_y+delta_y, ego_phi-delta_phi
    #         plt.plot(path_x, path_y, 'g.')
    #         plot_phi_line(path_x, path_y, path_phi, 'g')

    #     delta_, _, _ = tracking_info[:3]
    #     ax.plot(self.ref_path.path[0], self.ref_path.path[1], color='g')
    #     ind = self.ref_path.ref_index
    #     plt.plot(self.ref_path.path[0], self.ref_path.path[1], color='g')

    #     indexs, points = self.ref_path.find_closest_point(np.array([ego_x], np.float32), np.array([ego_y],np.float32))
    #     path_x, path_y, path_phi = points[0][0], points[1][0], points[2][0]
    #     plt.plot(path_x, path_y, 'g.')
    #     delta_x, delta_y, delta_phi = ego_x - path_x, ego_y - path_y, ego_phi - path_phi

    #     # plot real time traj
    #     # try:
    #     #     color = ['b', 'lime']
    #     #     for i, item in enumerate(real_time_traj):
    #     #         if i == path_index:
    #     #             plt.plot(item.path[0], item.path[1], color=color[i], alpha=1.0)
    #     #         else:
    #     #             plt.plot(item.path[0], item.path[1], color=color[i], alpha=0.3)
    #     #         indexs, points = item.find_closest_point(np.array([ego_x], np.float32), np.array([ego_y], np.float32))
    #     #         path_x, path_y, path_phi = points[0][0], points[1][0], points[2][0]
    #     #         plt.plot(path_x, path_y,  color=color[i])
    #     # except Exception:
    #     #     pass

    #     # for j, item_point in enumerate(self.real_path.feature_points_all):
    #     #     for k in range(len(item_point)):
    #     #         plt.scatter(item_point[k][0], item_point[k][1], c='g')

    #     # plot ego dynamics
    #     text_x, text_y_start = -110, 60
    #     ge = iter(range(0, 1000, 4))
    #     plt.text(text_x, text_y_start - next(ge), 'ego_x: {:.2f}m'.format(ego_x))
    #     plt.text(text_x, text_y_start - next(ge), 'ego_y: {:.2f}m'.format(ego_y))
    #     plt.text(text_x, text_y_start - next(ge), 'path_x: {:.2f}m'.format(path_x))
    #     plt.text(text_x, text_y_start - next(ge), 'path_y: {:.2f}m'.format(path_y))
    #     plt.text(text_x, text_y_start - next(ge), 'delta_: {:.2f}m'.format(delta_))
    #     plt.text(text_x, text_y_start - next(ge), 'delta_x: {:.2f}m'.format(delta_x))
    #     plt.text(text_x, text_y_start - next(ge), 'delta_y: {:.2f}m'.format(delta_y))
    #     plt.text(text_x, text_y_start - next(ge), r'ego_phi: ${:.2f}\degree$'.format(ego_phi))
    #     plt.text(text_x, text_y_start - next(ge), r'path_phi: ${:.2f}\degree$'.format(path_phi))
    #     plt.text(text_x, text_y_start - next(ge), r'delta_phi: ${:.2f}\degree$'.format(delta_phi))
        
    #     plt.text(text_x, text_y_start - next(ge), 'v_x: {:.2f}m/s'.format(ego_v_x))
    #     plt.text(text_x, text_y_start - next(ge), 'exp_v: {:.2f}m/s'.format(self.exp_v))
    #     plt.text(text_x, text_y_start - next(ge), 'v_y: {:.2f}m/s'.format(ego_v_y))
    #     plt.text(text_x, text_y_start - next(ge), 'yaw_rate: {:.2f}rad/s'.format(ego_r))
    #     plt.text(text_x, text_y_start - next(ge), 'yaw_rate bound: [{:.2f}, {:.2f}]'.format(-r_bound, r_bound))
        
    #     plt.text(text_x, text_y_start - next(ge), r'$\alpha_f$: {:.2f} rad'.format(ego_alpha_f))
    #     plt.text(text_x, text_y_start - next(ge), r'$\alpha_f$ bound: [{:.2f}, {:.2f}] '.format(-alpha_f_bound,
    #                                                                                                 alpha_f_bound))
    #     plt.text(text_x, text_y_start - next(ge), r'$\alpha_r$: {:.2f} rad'.format(ego_alpha_r))
    #     plt.text(text_x, text_y_start - next(ge), r'$\alpha_r$ bound: [{:.2f}, {:.2f}] '.format(-alpha_r_bound,
    #                                                                                                 alpha_r_bound))
    #     if self.action is not None:
    #         steer, a_x = self.action[0], self.action[1]
    #         plt.text(text_x, text_y_start - next(ge), r'steer: {:.2f}rad (${:.2f}\degree$)'.format(steer, steer * 180 / np.pi))
    #         plt.text(text_x, text_y_start - next(ge), 'a_x: {:.2f}m/s^2'.format(a_x))
    #     '''
    #     text_x, text_y_start = 70, 60
    #     ge = iter(range(0, 1000, 4))
        
    #     # done info
    #     plt.text(text_x, text_y_start - next(ge), 'done info: {}'.format(self.done_type))

    #     # reward info
    #     if self.reward_info is not None:
    #         for key, val in self.reward_info.items():
    #             plt.text(text_x, text_y_start - next(ge), '{}: {:.4f}'.format(key, val))
    #     '''
    #     # indicator for trajectory selection
    #     # text_x, text_y_start = -25, -65
    #     # ge = iter(range(0, 1000, 6))
    #     # if traj_return is not None:
    #     #     for i, value in enumerate(traj_return):
    #     #         if i==path_index:
    #     #             plt.text(text_x, text_y_start-next(ge), 'track_error={:.4f}, collision_risk={:.4f}'.format(value[0], value[1]), fontsize=14, color=color[i], fontstyle='italic')
    #     #         else:
    #     #             plt.text(text_x, text_y_start-next(ge), 'track_error={:.4f}, collision_risk={:.4f}'.format(value[0], value[1]), fontsize=12, color=color[i], fontstyle='italic')

    #     # save figure

    #     # manager = plt.get_current_fig_manager()
    #     # manager.window.showMaximized()
    #     plt.show()
    #     plt.pause(0.001)
    #     # name_index = str(name_index).zfill(3)
    #     # plt.savefig(f'result/f{name_index}.png')

if __name__ == '__main__':
    pass