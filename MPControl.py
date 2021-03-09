import numpy as np 
from vehicle import VehicleDynamics
from Reference import ReferencePath
from Env_utils import STEP_TIME, deal_with_phi


class ModelPredictiveControl:
    def __init__(self, obs, horizon, ref, task):  # init_x为Env 的 obs
        self.horizon = horizon
        self.obs = obs
        self.vehicle_dynamics = VehicleDynamics()
        self.task = task
        self.ref_path = ref.path
        self.ref = ref
        self.ego_info_dim = 6  #v_x, v_y, r, x, y, phi
        self.per_veh_info_dim = 4  # x, y, v_x, phi
        self.future_ref_list = None
        self.current_ref_point = None
        self.exp_v = 7.8

    def reset_init_ref(self, obs, ref_index):
        self.obs = obs
        self.ref_path = ReferencePath('left', ref_index)
 
    def reset_obs(self, obs):
        self.obs = obs

    def _update_future_ref(self):
        ego_list = self.obs[0]  # a list [v_x, v_y, r, x, y, phi]
        current_ref_point, future_ref_list = self.ref.future_ref_points(ego_list[3], ego_list[4], self.horizon)
        self.current_ref_point = current_ref_point
        self.future_ref_list = future_ref_list
        return current_ref_point, future_ref_list

    def compute_next_obs(self, obs, actions):
        ego_list = obs[0] # a list [v_x, v_y, r, x, y, phi]
        veh_recarray = obs[1]

        next_ego_list = self.ego_predict(ego_list, actions)
        next_veh_recarray = self.veh_predict(veh_recarray)

        next_obs = next_ego_list, next_veh_recarray
        return next_obs

    def ego_predict(self, ego_list, actions):
        ego_next_state, _ = self.vehicle_dynamics.prediction(ego_list, actions, STEP_TIME)
        v_x, v_y, r, x, y, phi = ego_next_state[0], ego_next_state[1], ego_next_state[2],\
                                     ego_next_state[3], ego_next_state[4], ego_next_state[5]
        ego_next_list = [v_x, v_y, r, x, y, phi]                          
        return ego_next_list

    def veh_predict(self, veh_recarray):
        def route_to_task(veh):
            if all(veh.route == ['1o', '4i']) or all(veh.route == ['2o', '1i']) :
                task = 'left'
            elif all(veh.route == ['4o', '1i']) or all(veh.route == ['3o', '4i']):
                task = 'right'
            else:
                task = 'straight'
            return task

        def predict_for_a_task(veh, task):
            veh_x, veh_y, veh_v, veh_phi = veh.x, veh.y, veh.v, veh.phi
            veh_phi_rad = veh_phi * np.pi / 180.

            zeros = np.zeros_like(veh_x)

            veh_x_delta = veh_v * STEP_TIME * np.cos(veh_phi_rad)
            veh_y_delta = veh_v * STEP_TIME * np.sin(veh_phi_rad)

            if task  == 'left':
                veh_phi_rad_delta = np.where(-25 < veh_x < 25, (veh_v / 26.875) * STEP_TIME, zeros)
            elif task == 'right':
                veh_phi_rad_delta = np.where(-25 < veh_y < 25, -(veh_v / 19.375) * STEP_TIME, zeros)
            else:
                veh_phi_rad_delta = zeros
            next_veh_x, next_veh_y, next_veh_v, next_veh_phi_rad = \
                veh_x + veh_x_delta, veh_y + veh_y_delta, veh_v, veh_phi_rad + veh_phi_rad_delta
            next_veh_phi_rad = np.where(next_veh_phi_rad > np.pi, next_veh_phi_rad - 2 * np.pi, next_veh_phi_rad)
            next_veh_phi_rad = np.where(next_veh_phi_rad <= -np.pi, next_veh_phi_rad + 2 * np.pi, next_veh_phi_rad)
            next_veh_phi = next_veh_phi_rad * 180 / np.pi
            return next_veh_x, next_veh_y, next_veh_v, next_veh_phi
        if veh_recarray is not None:
            veh_copy = veh_recarray.copy()
            for veh in veh_copy:
                veh_task = route_to_task(veh)
                next_veh_x, next_veh_y, next_veh_v, next_veh_phi = predict_for_a_task(veh, veh_task)
                veh.x, veh.y, veh.v, veh.phi = next_veh_x, next_veh_y, next_veh_v, next_veh_phi
        return veh_copy
                
    def plant_model(self, obs, u):
        obs = self.compute_next_obs(obs, u)
        return obs

    def compute_loss(self, obs, actions, i):
        ego_list = obs[0] # a list [v_x, v_y, r, x, y, phi]
        veh_recarray = obs[1]


        steer, a_x = actions[0], actions[1]
        # rewards related to action
        punish_steer = np.square(steer)
        punish_a_x = np.square(a_x)

        # rewards related to ego stability  r
        punish_yaw_rate = np.square(ego_list[2])
        # rewards related to tracking error

        x_ref, y_ref, phi_ref = self.future_ref_list[i]

        # print('x, y, phi',ego_list[3:])
        # print('x_ref, y_ref, phi_ref',x_ref, y_ref, phi_ref)
        devi_x = (ego_list[3] - x_ref)**2
        devi_y = (ego_list[4] - y_ref)**2
        devi_phi = np.square(deal_with_phi(ego_list[5] - phi_ref))
        devi_v = np.square(ego_list[0] - self.exp_v)
        #x:{devi_x} y:{devi_y} phi:{devi_phi} v:{devi_v} yaw_rate:{punish_yaw_rate} steer:{steer}  a_x{a_x}
        loss = 20* devi_x + 20* devi_y + 0.01 * devi_phi + 30* devi_v \
                    #+ 120 * punish_yaw_rate + 800 * punish_steer + 50 * punish_a_x
        return loss

    def cost_function(self, u):
        u = u.reshape(self.horizon, 2) # u.shape (10,2)
        loss = 0.
        obs = self.obs
        for i in range(self.horizon):
            #u_i = u[i] * np.array([0.35, 5.]) - np.array([0., 2])   # u[0].shape (2,)
            u_i = u[i]
            obs = self.plant_model(obs, u_i)
            loss += self.compute_loss(obs, u_i, i)
        return loss

    def constraint_function(self, u):
        u = u.reshape(self.horizon, 2) # u.shape (10,2)
        obs = self.obs
        dist_array = np.array([])
        for i in range(self.horizon):
            #u_i = u[i] * np.array([0.35, 5.]) - np.array([0., 2])   # u[0].shape (2,)
            u_i = u[i] # u[i].shape (2,)
            obs = self.plant_model(obs, u_i)
            ego_list = obs[0] # a list [v_x, v_y, r, x, y, phi]
            veh_recarray = obs[1]
            if veh_recarray is not None:
                dist_array = np.concatenate((dist_array, np.sqrt((veh_recarray.x - ego_list[3])**2 + (veh_recarray.y - ego_list[4])**2)), axis=0)
        safe_distance = 5 * np.ones_like(dist_array)
        constraint_array = dist_array - safe_distance
        return constraint_array# ndarray



            # # punish_cond = np.logical_and(np.logical_and(dists * cos_values > -5., dists * np.abs(sin_values) < (L + W) / 2),
            # #                           dists < 10.)
            # punish_cond = np.logical_and(np.logical_and(cos_values < 0., dists * np.abs(sin_values) < (L + W) / 2),
            #                           dists < 10.)
            # veh2veh -= np.where(punish_cond, 10. - dists, np.zeros_like(veh_infos[:, 0]))

        
