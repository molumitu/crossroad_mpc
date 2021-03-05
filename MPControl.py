import numpy as np 
from vehicle import VehicleDynamics
from Reference import ReferencePath
from Env_utils import STEP_TIME


class ModelPredictiveControl:
    def __init__(self, init_x, horizon, task):
        self.horizon = horizon
        self.init_x = init_x
        # x is obs_vector
        self.vehicle_dynamics = VehicleDynamics()
        self.task = task
        self.ref_path = None
        self.ego_info_dim = 6  #v_x, v_y, r, x, y, phi
        self.per_veh_info_dim = 4  # x, y, phi, v_x

    def reset_init_ref(self, init_x, ref_index):
        self.init_x = init_x
        self.ref_path = ReferencePath('left', ref_index)
 
    def reset_init_state(self, init_x):
        self.init_x = init_x

    def compute_next_obses(self, obses, actions):
        ego_infos, tracking_infos, veh_infos = obses[:, :self.ego_info_dim], \
                                               obses[:, self.ego_info_dim:self.ego_info_dim + 3], \
                                               obses[:, self.ego_info_dim + 3:]

        next_ego_infos = self.ego_predict(ego_infos, actions)

        next_tracking_infos = self.ref_path.tracking_error_vector(next_ego_infos[:, 3],
                                                                  next_ego_infos[:, 4],
                                                                  next_ego_infos[:, 5],
                                                                  next_ego_infos[:, 0],
                                                                  0)
        next_veh_infos = self.veh_predict(veh_infos)
        next_obses = np.concatenate([next_ego_infos, next_tracking_infos, next_veh_infos], 1)
        return next_obses

    def ego_predict(self, ego_infos, actions):
        ego_next_infos, _ = self.vehicle_dynamics.prediction(ego_infos[:, :6], actions, STEP_TIME)
        return ego_next_infos

    def veh_predict(self, veh_infos):
        if self.task == 'left':
            veh_mode_list = ['dl'] * 0 + ['du'] * 0 + ['ud'] * 2 + ['ul'] * 0
        elif self.task == 'straight':
            veh_mode_list = ['dl'] * 2 + ['du'] * 2 + ['ud'] * 2 + ['ru'] * 3 + ['ur'] * 3
        else:
            assert self.task == 'right'
            veh_mode_list = ['dr'] * 2 + ['ur'] * 3 + ['lr'] * 3

        predictions_to_be_concat = []

        for vehs_index in range(len(veh_mode_list)):
            predictions_to_be_concat.append(self.predict_for_a_mode(
                veh_infos[:, vehs_index * self.per_veh_info_dim:(vehs_index + 1) * self.per_veh_info_dim],
                veh_mode_list[vehs_index]))
        return np.concatenate(predictions_to_be_concat, 1)

    def predict_for_a_mode(self, vehs, mode):
        veh_xs, veh_ys, veh_vs, veh_phis = vehs[:, 0], vehs[:, 1], vehs[:, 2], vehs[:, 3]
        veh_phis_rad = veh_phis * np.pi / 180.

        zeros = np.zeros_like(veh_xs)

        veh_xs_delta = veh_vs * STEP_TIME * np.cos(veh_phis_rad)
        veh_ys_delta = veh_vs * STEP_TIME * np.sin(veh_phis_rad)

        if mode in ['dl', 'rd', 'ur', 'lu']:
            veh_phis_rad_delta = np.where(-25 < veh_xs < 25, (veh_vs / 26.875) * STEP_TIME, zeros)
        elif mode in ['dr', 'ru', 'ul', 'ld']:
            veh_phis_rad_delta = np.where(-25 < veh_ys < 25, -(veh_vs / 19.375) * STEP_TIME, zeros)
        else:
            veh_phis_rad_delta = zeros
        next_veh_xs, next_veh_ys, next_veh_vs, next_veh_phis_rad = \
            veh_xs + veh_xs_delta, veh_ys + veh_ys_delta, veh_vs, veh_phis_rad + veh_phis_rad_delta
        next_veh_phis_rad = np.where(next_veh_phis_rad > np.pi, next_veh_phis_rad - 2 * np.pi, next_veh_phis_rad)
        next_veh_phis_rad = np.where(next_veh_phis_rad <= -np.pi, next_veh_phis_rad + 2 * np.pi, next_veh_phis_rad)
        next_veh_phis = next_veh_phis_rad * 180 / np.pi
        return np.stack([next_veh_xs, next_veh_ys, next_veh_vs, next_veh_phis], 1)

    def plant_model(self, u, x):
        x_copy = x.copy()
        x_copy = self.compute_next_obses(x_copy[np.newaxis, :], u[np.newaxis, :])[0]
        return x_copy

    def compute_loss(self, obses, actions):
        ego_infos, tracking_infos, veh_infos = obses[:, :self.ego_info_dim], \
                                               obses[:, self.ego_info_dim:self.ego_info_dim + 3], \
                                               obses[:, self.ego_info_dim + 3:]
        steers, a_xs = actions[:, 0], actions[:, 1]
        # rewards related to action
        punish_steer = np.square(steers)
        punish_a_x = np.square(a_xs)

        # rewards related to ego stability  r
        punish_yaw_rate = np.square(ego_infos[:, 2])
        # rewards related to tracking error
        devi_y = np.square(tracking_infos[:, 0])
        devi_phi = np.square(tracking_infos[:, 1] * np.pi / 180.)
        devi_v = np.square(tracking_infos[:, 2])

        loss = 0.4*5 *2* devi_y + 0.1*10*5 * devi_phi  + 0.002 * punish_yaw_rate + \
                  0.001 * punish_steer + 0.005 * punish_a_x
        return loss

    def cost_function(self, u):
        u = u.reshape(self.horizon, 2) # u.shape (10,2)
        loss = 0.
        x = self.init_x.copy()
        for i in range(self.horizon):
            #u_i = u[i] * np.array([0.35, 5.]) - np.array([0., 2])   # u[0].shape (2,)
            u_i = u[i]
            loss += self.compute_loss(x[np.newaxis, :], u_i[np.newaxis, :])  #x[np.newaxis, :] 变为行向量
            x = self.plant_model(u_i, x)

        return loss

    def constraint_function(self, u):
        u = u.reshape(self.horizon, 2) # u.shape (10,2)
        x = self.init_x.copy()
        dist_list = []
        for i in range(self.horizon):
            #u_i = u[i] * np.array([0.35, 5.]) - np.array([0., 2])   # u[0].shape (2,)
            u_i = u[i] # u[i].shape (2,)
            obses = x[np.newaxis, :]
            ego_infos, tracking_infos, veh_infos = obses[:, :self.ego_info_dim], \
                                        obses[:, self.ego_info_dim:self.ego_info_dim + 3], \
                                        obses[:, self.ego_info_dim + 3:]
            L, W = 4.8, 2.
            veh2veh = np.zeros_like(veh_infos[:, 0])
            for veh_index in range(int(np.shape(veh_infos)[1] / self.per_veh_info_dim)):
                vehs = veh_infos[:, veh_index * self.per_veh_info_dim:(veh_index + 1) * self.per_veh_info_dim]
                #rela_phis_rad = np.arctan2(vehs[:, 1] - ego_infos[:, 4], vehs[:, 0] - ego_infos[:, 3])
                #ego_phis_rad = ego_infos[:, 5] * np.pi / 180.
                #cos_values, sin_values = np.cos(rela_phis_rad - ego_phis_rad), np.sin(rela_phis_rad - ego_phis_rad)
                dist = np.sqrt(np.square(vehs[:, 0] - ego_infos[:, 3]) + np.square(vehs[:, 1] - ego_infos[:, 4]))
                dist_list.append(float(dist))
            x = self.plant_model(u_i, x)
        dist_array = np.array(dist_list)
        safe_distance = 5 * np.ones_like(dist_array)
        constraint_list = dist_array - safe_distance
        return constraint_list# ndarray



            # # punish_cond = np.logical_and(np.logical_and(dists * cos_values > -5., dists * np.abs(sin_values) < (L + W) / 2),
            # #                           dists < 10.)
            # punish_cond = np.logical_and(np.logical_and(cos_values < 0., dists * np.abs(sin_values) < (L + W) / 2),
            #                           dists < 10.)
            # veh2veh -= np.where(punish_cond, 10. - dists, np.zeros_like(veh_infos[:, 0]))

        
