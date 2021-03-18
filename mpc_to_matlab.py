import numpy as np
from state_trans_LPF_py import state_trans_LPF_py

#####本来应该和matlab生成的c++代码保持一致，现在可能版本比较落后

def deal_with_phi(phi):
    return np.mod(phi+180,2*180)-180.


def mpc_cost_function(u, ego_list,future_ref_list, horizon, STEP_TIME, Q, R):
    exp_v = 8

    def compute_loss(ego_list, action, future_ref_list, i):

        steer, a_x = action
        punish_steer = (steer)**2
        punish_a_x = (a_x)**2
        punish_yaw_rate = (ego_list[2])**2

        x_ref, y_ref, phi_ref = future_ref_list[i]

        devi_x = (ego_list[3] - x_ref)**2
        devi_y = (ego_list[4] - y_ref)**2
        devi_phi = (deal_with_phi(ego_list[5] - phi_ref))**2
        devi_v = (ego_list[0] - exp_v)**2
        loss = Q[0]* devi_x + Q[1]* devi_y + Q[2] * devi_phi + Q[3]* devi_v + Q[4]*punish_yaw_rate + R[0] * punish_steer + R[1] * punish_a_x
        return loss


    u = u.reshape(horizon, 2) # u.shape (10,2)
    loss = 0.
    ego = ego_list
    for i in range(horizon):
        u_i = u[i]
        ego, next_params = state_trans_LPF_py(ego, u_i, STEP_TIME)
        loss += compute_loss(ego, u_i, future_ref_list, i)
    return loss


# #vehs_array 是一个n*10*4的array
# def mpc_constraints(u, ego_list, vehicles_array, n, horizon, STEP_TIME, safe_dist):
    
#     u = u.reshape(horizon, 2) # u.shape (10,2)
#     ego = ego_list
#     dist_array = np.zeros(n*horizon)
#     for i in range(horizon):
#         u_i = u[i] # u[i].shape (2,)
#         ego, next_params = state_trans_LPF_py(ego, u_i, STEP_TIME)
#         veh_current_array = vehicles_array[:,i,:]
#         dist_array[i*n:(i+1)*n] = np.sqrt((veh_current_array[:,0] - ego[3])**2 + (veh_current_array[:,1] - ego[4])**2)
#     safe_distance = safe_dist * np.ones_like(dist_array)
#     constraint_array = dist_array - safe_distance
#     return constraint_array# ndarray