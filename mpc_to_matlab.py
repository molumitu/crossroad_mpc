import numpy as np


#####本来应该和matlab生成的c++代码保持一致，现在可能版本比较落后

def state_trans(ego_list, action, STEP_TIME): 
    v_x, v_y, r, x, y, phi = ego_list[0], ego_list[1], ego_list[2], ego_list[3], ego_list[4], ego_list[5]
    phi = deal_with_phi(phi)
    phi = phi * np.pi / 180.
    steer, a_x = action[0], action[1]

    C_f=-149995.9 # so the K is equal to 0
    C_r=-85943.6  # rear wheel cornering stiffness [N/rad]
    a=1.06  # distance from CG to front axle [m]
    b=1.85  # distance from CG to rear axle [m]
    mass=1412.  # mass [kg]
    I_z=1536.7  # Polar moment of inertia at CG [kg*m^2]
    miu=1.0  # tire-road friction coefficient
    g=9.81 # acceleration of gravity [m/s^2]

    F_zf, F_zr = b * mass * g / (a + b),  a * mass * g / (a + b)
    F_xf = np.where(a_x < 0, mass * a_x / 2, np.zeros_like(a_x))
    F_xr = np.where(a_x < 0, mass * a_x / 2, mass * a_x)
    miu_f = np.sqrt(np.square(miu * F_zf) - np.square(F_xf)) / F_zf
    miu_r = np.sqrt(np.square(miu * F_zr) - np.square(F_xr)) / F_zr

    if STEP_TIME == 0.1:
        _freq = 10
    elif STEP_TIME == 0.01:
        _freq = 1
    tau = STEP_TIME / _freq
    for _ in range(_freq):
        alpha_f = np.arctan((v_y + a * r) / (v_x+1e-8)) - steer
        alpha_r = np.arctan((v_y - b * r) / (v_x+1e-8))
        F_yf = alpha_f * C_f
        F_yr = alpha_r * C_r
        next_state = [v_x + tau * (a_x + v_y * r - alpha_f * C_f * np.sin(steer) / mass),
                        v_y + tau * (-v_x * r +(F_yr + F_yf * np.cos(steer))/mass),
                        r + tau * (F_yf * a * np.cos(steer) - F_yr * b)/I_z,
                        x + tau * (v_x * np.cos(phi) - v_y * np.sin(phi)),
                        y + tau * (v_x * np.sin(phi) + v_y * np.cos(phi)),
                        deal_with_phi((phi + tau * r) * 180 / np.pi)]
        v_x, v_y, r, x, y, phi = next_state 
        phi = phi * np.pi /180

    return next_state, [alpha_f, alpha_r, miu_f, miu_r]

def deal_with_phi(phi):
    return np.mod(phi+180,2*180)-180.


def mpc_cost_function(u, ego_list,future_ref_list, horizon, STEP_TIME, Q, R):
    exp_v = 8

    def compute_loss(ego_list, action, future_ref_list, i):

        steer, a_x = action[0], action[1]
        punish_steer = np.square(steer)
        punish_a_x = np.square(a_x)
        punish_yaw_rate = np.square(ego_list[2])

        x_ref, y_ref, phi_ref = future_ref_list[i]

        devi_x = (ego_list[3] - x_ref)**2
        devi_y = (ego_list[4] - y_ref)**2
        devi_phi = np.square(deal_with_phi(ego_list[5] - phi_ref))
        devi_v = np.square(ego_list[0] - exp_v)
        loss = Q[0]* devi_x + Q[1]* devi_y + Q[2] * devi_phi + Q[3]* devi_v + Q[4]*punish_yaw_rate + R[0] * punish_steer + R[1] * punish_a_x\
                    #+ 120 * punish_yaw_rate 
        return loss


    u = u.reshape(horizon, 2) # u.shape (10,2)
    loss = 0.
    ego = ego_list
    for i in range(horizon):
        u_i = u[i]
        ego, next_params = state_trans(ego, u_i, STEP_TIME)
        loss += compute_loss(ego, u_i, future_ref_list, i)
    return loss


#vehs_array 是一个n*10*4的array
def mpc_constraints(u, ego_list, vehicles_array, n, horizon, STEP_TIME, safe_dist):
    
    u = u.reshape(horizon, 2) # u.shape (10,2)
    ego = ego_list
    dist_array = np.zeros(n*horizon)
    for i in range(horizon):
        u_i = u[i] # u[i].shape (2,)
        ego, next_params = state_trans(ego, u_i, STEP_TIME)
        veh_current_array = vehicles_array[:,i,:]
        dist_array[i*n:(i+1)*n] = np.sqrt((veh_current_array[:,0] - ego[3])**2 + (veh_current_array[:,1] - ego[4])**2)
    safe_distance = safe_dist * np.ones_like(dist_array)
    constraint_array = dist_array - safe_distance
    return constraint_array# ndarray