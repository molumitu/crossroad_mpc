import numpy as np
def state_trans_LPF_py(ego_list, action, STEP_TIME):

    v_x, v_y, r, x, y, phi, steer, a_x = ego_list
    phi = phi * np.pi / 180
    
    steer_input, a_x_input = action

    C_f=-96995.9
    C_r=-85943.6
    a=1.4
    b=1.6
    h = 0.35
    mass=1412.
    I_z=1536.7
    miu = 0.9
    g = 9.81
    
    T = 0.4
    freq = int(STEP_TIME/ 0.01)
    tau = STEP_TIME / freq
    eps = 1e-8
    for i in range(freq):
        signed_eps = np.sign(v_x) * eps
        

        
        F_zf = b * mass * g / (a+b) - mass * a_x * h / (a+b)
        F_zr = a * mass * g / (a+b) + mass * a_x * h / (a+b)
        if a_x >= 0:
            F_xf = 0
            F_xr = mass * a_x
        else:
            F_xf = mass * a_x / 2
            F_xr = mass * a_x / 2
        max_F_yf = np.sqrt((F_zf * miu)**2 - (F_xf)**2)
        max_F_yr = np.sqrt((F_zr * miu)**2 - (F_xr)**2)
        alpha_f = np.arctan((v_y + a * r) / (v_x + signed_eps)) * np.sign(v_x) - steer * np.tanh(4 * v_x)
        alpha_r = (np.arctan((v_y - b * r) / (v_x + signed_eps))) * np.sign(v_x)
        F_yf = np.clip(alpha_f * C_f, -max_F_yf, max_F_yf)
        F_yr = np.clip(alpha_r * C_r, -max_F_yr, max_F_yr)
        #(mass * v_y * v_x + tau * (a * C_f - b * C_r) * r - tau * C_f * steer * v_x - tau * mass * v_x**2 * r) / (mass * v_x - tau * (C_f + C_r)),...
        #(-I_z * r * v_x - tau * (a * C_f - b * C_r) * v_y + tau * a * C_f * steer * v_x) / (tau * (a**2 * C_f + b**2 * C_r) - I_z * v_x),...
        next_state = [v_x + tau * (a_x + v_y * r - F_yf * np.sin(steer) / mass) ,\
                        v_y + tau * (-v_x * r +(F_yr + F_yf * np.cos(steer))/mass) ,\
                        r + tau * (F_yf * a * np.cos(steer) - F_yr * b)/I_z ,\
                        x + tau * (v_x * np.cos(phi) - v_y * np.sin(phi))  ,\
                        y + tau * (v_x * np.sin(phi) + v_y * np.cos(phi))  ,\
                        (phi + tau * r) * 180 / np.pi,\
                        (1 - tau/T) * steer + tau/T * steer_input,\
                        (1 - tau/T) * a_x + tau/T * a_x_input
                        ]
        v_x, v_y, r, x, y, phi, steer, a_x = next_state
        phi = phi * np.pi /180
    # next_ego_list = np.concatenate((next_state, np.array([steer]) , np.array([a_x]))) 
    next_ego_list = next_state
    next_ego_param = [alpha_f * C_f +  max_F_yf, max_F_yf - alpha_f * C_f, alpha_r * C_r +  max_F_yr, max_F_yr - alpha_r * C_r] 
    return next_ego_list, next_ego_param



