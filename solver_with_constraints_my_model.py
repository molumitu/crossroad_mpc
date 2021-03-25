import numpy as np
import casadi as ca

solver_cache = dict()

####预先创建好了5个solver，分别可以处理0~5辆车

def create_solver_with_cons(Horizon, STEP_TIME, n_vehicles):
    key = Horizon, STEP_TIME, n_vehicles
    if key in solver_cache:
        return solver_cache[key]
    solver = _create_solver_with_cons(*key)
    solver_cache[key] = solver
    return solver

def _create_solver_with_cons(Horizon, STEP_TIME, n_vehicles):
    n_states = 8
    n_controls = 2
    safety_dist = 4.5
    n_vehicles = n_vehicles
    C_f = -96995.9
    C_r = -85943.6
    a = 1.4
    b = 1.6
    h = 0.35
    mass = 1412.
    I_z = 1536.7
    miu = 0.9
    gravity = 9.81
    T_LPF = 0.4
    Q1 = 10.
    Q2 = 10.
    Q3 = 0.
    R1 = 0.5
    R2 = 0.1

    obj = 0
    eps = 1e-8
    H = Horizon
    N = H

    U = ca.SX.sym('U', n_controls, Horizon)
    P = ca.SX.sym('P', 1, n_states + Horizon*3 + n_vehicles * 2 * Horizon)  # 初始状态
    Ref_x = ca.SX.sym('Ref_x', 1, Horizon)
    Ref_y = ca.SX.sym('Ref_y', 1, Horizon)
    Ref_phi = ca.SX.sym('Ref_phi', 1, Horizon)
    Ref_x = P[8:8+Horizon]
    Ref_y = P[8+Horizon*1:8+Horizon*2]
    Ref_phi = P[8+Horizon*2:8+Horizon*3]
    Vehs_x = P[8+Horizon*3: 8+Horizon*3 + n_vehicles*Horizon]
    Vehs_y = P[8+Horizon*3 + n_vehicles *Horizon: 8+Horizon*3 + n_vehicles*2*Horizon]



    G = ca.SX.sym('G', H, 1 + 4 + n_vehicles)


    v_x_c, v_y_c, r_c, x_c, y_c, phi_c, steer_c, a_x_c = P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7]
    phi_c = (phi_c) * ca.pi / 180.
    for i in range(Horizon):
        U_c = U[:, i]

        steer_input_c, a_x_input_c = U_c[0], U_c[1]


        freq = int(STEP_TIME / 0.01)
        tau = STEP_TIME / freq

        for j in range(freq):
            steer_c = (1 - tau/T_LPF) * steer_c + tau/T_LPF * steer_input_c
            a_x_c = (1 - tau/(T_LPF/2)) * a_x_c + tau/(T_LPF/2) * a_x_input_c
            signed_eps_c = ca.sign(v_x_c) * eps

            F_zf = b * mass * gravity/(a+b) - mass * a_x_c * h / (a+b)
            F_zr = a * mass * gravity/(a+b) + mass * a_x_c * h / (a+b)
            F_xf = ca.if_else(a_x_c >= 0, 0, mass * a_x_c / 2)
            F_xr = ca.if_else(a_x_c >= 0, mass * a_x_c, mass * a_x_c / 2)
            max_F_yf = ca.sqrt((F_zf * miu)**2 - (F_xf)**2)
            max_F_yr = ca.sqrt((F_zr * miu)**2 - (F_xr)**2)

            alpha_f = ca.atan((v_y_c + a * r_c) / (v_x_c + signed_eps_c)) * \
                ca.sign(v_x_c) - steer_c * ca.tanh(4 * v_x_c)
            alpha_r = ca.atan((v_y_c - b * r_c) /
                            (v_x_c + signed_eps_c)) * ca.sign(v_x_c)
            F_yf = ca.fmin(ca.fmax(alpha_f * C_f, -max_F_yf), max_F_yf)
            F_yr = ca.fmin(ca.fmax(alpha_r * C_r, -max_F_yr), max_F_yr)

            v_x_next = v_x_c + tau * \
                (a_x_c + v_y_c * r_c - F_yf * ca.sin(steer_c) / mass)
            v_y_next = v_y_c + tau * \
                (-v_x_c * r_c + (F_yr + F_yf * ca.cos(steer_c)) / mass)
            r_next = r_c + tau * (F_yf * a * ca.cos(steer_c) - F_yr * b) / I_z
            x_next = x_c + tau * (v_x_c * ca.cos(phi_c) - v_y_c * ca.sin(phi_c))
            y_next = y_c + tau * (v_x_c * ca.sin(phi_c) + v_y_c * ca.cos(phi_c))
            phi_next = (phi_c + tau * r_c)
            steer_next = steer_c
            a_x_next = a_x_c

            v_x_c = v_x_next
            v_y_c = v_y_next
            r_c = r_next
            x_c = x_next
            y_c = y_next
            phi_c = phi_next
            steer_c = steer_next
            a_x_c = a_x_next
        G[i, 0] = v_x_c
        G[i, 1] = alpha_f * C_f + max_F_yf
        G[i, 2] = max_F_yf - alpha_f * C_f
        G[i, 3] = alpha_r * C_r + max_F_yr
        G[i, 4] = max_F_yr - alpha_r * C_r
        for k in range(n_vehicles):
            G[i, k+5] = ca.sqrt((Vehs_x[k] - x_c)**2 +(Vehs_y[k] - y_c)**2) - safety_dist

        x_ref = Ref_x[i]
        y_ref = Ref_y[i]
        phi_ref = Ref_phi[i] / 180 * ca.pi  # 在本for循环内均用弧度制
        delta_phi = phi_next - phi_ref
        delta_phi = ca.mod(delta_phi + 2*ca.pi, 2*ca.pi) - ca.pi
        obj = obj + Q1*(x_next - x_ref)**2 + Q2*(y_next - y_ref)**2 + Q3 * \
            (delta_phi / ca.pi * 180)**2 + R1 * steer_next**2 + R2 * a_x_next**2


    nlp_prob = {'f': obj, 'x': ca.reshape(U, -1, 1), 'g': ca.reshape(G, -1, 1), 'p': P}
    opts_setting = {'ipopt.max_iter': 100, 'ipopt.print_level': 0, 'print_time': 0,\
                # 'ipopt.acceptable_tol': 1e-4, \
                'ipopt.acceptable_obj_change_tol': 1e-4}
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)
    return solver

for n_vehicles in range(6):
    create_solver_with_cons(20, 0.1, n_vehicles=n_vehicles)
