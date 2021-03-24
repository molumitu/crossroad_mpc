import sys
sys.path.insert(0, R'C:\Users\zgj_t\Desktop\crossroad_mpc\matlab_to_mpc\mpc\x64\Release')
import numpy as np 
from Env_utils import horizon 
from mpc import mpc_cost_function, mpc_constraints, mpc_alpha_constraints, state_trans_LPF, mpc_cost_function_jac, \
    mpc_constraints_jac, mpc_alpha_constraints_jac


def mpc_wrapper(u, ego_list, vehicles_xy_array, future_ref_array, Q, R, P):
    for arr in (u, ego_list, vehicles_xy_array, future_ref_array, Q, R, P):
        assert arr.flags.c_contiguous
    return mpc_cost_function_jac(u, ego_list, vehicles_xy_array, future_ref_array, Q, R, P)


def mpc_constraints_wrapper(u, ego_list, vehicles_xy_array, safe_dist):
    grad = mpc_constraints_jac(u, ego_list, vehicles_xy_array, safe_dist)[1]
    grad = grad.reshape(-1, horizon * 2)
    return grad

def mpc_alpha_constraints_wrapper(u, ego_list):
    grad = mpc_alpha_constraints_jac(u, ego_list)[1]
    grad = grad.reshape(-1, horizon * 2)
    return grad

def red_mpc_wrapper(u, ego_list, vehicles_xy_array, future_ref_array, Q, R, P):
    u = np.stack((np.zeros(horizon), np.array(u)), axis = 1)
    u = u.flatten()
    for arr in (u, ego_list, vehicles_xy_array, future_ref_array, Q, R, P):
        assert arr.flags.c_contiguous
    f, grad = mpc_cost_function_jac(u, ego_list, vehicles_xy_array, future_ref_array, Q, R, P)
    grad = grad[slice(1,horizon*2,2)]
    return f, grad


# def red_mpc_constraints_wrapper(u, ego_list, vehicles_xy_array, safe_dist):
#     u = np.stack((np.zeros(horizon), np.array(u)), axis = 1)
#     u = u.flatten()
#     grad = mpc_constraints_jac(u, ego_list, vehicles_xy_array, safe_dist)[1]
#     grad = grad.reshape(-1, horizon * 2)
#     grad = grad[:,slice(1,horizon*2,2)]
#     return grad



def red_mpc_constraints_jac_wrapper(u, ego_list, vehicles_xy_array, safe_dist):
    u = np.stack((np.zeros(horizon), np.array(u)), axis = 1)
    u = u.flatten()
    grad = mpc_constraints_jac(u, ego_list, vehicles_xy_array, safe_dist)[1]
    grad = grad.reshape(-1, horizon * 2)
    grad = grad[:,slice(1,horizon*2,2)]
    return grad

def red_mpc_constraints_wrapper(u, ego_list, vehicles_xy_array, safe_dist):
    u = np.stack((np.zeros(horizon), np.array(u)), axis = 1)
    u = u.flatten()
    g = mpc_constraints(u, ego_list, vehicles_xy_array, safe_dist)
    return g

def red_mpc_alpha_constraints_jac_wrapper(u, ego_list):
    u = np.stack((np.zeros(horizon), np.array(u)), axis = 1)
    u = u.flatten()
    grad = mpc_alpha_constraints_jac(u, ego_list)[1]
    grad = grad.reshape(-1, horizon * 2)
    grad = grad[:,slice(1,horizon*2,2)]
    return grad

def red_mpc_alpha_constraints_wrapper(u, ego_list):
    u = np.stack((np.zeros(horizon), np.array(u)), axis = 1)
    u = u.flatten()
    g = mpc_alpha_constraints(u, ego_list)
    return g