import sys
sys.path.insert(0, R"C:\Users\zgj_t\Desktop\crossroad_mpc")
from mpc_to_matlab import *
import mpc_cpp

def test_mpc():
    horizon = 20
    STEP_TIME = 0.1
    # Q = [20., 20., 0.01, 30., 0]
    Q = [0., 0., 0.01, 0., 0]
    R = [0., 0.]
    ego_list = [8, 0.3, 0.1, 0, 0, 90]
    np.random.seed(7355608)
    u = np.random.uniform(low = [-0.2, -1], high = [0.2, 3], size = [horizon,2])
    # print('u', u)
    future_ref_list = np.random.normal(size = (horizon,3))
    # print('future_ref_list', future_ref_list)
    result = mpc_cost_function(u.flatten(), ego_list,future_ref_list, horizon, STEP_TIME, Q, R)
    # print('result', result)
    n = 3
    safe_dist = 5
    vehicles_array = np.random.normal(size = (n,horizon,4))
    # print('vehicles_array', vehicles_array)
    result_constraint = mpc_constraints(u, ego_list, vehicles_array, n, horizon, STEP_TIME, safe_dist)
    # print('result_constraint', result_constraint)

    vehicles_array_cpp = vehicles_array[:,:,:2].copy()
    result_cpp = mpc_cpp.mpc_cost_function(u, ego_list,future_ref_list, Q, R) # horizon, STEP_TIME, 
    result_constraint_cpp = mpc_cpp.mpc_constraints(u, ego_list, vehicles_array_cpp, safe_dist).reshape(n+1,horizon).T.flatten() # horizon, STEP_TIME, 
    # assert abs(result - result_cpp) < 1e-8
    # assert np.std(result_constraint - result_constraint_cpp) < 1e-8
    # print('result_cpp', result_cpp)
    # print('result_constraint_cpp')
    # print(result_constraint_cpp)
    # print(result_constraint)
     
# test_mpc()