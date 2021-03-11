import sys
sys.path.insert(0, R"C:\Users\zgj_t\Desktop\crossroad_mpc")
from mpc_to_matlab import *
import mpc_cpp
import pytest

@pytest.mark.benchmark(group="mpc_cost")
def test_mpc_cost_python(benchmark):
    horizon = 20
    STEP_TIME = 0.1
    Q = np.array([0., 0., 0.01, 0., 0])
    R = np.array([0., 0.])
    ego_list = np.array([8, 0.3, 0.1, 0, 0, 90])
    np.random.seed(7355608)
    u = np.random.uniform(low = [-0.2, -1], high = [0.2, 3], size = [horizon,2])
    future_ref_list = np.random.normal(size = (horizon,3))  
    benchmark(mpc_cost_function, u.flatten(), ego_list,future_ref_list, horizon, STEP_TIME, Q, R)

@pytest.mark.benchmark(group="mpc_cost")
def test_mpc_cost_cpp(benchmark):
    horizon = 20
    STEP_TIME = 0.1
    Q = np.array([0., 0., 0.01, 0., 0])
    R = np.array([0., 0.])
    ego_list = np.array([8, 0.3, 0.1, 0, 0, 90])
    np.random.seed(7355608)
    u = np.random.uniform(low = [-0.2, -1], high = [0.2, 3], size = [horizon,2])
    future_ref_list = np.random.normal(size = (horizon,3))

    
    benchmark(mpc_cpp.mpc_cost_function, u, ego_list,future_ref_list, Q, R)

@pytest.mark.benchmark(group="mpc_constraints")
def test_mpc_constraints_python(benchmark):
    horizon = 20
    STEP_TIME = 0.1
    Q = np.array([0., 0., 0.01, 0., 0])
    R = np.array([0., 0.])
    ego_list = np.array([8, 0.3, 0.1, 0, 0, 90])
    np.random.seed(7355608)
    u = np.random.uniform(low = [-0.2, -1], high = [0.2, 3], size = [horizon,2])
    future_ref_list = np.random.normal(size = (horizon,3))
    n = 6
    vehicles_array = np.random.normal(size = (n,horizon,4))
    safe_dist = 5
    
    benchmark(mpc_constraints, u, ego_list, vehicles_array, n, horizon, STEP_TIME, safe_dist)

@pytest.mark.benchmark(group="mpc_constraints")
def test_mpc_constraints_cpp(benchmark):
    horizon = 20
    STEP_TIME = 0.1
    Q = np.array([0., 0., 0.01, 0., 0])
    R = np.array([0., 0.])
    ego_list = np.array([8, 0.3, 0.1, 0, 0, 90])
    np.random.seed(7355608)
    u = np.random.uniform(low = [-0.2, -1], high = [0.2, 3], size = [horizon,2])
    future_ref_list = np.random.normal(size = (horizon,3))
    n = 6
    vehicles_array = np.random.normal(size = (n,horizon,4))
    safe_dist = 5
    vehicles_array_cpp = vehicles_array[:,:,:2].copy()

    benchmark(mpc_cpp.mpc_constraints, u, ego_list, vehicles_array_cpp, safe_dist)
