from Reference import ReferencePath
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import minimize
from Env_new import Crossroad
from Env_utils import L,W
from MPControl import ModelPredictiveControl


def set_ego_init_state(ref):
    random_index = 0

    x, y, phi = ref.indexs2points(random_index)
    v = 8 
    if ref.task == 'left':
        routeID = 'dl'
    elif ref.task == 'straight':
        routeID = 'du'
    else:
        ref.task == 'right'
        routeID = 'dr'
    return dict(ego=dict(v_x=v,
                            v_y=0,
                            r=0,
                            x=x,
                            y=y,
                            phi=phi,
                            l=L,
                            w=W,
                            routeID=routeID,
                            ))    # 这里指出了自车的名字叫ego, 这里也可以加多车

def run_mpc():
    horizon_list = [10]
    horizon = horizon_list[0]
    task = 'left'
    ref = ReferencePath(task, ref_index=0)
    init_ego_state = set_ego_init_state(ref)

    env = Crossroad(init_ego_state = init_ego_state)
    obs = env.obs


    mpc = ModelPredictiveControl(obs, horizon, ref, task = 'left')
    bounds = [(-0.2, 0.2), (-7., 3.)] * horizon
    u_init = np.zeros((horizon, 2))
    tem_action = np.zeros((horizon, 2))
    mpc._update_future_ref()
    record_steer = []
    record_a_x = []
    record_delta_v = []
    record_delta_s = []
    record_delta_phi = []

    record_ego_x_list = []
    record_ego_y_list = []
    record_ego_v_x_list = []
    record_ego_phi_list = []
    # open_loop control
    # steer_action = np.loadtxt('action1.csv')
    # a_x_action = np.loadtxt('action2.csv')
    for name_index in range(120):
    # 控制量查表
        mpc._update_future_ref()
        ineq_cons = {'type': 'ineq',
            'fun' : mpc.constraint_function}
        results = minimize(mpc.cost_function,
                            #x0=u_init.flatten(),
                            x0 = tem_action.flatten(),
                            method = 'SLSQP',
                            bounds = bounds,
                            constraints = ineq_cons,
                            tol=0.01,
                            options={'disp': True} 
                            )
        mpc_action = results.x
        tem_action = np.concatenate((mpc_action[2:],mpc_action[-2:]),axis =0)
        if not results.success:
            print('fail')
            # import sys
            # sys.exit()
            mpc_action = [0.2, -6.]
        obs, reward, done, info = env.step(mpc_action[:2])
        #obs, reward, done, info = env.step(np.array([steer_action[name_index], a_x_action[name_index]]))
        record_steer.append(mpc_action[0])
        record_a_x.append(mpc_action[1])
        # record_delta_v.append(env.reward_info['delta_v'])
        # record_delta_s.append(abs(env.reward_info['delta_s']))
        # record_delta_phi.append(env.reward_info['delta_phi'])
        # record_result = np.stack((record_steer, record_a_x, record_delta_v, record_delta_s, record_delta_phi)).T
        record_ego_x_list.append(obs[0][3])
        record_ego_y_list.append(obs[0][4])
        record_ego_phi_list.append(obs[0][5])
        record_ego_v_x_list.append(obs[0][0])
        record_result = np.stack((record_ego_x_list, record_ego_y_list, record_ego_phi_list, record_ego_v_x_list, record_steer, record_a_x)).T
        mpc.reset_obs(obs)
        #env.render(name_index = name_index)

    #record data as csv
    import datetime
    current_time = datetime.datetime.now()
    np.savetxt(f'result/record_result{current_time:%Y_%m_%d_%H_%M_%S}.csv', record_result, delimiter = ',')




if __name__ == '__main__':
    run_mpc()


