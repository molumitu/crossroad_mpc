import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import minimize
from Env import Crossroad
from MPControl import ModelPredictiveControl


def run_mpc():
    horizon_list = [50]
    horizon = horizon_list[0]
    env = Crossroad(training_task='left')
    obs = env.reset()
    mpc = ModelPredictiveControl(obs, horizon, task = 'left')
    bounds = [(-0.3, 0.3), (-7., 3.)] * horizon
    u_init = np.zeros((horizon, 2))
    tem_action = np.zeros((horizon, 2))
    mpc.reset_init_ref(obs, env.ref_path.ref_index)
    record_steer = []
    record_a_x = []
    record_delta_v = []
    record_delta_s = []
    record_delta_phi = []
    # open_loop control
    # steer_action = np.loadtxt('action1.csv')
    # a_x_action = np.loadtxt('action2.csv')
    for name_index in range(2000):
    # 控制量查表
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
            mpc_action = [0.15, -6.]
        obs, reward, done, info = env.step(mpc_action[:2])
        #obs, reward, done, info = env.step(np.array([steer_action[name_index], a_x_action[name_index]]))
        record_steer.append(mpc_action[0])
        record_a_x.append(mpc_action[1])
        record_delta_v.append(env.reward_info['delta_v'])
        record_delta_s.append(abs(env.reward_info['delta_s']))
        record_delta_phi.append(env.reward_info['delta_phi'])
        record_result = np.stack((record_steer, record_a_x, record_delta_v, record_delta_s, record_delta_phi)).T
        mpc.reset_init_state(obs)
        #env.render(name_index = name_index)

    #record data as csv
    import datetime
    current_time = datetime.datetime.now()
    np.savetxt(f'result/record_result{current_time:%Y_%m_%d_%H_%M_%S}.csv', record_result, delimiter = ',')




if __name__ == '__main__':
    run_mpc()


