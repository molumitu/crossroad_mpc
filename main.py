from scipy.optimize.zeros import VALUEERR
from Reference import ReferencePath
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import minimize
from Env_new import Crossroad
from Env_utils import L,W
from MPControl import ModelPredictiveControl


def set_ego_init_state(ref):
    random_index = 3000

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
    ref = ReferencePath(task, ref_index=2)
    ref_1 = ReferencePath(task, ref_index=1)
    init_ego_state = set_ego_init_state(ref_1)

    env = Crossroad(init_ego_state = init_ego_state)
    obs = env.obs


    mpc = ModelPredictiveControl(obs, horizon, ref, task = 'left')
    mpc_1 = ModelPredictiveControl(obs, horizon, ref_1, task = 'left')
    bounds = [(-0.21, 0.21), (-7.1, 3.1)] * horizon
    u_init = np.zeros((horizon, 2))
    tem_action = np.zeros((horizon, 2))
    mpc._update_future_ref()
    mpc_1._update_future_ref()


    # record_steer = []  #0
    # record_a_x = []   #1
    # record_ego_x_list = [] #2
    # record_ego_y_list = [] #3
    # record_ego_v_x_list = [] #4
    # record_ego_phi_list = []#5

    # record_ref_x_list = []  #6:16
    # record_ref_y_list = []   #16:26
    # record_ref_phi_list = [] #26:36

    # tem_action_array = np.array([[]])  #36:56
    result_array = np.zeros((85,56))
    # open_loop control
    # steer_action = np.loadtxt('action1.csv')
    # a_x_action = np.loadtxt('action2.csv')
    for name_index in range(85):
    # 控制量查表
        mpc._update_future_ref()
        mpc_1._update_future_ref()
        ineq_cons = {'type': 'ineq',
            'fun' : mpc.constraint_function}
        ineq_cons_1 = {'type': 'ineq',
            'fun' : mpc_1.constraint_function}
        tem_action = tem_action.reshape(10,2)
        tem_action[:,0] = np.clip(tem_action[:,0], -0.2, 0.2)
        tem_action[:,1] = np.clip(tem_action[:,1], -7, 3)                   

        try:
            results_1 = minimize(mpc_1.cost_function,
                                #x0=u_init.flatten(),
                                x0 = tem_action.flatten(),
                                method = 'SLSQP',
                                bounds = bounds,
                                constraints = ineq_cons_1,
                                tol=0.01,
                                #options={'disp': True} 
                        )
            print('results_1.fun',results_1.fun)
            results_1_finished = 1
        except ValueError:
            results_1_finished = 0


        try:
            results = minimize(mpc.cost_function,
                                #x0=u_init.flatten(),
                                x0 = tem_action.flatten(),
                                method = 'SLSQP',
                                bounds = bounds,
                                constraints = ineq_cons,
                                tol=0.01,
                                #options={'disp': True} 
                                )
            print('results.fun',results.fun)
            results_finished = 1
        except ValueError:
            results_finished = 0

        if results_finished and results_1_finished:
            if results_1.fun <= results.fun:
                mpc_action = results_1.x
                print('I choose The index = 1')
            else:
                mpc_action = results.x
                print('I choose The index = 2')
        elif results_1_finished:
            mpc_action = results_1.x
            print('I choose The index = 1')
        else:
            mpc_action = results.x
            print('I choose The index = 2')
        



        


        tem_action = np.concatenate((mpc_action[2:],mpc_action[-2:]),axis =0)
        # if not results_1.success:
        #     print('fail')
        #     # import sys
        #     # sys.exit()
        #     mpc_action = [0, -6.]
        obs, reward, done, info = env.step(mpc_action[:2])
        #obs, reward, done, info = env.step(np.array([steer_action[name_index], a_x_action[name_index]]))

        # record_steer.append(mpc_action[0])
        # record_a_x.append(mpc_action[1])
        result_array[name_index,0] = mpc_action[0]
        result_array[name_index,1] = mpc_action[1]        

        # record_ego_x_list.append(obs[0][3])
        # record_ego_y_list.append(obs[0][4])
        # record_ego_phi_list.append(obs[0][5])
        # record_ego_v_x_list.append(obs[0][0])
        result_array[name_index,2] = obs[0][3]
        result_array[name_index,3] = obs[0][4]
        result_array[name_index,4] = obs[0][5]
        result_array[name_index,5] = obs[0][0]

        result_array[name_index,6:16] = [ref[0] for ref in mpc.future_ref_list]
        result_array[name_index,16:26] = [ref[1] for ref in mpc.future_ref_list]
        result_array[name_index,26:36] = [ref[2] for ref in mpc.future_ref_list]

        result_array[name_index,36:46] = mpc_action[slice(0,20,2)] # steer
        result_array[name_index,46:56] = mpc_action[slice(1,20,2)] # a_x

        mpc.reset_obs(obs)
        mpc_1.reset_obs(obs)
        #env.render(name_index = name_index)

    #record_result = np.stack((record_ego_x_list, record_ego_y_list, record_ego_phi_list, record_ego_v_x_list, record_steer, record_a_x)).T
    record_result = result_array
    import datetime
    current_time = datetime.datetime.now()
    np.savetxt(f'result/record_result{current_time:%Y_%m_%d_%H_%M_%S}.csv', record_result, delimiter = ',')




if __name__ == '__main__':
    run_mpc()


