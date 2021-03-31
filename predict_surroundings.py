import numpy as np 
from Utils import STEP_TIME, L, W

def route_to_task(veh_route):
    if veh_route == ('1o', '4i') or veh_route == ('2o', '1i') or veh_route == ('3o', '2i') or veh_route == ('4o', '3i'):
        task = 0   #左转
    elif veh_route == ('4o', '1i') or veh_route == ('1o', '2i') or veh_route == ('2o', '3i') or veh_route == ('3o', '4i'):
        task = 2  #右转
    else:
        task = 1  #直行
    return task

def veh_predict(veh, horizon):
    veh_x, veh_y, veh_v, veh_phi, veh_route = veh
    veh_task = route_to_task(veh_route)
    veh_array = np.zeros((horizon,4))

    veh_x_delta = veh_v * STEP_TIME * np.cos(veh_phi)
    veh_y_delta = veh_v * STEP_TIME * np.sin(veh_phi)

    rise = np.array([i+1 for i in range(horizon)])
    ones = np.ones(horizon)
    veh_x_array = veh_x * ones + rise * veh_x_delta
    veh_y_array = veh_y * ones + rise * veh_y_delta
    veh_v_array = veh_v * ones

    if veh_task  == 0:  # 左转
        veh_phi_rad_delta = np.where(-25 < veh_x < 25, (veh_v / 26.875) * STEP_TIME, 0)
    elif veh_task == 2:
        veh_phi_rad_delta = np.where(-25 < veh_y < 25, -(veh_v / 15.625) * STEP_TIME, 0)
    else:
        veh_phi_rad_delta = 0

    veh_phi_array = (veh_phi * ones + rise * veh_phi_rad_delta)
    veh_array[:,0] = veh_x_array
    veh_array[:,1] = veh_y_array
    veh_array[:,2] = veh_v_array
    veh_array[:,3] = veh_phi_array
    return veh_array

# 将车辆单圆模型转变为双圆模型
def double_circle_transfer(vehicles_array, n):
    offset = (L - W)/2
    veh_x_array = vehicles_array[:,:,0]
    veh_y_array = vehicles_array[:,:,1]
    veh_phi_array = vehicles_array[:,:,3]

    vehicles_xy_array_front = np.zeros_like(vehicles_array[:,:,:2])
    vehicles_xy_array_rear = np.zeros_like(vehicles_array[:,:,:2])
    vehicles_xy_array_front[:,:,0] = veh_x_array + offset * np.cos(veh_phi_array)
    vehicles_xy_array_front[:,:,1] = veh_y_array + offset * np.sin(veh_phi_array)
    vehicles_xy_array_rear[:,:,0] = veh_x_array - offset * np.cos(veh_phi_array)
    vehicles_xy_array_rear[:,:,1] = veh_y_array - offset * np.sin(veh_phi_array)
    return vehicles_xy_array_front, vehicles_xy_array_rear