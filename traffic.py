import copy
import os
import sys
from collections import defaultdict
from math import sqrt
import numpy as np
from typing import Dict
from Utils import shift_and_rotate_coordination, convert_car_coord_to_sumo_coord, \
    convert_sumo_coord_to_car_coord, xy2_edgeID_lane, STEP_TIME, CROSSROAD_SIZE,L,W

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import sumolib
from sumolib import checkBinary
import traci
from traci.exceptions import FatalTraCIError

dirname = os.path.dirname(__file__)
SUMOCFG_DIR = dirname + "/sumo_files/cross.sumocfg"
SUMO_BINARY = checkBinary('sumo-gui')


class Traffic(object):

    def __init__(self):
        self.step_time = str(STEP_TIME)
        seed = 6
        import time
        start = time.time()
        port = sumolib.miscutils.getFreeSocketPort()
        try:
            traci.start(
                [SUMO_BINARY, "-c", SUMOCFG_DIR,
                 "--step-length", self.step_time,
                 "--lateral-resolution", "3.75",
                 #"--random",
                 # "--start",
                 # "--quit-on-end",
                 "--no-warnings",
                 "--no-step-log",
                 #"--collision.action", "remove"
                 '--seed', str(int(seed))
                 ], port=port, numRetries=5)  # '--seed', str(int(seed))
        except FatalTraCIError:
            print('Retry by other port')
            # # port = sumolib.miscutils.getFreeSocketPort()
            # traci.start(
            #     [SUMO_BINARY, "-c", SUMOCFG_DIR,
            #      "--step-length", self.step_time,
            #      "--lateral-resolution", "1.25",
            #      # "--random",
            #      # "--start",
            #      # "--quit-on-end",
            #      "--no-warnings",
            #      "--no-step-log",
            #      '--seed', str(int(seed))
            #      ], port=port, numRetries=5)  # '--seed', str(int(seed))

        traci.vehicle.subscribeContext('collector',
                                       traci.constants.CMD_GET_VEHICLE_VARIABLE,
                                       999999, [traci.constants.VAR_POSITION,
                                                traci.constants.VAR_LENGTH,
                                                traci.constants.VAR_WIDTH,
                                                traci.constants.VAR_ANGLE,
                                                traci.constants.VAR_SIGNALS,
                                                traci.constants.VAR_SPEED,
                                                # traci.constants.VAR_TYPE,
                                                # traci.constants.VAR_EMERGENCY_DECEL,
                                                # traci.constants.VAR_LANE_INDEX,
                                                # traci.constants.VAR_LANEPOSITION,
                                                traci.constants.VAR_ACCELERATION,
                                                traci.constants.VAR_EDGES,
                                                # traci.constants.VAR_ROUTE_INDEX
                                                ],
                                       0, 999999999999)
        end = time.time()
        print("Sumo startup time: ", end - start)
        # 先让交通流运行一段时间
        while traci.simulation.getTime() < 20:   #这里的时间单位是秒
            traci.trafficlight.setPhase('0', 2)
            traci.simulationStep()

    def init_traffic(self, init_n_ego_dict:Dict[str, float or str]):
        self.sim_time = 20
        self.all_vehicles = traci.vehicle.getContextSubscriptionResults('collector')  # 最原始的信息



        self.collision_flag = False
        self.n_ego_collision_flag = {}
        self.collision_ego_id = None

        self.traffic_light = 0        
        
        init_egos_dict  = init_n_ego_dict.copy()

        # self.add_ego_vehicles({**init_n_ego_dict, **init_n_ego_dict_not_control})
        self.add_ego_vehicles(init_egos_dict)
        self.each_ego_vehicles_list = defaultdict(list)

        self._update_traffic_light()
        traci.simulationStep()
        self.all_vehicles = traci.vehicle.getContextSubscriptionResults('collector')  # 才开始包括自车的信息
        self.get_vehicles_for_each_ego(init_egos_dict.keys())

    def add_ego_vehicles(self, init_egos_dict):
        # add ego and move ego to the given position and remove conflict cars
        # 这时候self.all_vehicles里面还没有自车
        other_vehicles = copy.deepcopy(self.all_vehicles)

        for egoID, ego_dict in init_egos_dict.items():
            ego_v_x = ego_dict['v_x']
            ego_l = ego_dict['l']
            ego_x = ego_dict['x']
            ego_y = ego_dict['y']
            ego_phi = ego_dict['phi']
            ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo = convert_car_coord_to_sumo_coord(ego_x, ego_y, ego_phi, ego_l)
            edgeID, lane = '1o 4i', 0 #这个无所谓
            traci.vehicle.add(vehID=egoID, routeID=ego_dict['routeID'],typeID='self_car',departLane = lane, departPos = 0)
            traci.vehicle.moveToXY(egoID, edgeID, lane, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo*180/np.pi, keepRoute=1)
            traci.vehicle.setLength(egoID, ego_dict['l'])
            traci.vehicle.setWidth(egoID, ego_dict['w'])
            traci.vehicle.setSpeed(egoID, ego_v_x)


            # 有三种coord：
                #car_coord 
                #ego_coord 
                #sumo_coord
            for veh in other_vehicles:
                x_in_sumo, y_in_sumo = other_vehicles[veh][traci.constants.VAR_POSITION]
                a_in_sumo = other_vehicles[veh][traci.constants.VAR_ANGLE] /180 * np.pi
                veh_l = other_vehicles[veh][traci.constants.VAR_LENGTH]
                veh_v = other_vehicles[veh][traci.constants.VAR_SPEED]
                # veh_sig = other_vehicles[veh][traci.constants.VAR_SIGNALS]
                # 10: left and brake 9: right and brake 1: right 8: brake 0: no signal 2: left
                x, y, a = convert_sumo_coord_to_car_coord(x_in_sumo, y_in_sumo, a_in_sumo, veh_l)
                x_in_ego_coord, y_in_ego_coord, a_in_ego_coord = shift_and_rotate_coordination(x, y, a, ego_x, ego_y, ego_phi)
                ego_x_in_veh_coord, ego_y_in_veh_coord, ego_a_in_veh_coord = shift_and_rotate_coordination(ego_x, ego_y, ego_phi,x, y, a)
                #保证加入的自车和周围车辆之间相互不冲突
                if (-6 < x_in_ego_coord < 2 * (ego_v_x) + ego_l/2. + veh_l/2. + 2 and abs(y_in_ego_coord) < 3) or \
                        (-6 < ego_x_in_veh_coord < 2 * (veh_v) + ego_l/2. + veh_l/2. + 2 and abs(ego_y_in_veh_coord) <3):
                    traci.vehicle.remove(vehID=veh)

    def get_vehicles_for_each_ego(self, n_ego_dict_keys):
        self.each_ego_vehicles_list = defaultdict(list)
        veh_set = set()

        for egoID in n_ego_dict_keys:
            veh_info_dict = copy.deepcopy(self.all_vehicles)
            # 得到自车的状态信息
            length_ego = veh_info_dict[egoID][traci.constants.VAR_LENGTH]
            width_ego = veh_info_dict[egoID][traci.constants.VAR_WIDTH]
            route_ego = veh_info_dict[egoID][traci.constants.VAR_EDGES]
            x_in_sumo_ego, y_in_sumo_ego = veh_info_dict[egoID][traci.constants.VAR_POSITION]
            a_in_sumo_ego = veh_info_dict[egoID][traci.constants.VAR_ANGLE] /180 *np.pi
            x_ego, y_ego, a_ego = convert_sumo_coord_to_car_coord(x_in_sumo_ego, y_in_sumo_ego, a_in_sumo_ego, length_ego)
            v_ego = veh_info_dict[egoID][traci.constants.VAR_SPEED]
            route_ego = veh_info_dict[egoID][traci.constants.VAR_EDGES]
            if route_ego == ('1o', '4i') or route_ego == ('2o', '1i') or route_ego == ('3o', '2i') or route_ego == ('4o', '3i'):
                task = 0   #左转
            elif route_ego == ('4o', '1i') or route_ego == ('1o', '2i') or route_ego == ('2o', '3i') or route_ego == ('3o', '4i'):
                task = 2  #右转
            else:
                task = 1  #直行

            for veh in veh_info_dict:
                if veh != egoID  and veh != 'collector':
                    length = veh_info_dict[veh][traci.constants.VAR_LENGTH]
                    width = veh_info_dict[veh][traci.constants.VAR_WIDTH]
                    route = veh_info_dict[veh][traci.constants.VAR_EDGES]
                    acc = veh_info_dict[veh][traci.constants.VAR_ACCELERATION]
                    x_in_sumo, y_in_sumo = veh_info_dict[veh][traci.constants.VAR_POSITION]
                    a_in_sumo = veh_info_dict[veh][traci.constants.VAR_ANGLE] /180 *np.pi

                    # transfer x,y,a in car coord   # a means angle i.e. phi
                    x, y, a = convert_sumo_coord_to_car_coord(x_in_sumo, y_in_sumo, a_in_sumo, length)
                    v = veh_info_dict[veh][traci.constants.VAR_SPEED]
                    # transfer x,y,a in ego coord， 暂时还没用上
                    x_in_ego_coord, y_in_ego_coord, a_in_ego_coord = shift_and_rotate_coordination(x, y, a, x_ego, y_ego, a_ego)
                    #traci.vehicle.setColor(str(veh),(255,255,255))
                    if abs(y_ego) > 25:
                        if abs(x_ego - x) < 3 and abs(y_ego) - abs(y) > 0 and abs(y_ego) - abs(y) < 15:
                            if veh not in n_ego_dict_keys:
                                veh_set.add(veh)    
                            self.each_ego_vehicles_list[egoID].append([x, y, v, a, acc, route])
                    elif abs(x_ego) > 25:
                        if abs(y_ego - y) < 3 and abs(x_ego) - abs(x) > 0 and abs(x_ego) - abs(x) < 15:
                            if veh not in n_ego_dict_keys:
                                veh_set.add(veh)    
                            self.each_ego_vehicles_list[egoID].append([x, y, v, a, acc, route])
                    elif (sqrt((x_ego-x) ** 2 + (y_ego-y) ** 2) < 30) and task == 0 \
                        and x_in_ego_coord > -3 and y_in_ego_coord > -3 and abs(x)<27 and abs(y)<27: #左转
                        if veh not in n_ego_dict_keys:
                            veh_set.add(veh)    
                        self.each_ego_vehicles_list[egoID].append([x, y, v, a, acc, route])
                    elif (sqrt((x_ego-x) ** 2 + (y_ego-y) ** 2) < 20) and task == 1 \
                        and x_in_ego_coord > 0 and y_in_ego_coord > -5 and y_in_ego_coord < 5: #直行
                        if veh not in n_ego_dict_keys:
                            veh_set.add(veh)    
                        self.each_ego_vehicles_list[egoID].append([x, y, v, a, acc, route])
                    elif (sqrt((x_ego-x) ** 2 + (y_ego-y) ** 2) < 20) and task == 2 \
                        and x_in_ego_coord > -3 and y_in_ego_coord < 3: #右转
                        if veh not in n_ego_dict_keys:
                            veh_set.add(veh)    
                        self.each_ego_vehicles_list[egoID].append([x, y, v, a, acc, route])

        # for veh in veh_set:
        #     traci.vehicle.setColor(str(veh),(255,0,0))
        for veh in n_ego_dict_keys:
            traci.vehicle.setColor(str(veh),(255,0,255))

    @property
    def traffic_light(self):
        return traci.trafficlight.getPhase('0')

    @traffic_light.setter
    def traffic_light(self, phase):
        traci.trafficlight.setPhase('0', phase)

    def _update_traffic_light(self):
        sim_time  = self.sim_time % 130
        if sim_time < 5:
            self.traffic_light = 3  #下方来车绿色
        elif sim_time < 65:
            self.traffic_light = 0
        elif sim_time < 70:
            self.traffic_light = 1
        else:
            self.traffic_light = 2
# 2301
# 0123
# 1230
#

    def sim_step(self):
        self.sim_time += STEP_TIME
        self._update_traffic_light()
        traci.simulationStep()
        self.all_vehicles = traci.vehicle.getContextSubscriptionResults('collector')  # 最原始的信息

        # self.collision_check()
        # for egoID, collision_flag in self.n_ego_collision_flag.items():
        #     if collision_flag:
        #         self.collision_flag = True
        #         self.collision_ego_id = egoID

    def sync_ego_vehicles(self, n_ego_dict: Dict[str, Dict[str, float]]):
        """
        n_ego_dict由env传过来，在env中维护其增减操作

        Args:
            n_ego_dict_ (Dict[str, Dict[str, float]]): 所有自车状态
        """
        for egoID in n_ego_dict.keys():
            ego_v_x = n_ego_dict[egoID]['v_x']
            ego_x = n_ego_dict[egoID]['x']
            ego_y = n_ego_dict[egoID]['y']
            ego_phi = n_ego_dict[egoID]['phi']
            ego_l = L
            ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo = convert_car_coord_to_sumo_coord(ego_x, ego_y, ego_phi, ego_l)
            egdeID, lane = '1o 4i', 0
            keeproute = 1
            try:
                traci.vehicle.moveToXY(egoID, egdeID, lane, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo*180/np.pi, keeproute)
            except traci.exceptions.TraCIException:
                print(egoID, egdeID, lane, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo*180/np.pi, keeproute)
                traci.vehicle.moveToXY(egoID, egdeID, lane, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo*180/np.pi, keeproute)
            traci.vehicle.setSpeed(egoID, ego_v_x)

if __name__ == "__main__":
    pass