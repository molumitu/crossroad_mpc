#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import os
import sys
from collections import defaultdict
from math import fabs, cos, sin, pi ,sqrt
from Env_utils import shift_and_rotate_coordination, _convert_car_coord_to_sumo_coord, \
    _convert_sumo_coord_to_car_coord, xy2_edgeID_lane, STEP_TIME, CROSSROAD_SIZE

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
        seed = 73555608
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
            # port = sumolib.miscutils.getFreeSocketPort()
            traci.start(
                [SUMO_BINARY, "-c", SUMOCFG_DIR,
                 "--step-length", self.step_time,
                 "--lateral-resolution", "1.25",
                 # "--random",
                 # "--start",
                 # "--quit-on-end",
                 "--no-warnings",
                 "--no-step-log",
                 '--seed', str(int(seed))
                 ], port=port, numRetries=5)  # '--seed', str(int(seed))

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

    def add_ego_vehicles(self, n_ego_dict):
        all_vehicles = traci.vehicle.getContextSubscriptionResults('collector')

        other_vehicles = copy.deepcopy(all_vehicles)
        for ego_id in self.n_ego_dict.keys():
            if ego_id in other_vehicles:
                del other_vehicles[ego_id]


        for egoID, ego_dict in n_ego_dict.items():
            ego_v_x = ego_dict['v_x']
            ego_l = ego_dict['l']
            ego_x = ego_dict['x']
            ego_y = ego_dict['y']
            ego_phi = ego_dict['phi']
            ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo = _convert_car_coord_to_sumo_coord(ego_x, ego_y, ego_phi, ego_l)
            edgeID, lane = xy2_edgeID_lane(ego_x, ego_y)
            traci.vehicle.add(vehID=egoID, routeID=ego_dict['routeID'],typeID='self_car',departLane = lane, departPos = 0)
            traci.vehicle.moveToXY(egoID, edgeID, lane, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo, keepRoute=1)
            traci.vehicle.setLength(egoID, ego_dict['l'])
            traci.vehicle.setWidth(egoID, ego_dict['w'])
            traci.vehicle.setSpeed(egoID, ego_v_x)

            for veh in other_vehicles:
                x_in_sumo, y_in_sumo = other_vehicles[veh][traci.constants.VAR_POSITION]
                a_in_sumo = other_vehicles[veh][traci.constants.VAR_ANGLE]
                veh_l = other_vehicles[veh][traci.constants.VAR_LENGTH]
                veh_v = other_vehicles[veh][traci.constants.VAR_SPEED]
                # veh_sig = other_vehicles[veh][traci.constants.VAR_SIGNALS]
                # 10: left and brake 9: right and brake 1: right 8: brake 0: no signal 2: left

                x, y, a = _convert_sumo_coord_to_car_coord(x_in_sumo, y_in_sumo, a_in_sumo, veh_l)
                x_in_ego_coord, y_in_ego_coord, a_in_ego_coord = shift_and_rotate_coordination(x, y, a, ego_x,
                                                                                               ego_y, ego_phi)
                ego_x_in_veh_coord, ego_y_in_veh_coord, ego_a_in_veh_coord = shift_and_rotate_coordination(0, 0, 0,
                                                                                                           x_in_ego_coord,
                                                                                                           y_in_ego_coord,
                                                                                                           a_in_ego_coord)
                if (-5 < x_in_ego_coord < 1 * (ego_v_x) + ego_l/2. + veh_l/2. + 2 and abs(y_in_ego_coord) < 3) or \
                        (-5 < ego_x_in_veh_coord < 1 * (veh_v) + ego_l/2. + veh_l/2. + 2 and abs(ego_y_in_veh_coord) <3):
                    #traci.vehicle.moveToXY(veh, '4i', 1, -80, 1.85, 180, 2)
                    traci.vehicle.remove(vehID=veh)


    def init_traffic(self, init_n_ego_dict):
        self.sim_time = 0
        self.all_vehicles = traci.vehicle.getContextSubscriptionResults('collector')  # 最原始的信息
        self.n_ego_vehicles = defaultdict(list)
        self.n_ego_vehicles_list = defaultdict(list)  #返回的值比较简短

        self.collision_flag = False
        self.n_ego_collision_flag = {}
        self.collision_ego_id = None


        self.n_ego_dict = init_n_ego_dict

        self.traffic_light = 0
        traci.trafficlight.setPhase('0', self.traffic_light)
        

        # add ego and move ego to the given position and remove conflict cars
        self.add_ego_vehicles(init_n_ego_dict)



        self._update_traffic_light()
        traci.simulationStep()
        self.all_vehicles = traci.vehicle.getContextSubscriptionResults('collector')  # 最原始的信息
        self._update_n_ego_dict()
        self._get_vehicles()

    def _get_vehicles(self):
        self.n_ego_vehicles = defaultdict(list)  # 清零
        self.n_ego_vehicles_list = defaultdict(list)
        veh_infos = self.all_vehicles

        for egoID in self.n_ego_dict.keys():
            veh_info_dict = copy.deepcopy(veh_infos)
            # 得到自车的状态信息
            length_ego = veh_info_dict[egoID][traci.constants.VAR_LENGTH]
            width_ego = veh_info_dict[egoID][traci.constants.VAR_WIDTH]
            route_ego = veh_info_dict[egoID][traci.constants.VAR_EDGES]
            x_in_sumo_ego, y_in_sumo_ego = veh_info_dict[egoID][traci.constants.VAR_POSITION]
            a_in_sumo_ego = veh_info_dict[egoID][traci.constants.VAR_ANGLE]
            x_ego, y_ego, a_ego = _convert_sumo_coord_to_car_coord(x_in_sumo_ego, y_in_sumo_ego, a_in_sumo_ego, length_ego)
            v_ego = veh_info_dict[egoID][traci.constants.VAR_SPEED]

            for i, veh in enumerate(veh_info_dict):
                if veh != egoID and veh not in self.n_ego_dict.keys() and veh != 'collector':
                    length = veh_info_dict[veh][traci.constants.VAR_LENGTH]
                    width = veh_info_dict[veh][traci.constants.VAR_WIDTH]
                    route = veh_info_dict[veh][traci.constants.VAR_EDGES]
                    x_in_sumo, y_in_sumo = veh_info_dict[veh][traci.constants.VAR_POSITION]
                    a_in_sumo = veh_info_dict[veh][traci.constants.VAR_ANGLE]

                    # transfer x,y,a in car coord   # a means angle i.e. phi
                    x, y, a = _convert_sumo_coord_to_car_coord(x_in_sumo, y_in_sumo, a_in_sumo, length)
                    v = veh_info_dict[veh][traci.constants.VAR_SPEED]
                    traci.vehicle.setColor(str(veh),(255,255,255))
                    # if  (sqrt((x_ego-x) ** 2 + (y_ego-y) ** 2) < 80) and \
                    #     (((route[1] == '4i') and (x > -CROSSROAD_SIZE/2 - 10)) or ((route[1] == '1i') \
                    #         and (y > -CROSSROAD_SIZE/2) and (x < CROSSROAD_SIZE/2)\
                    #         ))\
                    #         and x < (x_ego + 15)  and y > (y_ego - 15):
                    if (sqrt((x_ego-x) ** 2 + (y_ego-y) ** 2) < 30) and\
                        (((route[1]=='1i' and route[0] != '4o' and (y > -CROSSROAD_SIZE/2 + 15) and (x < CROSSROAD_SIZE/2 + 2))\
                        or (route[1]=='4i'))and x < (x_ego)  and y > (y_ego)):
                        if veh not in self.n_ego_dict.keys():
                            traci.vehicle.setColor(str(veh),(255,0,0))
                        self.n_ego_vehicles[egoID].append(dict(veh_ID=veh, x=x, y=y, v=v, phi=a, l=length, w=width, route=route))
                        self.n_ego_vehicles_list[egoID].append([x, y, v, a, route])

    def _get_traffic_light(self):
        return traci.trafficlight.getPhase('0')

    def _update_n_ego_dict(self):
        n_ego_dict = self.n_ego_dict.copy()
        for egoID, ego_dict in n_ego_dict.items():
            ego_x = ego_dict['x']
            ego_y = ego_dict['y']
            #print('egoID',egoID,'ego_x', ego_x)
            if abs(ego_x) > 97 or abs(ego_y) > 97:
                del self.n_ego_dict[egoID]

    def _update_traffic_light(self):
        sim_time  = self.sim_time % 60
        if sim_time < 25:
            self.traffic_light = 0
        elif sim_time < 30:
            self.traffic_light = 1
        elif sim_time < 55:
            self.traffic_light = 2
        else:
            self.traffic_light = 3
        traci.trafficlight.setPhase('0', self.traffic_light)

    def sim_step(self):

        self.sim_time += STEP_TIME
        self._update_traffic_light()
        traci.simulationStep()
        self.all_vehicles = traci.vehicle.getContextSubscriptionResults('collector')  # 最原始的信息
        self._update_n_ego_dict()
        self._get_vehicles()

        # self.collision_check()
        # for egoID, collision_flag in self.n_ego_collision_flag.items():
        #     if collision_flag:
        #         self.collision_flag = True
        #         self.collision_ego_id = egoID

    def set_own_car(self, n_ego_dict_):  #n_ego_dict由env传过来，在env中维护其增减操作
        # assert len(self.n_ego_dict) == len(n_ego_dict_)
        # for egoID in self.n_ego_dict.keys():   #周车数量可变的情况下，不需要这个assert
        for egoID in n_ego_dict_.keys():
            self.n_ego_dict[egoID]['v_x'] = ego_v_x = n_ego_dict_[egoID]['v_x']
            self.n_ego_dict[egoID]['v_y'] = ego_v_y = n_ego_dict_[egoID]['v_y']
            self.n_ego_dict[egoID]['r'] = ego_r = n_ego_dict_[egoID]['r']
            self.n_ego_dict[egoID]['x'] = ego_x = n_ego_dict_[egoID]['x']
            self.n_ego_dict[egoID]['y'] = ego_y = n_ego_dict_[egoID]['y']
            self.n_ego_dict[egoID]['phi'] = ego_phi = n_ego_dict_[egoID]['phi']

            ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo = _convert_car_coord_to_sumo_coord(ego_x, ego_y, ego_phi,
                                                                                           self.n_ego_dict[egoID]['l'])
            egdeID, lane = xy2_edgeID_lane(ego_x, ego_y)
            keeproute = 1
            try:
                traci.vehicle.moveToXY(egoID, egdeID, lane, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo, keeproute)
            except traci.exceptions.TraCIException:
                print(egoID, egdeID, lane, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo, keeproute)
                traci.vehicle.moveToXY(egoID, egdeID, lane, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo, keeproute)
            traci.vehicle.setSpeed(egoID, ego_v_x)

    def collision_check(self):  # True: collision
        flag_dict = dict()
        for egoID, list_of_veh_dict in self.n_ego_vehicles.items():
            ego_x = self.n_ego_dict[egoID]['x']
            ego_y = self.n_ego_dict[egoID]['y']
            ego_phi = self.n_ego_dict[egoID]['phi']
            ego_l = self.n_ego_dict[egoID]['l']
            ego_w = self.n_ego_dict[egoID]['w']
            ego_lw = (ego_l - ego_w) / 2
            ego_x0 = (ego_x + cos(ego_phi / 180 * pi) * ego_lw)
            ego_y0 = (ego_y + sin(ego_phi / 180 * pi) * ego_lw)
            ego_x1 = (ego_x - cos(ego_phi / 180 * pi) * ego_lw)
            ego_y1 = (ego_y - sin(ego_phi / 180 * pi) * ego_lw)
            flag_dict[egoID] = False

            for veh in list_of_veh_dict:
                if fabs(veh['x'] - ego_x) < 10 and fabs(veh['y'] - ego_y) < 10:
                    surrounding_lw = (veh['l'] - veh['w']) / 2
                    surrounding_x0 = (veh['x'] + cos(veh['phi'] / 180 * pi) * surrounding_lw)
                    surrounding_y0 = (veh['y'] + sin(veh['phi'] / 180 * pi) * surrounding_lw)
                    surrounding_x1 = (veh['x'] - cos(veh['phi'] / 180 * pi) * surrounding_lw)
                    surrounding_y1 = (veh['y'] - sin(veh['phi'] / 180 * pi) * surrounding_lw)
                    collision_check_dis = ((veh['w'] + ego_w) / 2 + 0.5) ** 2
                    if (ego_x0 - surrounding_x0) ** 2 + (ego_y0 - surrounding_y0) ** 2 < collision_check_dis:
                        flag_dict[egoID] = True
                    elif (ego_x0 - surrounding_x1) ** 2 + (ego_y0 - surrounding_y1) ** 2 < collision_check_dis:
                        flag_dict[egoID] = True
                    elif (ego_x1 - surrounding_x1) ** 2 + (ego_y1 - surrounding_y1) ** 2 < collision_check_dis:
                        flag_dict[egoID] = True
                    elif (ego_x1 - surrounding_x0) ** 2 + (ego_y1 - surrounding_y0) ** 2 < collision_check_dis:
                        flag_dict[egoID] = True

        self.n_ego_collision_flag = flag_dict


if __name__ == "__main__":
    pass