from Reference import ReferencePath
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle as Rt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib import cm
# from tqdm import trange
import numpy as np
import json
import traci
import pickle
import os
from Utils import CROSSROAD_SIZE, LANE_NUMBER, LANE_WIDTH, convert_sumo_coord_to_car_coord, L, W
from map import static_view, update_light
extension = 40
ego_route_dict = {'ego1':'dl', 'ego2':'du', 'ego3':'dr' ,'ego4':'rd','ego5':'rl', 'ego6':'ru', 
                  'ego7':'ur' ,'ego8':'ud','ego9':'ul', 'ego10':'lu', 'ego11':'lr' ,'ego12':'ld'}
ego_route_dict_not_control = {'ego1_not_control':'dl', 'ego2_not_control':'du', 'ego3_not_control':'dr' ,'ego4_not_control':'rd',
                            'ego5_control':'rl', 'ego6_not_control':'ru', 'ego7_not_control':'ur' ,'ego8_not_control':'ud',
                            'ego9_not_control':'ul', 'ego10_not_control':'lu', 'ego11_not_control':'lr' ,'ego12_not_control':'ld'}

def rotate(phi: float):
    R = np.array([[np.cos(phi), -np.sin(phi)],
                  [np.sin(phi), np.cos(phi)]])
    return R

def veh2vis(xy: np.ndarray, phi: float):
    xy_vis = xy + rotate(phi) @ np.array([-L / 2, W / 2])
    phi_vis = phi * 180 / np.pi - 90
    return xy_vis, phi_vis

class ExpAnimation():

    def __init__(self) -> None:
        self.ego = None
        self.others = None
        self.traffic_light = None
        self.task_name = ''
    
    def load_exp_data(self, load_data_path:str):
        self.task_name = load_data_path.split('\\')[-1]
        print(self.task_name)
        with open(os.path.join(load_data_path, 'ego.pickle'), 'rb') as f:
            self.ego = pickle.load(f)
        with open(os.path.join(load_data_path, 'others.pickle'), 'rb') as f:
            self.others = pickle.load(f)
        with open(os.path.join(load_data_path, 'traffic_light.pickle'), 'rb') as f:
            self.traffic_light = pickle.load(f)
    
    def move_car(self, carPatch:Rt, xy, phi):
        xy_vis, phi_vis = veh2vis(xy, phi)
        carPatch.set_xy(xy_vis)
        carPatch.angle = phi_vis

    def move_ego(self, carPatch:Rt, detectionPatch:Circle, ego_xy, ego_phi):
        xy_vis, phi_vis = veh2vis(ego_xy, ego_phi)
        carPatch.set_xy(xy_vis)
        carPatch.angle = phi_vis
        detectionPatch.center = ego_xy
    
    def remove_env(self, carPatch:Rt):
        carPatch.remove()
    
    def add_env(self, ax, xy, phi, facecolor='blue'):
        p = ax.add_patch(Rt([0, 0], 1.6, 3.6, 0, facecolor=facecolor, edgecolor='black'))
        self.move_car(p, xy, phi)
        return p

    def animation_output(self, save_video_path:str, fps=10, frame_range=50, timeline:tuple=None, colorEgo='red', colorEnv='blue', colorTrack:bool =False, colorRange:int =50, xlim=None, ylim=None):

        if self.ego is None or self.others is None:
            print('fuck!')
            return
        
        metadata = dict(title='Demo', artist='Guojian Zhan',
        comment='2021_intersection_graduation')
        writer = FFMpegWriter(fps=fps, metadata=metadata)

        # ------------------ initialization -------------------
        n_frames = len(self.others)
        if colorTrack:
            if timeline:
                n = Normalize(timeline[0], timeline[1])
                n_frames = timeline[1] - timeline[0]
                Iter = range(timeline[0], timeline[1])
            else:
                n = Normalize(colorRange)
            cmap1 = cm.ScalarMappable(norm=n, cmap='plasma')
            self.task_name = f'{self.task_name}_coloredtrack'
        elif timeline:
            Iter = range(timeline[0], timeline[1])
            self.task_name = f'{self.task_name}_segs'
        else:
            Iter = range(n_frames)
        env_set = set()
        env_dict = dict()
        fig, main_axes = static_view()
        writer.setup(fig, os.path.join(save_video_path, f'{self.task_name}.mp4'))
        ego_keys = ego_route_dict.keys()
        ego_car = {}
        ego_detection = {}
        for egoID in ego_keys:
            ego_car[egoID] = main_axes.add_patch(Rt([CROSSROAD_SIZE + extension, -CROSSROAD_SIZE - extension], 1.6, 3.6, 0, facecolor=colorEgo, edgecolor='black'))
            ego_detection[egoID] = main_axes.add_patch(Circle((CROSSROAD_SIZE + extension, -CROSSROAD_SIZE - extension), radius= 30, alpha = 0.01, color = 'gray'))
        # ---------------------- update -----------------------


        for i in Iter:
            line = {}
            ref_points = {}
            egos = self.ego[i]
            env_cars = self.others[i]
            traffic_light = self.traffic_light[i]
            Lines = update_light(main_axes, traffic_light)
            for egoID in egos.keys():
                line[egoID] = main_axes.plot([], [],  alpha = 0.5, color = 'g')
                ref_points[egoID] = main_axes.scatter([], [], marker='.', s=30, color='r')
            for egoID, ego_info in egos.items():
                ego_dynamics = ego_info['dynamics']
                ego_ref_points = ego_info['ref']
                ego_route = ego_route_dict[egoID]
                ego_ref = ReferencePath(ego_route, extension=40)
                ego_xy = ego_dynamics[3:5]
                ego_phi = ego_dynamics[5]
                if colorTrack:
                    main_axes.scatter(ego_xy[0], ego_xy[1], marker='.', s=20, color=cmap1.to_rgba(i))
                self.move_ego(ego_car[egoID], ego_detection[egoID], ego_xy, ego_phi)
                ref_best_index = ego_dynamics[8]
                ref_points[egoID] = main_axes.scatter(ego_ref_points[:,0], ego_ref_points[:,1], marker='.', s=30, color='r')
                line[egoID] = main_axes.plot(ego_ref.path_list[ref_best_index][0], ego_ref.path_list[ref_best_index][1],  alpha = 0.5, color = 'g')

            # # ---------------- update of Env ------------------

            on_spot = set(env_cars.keys())
            to_remove = env_set - on_spot
            to_add = on_spot - env_set
            to_update = on_spot - to_add
            for e in to_remove:
                self.remove_env(env_dict.pop(e))
            for e in to_add:
                x_in_sumo, y_in_sumo = env_cars[e][traci.constants.VAR_POSITION]
                a_in_sumo = env_cars[e][traci.constants.VAR_ANGLE]/180 *np.pi
                x, y, a = convert_sumo_coord_to_car_coord(x_in_sumo, y_in_sumo, a_in_sumo, L)
                xy = [x, y]
                if e in ego_route_dict_not_control.keys():
                    env_dict[e] = self.add_env( ax = main_axes, 
                            xy = xy, 
                            phi = a, 
                            facecolor = colorEgo)
                else:
                    env_dict[e] = self.add_env( ax = main_axes, 
                                                xy = xy, 
                                                phi = a, 
                                                facecolor = colorEnv)
            for e in to_update:
                x_in_sumo, y_in_sumo = env_cars[e][traci.constants.VAR_POSITION]
                a_in_sumo = env_cars[e][traci.constants.VAR_ANGLE]/180 *np.pi
                x, y, a = convert_sumo_coord_to_car_coord(x_in_sumo, y_in_sumo, a_in_sumo, L)
                xy = [x, y]
                self.move_car(env_dict[e], xy = xy, phi=a)
            env_set = on_spot

            # ----------------- 界面对准中心 -----------------------
            c_x = 0
            c_y = 0
            frame_range = CROSSROAD_SIZE/2 + extension/2
            main_axes.set(xlim=[c_x-frame_range, c_x+frame_range], ylim=[c_y-frame_range, c_y+frame_range])
            writer.grab_frame()
            for egoID in egos.keys():
                line[egoID][0].remove()
                ref_points[egoID].remove()
            for l in Lines:
                l[0].remove()



        
        # if timeline and xlim:
        #     main_axes.set(xlim=xlim, ylim=ylim)
        #     main_axes.set_xlabel('X / m')
        #     main_axes.set_ylabel('Y / m')
        #     main_axes.set_position([0.1, 0.1, 0.8, 0.8])
        #     main_axes.scatter([0],[0], marker='s', c='r', s=80, label='ego_car')
        #     main_axes.scatter([0],[0], marker='s', c='b', s=80, label='env_car')
        #     main_axes.legend()
        #     cax = fig.add_axes([0.3,0.87,0.4,0.01])
        #     if colorTrack:
        #         fig.colorbar(cmap1, cax=cax, orientation='horizontal', label='time / 0.1s') # location='right', fraction=0.05, aspect=30, pad=0
        #     fig.savefig(os.path.join(tpath, f'{self.task_name}.png'), dpi=150)

        writer.finish()
        writer.cleanup()
        print('video export success!')


if __name__ == "__main__":
    pass