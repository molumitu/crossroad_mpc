import ctypes
ctypes.cdll.LoadLibrary(R"C:\ProgramData\Anaconda3\Lib\site-packages\bezier\extra-dll\bezier-b9fda8dc.dll")
import bezier
import matplotlib.pyplot as plt
import numpy as np
from Env_utils import CROSSROAD_SIZE, LANE_WIDTH, LANE_NUMBER, STEP_TIME, deal_with_phi

class ReferencePath(object):
    def __init__(self, task, mode=None, ref_index=None):
        self.mode = mode
        self.traj_mode = None
        self.exp_v = 8.
        self.task = task
        self.path_list = []
        self.path_len_list = []
        self._construct_ref_path(self.task)
        #self.ref_index = np.random.choice(len(self.path_list)) if ref_index is None else ref_index
        self.ref_index = 2
        self.path = self.path_list[self.ref_index]

    def set_path(self, traj_mode, path_index=None, path=None):
        self.traj_mode = traj_mode
        # if traj_mode == 'dyna_traj':
        #     self.path = path
        # elif traj_mode == 'static_traj':
        self.ref_index = path_index
        self.path = self.path_list[self.ref_index]

    def _construct_ref_path(self, task):
        sl = 40  # straight line length, equal to extensions
        meter_pointnum_ratio = 10 
        end_points_num = int(sl * meter_pointnum_ratio) + 1
        arc_points_num = 422+1
        # the length of the arc is 42.2152m
        R = CROSSROAD_SIZE/2 + LANE_WIDTH/2
        if task == 'left':
            end_offsets = [LANE_WIDTH*(i+0.5) for i in range(LANE_NUMBER)]
            start_offsets = [LANE_WIDTH*0.5]
            for end_offset in end_offsets:


                #---------start_straight_line---------------------------------------------------------
                start_points_num = int((sl + end_offset - LANE_WIDTH*0.5) * meter_pointnum_ratio) + 1
                start_straight_line_x = LANE_WIDTH/2 * np.ones(shape=(start_points_num,), dtype=np.float32)[:-1]
                start_straight_line_y = np.linspace(-CROSSROAD_SIZE/2 - sl, -CROSSROAD_SIZE/2 + end_offset- LANE_WIDTH*0.5 , start_points_num , dtype=np.float32)[:-1]

                #---------connected_arc_line-----------------------------------------------------------
                s_vals = np.linspace(0, np.pi/2, arc_points_num, dtype=np.float32)
                arc_line_x = R * np.cos(s_vals) - R + start_offsets
                arc_line_y = R * np.sin(s_vals) - R + end_offset

                #---------end_straight_line------------------------------------------------------------
                end_straight_line_x = np.linspace(-CROSSROAD_SIZE/2, -CROSSROAD_SIZE/2 - sl, end_points_num,                                                     dtype=np.float32)[1:]
                end_straight_line_y = end_offset * np.ones(shape=(end_points_num,), dtype=np.float32)[1:]

                #---------------------------------------------------------------------------------------

                total_x = np.concatenate((start_straight_line_x,arc_line_x,end_straight_line_x), axis=0)
                total_y = np.concatenate((start_straight_line_y,arc_line_y,end_straight_line_y), axis=0)
                
                xs_1, ys_1 = total_x[:-1], total_y[:-1]  #除去最后一个点之后，剩余的部分
                xs_2, ys_2 = total_x[1:], total_y[1:]    #除去第一个点之后，剩余的部分
                phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1) * 180 / np.pi

                planned_trj = total_x, total_y, np.concatenate((phis_1, phis_1[-1:]), axis=0)
            
                self.path_list.append(planned_trj)
                self.path_len_list.append(len(total_x))
        
    def find_closest_point(self, xs, ys, ratio=1):  # radio用来将轨迹点稀疏化，但是不需要
        # path_len = len(self.path[0])
        # reduced_idx = np.arange(0, path_len, ratio)
        # reduced_len = len(reduced_idx)
        # reduced_path_x, reduced_path_y = self.path[0][reduced_idx], self.path[1][reduced_idx]

        # xs_tile = np.tile(np.reshape(xs, (-1, 1)), [1, reduced_len])
        # ys_tile = np.tile(np.reshape(ys, (-1, 1)), [1, reduced_len])
        # pathx_tile = np.tile(np.reshape(reduced_path_x, (1, -1)), [len(xs), 1])
        # pathy_tile = np.tile(np.reshape(reduced_path_y, (1, -1)), [len(xs), 1])

        # dist_array = np.square(xs_tile - pathx_tile) + np.square(ys_tile - pathy_tile)
        # indexs = np.argmin(dist_array, 1) * ratio
        xs_array = float(xs) * np.ones_like(self.path[0])
        ys_array = float(ys) * np.ones_like(self.path[1])
        dist_array = np.square(xs_array - self.path[0]) + np.square(ys_array - self.path[1])
        indexs = np.argmin(dist_array,0)
        return indexs, self.indexs2points(indexs)

    def future_n_data(self, current_indexs, n):  # 用于在确定当前点后，找到接下来预测时域中的n=horizons个参考点
        future_data_list = []
        current_indexs = np.array(current_indexs, np.int32)
        for _ in range(n):
            current_indexs += 80
            current_indexs = np.where(current_indexs >= len(self.path[0]) - 5, len(self.path[0]) - 5, current_indexs)
            future_data_list.append(self.indexs2points(current_indexs))
        return future_data_list

    def indexs2points(self, indexs):
        indexs = np.where(indexs >= 0, indexs, 0)
        indexs = np.where(indexs < len(self.path[0]), indexs, len(self.path[0])-1)
        points = self.path[0][indexs], self.path[1][indexs], self.path[2][indexs]

        return points[0], points[1], points[2]

    def tracking_error_vector(self, ego_xs, ego_ys, ego_phis, ego_vs, n=0):
        def two2one(ref_xs, ref_ys):
            if self.task == 'left':
                # delta_ = np.sqrt(np.square(ego_xs - (-CROSSROAD_SIZE/2)) + np.square(ego_ys - (-CROSSROAD_SIZE/2))) - \
                #          np.sqrt(np.square(ref_xs - (-CROSSROAD_SIZE/2)) + np.square(ref_ys - (-CROSSROAD_SIZE/2)))
                delta_ = np.sqrt(np.square(ego_xs - ref_xs) + np.square(ego_ys - ref_ys))
                delta_ = np.where(ego_ys < -CROSSROAD_SIZE/2, ego_xs - ref_xs, delta_)
                delta_ = np.where(ego_xs < -CROSSROAD_SIZE/2, ego_ys - ref_ys, delta_)
                return -delta_

        indexs, current_points = self.find_closest_point(ego_xs, ego_ys)
        #n_future_data = self.future_n_data(indexs, n)

        tracking_error = np.stack([two2one(current_points[0], current_points[1]),
                                            deal_with_phi(ego_phis - current_points[2]),
                                            ego_vs - self.exp_v], 1)

        # final = tracking_error
        # if n > 0:
        #     future_points = np.concatenate([np.stack([ref_point[0] - ego_xs,
        #                                             ref_point[1] - ego_ys,
        #                                             deal_with_phi(ego_phis - ref_point[2])], 1)
        #                                 for ref_point in n_future_data], 1)
        #     final = np.concatenate([final, future_points], 1)

        return tracking_error