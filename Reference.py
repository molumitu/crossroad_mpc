import ctypes
from sys import path
ctypes.cdll.LoadLibrary(R"C:\ProgramData\Anaconda3\Lib\site-packages\bezier\extra-dll\bezier-b9fda8dc.dll")
import bezier
import matplotlib.pyplot as plt
import numpy as np
from Env_utils import CROSSROAD_SIZE, LANE_WIDTH, LANE_NUMBER, STEP_TIME, deal_with_phi

class ReferencePath(object):
    def __init__(self, task):
        self.exp_v = 6.
        self.task = task
        self.path_list = []
        self.path_len_list = []
        self._construct_ref_path(self.task)


    def _construct_ref_path(self, task):
        sl = 40  # straight line length, equal to extensions
        meter_pointnum_ratio = 100 
        end_points_num = int(2.5*sl * meter_pointnum_ratio) + 1
        arc_points_num = int((4222 +1) * 1.5) 
        # the length of the arc is 42.2152m
        R = CROSSROAD_SIZE/2 + LANE_WIDTH/2
        if task == 'left':
            end_offsets = [LANE_WIDTH*(i+0.5) for i in range(LANE_NUMBER)]
            start_offsets = [LANE_WIDTH*0.5]
            for end_offset in end_offsets:


                #---------start_straight_line---------------------------------------------------------
                start_points_num = int((sl+ end_offset- LANE_WIDTH*0.5) * meter_pointnum_ratio) + 1
                start_straight_line_x = LANE_WIDTH/2 * np.ones(shape=(start_points_num,))[:-1]
                start_straight_line_y = np.linspace(-CROSSROAD_SIZE/2 - sl, -CROSSROAD_SIZE/2 + end_offset- LANE_WIDTH*0.5 , start_points_num)[:-1]
                
                #---------connected_arc_line-----------------------------------------------------------
                s_vals = np.linspace(0, np.pi/2, arc_points_num)
                arc_line_x = R * np.cos(s_vals) - R + start_offsets
                arc_line_y = R * np.sin(s_vals) - R + end_offset

                #---------end_straight_line------------------------------------------------------------
                end_straight_line_x = np.linspace(-CROSSROAD_SIZE/2, -CROSSROAD_SIZE/2 - 2.5*sl, end_points_num,                                                     dtype=np.float32)[1:]
                end_straight_line_y = end_offset * np.ones(shape=(end_points_num,))[1:]

                #---------------------------------------------------------------------------------------

                total_x = np.concatenate((start_straight_line_x,arc_line_x,end_straight_line_x), axis=0)
                total_y = np.concatenate((start_straight_line_y,arc_line_y,end_straight_line_y), axis=0)
                
                xs_1, ys_1 = total_x[:-1], total_y[:-1]  #除去最后一个点之后，剩余的部分
                xs_2, ys_2 = total_x[1:], total_y[1:]    #除去第一个点之后，剩余的部分
                phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1) * 180 / np.pi

                planned_trj = total_x, total_y, np.concatenate((phis_1, phis_1[-1:]), axis=0)
            
                self.path_list.append(planned_trj)
                self.path_len_list.append(len(total_x))
        

            path_red_points_nums = int((sl- 20) * meter_pointnum_ratio) + 1
            path_red_line_x = LANE_WIDTH/2 * np.ones(shape=(path_red_points_nums,))
            path_red_line_y = np.linspace(-CROSSROAD_SIZE/2 - sl, -CROSSROAD_SIZE/2 - 20 , path_red_points_nums)

            a_brake = -2
            v0 = 6
            t = np.linspace(0,3,30*60)[1:]
            s_y = v0 * t + 1/2*a_brake*t**2 - CROSSROAD_SIZE/2 -20
            s_x = LANE_WIDTH/2 * np.ones(shape=(len(t),))

            total_len = path_red_points_nums + len(t)


            path_red_phi = 90 * np.ones(shape=(total_len,))
            path_red_x = np.concatenate((path_red_line_x, s_x))
            path_red_y = np.concatenate((path_red_line_y, s_y))
            path_red = path_red_x, path_red_y, path_red_phi
            self.path_list.append(path_red)
            self.path_len_list.append(total_len)
        
    def find_closest_point(self, xs, ys, path_index):  # radio用来将轨迹点稀疏化，但是不需要
        xs_array = float(xs) * np.ones_like(self.path_list[path_index][0])
        ys_array = float(ys) * np.ones_like(self.path_list[path_index][1])
        dist_array = np.square(xs_array - self.path_list[path_index][0]) + np.square(ys_array - self.path_list[path_index][1])
        indexs = np.argmin(dist_array,0)
        return indexs, self.indexs2points(indexs, path_index)

    def future_ref_points(self, ego_xs, ego_ys, n, path_index):  # 用于在确定当前点后，找到接下来预测时域中的n=horizons个参考点
        current_index, current_point = self.find_closest_point(ego_xs, ego_ys, path_index)
        future_ref_list = []

        # current_index = current_index - path_index * 30
        
        # # # #### 给单点， 作切线当作ref
        # next_x, next_y, next_phi = self.indexs2points(np.array(current_index + 80), path_index)
        # next_phi_rad = next_phi / 180. * np.pi
        # for _ in range(n):
        #     future_ref_list.append((next_x, next_y, next_phi))
        #     next_x +=  0.6* np.cos(next_phi_rad)
        #     next_y +=  0.6* np.sin(next_phi_rad)

        ## 给未来horizon个ref_points
        for _ in range(n):
            current_index = current_index + 60
            #current_index = np.where(current_index >= len(self.path_list[path_index][0]) - 1, len(self.path_list[path_index][0]) - 1, current_index)  # 避免仿真末尾报错
            future_ref_list.append(self.indexs2points(current_index, path_index))


        # # #### 给单点作为horizon
        # current_indexs = np.array(current_index+ 60 * 10)
        # for _ in range(n):
        #     current_indexs += 0
        #     current_indexs = np.where(current_indexs >= len(self.path_list[path_index][0]) - 1, len(self.path_list[path_index][0]) - 1, current_indexs)  # 避免仿真末尾报错
        #     future_ref_list.append(self.indexs2points(current_indexs, path_index))
        return current_point, future_ref_list


    def multi_future_ref_points(self, ego_xs, ego_ys, n):
        _, future_ref_list = self.future_ref_points(ego_xs, ego_ys, n, path_index = 0)
        _, future_ref_list_1 = self.future_ref_points(ego_xs, ego_ys, n, path_index = 1)
        _, future_ref_list_2 = self.future_ref_points(ego_xs, ego_ys, n, path_index = 2)
        _, future_ref_list_3 = self.future_ref_points(ego_xs, ego_ys, n, path_index = 3)
        return [future_ref_list, future_ref_list_1, future_ref_list_2, future_ref_list_3]


    def indexs2points(self, index, path_index):  #根据index 得到轨迹点的 x_ref, y_ref, phi_ref
        index = np.where(index >= 0, index, 0)
        index = np.where(index < len(self.path_list[path_index][0]), index, len(self.path_list[path_index][0])-1)  # 避免仿真末尾报错
        point = self.path_list[path_index][0][index], self.path_list[path_index][1][index], self.path_list[path_index][2][index]
        return point



if __name__ == "__main__":
    ref = ReferencePath('left')
    print(ref.path_len_list)
    print(ref.path_list[1][0])
    plt.scatter(ref.path_list[1][0], ref.path_list[1][1])
    plt.show()
    plt.axis('equal')
    print(ref.multi_future_ref_points(1.875, -34.856, 20)[0][0])
