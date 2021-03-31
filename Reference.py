import ctypes
from sys import path
import os 
print(os.environ['path'])
os.environ['path'] += os.pathsep + R"C:\ProgramData\Anaconda3\Lib\site-packages\bezier\extra-dll"

# ctypes.cdll.LoadLibrary(R"C:\ProgramData\Anaconda3\Lib\site-packages\bezier\extra-dll\bezier-b9fda8dc.dll")
import bezier
import matplotlib.pyplot as plt
import numpy as np
from Env_utils import CROSSROAD_SIZE, LANE_WIDTH, LANE_NUMBER, STEP_TIME
from enum import Enum

class ReferencePath(object):
    def __init__(self, routeID):
        self.exp_v = 6.
        self.routeID = routeID
        self.routes_num = self.cal_route_num()
        self.path_list = []
        self.path_len_list = []
        self._construct_ref_path()


    def cal_route_num(self):
        routes_ID_to_routes_num_dict = {'dl':3, 'rd':3, 'ur':3, 'lu':3,
                                        'du':3, 'rl':3, 'ud':3, 'lr':3,
                                        'dr':1, 'ru':1, 'ul':1, 'ld':1}
        return routes_ID_to_routes_num_dict[self.routeID]

    def _construct_ref_path(self):
        sl = 75  # straight line length, equal to extensions
        meter_pointnum_ratio = 100
        end_points_num = int(sl * meter_pointnum_ratio) + 1

        if self.routeID in ('dl', 'rd', 'ur', 'lu'):
 
            arc_points_num = int((4222 +1)) 
            # the length of the arc is 42.2152m
            R = CROSSROAD_SIZE/2 + LANE_WIDTH/2
            end_offsets = [LANE_WIDTH*(i+0.5) for i in range(LANE_NUMBER)]
            start_offsets = LANE_WIDTH*0.5
            for end_offset in end_offsets:
                #---------start_straight_line---------------------------------------------------------
                start_points_num = int((sl+ end_offset- LANE_WIDTH*0.5) * meter_pointnum_ratio) + 1
                start_straight_line_x = LANE_WIDTH/2 * np.ones(shape=(start_points_num,))[:-1]
                start_straight_line_y = np.linspace(-CROSSROAD_SIZE/2 - sl, -CROSSROAD_SIZE/2 + end_offset- LANE_WIDTH*0.5 , start_points_num)[:-1]
                
                #---------connected_arc_line-----------------------------------------------------------
                s_vals = np.linspace(0, np.pi/2, arc_points_num)
                arc_line_x = R * np.cos(s_vals) - CROSSROAD_SIZE/2
                arc_line_y = R * np.sin(s_vals) - CROSSROAD_SIZE/2 + end_offset - LANE_WIDTH*0.5

                #---------end_straight_line------------------------------------------------------------
                end_straight_line_x = np.linspace(-CROSSROAD_SIZE/2, -CROSSROAD_SIZE/2 - sl, end_points_num,                                                     dtype=np.float32)[1:]
                end_straight_line_y = end_offset * np.ones(shape=(end_points_num,))[1:]

                #---------------------------------------------------------------------------------------

                total_x = np.concatenate((start_straight_line_x,arc_line_x,end_straight_line_x), axis=0)
                total_y = np.concatenate((start_straight_line_y,arc_line_y,end_straight_line_y), axis=0)
                
                xs_1, ys_1 = total_x[:-1], total_y[:-1]  #除去最后一个点之后，剩余的部分
                xs_2, ys_2 = total_x[1:], total_y[1:]    #除去第一个点之后，剩余的部分
                phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1) ##弧度制
                planned_trj = total_x, total_y, np.concatenate((phis_1, phis_1[-1:]), axis=0)
                planned_trj = coordinate_trans(planned_trj, self.routeID)
                self.path_list.append(planned_trj)
                self.path_len_list.append(len(total_x))
        
        
            path_red_points_nums = int((sl- 12) * meter_pointnum_ratio) + 1
            path_red_line_x = LANE_WIDTH/2 * np.ones(shape=(path_red_points_nums,))
            path_red_line_y = np.linspace(-CROSSROAD_SIZE/2 - sl, -CROSSROAD_SIZE/2 - 12 , path_red_points_nums)

            a_brake = -2
            v0 = 6
            t = np.linspace(0,3,30*60)[1:] #这里60对应于下面current_index + 60
            s_y = v0 * t + 1/2*a_brake*t**2 - CROSSROAD_SIZE/2 -12
            s_x = LANE_WIDTH/2 * np.ones(shape=(len(t),))

            total_len_path_red = path_red_points_nums + len(t)
            path_red_phi = 90 * np.ones(shape=(total_len_path_red,))
            path_red_x = np.concatenate((path_red_line_x, s_x))
            path_red_y = np.concatenate((path_red_line_y, s_y))
            path_red = path_red_x, path_red_y, path_red_phi
            path_red = coordinate_trans(path_red, self.routeID)
            self.path_list.append(path_red)
            self.path_len_list.append(total_len_path_red)

        elif self.routeID in ('du', 'rl', 'ud', 'lr'):
 
            #---------start_straight_line---------------------------------------------------------
            start_points_num = int(sl * meter_pointnum_ratio) + 1
            start_straight_line_x = LANE_WIDTH * 1.5 * np.ones(shape=(start_points_num,))[:-1]
            start_straight_line_y = np.linspace(-CROSSROAD_SIZE/2 - sl, -CROSSROAD_SIZE/2 , start_points_num)[:-1]
            
            #---------connected_arc_line-----------------------------------------------------------
            connect_points_num = int(CROSSROAD_SIZE * meter_pointnum_ratio) + 1
            connect_line_x = LANE_WIDTH * 1.5 * np.ones(shape=(connect_points_num,))
            connect_line_y = np.linspace(-CROSSROAD_SIZE/2 , CROSSROAD_SIZE/2 , connect_points_num)

            #---------end_straight_line------------------------------------------------------------
            end_straight_line_x = LANE_WIDTH * 1.5 * np.ones(shape=(end_points_num,))[1:]
            end_straight_line_y = np.linspace(CROSSROAD_SIZE/2, CROSSROAD_SIZE/2 + sl, end_points_num)[1:]

            #---------------------------------------------------------------------------------------

            total_x = np.concatenate((start_straight_line_x,connect_line_x,end_straight_line_x), axis=0)
            total_y = np.concatenate((start_straight_line_y,connect_line_y,end_straight_line_y), axis=0)
            
            xs_1, ys_1 = total_x[:-1], total_y[:-1]  #除去最后一个点之后，剩余的部分
            xs_2, ys_2 = total_x[1:], total_y[1:]    #除去第一个点之后，剩余的部分
            phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1)

            planned_trj = total_x, total_y, np.concatenate((phis_1, phis_1[-1:]), axis=0)
            planned_trj = coordinate_trans(planned_trj, self.routeID)
            self.path_list.append(planned_trj)
            self.path_len_list.append(len(total_x))



            ##################右贝塞尔曲线#####################################################
            #---------start_straight_line---------------------------------------------------------
            start_points_num = int(sl * meter_pointnum_ratio) + 1
            start_straight_line_x = LANE_WIDTH * 1.5 * np.ones(shape=(start_points_num,))[:-1]
            start_straight_line_y = np.linspace(-CROSSROAD_SIZE/2 - sl, -CROSSROAD_SIZE/2 , start_points_num)[:-1]
            
            #---------connected_arc_line-----------------------------------------------------------
            connect_points_num = int(CROSSROAD_SIZE * meter_pointnum_ratio) + 1
            control_ext = 15
            offset = 5
            control_point1 = LANE_WIDTH * 1.5 , -CROSSROAD_SIZE/2 + offset
            control_point2 = LANE_WIDTH * 1.5, -CROSSROAD_SIZE/2 + control_ext+ offset
            control_point3 = LANE_WIDTH * 2.5,CROSSROAD_SIZE/2 - control_ext- offset
            control_point4 = LANE_WIDTH * 2.5,CROSSROAD_SIZE/2- offset

            node = np.asfortranarray([[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                                [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]])
            curve = bezier.Curve(node, degree=3)
            s_vals = np.linspace(0, 1.0, connect_points_num)
            trj_data = curve.evaluate_multi(s_vals)
            connect_line_x = trj_data[0]
            connect_line_y = trj_data[1]

            #---------end_straight_line------------------------------------------------------------
            end_straight_line_x = LANE_WIDTH * 2.5 * np.ones(shape=(end_points_num,))[1:]
            end_straight_line_y = np.linspace(CROSSROAD_SIZE/2, CROSSROAD_SIZE/2 + sl, end_points_num)[1:]

            #---------------------------------------------------------------------------------------

            total_x = np.concatenate((start_straight_line_x,connect_line_x,end_straight_line_x), axis=0)
            total_y = np.concatenate((start_straight_line_y,connect_line_y,end_straight_line_y), axis=0)
            
            xs_1, ys_1 = total_x[:-1], total_y[:-1]  #除去最后一个点之后，剩余的部分
            xs_2, ys_2 = total_x[1:], total_y[1:]    #除去第一个点之后，剩余的部分
            phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1)

            planned_trj = total_x, total_y, np.concatenate((phis_1, phis_1[-1:]), axis=0)
            planned_trj = coordinate_trans(planned_trj, self.routeID)
            self.path_list.append(planned_trj)
            self.path_len_list.append(len(total_x))



            ##################左贝塞尔曲线#####################################################
            #---------start_straight_line---------------------------------------------------------
            start_points_num = int(sl * meter_pointnum_ratio) + 1
            start_straight_line_x = LANE_WIDTH * 1.5 * np.ones(shape=(start_points_num,))[:-1]
            start_straight_line_y = np.linspace(-CROSSROAD_SIZE/2 - sl, -CROSSROAD_SIZE/2 , start_points_num)[:-1]
            
            #---------connected_arc_line-----------------------------------------------------------
            connect_points_num = int(CROSSROAD_SIZE * meter_pointnum_ratio) + 1
            control_ext = 15
            offset = 5
            control_point1 = LANE_WIDTH * 1.5 , -CROSSROAD_SIZE/2 + offset
            control_point2 = LANE_WIDTH * 1.5, -CROSSROAD_SIZE/2 + control_ext+ offset
            control_point3 = LANE_WIDTH * 0.5,CROSSROAD_SIZE/2 - control_ext- offset
            control_point4 = LANE_WIDTH * 0.5,CROSSROAD_SIZE/2- offset

            node = np.asfortranarray([[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                                [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]])
            curve = bezier.Curve(node, degree=3)
            s_vals = np.linspace(0, 1.0, connect_points_num)
            trj_data = curve.evaluate_multi(s_vals)
            connect_line_x = trj_data[0]
            connect_line_y = trj_data[1]

            #---------end_straight_line------------------------------------------------------------
            end_straight_line_x = LANE_WIDTH * 0.5 * np.ones(shape=(end_points_num,))[1:]
            end_straight_line_y = np.linspace(CROSSROAD_SIZE/2, CROSSROAD_SIZE/2 + sl, end_points_num)[1:]

            #---------------------------------------------------------------------------------------

            total_x = np.concatenate((start_straight_line_x,connect_line_x,end_straight_line_x), axis=0)
            total_y = np.concatenate((start_straight_line_y,connect_line_y,end_straight_line_y), axis=0)
            
            xs_1, ys_1 = total_x[:-1], total_y[:-1]  #除去最后一个点之后，剩余的部分
            xs_2, ys_2 = total_x[1:], total_y[1:]    #除去第一个点之后，剩余的部分
            phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1)

            planned_trj = total_x, total_y, np.concatenate((phis_1, phis_1[-1:]), axis=0)
            planned_trj = coordinate_trans(planned_trj, self.routeID)
            self.path_list.append(planned_trj)
            self.path_len_list.append(len(total_x))


            path_red_points_nums = int((sl- 12) * meter_pointnum_ratio) + 1
            path_red_line_x = LANE_WIDTH*1.5 * np.ones(shape=(path_red_points_nums,))
            path_red_line_y = np.linspace(-CROSSROAD_SIZE/2 - sl, -CROSSROAD_SIZE/2 - 12 , path_red_points_nums)

            a_brake = -2
            v0 = 6
            t = np.linspace(0,3,30*60)[1:]
            s_y = v0 * t + 1/2*a_brake*t**2 - CROSSROAD_SIZE/2 -12
            s_x = LANE_WIDTH * 1.5 * np.ones(shape=(len(t),))

            total_len = path_red_points_nums + len(t)

            path_red_phi = 90 * np.ones(shape=(total_len,))
            path_red_x = np.concatenate((path_red_line_x, s_x))
            path_red_y = np.concatenate((path_red_line_y, s_y))
            path_red = path_red_x, path_red_y, path_red_phi
            path_red = coordinate_trans(path_red, self.routeID)
            self.path_list.append(path_red)
            self.path_len_list.append(total_len)


        elif self.routeID in ('dr', 'ru', 'ul', 'ld'):  #右转

            arc_points_num = int((2454 +1)) 
            # the length of the arc is 24.5437m
            R = CROSSROAD_SIZE/2 - 2.5 * LANE_WIDTH
            end_offset = LANE_WIDTH*(-2.5)
            start_offset = LANE_WIDTH*2.5

            #---------start_straight_line---------------------------------------------------------
            start_points_num = int((sl) * meter_pointnum_ratio) + 1
            start_straight_line_x = LANE_WIDTH* 2.5 * np.ones(shape=(start_points_num,))[:-1]
            start_straight_line_y = np.linspace(-CROSSROAD_SIZE/2 - sl, -CROSSROAD_SIZE/2 , start_points_num)[:-1]
            
            #---------connected_arc_line-----------------------------------------------------------
            s_vals = np.linspace(0, np.pi/2, arc_points_num)
            arc_line_x = - R * np.cos(s_vals) + CROSSROAD_SIZE/2
            arc_line_y = R * np.sin(s_vals) - CROSSROAD_SIZE/2

            #---------end_straight_line------------------------------------------------------------
            end_straight_line_x = np.linspace(CROSSROAD_SIZE/2, CROSSROAD_SIZE/2 + sl, end_points_num,                                                     dtype=np.float32)[1:]
            end_straight_line_y = end_offset * np.ones(shape=(end_points_num,))[1:]

            #---------------------------------------------------------------------------------------

            total_x = np.concatenate((start_straight_line_x,arc_line_x,end_straight_line_x), axis=0)
            total_y = np.concatenate((start_straight_line_y,arc_line_y,end_straight_line_y), axis=0)
            
            xs_1, ys_1 = total_x[:-1], total_y[:-1]  #除去最后一个点之后，剩余的部分
            xs_2, ys_2 = total_x[1:], total_y[1:]    #除去第一个点之后，剩余的部分
            phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1)

            planned_trj = total_x, total_y, np.concatenate((phis_1, phis_1[-1:]), axis=0)
            planned_trj = coordinate_trans(planned_trj, self.routeID)
            self.path_list.append(planned_trj)
            self.path_len_list.append(len(total_x))

        
    def find_closest_point(self, xs, ys, path_index):
        xs_array = float(xs) * np.ones_like(self.path_list[path_index][0])
        ys_array = float(ys) * np.ones_like(self.path_list[path_index][1])
        dist_array = np.square(xs_array - self.path_list[path_index][0]) + np.square(ys_array - self.path_list[path_index][1])
        indexs = np.argmin(dist_array,0)
        return indexs, self.indexs2points(indexs, path_index)


    def future_ref_points(self, ego_xs, ego_ys, n, path_index):  # 用于在确定当前点后，找到接下来预测时域中的n=horizons个参考点
        current_index, current_point = self.find_closest_point(ego_xs, ego_ys, path_index)
        future_ref_list = []
        
        # #### 作切线当作ref
        # next_x, next_y, next_phi = self.indexs2points(np.array(current_index + 80), path_index)
        # next_phi_rad = next_phi
        # for _ in range(n):
        #     future_ref_list.append((next_x, next_y, next_phi))
        #     next_x +=  0.6* np.cos(next_phi_rad)
        #     next_y +=  0.6* np.sin(next_phi_rad)

        ##### 给未来horizon个ref_points
        for _ in range(n):
            current_index = current_index + 60
            future_ref_list.append(self.indexs2points(current_index, path_index))


        ###### 给单点作为horizon
        # current_indexs = np.array(current_index+ 60 * 10)
        # for _ in range(n):
        #     current_indexs += 0
        #     current_indexs = np.where(current_indexs >= len(self.path_list[path_index][0]) - 1, len(self.path_list[path_index][0]) - 1, current_indexs)  # 避免仿真末尾报错
        #     future_ref_list.append(self.indexs2points(current_indexs, path_index))
        return current_point, future_ref_list


    def multi_future_ref_points(self, ego_xs, ego_ys, n):
        multi_future_ref_list = []
        for i in range(self.routes_num):
            _, future_ref_list = self.future_ref_points(ego_xs, ego_ys, n, path_index = i)
            multi_future_ref_list.append(future_ref_list)
        return multi_future_ref_list


    def indexs2points(self, index, path_index):  #根据index 得到轨迹点的 x_ref, y_ref, phi_ref
        index = np.where(index >= 0, index, 0)
        index = np.where(index < len(self.path_list[path_index][0]), index, len(self.path_list[path_index][0])-1)  # 避免仿真末尾报错
        point = self.path_list[path_index][0][index], self.path_list[path_index][1][index], self.path_list[path_index][2][index]
        return point


class routeID(Enum):
    dl = 'dl'
    rd = 'rd'
    ur = 'ur'
    lu = 'lu'
    du = 'du'
    rl = 'rl'
    ud = 'ud'
    lr = 'lr'
    dr = 'dr'
    ru = 'ru'
    ul = 'ul'
    ld = 'ld'

def coordinate_trans(path,route_ID):
    rotate_dict = {'dl':0, 'rd':np.pi/2, 'ur':np.pi, 'lu':np.pi*3/2,     # 定义逆时针方向为正
                    'du':0, 'rl':np.pi/2, 'ud':np.pi, 'lr':np.pi*3/2,
                    'dr':0, 'ru':np.pi/2, 'ul':np.pi, 'ld':np.pi*3/2}
    x, y, phi = path
    theta = rotate_dict[route_ID]
    x_trans = x*np.cos(theta) - y*np.sin(theta)
    y_trans = x*np.sin(theta) + y*np.cos(theta)
    phi_trans = phi + theta
    return x_trans, y_trans, phi_trans 
    
if __name__ == "__main__":
    ref = ReferencePath('du')
    #print(ref.path_len_list)
    #print(ref.path_list[1][0])
    plt.plot(ref.path_list[2][0], ref.path_list[2][1])

    # plt.xlim([4,10])
    # plt.ylim([-30,30])
    x_r, y_r, phi_r = coordinate_trans(ref.path_list[2],'rl')
    x_u, y_u, phi_u = coordinate_trans(ref.path_list[2],'ud')
    x_l, y_l, phi_l = coordinate_trans(ref.path_list[2],'lr')
    plt.plot(x_r, y_r)
    plt.plot(x_u, y_u)
    plt.plot(x_l, y_l)
    plt.show()
    #print(ref.multi_future_ref_points(1.875, -34.856, 20)[0][0])

                                                                                                                
