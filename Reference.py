import os 
os.environ['path'] += os.pathsep + R"C:\ProgramData\Anaconda3\Lib\site-packages\bezier\extra-dll"
import bezier
import matplotlib.pyplot as plt
import numpy as np
from Utils import CROSSROAD_SIZE, LANE_WIDTH, LANE_NUMBER, STEP_TIME
class ReferencePath(object):
    extension = 75
    meter_pointnum_ratio = 100
    stop_dist = 12
    def __init__(self, routeID):
        self.exp_v = 6.
        self.routeID = routeID
        self.routes_num = self.cal_route_num()
        self.path_list = []
        self.path_len_list = []
        self._construct_ref_path()
        self._construct_red_ref_path()

    def cal_route_num(self):
        routes_ID_to_routes_num_dict = {'dl':4, 'rd':4, 'ur':4, 'lu':4,
                                        'du':4, 'rl':4, 'ud':4, 'lr':4,
                                        'dr':2, 'ru':2, 'ul':2, 'ld':2}
        return routes_ID_to_routes_num_dict[self.routeID]

    def _construct_red_ref_path(self):
        red_path_equal_nums = int((self.extension- self.stop_dist) * self.meter_pointnum_ratio) + 1  #匀速路段
        a_brake = -2
        brake_time = 3
        exp_v = 6
        t_brake_nums = np.linspace(0,brake_time,brake_time * exp_v * self.meter_pointnum_ratio)[1:]

        if self.routeID in ('dl', 'rd', 'ur', 'lu'):
            red_x = LANE_WIDTH * 0.5
        elif self.routeID in ('du', 'rl', 'ud', 'lr'):
            red_x = LANE_WIDTH * 1.5
        elif self.routeID in ('dr', 'ru', 'ul', 'ld'):
            red_x = LANE_WIDTH * 2.5
        else:
            return

        red_path_equal_x = red_x * np.ones(shape=(red_path_equal_nums,))
        red_path_equal_y = np.linspace(-CROSSROAD_SIZE/2 - self.extension, -CROSSROAD_SIZE/2 - self.stop_dist , red_path_equal_nums)
        red_path_brake_y = exp_v * t_brake_nums + 1/2*a_brake*t_brake_nums**2 - CROSSROAD_SIZE/2 - self.stop_dist
        red_path_brake_x = red_x * np.ones(shape=(len(t_brake_nums),))
        total_len_red_path = red_path_equal_nums + len(t_brake_nums)
        red_path_phi = 90 * np.ones(shape=(total_len_red_path,))
        red_path_x = np.concatenate((red_path_equal_x, red_path_brake_x))
        red_path_y = np.concatenate((red_path_equal_y, red_path_brake_y))
        red_path = coordinate_trans((red_path_x, red_path_y, red_path_phi), self.routeID)
        self.path_list.append(red_path)
        self.path_len_list.append(total_len_red_path)

    def _construct_ref_path(self):
        end_points_num = int(self.extension * self.meter_pointnum_ratio) + 1


        if self.routeID in ('dl', 'rd', 'ur', 'lu'):
 
            arc_points_num = int((4222 +1)) 
            # the length of the arc is 42.2152m
            R = CROSSROAD_SIZE/2 + 0.5 * LANE_WIDTH
            for i in range(LANE_NUMBER):
                #---------start_straight_line---------------------------------------------------------
                start_points_num = int((self.extension+ i * LANE_WIDTH ) * self.meter_pointnum_ratio) + 1
                start_straight_line_x = 0.5 * LANE_WIDTH * np.ones((start_points_num,))[:-1]
                start_straight_line_y = np.linspace(-CROSSROAD_SIZE/2 - self.extension, -CROSSROAD_SIZE/2 + i*LANE_WIDTH , start_points_num)[:-1]
                
                #---------connected_arc_line-----------------------------------------------------------
                s_vals = np.linspace(0, np.pi/2, int(arc_points_num))
                arc_line_x = R * np.cos(s_vals) - CROSSROAD_SIZE/2
                arc_line_y = R * np.sin(s_vals) - CROSSROAD_SIZE/2 + i * LANE_WIDTH

                #---------end_straight_line------------------------------------------------------------
                end_straight_line_x = np.linspace(-CROSSROAD_SIZE/2, -CROSSROAD_SIZE/2 - self.extension, end_points_num)[1:]
                end_straight_line_y = ((i + 0.5) * LANE_WIDTH) * np.ones((end_points_num,))[1:]

                #---------------------------------------------------------------------------------------
                total_x = np.concatenate((start_straight_line_x,arc_line_x,end_straight_line_x), axis=0)
                total_y = np.concatenate((start_straight_line_y,arc_line_y,end_straight_line_y), axis=0)
                
                xs_1, ys_1 = total_x[:-1], total_y[:-1]  #除去最后一个点之后，剩余的部分
                xs_2, ys_2 = total_x[1:], total_y[1:]    #除去第一个点之后，剩余的部分
                phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1) ##弧度制
                total_phi = np.concatenate((phis_1, phis_1[-1:]), axis=0)
                planned_path = coordinate_trans((total_x, total_y, total_phi), self.routeID)
                self.path_list.append(planned_path)
                self.path_len_list.append(len(total_x))
        
        elif self.routeID in ('du', 'rl', 'ud', 'lr'):
            #---------straight_line-----------------------------------------------------------
            total_nums = int((self.extension * 2 + CROSSROAD_SIZE) * self.meter_pointnum_ratio) + 1
            total_x = LANE_WIDTH * (1.5) * np.ones(shape=(total_nums,))
            total_y = np.linspace(-CROSSROAD_SIZE/2 - self.extension , CROSSROAD_SIZE/2 + self.extension, total_nums)
       
            xs_1, ys_1 = total_x[:-1], total_y[:-1]
            xs_2, ys_2 = total_x[1:], total_y[1:]
            phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1)
            total_phi = np.concatenate((phis_1, phis_1[-1:]), axis=0)
            planned_path = coordinate_trans((total_x, total_y, total_phi), self.routeID)
            self.path_list.append(planned_path)
            self.path_len_list.append(len(total_x))
 
            #---------start_straight_line---------------------------------------------------------
            start_points_num = int(self.extension * self.meter_pointnum_ratio) + 1
            start_straight_line_x = LANE_WIDTH * 1.5 * np.ones(shape=(start_points_num,))[:-1]
            start_straight_line_y = np.linspace(-CROSSROAD_SIZE/2 - self.extension, -CROSSROAD_SIZE/2 , start_points_num)[:-1]
            
            for i in [0,2]:
                #---------connected_bezier_line-----------------------------------------------------------
                connect_points_num = int(CROSSROAD_SIZE * self.meter_pointnum_ratio) + 1
                control_ext = 15
                control_point1 = LANE_WIDTH * 1.5 , -CROSSROAD_SIZE/2
                control_point2 = LANE_WIDTH * 1.5 , -CROSSROAD_SIZE/2 + control_ext
                control_point3 = LANE_WIDTH * (i+0.5), CROSSROAD_SIZE/2 - control_ext
                control_point4 = LANE_WIDTH * (i+0.5), CROSSROAD_SIZE/2

                node = np.asfortranarray([[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                                    [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]])
                curve = bezier.Curve(node, degree=3)
                connect_points_num = int(curve.length * self.meter_pointnum_ratio) + 1
                s_vals = np.linspace(0, 1.0, connect_points_num)
                #curve.length 50.16379702950708m
                trj_data = curve.evaluate_multi(s_vals)
                connect_line_x = trj_data[0]
                connect_line_y = trj_data[1]

                #---------end_straight_line------------------------------------------------------------
                end_straight_line_x = LANE_WIDTH * (i+0.5) * np.ones(shape=(end_points_num,))[1:]
                end_straight_line_y = np.linspace(CROSSROAD_SIZE/2, CROSSROAD_SIZE/2 + self.extension, end_points_num)[1:]

                #---------------------------------------------------------------------------------------

                total_x = np.concatenate((start_straight_line_x,connect_line_x,end_straight_line_x), axis=0)
                total_y = np.concatenate((start_straight_line_y,connect_line_y,end_straight_line_y), axis=0)
                
                xs_1, ys_1 = total_x[:-1], total_y[:-1]
                xs_2, ys_2 = total_x[1:], total_y[1:]
                phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1)
                total_phi = np.concatenate((phis_1, phis_1[-1:]), axis=0)
                planned_path = coordinate_trans((total_x, total_y, total_phi), self.routeID)
                self.path_list.append(planned_path)
                self.path_len_list.append(len(total_x))

        elif self.routeID in ('dr', 'ru', 'ul', 'ld'):  #右转

            arc_points_num = int((2454 +1)) 
            # the length of the arc is 24.5437m
            R = CROSSROAD_SIZE/2 - 2.5 * LANE_WIDTH
            end_offset = LANE_WIDTH*-2.5
            start_offset = LANE_WIDTH*2.5

            #---------start_straight_line---------------------------------------------------------
            start_points_num = int((self.extension) * self.meter_pointnum_ratio) + 1
            start_straight_line_x = LANE_WIDTH* 2.5 * np.ones(shape=(start_points_num,))[:-1]
            start_straight_line_y = np.linspace(-CROSSROAD_SIZE/2 - self.extension, -CROSSROAD_SIZE/2 , start_points_num)[:-1]
            
            #---------connected_arc_line-----------------------------------------------------------
            s_vals = np.linspace(0, np.pi/2, arc_points_num)
            arc_line_x = - R * np.cos(s_vals) + CROSSROAD_SIZE/2
            arc_line_y = R * np.sin(s_vals) - CROSSROAD_SIZE/2

            #---------end_straight_line------------------------------------------------------------
            end_straight_line_x = np.linspace(CROSSROAD_SIZE/2, CROSSROAD_SIZE/2 + self.extension, end_points_num,                                                     dtype=np.float32)[1:]
            end_straight_line_y = end_offset * np.ones(shape=(end_points_num,))[1:]

            #---------------------------------------------------------------------------------------
            total_x = np.concatenate((start_straight_line_x,arc_line_x,end_straight_line_x), axis=0)
            total_y = np.concatenate((start_straight_line_y,arc_line_y,end_straight_line_y), axis=0)
            
            xs_1, ys_1 = total_x[:-1], total_y[:-1]
            xs_2, ys_2 = total_x[1:], total_y[1:]
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
        # for _ in range(n):
        #     future_ref_list.append((next_x, next_y, next_phi))
        #     next_x +=  0.6* np.cos(next_phi)
        #     next_y +=  0.6* np.sin(next_phi)

        ##### 给未来horizon个ref_points
        for _ in range(n):
            current_index = current_index + 60
            future_ref_list.append(self.indexs2points(current_index, path_index))

        # ##### 给单点作为horizon
        # current_indexs = np.array(current_index+ 60 * 8)
        # for _ in range(n):
        #     current_indexs += 0
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

def coordinate_trans(path,route_ID):
    rotate_dict = {'dl':0, 'rd':np.pi/2, 'ur':np.pi, 'lu':np.pi*3/2,     # 定义逆时针方向为正
                    'du':0, 'rl':np.pi/2, 'ud':np.pi, 'lr':np.pi*3/2,
                    'dr':0, 'ru':np.pi/2, 'ul':np.pi, 'ld':np.pi*3/2}
    x, y, phi = path
    theta = rotate_dict[route_ID]
    x_trans = x * np.cos(theta) - y * np.sin(theta)
    y_trans = x * np.sin(theta) + y * np.cos(theta)
    phi_trans = phi + theta
    return x_trans, y_trans, phi_trans 
    
if __name__ == "__main__":
    ref1 = ReferencePath('du') #直行
    ref2 = ReferencePath('dl') #左转
    ref3 = ReferencePath('dr') #右转



    plt.plot(ref2.path_list[0][0],ref2.path_list[0][1])
    plt.plot(ref2.path_list[1][0],ref2.path_list[1][1])
    plt.plot(ref2.path_list[2][0],ref2.path_list[2][1])
    plt.axis('equal')
    plt.show()

    # plt.plot(ref1.path_list[0][0], ref1.path_list[0][1])
    # plt.plot(ref1.path_list[1][0], ref1.path_list[1][1])
    # plt.plot(ref1.path_list[2][0], ref1.path_list[2][1])
    # plt.plot(ref2.path_list[0][0], ref2.path_list[0][1])
    # plt.plot(ref2.path_list[1][0], ref2.path_list[1][1])
    # plt.plot(ref2.path_list[2][0], ref2.path_list[2][1])
    # plt.plot(ref3.path_list[0][0], ref3.path_list[0][1])

    # x_r10, y_r10, phi_r10 = coordinate_trans(ref1.path_list[0],'rl')
    # x_u10, y_u10, phi_u10 = coordinate_trans(ref1.path_list[0],'ud')
    # x_l10, y_l10, phi_l10 = coordinate_trans(ref1.path_list[0],'lr')
    # x_r11, y_r11, phi_r11 = coordinate_trans(ref1.path_list[1],'rl')
    # x_u11, y_u11, phi_u11 = coordinate_trans(ref1.path_list[1],'ud')
    # x_l11, y_l11, phi_l11 = coordinate_trans(ref1.path_list[1],'lr')
    # x_r12, y_r12, phi_r12 = coordinate_trans(ref1.path_list[2],'rl')
    # x_u12, y_u12, phi_u12 = coordinate_trans(ref1.path_list[2],'ud')
    # x_l12, y_l12, phi_l12 = coordinate_trans(ref1.path_list[2],'lr')


    # x_r20, y_r20, phi_r20 = coordinate_trans(ref2.path_list[0],'rd')
    # x_u20, y_u20, phi_u20 = coordinate_trans(ref2.path_list[0],'ur')
    # x_l20, y_l20, phi_l20 = coordinate_trans(ref2.path_list[0],'lu')
    # x_r21, y_r21, phi_r21 = coordinate_trans(ref2.path_list[1],'rd')
    # x_u21, y_u21, phi_u21 = coordinate_trans(ref2.path_list[1],'ur')
    # x_l21, y_l21, phi_l21 = coordinate_trans(ref2.path_list[1],'lu')
    # x_r22, y_r22, phi_r22 = coordinate_trans(ref2.path_list[2],'rd')
    # x_u22, y_u22, phi_u22 = coordinate_trans(ref2.path_list[2],'ur')
    # x_l22, y_l22, phi_l22 = coordinate_trans(ref2.path_list[2],'lu')
    # x_r3, y_r3, phi_r3 = coordinate_trans(ref3.path_list[0],'ru')
    # x_u3, y_u3, phi_u3 = coordinate_trans(ref3.path_list[0],'ul')
    # x_l3, y_l3, phi_l3 = coordinate_trans(ref3.path_list[0],'ld')
    # plt.plot(x_r10, y_r10)
    # plt.plot(x_u10, y_u10)
    # plt.plot(x_l10, y_l10)
    # plt.plot(x_r20, y_r20)
    # plt.plot(x_u20, y_u20)
    # plt.plot(x_l20, y_l20)

    # plt.plot(x_r11, y_r11)
    # plt.plot(x_u11, y_u11)
    # plt.plot(x_l11, y_l11)
    # plt.plot(x_r21, y_r21)
    # plt.plot(x_u21, y_u21)
    # plt.plot(x_l21, y_l21)

    # plt.plot(x_r12, y_r12)
    # plt.plot(x_u12, y_u12)
    # plt.plot(x_l12, y_l12)
    # plt.plot(x_r22, y_r22)
    # plt.plot(x_u22, y_u22)
    # plt.plot(x_l22, y_l22)
    # plt.plot(x_r3, y_r3)
    # plt.plot(x_u3, y_u3)
    # plt.plot(x_l3, y_l3)


    # plot red line----------------------------------------------------------
    # plt.plot(ref1.path_list[-1][0], ref1.path_list[-1][1])
    # x_r14, y_r14, phi_r14 = coordinate_trans(ref1.path_list[-1],'rl')
    # x_u14, y_u14, phi_u14 = coordinate_trans(ref1.path_list[-1],'ud')
    # x_l14, y_l14, phi_l14 = coordinate_trans(ref1.path_list[-1],'lr')
    # x_r24, y_r24, phi_r24 = coordinate_trans(ref2.path_list[-1],'rd')
    # x_u24, y_u24, phi_u24 = coordinate_trans(ref2.path_list[-1],'ur')
    # x_l24, y_l24, phi_l24 = coordinate_trans(ref2.path_list[-1],'lu')
    # plt.plot(x_r14, y_r14)
    # plt.plot(x_u14, y_u14)
    # plt.plot(x_l14, y_l14)
    # plt.plot(x_r24, y_r24)
    # plt.plot(x_u24, y_u24)
    # plt.plot(x_l24, y_l24)
    # plt.plot(ref2.path_list[-1][0], ref2.path_list[-1][1])
    # plt.axis('equal')
    # plt.show()

                                                                                                                
