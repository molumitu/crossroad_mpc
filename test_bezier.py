import ctypes
import matplotlib.pyplot as plt
ctypes.cdll.LoadLibrary(R"C:\ProgramData\Anaconda3\Lib\site-packages\bezier\extra-dll\bezier-b9fda8dc.dll")
import bezier
LANE_WIDTH = 3.75
CROSSROAD_SIZE = 50
LANE_NUMBER = 3
control_ext = 10
extension = 75
meter_pointnum_ratio = 100
import numpy as np

end_offsets = [LANE_WIDTH*(i+0.5) for i in range(LANE_NUMBER)]
start_offsets = [LANE_WIDTH*0.5]
for start_offset in start_offsets:
    for end_offset in end_offsets:
        control_point1 = start_offset, -CROSSROAD_SIZE/2
        control_point2 = start_offset, -CROSSROAD_SIZE/2 + control_ext
        control_point3 = -CROSSROAD_SIZE/2 + control_ext, end_offset
        control_point4 = -CROSSROAD_SIZE/2, end_offset

        node = np.asfortranarray([[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                                    [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]],
                                    dtype=np.float32)
        curve = bezier.Curve(node, degree=3)
        s_vals = np.linspace(0, 1.0, int(np.pi/2*(CROSSROAD_SIZE/2+LANE_WIDTH/2)) * self.meter_pointnum_ratio)
        trj_data = curve.evaluate_multi(s_vals)


control_point1 = LANE_WIDTH * 1.5, -CROSSROAD_SIZE/2
control_point2 = LANE_WIDTH * 1.5, -CROSSROAD_SIZE/2 + control_ext
control_point3 = LANE_WIDTH * 2.5,CROSSROAD_SIZE/2 - control_ext
control_point4 = LANE_WIDTH * 2.5,CROSSROAD_SIZE/2

node = np.asfortranarray([[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                    [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]])
curve = bezier.Curve(node, degree=3)
s_vals = np.linspace(0, 1.0, int(extension * meter_pointnum_ratio) + 1)
trj_data = curve.evaluate_multi(s_vals)


if __name__ == "__main__":
    print(trj_data)
    plt.plot(trj_data[0], trj_data[1])
    plt.show()
    plt.axis('equal')