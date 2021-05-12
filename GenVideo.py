from animation import ExpAnimation
import os

data_path = R'C:\Users\zgj_t\Desktop\crossroad_mpc\visualization'



def Gen_Video(test_name, time_name, time_length):
    AniTool = ExpAnimation()
    load_data_path = os.path.join(data_path, test_name, time_name)
    if not os.path.exists(load_data_path):
        os.makedirs(load_data_path)

    save_video_path = os.path.join(data_path, test_name, time_name)
    if not os.path.exists(save_video_path):
        os.makedirs(save_video_path)
    AniTool.load_exp_data(load_data_path)
    AniTool.animation_output(save_video_path, timeline=(0, time_length), colorTrack=True)
    # AniTool.car_state_plot(os.path.join(data_path, exp_name), time_line=(0, 315))


if __name__ == "__main__":
    time_name = '2021_04_16_14_57_39'
    Gen_Video(time_name)
