from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle as Rt
from matplotlib.colors import Normalize
from matplotlib import cm
from tqdm import trange
import numpy as np
import json
import traci
import pickle
import os
from Utils import CROSSROAD_SIZE, LANE_NUMBER, LANE_WIDTH
from Reference import ReferencePath


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

##----------------------------------静态路径规划----------------------------------------------
ref1 = ReferencePath('du', extension =40) #直行
ref2 = ReferencePath('dl', extension =40) #左转
ref3 = ReferencePath('dr', extension =40) #右转

##----------------------左转---------------------------------------------------------------##
x_r20, y_r20, phi_r20 = coordinate_trans(ref2.path_list[0],'rd')
x_u20, y_u20, phi_u20 = coordinate_trans(ref2.path_list[0],'ur')
x_l20, y_l20, phi_l20 = coordinate_trans(ref2.path_list[0],'lu')

x_r21, y_r21, phi_r21 = coordinate_trans(ref2.path_list[1],'rd')
x_u21, y_u21, phi_u21 = coordinate_trans(ref2.path_list[1],'ur')
x_l21, y_l21, phi_l21 = coordinate_trans(ref2.path_list[1],'lu')

x_r22, y_r22, phi_r22 = coordinate_trans(ref2.path_list[2],'rd')
x_u22, y_u22, phi_u22 = coordinate_trans(ref2.path_list[2],'ur')
x_l22, y_l22, phi_l22 = coordinate_trans(ref2.path_list[2],'lu')


##----------------------直行---------------------------------------------------------------##
x_r10, y_r10, phi_r10 = coordinate_trans(ref1.path_list[0],'rl')
x_u10, y_u10, phi_u10 = coordinate_trans(ref1.path_list[0],'ud')
x_l10, y_l10, phi_l10 = coordinate_trans(ref1.path_list[0],'lr')

x_r11, y_r11, phi_r11 = coordinate_trans(ref1.path_list[1],'rl')
x_u11, y_u11, phi_u11 = coordinate_trans(ref1.path_list[1],'ud')
x_l11, y_l11, phi_l11 = coordinate_trans(ref1.path_list[1],'lr')

x_r12, y_r12, phi_r12 = coordinate_trans(ref1.path_list[2],'rl')
x_u12, y_u12, phi_u12 = coordinate_trans(ref1.path_list[2],'ud')
x_l12, y_l12, phi_l12 = coordinate_trans(ref1.path_list[2],'lr')

##----------------------右转---------------------------------------------------------------##
x_r3, y_r3, phi_r3 = coordinate_trans(ref3.path_list[0],'ru')
x_u3, y_u3, phi_u3 = coordinate_trans(ref3.path_list[0],'ul')
x_l3, y_l3, phi_l3 = coordinate_trans(ref3.path_list[0],'ld')






def static_view():
    '''
        create static view of simulation scene
    '''
    # plot basic map
    extension = 40

    lane_edge_width = 1.5
    dotted_line_style = '--'
    solid_line_style = '-'

    fig = plt.figure(figsize=(8,8), dpi=100)
    ax = fig.add_axes([0.05,0.05,0.9,0.9])
    # ax.set_title("Intersection")
    ax.set_xlim([-CROSSROAD_SIZE / 2 - extension, CROSSROAD_SIZE / 2 + extension])
    ax.set_ylim([-CROSSROAD_SIZE / 2 - extension, CROSSROAD_SIZE / 2 + extension])
    ax.axis("equal")
    ax.axis('off')

    # ---------rectangle edge------------ 
    # ax.add_patch(plt.Rectangle((-CROSSROAD_SIZE / 2 - extension, -CROSSROAD_SIZE / 2 - extension),
    #                             CROSSROAD_SIZE + 2 * extension, CROSSROAD_SIZE + 2 * extension, edgecolor='black',
    #                             facecolor='none'))

    # ----------arrow--------------------
    ax.arrow(LANE_WIDTH/2, -CROSSROAD_SIZE / 2 - 10, 0, 5, color='gray')
    ax.arrow(LANE_WIDTH/2, -CROSSROAD_SIZE / 2 - 10 +5, -0.5, 0, color='gray', head_width=1)
    ax.arrow(LANE_WIDTH*1.5, -CROSSROAD_SIZE / 2 - 10, 0, 5, color='gray', head_width=1)
    ax.arrow(LANE_WIDTH*2.5, -CROSSROAD_SIZE / 2 - 10, 0, 5, color='gray')
    ax.arrow(LANE_WIDTH*2.5, -CROSSROAD_SIZE / 2 - 10 + 5, 0.5, 0, color='gray', head_width=1)

    # ----------horizon--------------
    ax.plot([-CROSSROAD_SIZE / 2 - extension, -CROSSROAD_SIZE / 2], [0, 0], color='orange', linewidth = lane_edge_width)
    ax.plot([CROSSROAD_SIZE / 2 + extension, CROSSROAD_SIZE / 2], [0, 0], color='orange', linewidth = lane_edge_width)

    #
    for i in range(1, LANE_NUMBER + 1):
        linestyle = dotted_line_style if i < LANE_NUMBER else solid_line_style
        ax.plot([-CROSSROAD_SIZE / 2 - extension, -CROSSROAD_SIZE / 2], [i * LANE_WIDTH, i * LANE_WIDTH],
                    linestyle=linestyle, color='black', linewidth = lane_edge_width)
        ax.plot([CROSSROAD_SIZE / 2 + extension, CROSSROAD_SIZE / 2], [i * LANE_WIDTH, i * LANE_WIDTH],
                    linestyle=linestyle, color='black', linewidth = lane_edge_width)
        ax.plot([-CROSSROAD_SIZE / 2 - extension, -CROSSROAD_SIZE / 2], [-i * LANE_WIDTH, -i * LANE_WIDTH],
                    linestyle=linestyle, color='black', linewidth = lane_edge_width)
        ax.plot([CROSSROAD_SIZE / 2 + extension, CROSSROAD_SIZE / 2], [-i * LANE_WIDTH, -i * LANE_WIDTH],
                    linestyle=linestyle, color='black', linewidth = lane_edge_width)

    # ----------vertical----------------
    ax.plot([0, 0], [-CROSSROAD_SIZE / 2 - extension, -CROSSROAD_SIZE / 2], color='orange', linewidth = lane_edge_width)
    ax.plot([0, 0], [CROSSROAD_SIZE / 2 + extension, CROSSROAD_SIZE / 2], color='orange', linewidth = lane_edge_width)

    #
    for i in range(1, LANE_NUMBER + 1):
        linestyle = dotted_line_style if i < LANE_NUMBER else solid_line_style
        ax.plot([i * LANE_WIDTH, i * LANE_WIDTH], [-CROSSROAD_SIZE / 2 - extension, -CROSSROAD_SIZE / 2],
                    linestyle=linestyle, color='black', linewidth = lane_edge_width)
        ax.plot([i * LANE_WIDTH, i * LANE_WIDTH], [CROSSROAD_SIZE / 2 + extension, CROSSROAD_SIZE / 2],
                    linestyle=linestyle, color='black', linewidth = lane_edge_width)
        ax.plot([-i * LANE_WIDTH, -i * LANE_WIDTH], [-CROSSROAD_SIZE / 2 - extension, -CROSSROAD_SIZE / 2],
                    linestyle=linestyle, color='black', linewidth = lane_edge_width)
        ax.plot([-i * LANE_WIDTH, -i * LANE_WIDTH], [CROSSROAD_SIZE / 2 + extension, CROSSROAD_SIZE / 2],
                    linestyle=linestyle, color='black', linewidth = lane_edge_width)

    # # ----------stop line--------------
    # ax.plot([0, 2 * LANE_WIDTH], [-CROSSROAD_SIZE / 2, -CROSSROAD_SIZE / 2],
    #          color='black')
    # ax.plot([-2 * LANE_WIDTH, 0], [CROSSROAD_SIZE / 2, CROSSROAD_SIZE / 2],
    #          color='black')
    # ax.plot([-CROSSROAD_SIZE / 2, -CROSSROAD_SIZE / 2], [0, -2 * LANE_WIDTH],
    #          color='black')
    # ax.plot([CROSSROAD_SIZE / 2, CROSSROAD_SIZE / 2], [2 * LANE_WIDTH, 0],
    #          color='black')



    ##----------connection--------------
    ax.plot([LANE_NUMBER * LANE_WIDTH, CROSSROAD_SIZE / 2], [-CROSSROAD_SIZE / 2, -LANE_NUMBER * LANE_WIDTH],
                color='black', linewidth = lane_edge_width)
    ax.plot([LANE_NUMBER * LANE_WIDTH, CROSSROAD_SIZE / 2], [CROSSROAD_SIZE / 2, LANE_NUMBER * LANE_WIDTH],
                color='black', linewidth = lane_edge_width)
    ax.plot([-LANE_NUMBER * LANE_WIDTH, -CROSSROAD_SIZE / 2], [-CROSSROAD_SIZE / 2, -LANE_NUMBER * LANE_WIDTH],
                color='black', linewidth = lane_edge_width)
    ax.plot([-LANE_NUMBER * LANE_WIDTH, -CROSSROAD_SIZE / 2], [CROSSROAD_SIZE / 2, LANE_NUMBER * LANE_WIDTH],
                color='black', linewidth = lane_edge_width)

    # from matplotlib.patches import Arc
    # Radius = CROSSROAD_SIZE - 2 * LANE_NUMBER * LANE_WIDTH
    # quarter_circle_1 = Arc((25,-25),Radius,Radius,90,0,90, linewidth = lane_edge_width)
    # quarter_circle_2 = Arc((25,25),Radius,Radius,180,0,90, linewidth = lane_edge_width)
    # quarter_circle_3 = Arc((-25,25),Radius,Radius,270,0,90, linewidth = lane_edge_width)
    # quarter_circle_4 = Arc((-25,-25),Radius,Radius,360,0,90, linewidth = lane_edge_width)
    # ax.add_patch(quarter_circle_1)
    # ax.add_patch(quarter_circle_2)
    # ax.add_patch(quarter_circle_3)
    # ax.add_patch(quarter_circle_4)
    return fig, ax

def update_light(ax, traffic_light):
    light_line_width = 2
    if traffic_light == 0:
        v_color, h_color = 'green', 'red'
    elif traffic_light == 1:
        v_color, h_color = 'orange', 'red'
    elif traffic_light == 2:
        v_color, h_color = 'red', 'green'
    elif traffic_light == 3:
        v_color, h_color = 'red', 'orange'
    elif traffic_light == 4:
        v_color, h_color = 'green', 'green'

    Lines = []


    # top vertical
    Lines.append(ax.plot([0, (LANE_NUMBER-1)*LANE_WIDTH], [-CROSSROAD_SIZE / 2, -CROSSROAD_SIZE / 2],
                color=v_color, linewidth=light_line_width))
    Lines.append(ax.plot([(LANE_NUMBER-1)*LANE_WIDTH, LANE_NUMBER * LANE_WIDTH], [-CROSSROAD_SIZE / 2, -CROSSROAD_SIZE / 2],
                color='green', linewidth=light_line_width))

    # down vertical
    Lines.append(ax.plot([-(LANE_NUMBER-1)*LANE_WIDTH, 0], [CROSSROAD_SIZE / 2, CROSSROAD_SIZE / 2],
                color=v_color, linewidth=light_line_width))
    Lines.append(ax.plot([-LANE_NUMBER * LANE_WIDTH, -(LANE_NUMBER-1)*LANE_WIDTH], [CROSSROAD_SIZE / 2, CROSSROAD_SIZE / 2],
            color='green', linewidth=light_line_width))
            
    # left horizon
    Lines.append(ax.plot([-CROSSROAD_SIZE / 2, -CROSSROAD_SIZE / 2], [0, -(LANE_NUMBER-1)*LANE_WIDTH],
                color=h_color, linewidth=light_line_width))
    Lines.append(ax.plot([-CROSSROAD_SIZE / 2, -CROSSROAD_SIZE / 2], [-(LANE_NUMBER-1)*LANE_WIDTH, -LANE_NUMBER * LANE_WIDTH],
                color='green', linewidth=light_line_width))
    # right horizon
    Lines.append(ax.plot([CROSSROAD_SIZE / 2, CROSSROAD_SIZE / 2], [(LANE_NUMBER-1)*LANE_WIDTH, 0],
                color=h_color, linewidth=light_line_width))
    Lines.append(ax.plot([CROSSROAD_SIZE / 2, CROSSROAD_SIZE / 2], [LANE_NUMBER * LANE_WIDTH, (LANE_NUMBER-1)*LANE_WIDTH],
                color='green', linewidth=light_line_width))



    if traffic_light == 2 or traffic_light == 4:
        extension = 40
        Lines.append(ax.plot( [LANE_WIDTH*0.5, LANE_WIDTH*0.5],[-CROSSROAD_SIZE/2-extension, -CROSSROAD_SIZE/2], alpha = 0.25, color = 'r'))
        Lines.append(ax.plot( [LANE_WIDTH*1.5, LANE_WIDTH*1.5],[-CROSSROAD_SIZE/2-extension, -CROSSROAD_SIZE/2], alpha = 0.25, color = 'r'))
        Lines.append(ax.plot( [LANE_WIDTH*-0.5, LANE_WIDTH*-0.5],[CROSSROAD_SIZE/2+extension, CROSSROAD_SIZE/2], alpha = 0.25, color = 'r'))
        Lines.append(ax.plot( [LANE_WIDTH*-1.5, LANE_WIDTH*-1.5],[CROSSROAD_SIZE/2+extension, CROSSROAD_SIZE/2], alpha = 0.25, color = 'r'))



        Lines.append(ax.plot(ref3.path_list[0][0], ref3.path_list[0][1], alpha = 0.25, color = 'g'))
        Lines.append(ax.plot(x_r10, y_r10, alpha = 0.25, color = 'g'))
        Lines.append(ax.plot(x_l10, y_l10, alpha = 0.25, color = 'g'))
        Lines.append(ax.plot(x_r20, y_r20, alpha = 0.25, color = 'g'))
        Lines.append(ax.plot(x_l20, y_l20, alpha = 0.25, color = 'g'))

        Lines.append(ax.plot(x_r11, y_r11, alpha = 0.25, color = 'g'))
        Lines.append(ax.plot(x_l11, y_l11, alpha = 0.25, color = 'g'))
        Lines.append(ax.plot(x_r21, y_r21, alpha = 0.25, color = 'g'))
        Lines.append(ax.plot(x_l21, y_l21, alpha = 0.25, color = 'g'))

        Lines.append(ax.plot(x_r12, y_r12, alpha = 0.25, color = 'g'))
        Lines.append(ax.plot(x_l12, y_l12, alpha = 0.25, color = 'g'))
        Lines.append(ax.plot(x_r22, y_r22, alpha = 0.25, color = 'g'))
        Lines.append(ax.plot(x_l22, y_l22, alpha = 0.25, color = 'g'))

        Lines.append(ax.plot(x_r3, y_r3, alpha = 0.25, color = 'g'))
        Lines.append(ax.plot(x_u3, y_u3, alpha = 0.25, color = 'g'))
        Lines.append(ax.plot(x_l3, y_l3, alpha = 0.25, color = 'g'))

        
    if traffic_light == 0 or traffic_light == 4:
        extension = 40
        ##---------red light----------------------------
        Lines.append(ax.plot([-CROSSROAD_SIZE/2-extension, -CROSSROAD_SIZE/2], [LANE_WIDTH*-0.5, LANE_WIDTH*-0.5], alpha = 0.25, color = 'r'))
        Lines.append(ax.plot([-CROSSROAD_SIZE/2-extension, -CROSSROAD_SIZE/2], [LANE_WIDTH*-1.5, LANE_WIDTH*-1.5], alpha = 0.25, color = 'r'))
        Lines.append(ax.plot([CROSSROAD_SIZE/2, CROSSROAD_SIZE/2+extension], [LANE_WIDTH*0.5, LANE_WIDTH*0.5], alpha = 0.25, color = 'r'))
        Lines.append(ax.plot([CROSSROAD_SIZE/2, CROSSROAD_SIZE/2+extension], [LANE_WIDTH*1.5, LANE_WIDTH*1.5], alpha = 0.25, color = 'r'))

        ##---------green light----------------------------
        Lines.append(ax.plot(ref1.path_list[0][0], ref1.path_list[0][1], alpha = 0.25, color = 'g'))
        Lines.append(ax.plot(ref1.path_list[1][0], ref1.path_list[1][1], alpha = 0.25, color = 'g'))
        Lines.append(ax.plot(ref1.path_list[2][0], ref1.path_list[2][1], alpha = 0.25, color = 'g'))
        Lines.append(ax.plot(ref2.path_list[0][0], ref2.path_list[0][1], alpha = 0.25, color = 'g'))
        Lines.append(ax.plot(ref2.path_list[1][0], ref2.path_list[1][1], alpha = 0.25, color = 'g'))
        Lines.append(ax.plot(ref2.path_list[2][0], ref2.path_list[2][1], alpha = 0.25, color = 'g'))
        Lines.append(ax.plot(ref3.path_list[0][0], ref3.path_list[0][1], alpha = 0.25, color = 'g'))

        Lines.append(ax.plot(x_u10, y_u10, alpha = 0.25, color = 'g'))
        Lines.append(ax.plot(x_u20, y_u20, alpha = 0.25, color = 'g'))
        Lines.append(ax.plot(x_u11, y_u11, alpha = 0.25, color = 'g'))
        Lines.append(ax.plot(x_u21, y_u21, alpha = 0.25, color = 'g'))
        Lines.append(ax.plot(x_u12, y_u12, alpha = 0.25, color = 'g'))
        Lines.append(ax.plot(x_u22, y_u22, alpha = 0.25, color = 'g'))

        Lines.append(ax.plot(x_r3, y_r3, alpha = 0.25, color = 'g'))
        Lines.append(ax.plot(x_u3, y_u3, alpha = 0.25, color = 'g'))
        Lines.append(ax.plot(x_l3, y_l3, alpha = 0.25, color = 'g'))

    return Lines
if __name__ == "__main__":
    fig, ax = static_view()
    update_light(ax, 0)
    fig.savefig('view0.png')
    fig, ax = static_view()
    update_light(ax, 2)
    fig.savefig('view2.png')