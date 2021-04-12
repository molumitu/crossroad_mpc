from Reference import ReferencePath
import numpy as np
from Utils import L, W


def generate_ego_init_state(routeID, index):
    ref = ReferencePath(routeID)
    x, y, phi = ref.indexs2points(index, path_index=0)
    steer = 0.
    a_x = 0.
    v = 6
    ego_state = {'v_x':v,
                "v_y":0,
                'r':0,
                'x':x,
                'y':y,
                'phi':phi,
                'steer' : steer,
                'a_x' : a_x,
                'l':L,
                'w':W,
                'routeID':routeID
                }
    return ego_state, ref

if __name__ == "__main__" :
    init_ego_state = {}
    init_ego_ref = {}
    init_ego_state['ego'], init_ego_ref['ego'] = generate_ego_init_state('dl', 1500)
    init_ego_state['ego1'], init_ego_ref['ego1'] = generate_ego_init_state('dl', 120)
    print(init_ego_state)
    for egoID in init_ego_ref.keys():
        print(init_ego_state[egoID])