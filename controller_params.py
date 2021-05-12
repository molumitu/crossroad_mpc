from dataclasses import dataclass
import dataclasses
from predict_surroundings import route_to_task
from Utils import horizon, Nc
import numpy as np

@dataclass
class MPC_Param:
    bounds : tuple = ((-0.28, 0.28), (-6.1, 2)) * Nc
    red_bounds : tuple = ((-9.1, 0),) * Nc 
    # Q : tuple= (30, 30, 1, 0., 0)
    # R : tuple = (0.1,0.05)
    Q : tuple = (25.*10, 25.*10, 100., 0., 0., 200.)
    R : tuple = (88.9/2,0.25)
    P : tuple = (5000.,)
    route_weight_list : tuple = (0.9, 0.95, 1)
    safe_dist_front: float = 3.
    safe_dist_rear: float = 3.
    red_safe_dist_front: float = 3 + 3.0
    red_safe_dist_rear: float = 3.
    

    # def __post_init__(self):
    #     self.Q = np.array(self.Q)
    #     self.R = np.array(self.R)
    #     self.P = np.array(self.P)

class Param_Set:
    def __init__(self):
        self.param_left = MPC_Param()
        self.param_right = MPC_Param(P = 0)
        self.param_straight = MPC_Param(R = (889,0.25), P = 2000, route_weight_list=(0.1,1.2,1))

    def select_param_by_task(self, task):   # task 0:left, 1:straight, 2:right
        if task == 0:
            return self.param_left
        elif task == 1:
            return self.param_straight
        elif task == 2:
            return self.param_right
    def to_dict(self):
        return {
            'left': dataclasses.asdict(self.param_left),
            'straight': dataclasses.asdict(self.param_straight),
            'right': dataclasses.asdict(self.param_right)
        }



 


