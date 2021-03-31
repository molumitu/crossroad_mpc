from dataclasses import dataclass
from Utils import horizon
import numpy as np

@dataclass
class IPOPT_Param:
    # horizon: int = 20
    routes_num: int = 3
    Q : tuple= (30, 30, 0.1, 0., 0)
    R : tuple = (0.5,0.1)
    P : tuple = (0,)
    safety_dist = 5
    lbx : tuple = (-0.26, -6.1) * horizon
    ubx : tuple =  (0.26, 2.8) * horizon
    red_lbx : tuple = (-0., -6.1) * horizon
    red_ubx : tuple = (0., 2.8) * horizon

    


    def __post_init__(self):
        self.Q = np.array(self.Q)
        self.R = np.array(self.R)
        self.P = np.array(self.P)
        self.lbx = list(self.lbx)
        self.ubx = list(self.ubx)
        self.red_lbx = list(self.red_lbx)
        self.red_ubx = list(self.red_ubx)