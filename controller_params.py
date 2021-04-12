from dataclasses import dataclass
from Utils import horizon, Nc
import numpy as np

@dataclass
class MPC_Param:
    bounds : tuple = ((-0.28, 0.28), (-6.1, 2.8)) * Nc
    red_bounds : tuple = ((-9.1, 0),) * Nc
    # Q : tuple= (30, 30, 1, 0., 0)
    # R : tuple = (0.1,0.05)
    Q : tuple= (25, 25, 100, 0., 0)
    R : tuple = (88.9,0.25)
    P : tuple = (500,)

    def __post_init__(self):
        self.Q = np.array(self.Q)
        self.R = np.array(self.R)
        self.P = np.array(self.P)

 


