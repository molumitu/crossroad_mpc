from dataclasses import dataclass
from Env_utils import horizon
import numpy as np

@dataclass
class MPC_Param:
    # horizon: int = 20
    bounds : tuple = ((-0.28, 0.28), (-6.1, 2.8)) * horizon
    red_bounds : tuple = ((-6.1, 0),) * horizon
    Q : tuple= (30, 30, 0.1, 0., 0)
    R : tuple = (0.2,0.1)
    P : tuple = (100,)
    safety_dist = 5


    def __post_init__(self):
        self.Q = np.array(self.Q)
        self.R = np.array(self.R)
        self.P = np.array(self.P)

 
# mpc_params = MPC_Param()
# print(np.array(mpc_params.P))
# print(mpc_params)

