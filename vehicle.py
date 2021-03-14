from Env_utils import STEP_TIME, deal_with_phi
import numpy as np

###### 务必和matlab保持一致
class VehicleDynamics():
    def __init__(self, ):
        self.vehicle_params = dict(#C_f=-128915.5,  # front wheel cornering stiffness [N/rad]
                                   C_f=-96995.9, # so the K is equal to 0
                                   C_r=-85943.6,  # rear wheel cornering stiffness [N/rad]
                                   a=1.4,  # distance from CG to front axle [m]
                                   b=1.6,  # distance from CG to rear axle [m]
                                   h=0.35,
                                   mass=1412.,  # mass [kg]
                                   I_z=1536.7,  # Polar moment of inertia at CG [kg*m^2]
                                   miu=0.9,  # tire-road friction coefficient 沥青路面
                                   g=9.81,  # acceleration of gravity [m/s^2]
                                   )
        # a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
        #                 self.vehicle_params['mass'], self.vehicle_params['g']
        # F_zf_staic , F_zr_static = b * mass * g / (a + b), a * mass * g / (a + b)
        # self.vehicle_params.update(dict(F_zf_staic = F_zf_staic, F_zr_static = F_zr_static))

    def state_trans(self, states, actions, tau): 
        v_x, v_y, r, x, y, phi = states[0], states[1], states[2], states[3], states[4], states[5]
        steer, a_x = states[6], states[7]

        phi = phi * np.pi / 180.
        steer_input, a_x_input = actions[0], actions[1]

        C_f = np.array(self.vehicle_params['C_f'])
        C_r = np.array(self.vehicle_params['C_r'])
        a = np.array(self.vehicle_params['a'])
        b = np.array(self.vehicle_params['b'])
        h = np.array(self.vehicle_params['h'])
        mass = np.array(self.vehicle_params['mass'])
        I_z = np.array(self.vehicle_params['I_z'])
        miu = np.array(self.vehicle_params['miu'])
        g = np.array(self.vehicle_params['g'])

        _freq = int(tau / 0.01)
        tau = tau / _freq

        T = 0.4
        eps = 1e-8
        for _ in range(_freq):
            eps_signed = np.copysign(eps, v_x)

            steer = (1 - tau/T) * steer + tau/T * steer_input
            a_x =  (1 - tau/T) * a_x + tau/T * a_x_input

            F_zf = b * mass * g / (a + b) - mass * a_x * h / (a + b)
            F_zr = a * mass * g / (a + b) + mass * a_x * h / (a + b)
            F_xf = np.where(a_x < 0, mass * a_x / 2, np.zeros_like(a_x))
            F_xr = np.where(a_x < 0, mass * a_x / 2, mass * a_x)
            max_F_yf = np.sqrt(np.square(miu * F_zf) - np.square(F_xf))
            max_F_yr = np.sqrt(np.square(miu * F_zr) - np.square(F_xr))

            alpha_f = np.arctan((v_y + a * r) / (v_x+eps_signed)) * np.sign(v_x) - steer * np.tanh(4* v_x)
            alpha_r = (np.arctan((v_y - b * r) / (v_x+eps_signed))) * np.sign(v_x)
            F_yf = np.clip(alpha_f * C_f, -max_F_yf, max_F_yf)
            F_yr = np.clip(alpha_r * C_r, -max_F_yr, max_F_yr)

            next_state = [v_x + tau * (a_x + v_y * r - F_yf * np.sin(steer) / mass),
                          v_y + tau * (-v_x * r +(F_yr + F_yf * np.cos(steer))/mass),
                          r + tau * (F_yf * a * np.cos(steer) - F_yr * b)/I_z,
                        #   (mass * v_y * v_x + tau * (a * C_f - b * C_r) * r - tau * C_f * steer * v_x - tau * mass * v_x**2 * r) / (mass * v_x - tau * (C_f + C_r)),
                        #   (-I_z * r * v_x - tau * (a * C_f - b * C_r) * v_y + tau * a * C_f * steer * v_x) / (tau * (a**2 * C_f + b**2 * C_r) - I_z * v_x),
                          x + tau * (v_x * np.cos(phi) - v_y * np.sin(phi)),
                          y + tau * (v_x * np.sin(phi) + v_y * np.cos(phi)),
                          deal_with_phi((phi + tau * r) * 180 / np.pi)]
            v_x, v_y, r, x, y, phi = next_state
            phi = phi * np.pi /180
        next_state = next_state + [steer, a_x]
        return next_state

    # def prediction(self, x, u, STEP_TIME): 
    #     return self.state_trans(x, u, STEP_TIME)

        # next_state = np.array([[v_x, v_y, r, x, y, phi]])
        # phi 输入输出的单位都是°
