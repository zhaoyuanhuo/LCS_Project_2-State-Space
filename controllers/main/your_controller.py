# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

import math

# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        # These can be ignored in P1
        self.lr = 3.32
        self.lf = 1.01
        self.Ca = 20000
        self.Iz = 29526.2
        self.m = 4500
        self.g = 9.81
        # Add additional member variables according to your need here.
        # constraints
        self.F_max = 16000.0
        self.F_min = 0.0
        self.delta_min = -math.pi / 6
        self.delta_max = math.pi / 6

        # pid params
        self.kp_x = 50000.0
        self.ki_x = 150.0
        self.kd_x = 100.0
        self.kp_psi = 5.0
        self.ki_psi = 0.1
        self.kd_psi = 0.3

        #
        self.sum_error_x = 0.0
        self.error_x_old = 0.0
        self.sum_error_psi = 0.0
        self.error_psi_old = 0.0

        self.long_look_ahead = 600
        self.lat_look_ahead = 80

        # lateral
        self.x_est = np.array([[0.0],
                               [0.0],
                               [0.0],
                               [0.0]])
        self.delta = 0.0
        self.F = 0.0

        # self.Ko = np.array([[387.4, 11.8987],
        #                     [37235.6, 2596.4],
        #                     [9.4773, 390.3],
        #                     [1875.2, 37792.6]])
        # self.Kc = np.array([[4603.0, 1026.0, -19741.0, -6602.0],
        #                     [0.0, 0.0, 0.0, 0.0]])

        self.Ko = np.array([[379.484, 23.695],
                            [338.779, 86.207],
                            [11.898, 377.788],
                            [257.402, 333.149]])
        self.Kc = np.array([[0.4628, 0.32255, 4.5653, 2.8211],
                            [0.0, 0.0, 0.0, 0.0]])

    def inertial2global(self, x, y, psi):
        # convert (x, y) from inertial frame to global frame
        # psi_ = wrapToPi(psi)
        xy_inertial = np.array([[x],
                                [y]])
        convert_mat = np.array([[math.cos(psi_), -math.sin(psi_)],
                                [math.sin(psi_), math.cos(psi_)]])
        XY_global = np.matmul(convert_mat, xy_inertial)
        return XY_global[0][0], XY_global[1][0]

    def global2inertial(self, X, Y, psi):
        # convert (X, Y) from global frame to inertial frame
        # psi_ = wrapToPi(psi)
        XY_global = np.array([[X],
                              [Y]])
        convert_mat = np.array([[math.cos(psi), -math.sin(psi)],
                                [math.sin(psi), math.cos(psi)]])
        convert_mat = np.linalg.inv(convert_mat)

        xy_inertial = np.matmul(convert_mat, XY_global)
        return xy_inertial[0][0], xy_inertial[1][0]

    def wrapAngle(self, theta):
        return (theta + 2 * math.pi) % (2 * math.pi)

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Design your controllers in the spaces below.
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta).

        # preprocessing the reference trajectory
        # lateral preprocessing
        long_look_ahead = self.long_look_ahead
        lat_look_ahead = self.lat_look_ahead
        XTE, nn_idx = closestNode(X, Y, trajectory)
        nn_lat_next_idx = nn_idx + lat_look_ahead
        if nn_lat_next_idx >= len(trajectory) - 1:
            # print("lat near end")
            nn_lat_next_idx = len(trajectory) - 1
        X_next_ref = trajectory[nn_lat_next_idx][0]
        Y_next_ref = trajectory[nn_lat_next_idx][1]
        psi_ref = math.atan2(Y_next_ref - Y, X_next_ref - X)
        speed_scale = 1.1
        longi_scale = 1.0

        # longitude lookahead
        #   1. comparing with current psi, to determine if there is a curb ahead
        #   2. generate reference xdot, for longitudinal controller
        nn_long_next_idx = nn_idx + long_look_ahead
        if nn_long_next_idx >= len(trajectory) - 1:
            long_look_ahead = len(trajectory) - 1 - nn_idx
            nn_long_next_idx = len(trajectory) - 1
        X_long_next_ref = trajectory[nn_long_next_idx][0]
        Y_long_next_ref = trajectory[nn_long_next_idx][1]
        Xdot_ref = (X_long_next_ref - X) / (delT * long_look_ahead)
        Ydot_ref = (Y_long_next_ref - Y) / (delT * long_look_ahead)
        xdot_ref, ydot_ref = self.global2inertial(Xdot_ref, Ydot_ref, psi)
        # state machine
        # straight line boost
        psi_long_ref = math.atan2(Y_long_next_ref - Y, X_long_next_ref - X)
        error_psi_long = self.wrapAngle(psi_long_ref) - self.wrapAngle(psi)
        if np.abs(error_psi_long) < 20 * math.pi / 180:  # straight
            # print("straight!")
            longi_scale = 4.0
            self.kd_x = 5.0
            self.lat_look_ahead = 30
        elif np.abs(error_psi_long) < 30 * math.pi / 180:  # curb
            # print("small angle is", np.abs(error_psi_long))
            longi_scale = 3.0
            self.kd_x = 100.0
            self.lat_look_ahead = 75
        elif np.abs(error_psi_long) < 45 * math.pi / 180:  # curb
            # print("median angle is", np.abs(error_psi_long))
            longi_scale = 0.8
            self.kd_x = 50.0
            self.lat_look_ahead = 150
        elif np.abs(error_psi_long) < 85 * math.pi / 180:  # curb
            # print("large angle is", np.abs(error_psi_long))
            longi_scale = 0.7
            self.kd_x = 5.0
            self.lat_look_ahead = 180
        else:
            # print("super large angle is", np.abs(error_psi_long))
            longi_scale = 0.7
            self.kd_x = 5.0
            self.lat_look_ahead = 200
        # ---------------|Lateral Controller|-------------------------
        """
        Please design your lateral controller below.
        """
        # generate error state for lateral controller
        sys_A = np.array([[0.0, 1.0, 0.0, 0.0],
                          [0.0, -160/(9*xdot), 160/9, 308/(15*xdot)],
                          [0.0, 0.0, 0.0, 1.0],
                          [0.0, 3.12942/xdot, -3.12942, -16.31432]])
        sys_B = np.array([[0.0, 0.0],
                          [8.888889, 0.0],
                          [0.0, 0.0],
                          [1.368276, 0.0]])
        sys_C = np.array([[1.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0]])
        sys_control = np.array([[self.delta],
                                [self.F]])
        # print("XTE_curr", XTE)
        sys_output = np.array([[XTE],
                               [self.wrapAngle(psi) - self.wrapAngle(psi_ref)]]) # y: observation
        x_est_dot = sys_A @ self.x_est + sys_B @ sys_control + self.Ko @ (sys_output - sys_C@self.x_est)
        # print("est xdot: ", x_est_dot[0][0], " ", x_est_dot[1][0])
        self.x_est += delT * x_est_dot
        sys_control_next = - np.matmul(self.Kc, self.x_est)
        delta = sys_control_next[0][0]
        delta = clamp(delta, self.delta_min, self.delta_max)
        print(delta)
        # print("est x: ", self.x_est[0][0], " ", self.x_est[2][0])
        # print("computed input ", sys_control_next[1][0], " ", sys_control_next[0][0])

        # ---------------|Longitudinal Controller|-------------------------
        """
        Please design your longitudinal controller below.
        """
        error_x = xdot_ref * speed_scale * longi_scale - xdot
        self.sum_error_x += error_x * delT
        F = self.kp_x * error_x + \
            self.ki_x * self.sum_error_x + \
            self.kd_x * (error_x - self.error_x_old) / delT
        F = clamp(F, self.F_min, self.F_max)
        # # print("longi force: ", F, "XTE= ", XTE)
        # self.error_x_old = error_x
        # print("ref v= ", xdot_ref, "; real v= ", xdot)
        # print("ref angle= ", psi_ref, "; real angle= ", psi)
        # print("lateral angle= ", delta, "; longi force= ", F, "; XTE= ", XTE)

        # Return all states and calculated control inputs (F, delta)

        self.delta = delta
        self.F = F
        return X, Y, xdot, ydot, psi, psidot, F, delta