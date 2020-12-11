# December 8, 2020
# This script tests the derived forward kinematics of the trans-radial prosthetic

import numpy as np

# Defining joint variables (Angles must be in radians)
theta_1 = 90 * (np.pi / 180)    # This joint defines the pronation/suppination motion
theta_2 = 0 * (np.pi / 180)    # This joint defines the radial/ulnar rotation
theta_3 = 90 * (np.pi / 180)    # This joint defines the wrist flexion/extension

# Defining link lengths (Measurements are in cm)
a1 = 6
a2 = 7
a3 = 8

# Defining the rotation matrices
r_0_1 = np.array([[np.cos(theta_1), 0,  np.sin(theta_1)],
                  [np.sin(theta_1), 0, -np.cos(theta_1)],
                  [0,               1,               0]])

r_1_2 = np.array([[-np.sin(theta_2), 0, np.cos(theta_2)],
                  [np.cos(theta_2),  0, np.sin(theta_2)],
                  [0,                1,               0]])

r_2_3 = np.array([[np.cos(theta_3), -np.sin(theta_3), 0],
                  [np.sin(theta_3), np.cos(theta_3),  0],
                  [0,                0,               1]])

# Defining the displacement vectors
d_0_1 = np.array([[0], [0], [a1]])
d_1_2 = np.array([[-a2*np.sin(theta_2)], [a2*np.cos(theta_2)], [0]])
d_2_3 = np.array([[a3*np.cos(theta_3)], [a3*np.sin(theta_3)], [0]])

# Constructing homogeneous transformation matrices
bottom_row = np.array([[0, 0, 0, 1]]).reshape([1, 4])

H_0_1 = np.concatenate([r_0_1, d_0_1], axis=-1)
H_1_2 = np.concatenate([r_1_2, d_1_2], axis=-1)
H_2_3 = np.concatenate([r_2_3, d_2_3], axis=-1)

H_0_1 = np.concatenate([H_0_1, bottom_row], axis=0)
H_1_2 = np.concatenate([H_1_2, bottom_row], axis=0)
H_2_3 = np.concatenate([H_2_3, bottom_row], axis=0)

H_0_3 = np.dot(H_0_1, np.dot(H_1_2, H_2_3))
r_0_3 = H_0_3[:3, :3]
d_0_3 = H_0_3[:3, -1]
import code; code.interact(local=dict(globals(), **locals()))



