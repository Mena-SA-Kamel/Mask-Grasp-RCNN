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

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fig, ax = plt.subplots()

w = 15
h = 5
theta = 60 * (np.pi / 180)

# Rotation about the Z axis of the surface normal
r_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta),  0],
                  [0,                0,               1]])

box_vert_obj_frame = np.array([[w/2, -h/2, 0],
                               [w/2, h/2, 0],
                               [-w/2, h/2, 0],
                               [-w/2, -h/2, 0]])

# Defining the grasp region vertices in the approach vector frame of reference
rotated_points = np.dot(r_z, box_vert_obj_frame.T).T
box_original = Polygon(box_vert_obj_frame[:, :2], True)
box_rotated = Polygon(rotated_points[:, :2], True, color='r')


ax.add_patch(box_original)
ax.add_patch(box_rotated)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
plt.show()

approach_vector_orientation = np.array([[1, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 1]])
approach_vector_displacement = np.array([[0], [0], [0]])
HTM_camera_to_object = np.concatenate([approach_vector_orientation, approach_vector_displacement], axis=-1)
HTM_camera_to_object = np.concatenate([HTM_camera_to_object, np.array([[0, 0, 0, 1]])], axis=0)

rotated_points = np.concatenate([rotated_points, np.ones((4, 1))], axis=-1)
rotated_points_camera_frame = np.dot(HTM_camera_to_object, rotated_points.T)

fig = plt.figure()
ax = Axes3D(fig)
rotated_points_camera_frame = rotated_points_camera_frame.T
x = rotated_points_camera_frame[:, 0]
y = rotated_points_camera_frame[:, 1]
z = rotated_points_camera_frame[:, 2]
verts = [list(zip(x,y,z))]
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.add_collection3d(Poly3DCollection(verts))
plt.show()
import code; code.interact(local=dict(globals(), **locals()))


