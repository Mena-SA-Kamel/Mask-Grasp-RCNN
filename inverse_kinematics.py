# December 8, 2020
# This script tests the derived inverse kinematics of the trans-radial prosthetic

import numpy as np


def wrap_angle_around_90(angles):
    angles %= 360
    angles[np.logical_and((angles > 90), (angles < 180))] -= 180
    theta_bet_180_270 = angles[np.logical_and((angles >= 180), (angles < 270))]
    # angles[np.logical_and((angles >= 180), (angles < 270))] = 180 - theta_bet_180_270
    angles[np.logical_and((angles >= 180), (angles < 270))] = theta_bet_180_270 - 180
    angles[np.logical_and((angles >= 270), (angles < 360))] -= 360
    return angles

orientation_matrix = np.array( [[ -1,0,0],
                                [ 0, 1, 0],
                                [0,0,-1]])

# orientation_matrix = np.array( [[0, 1, 0],
#                                 [-1, 0, 0],
#                                 [0, 0, 1]])

c = orientation_matrix[0, 2]
f = orientation_matrix[1, 2]
i = orientation_matrix[2, 2]
g = orientation_matrix[2, 0]
h = orientation_matrix[2, 1]
s2 = i
c2 = np.sqrt(1 - s2 ** 2)
c2 = np.maximum(0.0001, c2)
s1 = f / c2
c1 = c / c2
s3 = -h / c2
c3 = g / c2

theta_1 = np.arctan2(s1, c1) / (np.pi / 180)
theta_2 = np.arctan2(s2, c2) / (np.pi / 180)
theta_3 = np.arctan2(s3, c3) / (np.pi / 180)

ps_home = 0
ur_home = 90
fe_home = -90

theta_1_corrected = theta_1 - ps_home
theta_2_corrected = -1*(ur_home - theta_2)
theta_3_corrected = theta_3 - fe_home

theta_1_wrapped = wrap_angle_around_90(np.array([theta_1_corrected]))[0]
theta_2_wrapped = wrap_angle_around_90(np.array([theta_2_corrected]))[0]
theta_3_wrapped = wrap_angle_around_90(np.array([theta_3_corrected]))[0]

print ('Theta 1 - Pronation/ Supination: ', theta_1_wrapped,
       '\nTheta 2 - Radial/ Ulnar: ', theta_2_wrapped,
       '\nTheta 3 - Flexion/Extension: ', theta_3_wrapped)




