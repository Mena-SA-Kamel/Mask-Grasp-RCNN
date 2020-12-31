# December 8, 2020
# This script tests the derived inverse kinematics of the trans-radial prosthetic

import numpy as np

orientation_matrix = np.array( [[ 0.43488047, -0.59604114,  0.67499181],
                                [ 0.74711572,  0.65727303,  0.09904674],
                                [-0.50268984,  0.4612235,   0.73114691]])

# orientation_matrix = np.array( [[0, 1, 0],
#                                 [-1, 0, 0],
#                                 [0, 0, 1]])

c = orientation_matrix[0, -1]
f = orientation_matrix[1, -1]
g, h, i = orientation_matrix[2].tolist()

s2 = i
c2 = np.sqrt(1 - s2**2)

c2 = np.maximum(0.0001, c2)

s1 = f/c2
c1 = c/c2

s3 = -h/c2
c3 = g/c2


theta_1 = np.arctan2(s1, c1) / (np.pi / 180)
theta_2 = np.arctan2(s2, c2) / (np.pi / 180)
theta_3 = np.arctan2(s3, c3) / (np.pi / 180)

print ('Theta 1 - Pronation/ Supination: ', theta_1,
       '\nTheta 2 - Radial/ Ulnar: ', theta_2,
       '\nTheta 3 - Flexion/Extension: ', theta_3)




