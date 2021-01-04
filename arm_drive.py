import serial
import numpy as np

ser = serial.Serial('COM6', 115200, timeout=1)

theta1 = -46.5 # pronate/supinate
theta2 = +19.5 # ulnar/radial
theta3 = -0.39 # flexion/extension
print('Original Values: ', theta1, theta2, theta3)

# Joint center positions as an 8 bit integer
ps_home = 100
ur_home = 170
fe_home = 162

# Defining the physical joint limits
pronate_supinate_limit = [-43, 90]
ulnar_radial_limit = [-13, 21]
flexion_extension_limit = [-42, 30]

INDEX = 0
MIDDLE = 1
RING = 2
THUMB= 3
OPPOSE = 4
FLEX = 5
ULNAR = 6
SUPPINATE = 7

def orient_wrist(theta1, theta2, theta3):
    # This function takes the joint angles, theta1, theta2, theta3, for
    # pronate/supinate, ulnar/radial, and flexion/extension, respectively.

    # Clipping angles to the physical joint limits
    theta1 = np.minimum(theta1, pronate_supinate_limit[1])
    theta1 = np.maximum(theta1, pronate_supinate_limit[0])
    theta2 = np.minimum(theta2, ulnar_radial_limit[1])
    theta2 = np.maximum(theta2, ulnar_radial_limit[0])
    theta3 = np.minimum(theta3, flexion_extension_limit[1])
    theta3 = np.maximum(theta3, flexion_extension_limit[0])

    data_input_range = 256
    mid_point = data_input_range//2

    if np.sign(theta1) == 1:
        joint1 = np.interp(theta1, (0, pronate_supinate_limit[1]), (ps_home, 255))
    else:
        joint1 = np.interp(theta1, (pronate_supinate_limit[0], 0), (0, ps_home))

    if np.sign(theta2) == 1:
        joint2 = np.interp(theta2, (0, ulnar_radial_limit[1]), (ur_home, 0))
    else:
        joint2 = np.interp(theta2, (ulnar_radial_limit[0], 0), (255, ur_home))

    if np.sign(theta3) == 1:
        joint3 = np.interp(theta3, (0, flexion_extension_limit[1]), (fe_home, 255))
    else:
        joint3 = np.interp(theta3, (flexion_extension_limit[0], 0), (0, fe_home))

    joint_output = np.array([joint1, joint2, joint3]).astype('uint8')

    print('Clipped values: ', theta1, theta2, theta3)
    print('Output values: ', joint_output)

    return joint_output

joint1, joint2, joint3 = orient_wrist(theta1, theta2, theta3).tolist()
string_command = 'w %d %d %d' %(joint3, joint2, joint1)
# pronate/supinate, ulnar/radial, and flexion/extension
home_command = 'w %d %d %d' %(fe_home, ur_home, ps_home)
import code; code.interact(local=dict(globals(), **locals()))
ser.write(string_command.encode())
ser.write(b'h')
ser.close()
