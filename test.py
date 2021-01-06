import numpy as np
import serial


def wrap_angle_around_90(angles):
    angles %= 360
    angles[np.logical_and((angles >= 90), (angles < 180))] -= 180
    theta_bet_180_270 = angles[np.logical_and((angles >= 180), (angles < 270))]
    angles[np.logical_and((angles >= 180), (angles < 270))] = 180 - theta_bet_180_270
    angles[np.logical_and((angles >= 270), (angles < 360))] -= 360
    return angles

def orient_wrist(theta1, theta2, theta3):
    # This function takes the joint angles, theta1, theta2, theta3, for
    # pronate/supinate, ulnar/radial, and flexion/extension, respectively.

    # Joint center positions as an 8 bit integer
    ps_home = 100
    ur_home = 170
    fe_home = 162

    # Defining the physical joint limits
    pronate_supinate_limit = [-43, 90]
    ulnar_radial_limit = [-13, 21]
    flexion_extension_limit = [-42, 30]

    # Clipping angles to the physical joint limits
    theta1 = np.minimum(theta1, pronate_supinate_limit[1])
    theta1 = np.maximum(theta1, pronate_supinate_limit[0])
    theta2 = np.minimum(theta2, ulnar_radial_limit[1])
    theta2 = np.maximum(theta2, ulnar_radial_limit[0])
    theta3 = np.minimum(theta3, flexion_extension_limit[1])
    theta3 = np.maximum(theta3, flexion_extension_limit[0])

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

def derive_motor_angles_v8(orientation_matrix):
    a = orientation_matrix[0, 0]
    b = orientation_matrix[1, 0]
    c = orientation_matrix[2, 0]
    d = orientation_matrix[2, 1]
    e = orientation_matrix[2, 2]

    s2 = c
    c2 = np.sqrt(1 - s2 ** 2)
    c2 = np.maximum(0.0001, c2)
    s1 = b / c2
    c1 = a / c2
    s3 = d / c2
    c3 = e / c2

    theta_1 = np.arctan2(s1, c1) / (np.pi / 180)
    theta_2 = np.arctan2(s2, c2) / (np.pi / 180)
    theta_3 = np.arctan2(s3, c3) / (np.pi / 180)
    #
    # theta_1 = wrap_angle_around_90(np.array([theta_1]))[0]
    # theta_2 = wrap_angle_around_90(np.array([theta_2]))[0]
    # theta_3 = wrap_angle_around_90(np.array([theta_3]))[0]
    print('Theta 1 - Pronation/ Supination: ', theta_1,
          '\nTheta 2 - Ulnar/ Radial: ', theta_2,
          '\nTheta 3 - Flexion/Extension: ', theta_3)

    return [theta_1, theta_2, theta_3]

vector = np.array([[1,0,0],
                   [0, 0.707, -0.707],
                   [0, 0.707, 0.707]])

theta1, theta2, theta3 = derive_motor_angles_v8(vector)
joint1, joint2, joint3 = orient_wrist(theta1, theta2, theta3).tolist()
string_command = 'w %d %d %d' % (joint3, joint2, joint1)

ser = serial.Serial('COM6', 115200, timeout=1)
import code; code.interact(local=dict(globals(), **locals()))
ser.write(string_command.encode())
# ser.write(b'h')
# time.sleep(3)
# ser.write(b'j 0 400')


import code; code.interact(local=dict(globals(), **locals()))
