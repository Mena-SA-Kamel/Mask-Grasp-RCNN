import numpy as np
import serial
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def wrap_angle_around_90(angles):
    angles %= 360
    angles[np.logical_and((angles > 90), (angles < 180))] -= 180
    theta_bet_180_270 = angles[np.logical_and((angles >= 180), (angles < 270))]
    # angles[np.logical_and((angles >= 180), (angles < 270))] = 180 - theta_bet_180_270
    angles[np.logical_and((angles >= 180), (angles < 270))] = theta_bet_180_270 - 180
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

def generate_random_poses():
    r = R.from_euler('zyx', [
        [90, 0, 0],
        [0, 0, 0],
        [0, 0, 0]], degrees=True)
    r.as_matrix()

def derive_motor_angles_v0(orientation_matrix):
    orientation_matrix = np.around(orientation_matrix, decimals=2)
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


    # theta_1 = wrap_angle_around_90(np.array([theta_1]))[0]
    # theta_2 = wrap_angle_around_90(np.array([theta_2]))[0]
    # theta_3 = wrap_angle_around_90(np.array([theta_3]))[0]
    print('Theta 1 - Pronation/ Supination: ', theta_1,
          '\nTheta 2 - Ulnar/ Radial: ', theta_2,
          '\nTheta 3 - Flexion/Extension: ', theta_3)

    return [theta_1, theta_2, theta_3]

def derive_motor_angles(orientation_matrix):
    a = orientation_matrix[0, 1]
    b = orientation_matrix[1, 1]
    c = orientation_matrix[2, 1]
    d = orientation_matrix[2, 0]
    e = orientation_matrix[2, 2]

    s2 = c
    c2 = np.sqrt(1 - s2 ** 2)
    c2 = np.maximum(0.0001, c2)
    s1 = b / c2
    c1 = a / c2
    s3 = e / c2
    c3 = d / c2

    theta_1 = np.arctan2(s1, c1) / (np.pi / 180)
    theta_2 = np.arctan2(s2, c2) / (np.pi / 180)
    theta_3 = np.arctan2(s3, c3) / (np.pi / 180)

    theta_1 = wrap_angle_around_90(np.array([theta_1]))[0]
    theta_2 = wrap_angle_around_90(np.array([theta_2]))[0]
    theta_3 = wrap_angle_around_90(np.array([theta_3]))[0]
    print('Theta 1 - Pronation/ Supination: ', theta_1,
          '\nTheta 2 - Ulnar/ Radial: ', theta_2,
          '\nTheta 3 - Flexion/Extension: ', theta_3)

    return [theta_1, theta_2, theta_3]

def derive_motor_angles_v2(orientation_matrix):
    a = orientation_matrix[0, 1]
    b = orientation_matrix[1, 1]
    c = orientation_matrix[2, 1]
    d = orientation_matrix[2, 0]
    e = orientation_matrix[2, 2]

    s2 = c
    c2 = np.sqrt(1 - s2 ** 2)
    c2 = np.maximum(0.0001, c2)
    s1 = -b / c2
    c1 = -a / c2
    s3 = e / c2
    c3 = -d / c2

    theta_1 = np.arctan2(s1, c1) / (np.pi / 180)
    theta_2 = np.arctan2(s2, c2) / (np.pi / 180)
    theta_3 = np.arctan2(s3, c3) / (np.pi / 180)

    theta_1 = wrap_angle_around_90(np.array([theta_1]))[0]
    theta_2 = wrap_angle_around_90(np.array([theta_2]))[0]
    theta_3 = wrap_angle_around_90(np.array([theta_3]))[0]
    print('Theta 1 - Pronation/ Supination: ', theta_1,
          '\nTheta 2 - Ulnar/ Radial: ', theta_2,
          '\nTheta 3 - Flexion/Extension: ', theta_3)

    return [theta_1, theta_2, theta_3]

# vector = np.array([[1,     0,      0],
#                    [0, 0.707, -0.707],
#                    [0, 0.707, 0.707]])

calibration_matrix = np.array([[0, 0, 1],
                              [0, -1, 0],
                               [1, 0, 0]])

cal_matrix_inverse = np.linalg.inv(calibration_matrix)
# vector = np.array([[1, 0, 0],
#                    [0, 1, 0],
#                    [0, 0, 1]])
vector = R.from_euler('y', -90, degrees=True).as_matrix()
# r = R.from_euler('zyx', [
# [90, 0, 0],
# [0, 45, 0],
# [45, 60, 30]], degrees=True)
# vector = R.from_euler('z', -90, degrees=True).as_matrix()


vector_new = np.dot(vector, calibration_matrix)


theta1, theta2, theta3 = derive_motor_angles_v0(vector_new)
joint1, joint2, joint3 = orient_wrist(theta1, theta2, theta3).tolist()
import code; code.interact(local=dict(globals(), **locals()))
# string_command = 'w %d %d %d' % (joint3, joint2, joint1)
#
# ser = serial.Serial('COM6', 115200, timeout=1)
# import code; code.interact(local=dict(globals(), **locals()))
# ser.write(string_command.encode())
# ser.write(b'h')
# time.sleep(3)
# ser.write(b'j 0 400')

#
#
# reference_center = np.array([0, 0, 0]).reshape([-1, 1]) # Position of the home joint of the arm
# end_effector_center = np.array([0, 0, 0]).reshape([-1, 1])
#
# # Defining the desired orientation
# home_orientation = np.array([[1,0,0],
#                              [0,1,0],
#                              [0,0,1]])
#
# theta1, theta2, theta3
# t1 = theta1 * (np.pi / 180)
# t2 = theta2 * (np.pi / 180)
# t3 = theta3 * (np.pi / 180)
#
# inverse_kinematics = np.array([[(-np.cos(t1)*np.sin(t2)*np.cos(t3) + np.sin(t1)*np.sin(t3)), np.cos(t2)*np.cos(t1), (-np.cos(t1)*np.sin(t2)*np.sin(t3) - np.sin(t1)*np.cos(t3))],
#                                [(-np.sin(t1)*np.sin(t2)*np.cos(t3) - np.cos(t1)*np.sin(t3)), np.sin(t1)*np.cos(t2), (-np.sin(t1)*np.sin(t2)*np.sin(t3) + np.cos(t1)*np.cos(t3))],
#                                [np.cos(t2)*np.cos(t3),                                                  np.sin(t2),                                       np.cos(t2)*np.sin(t3)]])
#
# end_effector_orientation = inverse_kinematics
#
# vx, vy, vz = np.split(home_orientation, indices_or_sections=3, axis=-1)
# ee_vx, ee_vy, ee_vz = np.split(end_effector_orientation, indices_or_sections=3, axis=-1)
#
# fig = plt.figure()
# ax = plt.axes(projection="3d")
# x, y, z = np.array(list(zip(reference_center, vx+reference_center)))
# ax.plot(x.flatten(), y.flatten(), z.flatten(), '-k', linewidth=3, color='r')
# x, y, z = np.array(list(zip(reference_center, vy+reference_center)))
# ax.plot(x.flatten(), y.flatten(), z.flatten(), '-k', linewidth=3, color='g')
# x, y, z = np.array(list(zip(reference_center, vz+reference_center)))
# ax.plot(x.flatten(), y.flatten(), z.flatten(), '-k', linewidth=3, color='b')
#
# x, y, z = np.array(list(zip(end_effector_center, ee_vx+end_effector_center)))
# ax.plot(x.flatten(), y.flatten(), z.flatten(), '--k', linewidth=3, color='r')
# x, y, z = np.array(list(zip(end_effector_center, ee_vy+end_effector_center)))
# ax.plot(x.flatten(), y.flatten(), z.flatten(), '--k', linewidth=3, color='g')
# x, y, z = np.array(list(zip(end_effector_center, ee_vz+end_effector_center)))
# ax.plot(x.flatten(), y.flatten(), z.flatten(), '--k', linewidth=3, color='b')
#
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# minmax = [-2, 2]
# ax.set_xlim(minmax)
# ax.set_ylim(minmax)
# ax.set_zlim(minmax)
#
#
#
# plt.show(block=False)
#
#
# string_command = 'w %d %d %d' % (joint3, joint2, joint1)
# #
# ser = serial.Serial('COM6', 115200, timeout=1)
# import code; code.interact(local=dict(globals(), **locals()))
# # import code; code.interact(local=dict(globals(), **locals()))
# ser.write(string_command.encode())
# # ser.write(b'h')
# # time.sleep(3)
# # ser.write(b'j 0 400')