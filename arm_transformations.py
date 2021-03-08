import pyrealsense2 as rs
import numpy as np
import cv2
import intel_realsense_IMU

import serial
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def compute_dt(previous_millis, frame_number, zero_timer=False):
    current_millis = int(round(time.time() * 1000))
    if frame_number == 0 or zero_timer:
        previous_millis = current_millis
    dt = (current_millis - previous_millis) / 1000
    previous_millis = current_millis
    return [previous_millis, dt]


def compute_wrist_orientation(approach_vector, theta):
    # Computes the wrist orientation in the camera frame
    # Need to get the projection of vx onto the approach vector (normal), to get the projection of vx on
    # the plane perperndicular to the surface normal
    vz = approach_vector
    vx = np.array([1, 0, 0])
    proj_n_vx = (np.dot(vx, approach_vector) / (np.linalg.norm(approach_vector) ** 2)) * approach_vector
    vx = vx - proj_n_vx
    vx = vx / np.linalg.norm(vx)
    vy = np.cross(vz, vx)
    vy = vy / np.linalg.norm(vy)
    vx = vx.reshape([3, 1])
    vy = vy.reshape([3, 1])
    vz = vz.reshape([3, 1])
    V = np.concatenate([vx, vy, vz], axis=-1)
    return V

def wrap_angle_around_90(angles):
    angles %= 360
    angles[np.logical_and((angles > 90), (angles < 180))] -= 180
    theta_bet_180_270 = angles[np.logical_and((angles >= 180), (angles < 270))]
    # angles[np.logical_and((angles >= 180), (angles < 270))] = 180 - theta_bet_180_270
    angles[np.logical_and((angles >= 180), (angles < 270))] = theta_bet_180_270 - 180
    angles[np.logical_and((angles >= 270), (angles < 360))] -= 360
    return angles

def derive_motor_angles_v0(orientation_matrix):
    orientation_matrix = np.around(orientation_matrix, decimals=2)

    b = orientation_matrix[0, 1]
    e = orientation_matrix[1, 1]
    i = orientation_matrix[2, 2]
    g = orientation_matrix[2, 0]
    h = orientation_matrix[2, 1]

    s2 = h
    c2 = np.sqrt(1 - s2 ** 2) # c2 is either a positive or negative angle. We need to try those angle combinations,
                                # and choose the angle combination with the lowest sum of angles
    angle_signs = [1, -1]
    join_sum = []
    joint_combinations = np.zeros((2,3))
    for j, angle_sign in enumerate(angle_signs):
        c2_i = c2 * angle_sign
        if c2_i == 0:
            print("Potentially Gimbal lock")
        c2_i = np.sign(c2_i)*np.maximum(0.0001, np.abs(c2_i)) # Gimbal lock case - Need to look into this
        s1 = e / c2_i
        c1 = b / c2_i
        s3 = i / c2_i
        c3 = g / c2_i
        theta_1 = np.arctan2(s1, c1) / (np.pi / 180)
        theta_2 = np.arctan2(s2, c2_i) / (np.pi / 180)
        theta_3 = np.arctan2(s3, c3) / (np.pi / 180)
        joint_combinations[j,:] = np.array([theta_1,theta_2, theta_3])
        join_sum.append(np.sum(np.abs(np.array([theta_1,theta_2, theta_3]))))
    theta_1, theta_2, theta_3 = joint_combinations[np.argmin(join_sum)] # Choosing the angle combo with the least
                                                                        # deviation required

    print(joint_combinations[np.argmin(join_sum)])
    theta_1_wrapped = wrap_angle_around_90(np.array([theta_1]))[0]
    theta_2_wrapped = wrap_angle_around_90(np.array([theta_2]))[0]
    theta_3_wrapped = wrap_angle_around_90(np.array([theta_3]))[0]
    return [theta_1_wrapped, theta_2_wrapped, theta_3_wrapped]


# def derive_motor_angles_v0(orientation_matrix):
#     orientation_matrix = np.around(orientation_matrix, decimals=2)
#     a = orientation_matrix[0, 0]
#     b = orientation_matrix[0, 1]
#     c = orientation_matrix[0, 2]
#     d = orientation_matrix[1, 0]
#     e = orientation_matrix[1, 1]
#     f = orientation_matrix[1, 2]
#     i = orientation_matrix[2, 2]
#     g = orientation_matrix[2, 0]
#     h = orientation_matrix[2, 1]
#
#     # value of i is what determines Gimbal lock: this is the case if i = +- 1, leading theta2 to eval to +/- 90 degrees
#
#     if np.abs(i) != 1:
#         theta_2_1 = np.arcsin(h)
#         theta_2_2 = np.pi - theta_2_1
#
#         theta_3_1 = np.arctan2(i / np.cos(theta_2_1), g / np.cos(theta_2_1))
#         theta_3_2 = np.arctan2(i / np.cos(theta_2_2), g / np.cos(theta_2_2))
#
#         theta_1_1 = np.arctan2(e / np.cos(theta_2_1), b / np.cos(theta_2_1))
#         theta_1_2 = np.arctan2(e / np.cos(theta_2_2), b / np.cos(theta_2_2))
#
#         theta_1_1 /= (np.pi / 180)
#         theta_1_2 /= (np.pi / 180)
#         theta_2_1 /= (np.pi / 180)
#         theta_2_2 /= (np.pi / 180)
#         theta_3_1 /= (np.pi / 180)
#         theta_3_2 /= (np.pi / 180)
#         angle_set_1 = np.array([theta_1_1, theta_2_1, theta_3_1])
#         angle_set_2 = np.array([theta_1_2, theta_2_2, theta_3_2])
#
#         if np.sum(np.abs(angle_set_1)) < np.sum(np.abs(angle_set_2)):
#             final_angles = angle_set_1
#         else:
#             final_angles = angle_set_2
#         theta_1, theta_2, theta_3 = final_angles
#         # print('First Set of Angles : ', ('ps', theta_1_1, 'ur', theta_2_1, 'fe', theta_3_1), '\n')
#         # print('Second Set of Angles : ', ('ps', theta_1_2, 'ur', theta_2_2, 'fe', theta_3_2), '\n')
#         # print('Final Set of Angles : ', ('ps', theta_1, 'ur', theta_2, 'fe', theta_3), '\n')
#     else:
#         print ('CASE NOT CORRECTED')
#         theta_1 = 0
#         if i == -1:
#             theta_2 = -np.pi/2
#             theta_3 = theta_1 + np.arctan2(c, a)
#             # theta_3 = theta_1 + np.arctan2(-a, -b)
#         else:
#             theta_2 = np.pi/2
#             theta_3 = -theta_1 + np.arctan2(-d, -a)
#             # theta_3 = -theta_1 + np.arctan2(a, b)
#         theta_1 /= (np.pi / 180)
#         theta_2 /= (np.pi / 180)
#         theta_3 /= (np.pi / 180)
#
#
#     ps_home = 0
#     ur_home = 0
#     fe_home = -0
#
#     theta_1_corrected = theta_1 - ps_home
#     theta_2_corrected = -1*(ur_home - theta_2)
#     theta_3_corrected = theta_3 - fe_home
#     # print('Corrected Set of Angles : ', ('ps', theta_1_corrected, 'ur', theta_2_corrected, 'fe', theta_3_corrected),'\n')
#
#     theta_1_wrapped = wrap_angle_around_90(np.array([theta_1_corrected]))[0]
#     theta_2_wrapped = wrap_angle_around_90(np.array([theta_2_corrected]))[0]
#     theta_3_wrapped = wrap_angle_around_90(np.array([theta_3_corrected]))[0]
#     # print('Wrapped Set of Angles : ', ('ps', theta_1_wrapped, 'ur', theta_2_wrapped, 'fe', theta_3_wrapped),'\n')
#
#     # print('\n'*3)
#     return [theta_1_wrapped, theta_2_wrapped, theta_3_wrapped]

def orient_wrist(theta1, theta2, theta3):
    # This function takes the joint angles, theta1, theta2, theta3, for
    # pronate/supinate, ulnar/radial, and flexion/extension, respectively, and computes the 8 bit integers
    # needed to drive the hand

    # Joint center positions as an 8 bit integer
    ps_home = 102
    ur_home = 180
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

    #
    # print('Clipped values: ', theta1, theta2, theta3)
    # print('Output values: ', joint_output)
    return joint_output

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# different resolutions of color and depth streams
image_width = 640
image_height = 480
fps = 15
frame_count = 0
cam_angles_t_1 = np.zeros(3,)
previous_millis = 0

config = rs.config()
config.enable_stream(rs.stream.depth, image_width, image_height, rs.format.z16, fps)
config.enable_stream(rs.stream.color, image_width, image_height, rs.format.rgb8, fps)
config.enable_stream(rs.stream.accel)
config.enable_stream(rs.stream.gyro)
# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
# print("Depth Scale is: ", depth_scale)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)
colorizer = rs.colorizer()

ser = serial.Serial('COM6', 115200, timeout=0.05)
ser.flushInput()

counter = 0
num_samples_for_yaw = 10 # Avg first 200 samples to get a good measure of where the home yaw position is
yaw_sum = 0
previous_millis_move = 0
current_millis_move = 0
frame_count_move = 0
update_count = 0

# Streaming loop
for i in list(range(20)):
    frames = pipeline.wait_for_frames()
# cv2.namedWindow('MASK-GRASP RCNN OUTPUT', cv2.WINDOW_AUTOSIZE)

while True:
    try:
        current_millis = int(round(time.time() * 1000))
        if frame_count == 0:
            previous_millis= current_millis
        if (current_millis - previous_millis) >= 50:
            dt = current_millis - previous_millis
            previous_millis = current_millis
            update_count+=1
        else:
            frame_count += 1
            continue

        frames = pipeline.wait_for_frames()
        # previous_millis, dt = compute_dt(previous_millis, frame_count)
        # if dt > 2:
        #     previous_millis, dt = compute_dt(previous_millis, frame_count, zero_timer=True)
        # print (dt)
        # Getting Camera Pose
        cam_angles_t = intel_realsense_IMU.camera_orientation_realsense(frames, dt, cam_angles_t_1)
        cam_angles_t_1 = np.array(cam_angles_t)

        frame_count += 1
        ser.write(b'r') # Arm will only respond when it is done moving all the joints
        input_bytes = ser.readline()
        while input_bytes == b'':
            # print('did not receive a ping back yet')
            serial_read = ser.write(b'r')
            input_bytes = ser.readline()

        decoded_bytes = np.array(input_bytes.decode().replace('\r','').replace('\n','').split('\t'), dtype='float32')
        theta_pitch, theta_roll, theta_yaw = decoded_bytes.tolist()
        if counter < num_samples_for_yaw:
            yaw_sum += theta_yaw
            counter += 1
            print("WARNING: ZEROING MAGNETOMETER!...")
            continue
        elif counter == num_samples_for_yaw:
            yaw_resting = yaw_sum / (num_samples_for_yaw)
            print('DONE CALIBRATION')
        else:
            theta_yaw -= yaw_resting


        approach_vector = np.array([0,1,0])
        approach_vector = approach_vector/np.linalg.norm(approach_vector)
        theta = 30 # Relative to the positive x-axis
        # theta = theta + 90  # Relative to the positive Y axis: CW rotaion = positive, CCW = negative
        theta = wrap_angle_around_90(np.array([theta]))[0]
        theta = theta * (np.pi / 180)  # network outputs positive angles in bottom right quadrant
        V = compute_wrist_orientation(approach_vector, theta)
        rotation_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                               [np.sin(theta), np.cos(theta), 0],
                               [0, 0, 1]])
        grasp_orientation_camera = np.dot(V, rotation_z)


        cam_pitch, cam_yaw, cam_roll = cam_angles_t
        arm_pitch, arm_roll, arm_yaw = [theta_pitch, theta_roll, theta_yaw]
        # print (arm_pitch, arm_roll, arm_yaw)

        R_shoulder_camera = R.from_euler('xyz', [[-0, 0, 0]],
                                         degrees=True).as_matrix().squeeze()
        R_shoulder_elbow = R.from_euler('xyz', [[0,0,0]],
                                        degrees=True).as_matrix().squeeze()
        R_elbow_shoulder = np.linalg.inv(R_shoulder_elbow)
        grasp_orientation_shoulder = np.dot(R_shoulder_camera, grasp_orientation_camera)

        calibration_matrix = np.array([[1,0,0],
                                       [0,1,0],
                                       [0,0,1]])
        calibration_matrix_inv = np.linalg.inv(calibration_matrix)
        grasp_orientation_elbow = np.dot(np.dot(R_elbow_shoulder, grasp_orientation_shoulder), calibration_matrix_inv)

        # theta1 : pronate/supinate
        # theta2 : ulnar/radial
        # theta3 : flexion/extension
        theta1, theta2, theta3 = derive_motor_angles_v0(grasp_orientation_elbow)
        theta1 = theta1 - arm_roll
        theta3 = theta3 - arm_pitch
        joint1, joint2, joint3 = orient_wrist(theta1, theta2, theta3).tolist()
        string_command = 'w %d %d %d' % (joint3, joint2, joint1)
        if update_count%1 == 0:
            print('ps', theta1, 'ur', theta2, 'fe', theta3)
            ser.write(string_command.encode())

        frame_count_move+=1
        counter += 1
    except:
        print("Keyboard Interrupt")
        break