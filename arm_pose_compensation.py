import numpy as np
import serial
import time
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def wrap_angle_around_90(angles):
    angles %= 360
    angles[np.logical_and((angles > 90), (angles < 180))] -= 180
    theta_bet_180_270 = angles[np.logical_and((angles >= 180), (angles < 270))]
    # angles[np.logical_and((angles >= 180), (angles < 270))] = 180 - theta_bet_180_270
    angles[np.logical_and((angles >= 180), (angles < 270))] = theta_bet_180_270 - 180
    angles[np.logical_and((angles >= 270), (angles < 360))] -= 360
    return angles

def initialize_serial_link(com_port, baud_rate):
    ser = serial.Serial(com_port, baud_rate)
    ser.flushInput()
    return ser

def live_plotter(x_vec, acc_history, gyro_history, mag_history, sys_history, axes_objects, identifier='', pause_time=0.0001):
    acc_x, acc_y, acc_z = np.split(acc_history, indices_or_sections=3, axis=0)
    gyro_x, gyro_y, gyro_z = np.split(gyro_history, indices_or_sections=3, axis=0)
    sys_x, sys_y, sys_z = np.split(sys_history, indices_or_sections=3, axis=0)
    ax = axes_objects[0]
    line1, line2, line3, line4, line5, line6, line7, line8, line9, line10 = axes_objects
    if ax == []:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec, acc_x.squeeze(), '-o', alpha=0.8, label='Accelerometer - X')
        line2, = ax.plot(x_vec, acc_y.squeeze(), '-o', alpha=0.8, label='Accelerometer - Y')
        line3, = ax.plot(x_vec, acc_z.squeeze(), '-o', alpha=0.8, label='Accelerometer - Z')
        line4, = ax.plot(x_vec, gyro_x.squeeze(), '-o', alpha=0.8, label='Gyro - X')
        line5, = ax.plot(x_vec, gyro_y.squeeze(), '-o', alpha=0.8, label='Gyro - Y')
        line6, = ax.plot(x_vec, gyro_z.squeeze(), '-o', alpha=0.8, label='Gyro - Z')
        line7, = ax.plot(x_vec, mag_history.squeeze(), '-o', alpha=0.8, label='Magnetometer - Yaw - Z')
        line8, = ax.plot(x_vec, sys_x.squeeze(), '-o', alpha=0.8, label='System - Pitch - X')
        line9, = ax.plot(x_vec, sys_y.squeeze(), '-o', alpha=0.8, label='System - Roll - Y')
        line10, = ax.plot(x_vec, sys_z.squeeze(), '-o', alpha=0.8, label='Sys - Z')

        # update plot label/title
        plt.ylabel('Acceleration')
        plt.title('RealSense IMU Output'.format(identifier))
        plt.legend()
        plt.show()

    line8.set_ydata(sys_x)
    line9.set_ydata(sys_y)
    line10.set_ydata(mag_history)
    plt.pause(pause_time)
    plt.ylim([-180, 180])

    # return line so we can update it again in the next iteration
    return [line1, line2, line3, line4, line5, line6, line7, line8, line9, line10]

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

    # ps_home = 0
    # ur_home = 90
    # fe_home = -90
    ps_home = 0
    ur_home = 0
    fe_home = -0

    theta_1_corrected = theta_1 - ps_home
    theta_2_corrected = -1*(ur_home - theta_2)
    theta_3_corrected = theta_3 - fe_home

    theta_1_wrapped = wrap_angle_around_90(np.array([theta_1_corrected]))[0]
    theta_2_wrapped = wrap_angle_around_90(np.array([theta_2_corrected]))[0]
    theta_3_wrapped = wrap_angle_around_90(np.array([theta_3_corrected]))[0]
    return [theta_1_wrapped, theta_2_wrapped, theta_3_wrapped]

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

ser = initialize_serial_link('COM6', 115200)

data_stream_size = 100
x_vals = np.linspace(0,1,data_stream_size+1)[0:-1]
acc_history = gyro_history = sys_history = np.array(np.zeros([3, len(x_vals)]))
mag_history = np.zeros(len(x_vals))
axes_objects = [[],[],[],[],[],[],[],[],[],[]]
num_samples = 200
orientations = ['+x', '-x', '+y', '-y', '+z', '-z']
counter = 0
ax_scale = 1.004
ax_bias = 2.1
ay_scale = 1.0004
ay_bias = 2.55
az_scale = 1.1
az_bias = 2.1

mag_A_matrix = np.array([[1.0082,   -0.0171,    0.0179],
                         [-0.0171,    1.0053,    0.0296],
                         [0.0179,    0.0296,    0.9881]])
mag_b_vector = np.array([57.3756,   78.6094,    9.1631]).reshape([1, 3])

previous_millis = 0
gyro_previous = np.array([0, 0, 0])
sys_previous = np.array([0, 0, 0])
gz_history = []
num_meas = 700
magnetomer_readings = np.zeros([num_meas, 3])
yaw_history = 0
num_samples_for_yaw = 50 # Avg first 20 samples to get a good measure of where the home yaw position is
sum_yaw = 0
yaw_cal = []

while True:
    try:
        current_millis = int(round(time.time() * 1000))
        if counter == 0:
            previous_millis = current_millis
        dt = (current_millis - previous_millis) / 1000
        previous_millis = current_millis
        input_bytes = ser.readline()
        counter += 1

        decoded_bytes = np.array(input_bytes.decode().replace('\r','').replace('\n','').split('\t'), dtype='float32')
        ay, ax, az, gy, gx, gz, mx, my, mz, T = decoded_bytes.tolist()

        # Accelerometer
        ax = (-ax - ax_bias)/ax_scale
        ay = (-ay - ay_bias)/ay_scale
        az = (az - az_bias)/az_scale

        acc_theta_x = np.arctan2(az, ay) * (180 / np.pi)
        acc_theta_y = np.arctan2(-az, ax) * (180 / np.pi)
        acc_theta_z = np.arctan2(ay, ax) * (180 / np.pi)
        acc_history[:, -1] = np.array([-(acc_theta_x+90), -(acc_theta_y-90), acc_theta_z]).squeeze()

        # Gyro
        gyro = np.array([gx,gy,gz])
        gyro_current = gyro_previous + gyro * dt
        gyro_history[:, -1] = gyro_current
        # gyro_history[:, -1] = gyro*(180 / np.pi)
        gz_history.append([gyro_history[-1, -1]])

        # Complementary filter
        G = 0.9
        A = 0.1
        sys_history[:, -1] = (sys_history[:, -2] + (gyro * dt * (180 / np.pi))) * G + acc_history[:, -1] * A

        # All measurements in the accelerometer frame of reference
        # Magnetometer
        mag_uncalibrated = np.array([mx, my, mz]).reshape([1, 3])
        mag_calibrated = np.dot((mag_uncalibrated - mag_b_vector), mag_A_matrix)

        mx, my, mz = mag_calibrated.squeeze().tolist()
        memory = my
        my = mx
        mx = memory
        mz = -mz

        theta_pitch = sys_history[0, -1]*(np.pi/180)
        theta_roll = sys_history[1, -1]*(np.pi/180)

        # Yaw tilt compensation
        x_heading = mx*np.cos(theta_roll) + mz*(np.sin(theta_roll))
        y_heading = mx*(np.sin(theta_roll)*np.sin(theta_pitch)) + my*np.cos(theta_pitch) - mz*(np.cos(theta_roll)*np.sin(theta_pitch))

        theta_yaw = np.arctan2(y_heading, x_heading) * (180 / np.pi)

        # Low pass filtering the yaw data
        mag_history[-1] = 0.1*mag_history[-2] + 0.9*(theta_yaw - yaw_history)
        if counter==num_samples_for_yaw:
            yaw_history = np.sum(mag_history) / num_samples_for_yaw
            print('DONE CALIBRATION')
        elif counter<num_samples_for_yaw:
            print('CALIBRATING MAG')

        # axes_objects = live_plotter(x_vals, acc_history, gyro_history, mag_history, sys_history, axes_objects)
        acc_history = np.append(acc_history[:, 1:], np.zeros([3, 1]), axis=-1)
        gyro_history = np.append(gyro_history[:, 1:], np.zeros([3, 1]), axis=-1)
        sys_history = np.append(sys_history[:, 1:], np.zeros([3, 1]), axis=-1)
        mag_history = np.append(mag_history[1:], 0)
        gyro_previous = gyro_current

        if counter<num_samples_for_yaw+2:
            continue
        # theta_pitch -> Rotation about x (flexion/extension)
        # theta_roll -> Rotation about z (pronate/supinate)
        # theta_yaw -> Rotation about y (ulnar/radial)
        # arm_orientation = R.from_euler('xyz', [[theta_pitch, theta_roll, theta_yaw]], degrees=True).as_matrix()

        theta_pitch = theta_pitch * (180/np.pi)
        theta_roll = theta_roll * (180/np.pi)
        theta_yaw = theta_yaw - yaw_history
        # arm_orientation = R.from_euler('xyz', [[theta_pitch, 0, theta_roll]], degrees=True).as_matrix().squeeze()
        # arm_orientation = R.from_euler('xyz', [[theta_pitch, theta_yaw, theta_roll]], degrees=True).as_matrix().squeeze()
        arm_orientation = R.from_euler('xyz', [[0, theta_yaw, 0]], degrees=True).as_matrix().squeeze()
        inverse_arm_orientation = np.linalg.inv(arm_orientation)

        calibration_matrix = np.array([[0, 0, 1],
                                       [0, -1, 0],
                                       [1, 0, 0]])
        inverse_arm_orientation = np.dot(inverse_arm_orientation, calibration_matrix)
        theta1, theta2, theta3 = derive_motor_angles_v0(inverse_arm_orientation)
        joint1, joint2, joint3 = orient_wrist(theta1, theta2, theta3).tolist()
        string_command = 'w %d %d %d' % (joint3, joint2, joint1)
        print (theta1, theta2, theta3)
        ser.write(string_command.encode())
        # print (dt)
    except:
        print("Keyboard Interrupt")
        break