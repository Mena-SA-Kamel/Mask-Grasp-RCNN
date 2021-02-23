import numpy as np
import serial
import time
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import os

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

    ps_home = 0
    # ur_home = 90
    # fe_home = -90
    # ps_home = 0
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

def compute_hand_aperture(grasp_box_width):
    # This function computes the motor command required to acheive an aperture of size grasp_box_width
    coef_path = 'aperture_mapping.txt'
    grasp_box_width = np.minimum(grasp_box_width, 90)
    grasp_box_width = np.maximum(grasp_box_width, 0)
    if os.path.exists(coef_path):
        coeffs = np.loadtxt(coef_path)
    else:
        motor_commands = np.arange(0, 1100, 100)
        aperture_size = np.array([90, 88, 75, 60, 40, 30, 18, 10, 8, 5.9, 5.4])
        coeffs = np.polynomial.polynomial.polyfit(aperture_size, motor_commands, 4)
        np.savetxt(coef_path, coeffs)
    motor_command = np.polynomial.polynomial.polyval(grasp_box_width, coeffs)
    motor_command = int(np.minimum(motor_command, 1000))
    return motor_command

def calibrate_magnetometer(mx, my, mz):
    # Calibrates the magnetometer for hard and soft sources
    mag_A_matrix = np.array([[0.9953, -0.0146, 0.0201],
                             [-0.0146, 1.0135, 0.0253],
                             [0.0201, 0.0253, 0.9926]])
    mag_b_vector = np.array([57.4634, 92.6249, 9.6638]).reshape([1, 3])
    # All measurements in the accelerometer frame of reference
    # Magnetometer
    mag_uncalibrated = np.array([mx, my, mz]).reshape([1, 3])
    mag_calibrated = np.dot((mag_uncalibrated - mag_b_vector), mag_A_matrix)

    mx, my, mz = mag_calibrated.squeeze().tolist()
    memory = my
    my = mx
    mx = memory
    mz = -mz
    return [mx, my, mz]

def calibrate_accelerometer(ax, ay, az):
    # Calibrate the accelerometer
    ax_scale = 1.004
    ax_bias = 2.1
    ay_scale = 1.0004
    ay_bias = 2.55
    az_scale = 1.1
    az_bias = 2.1
    # Accelerometer
    ax = (-ax - ax_bias) / ax_scale
    ay = (-ay - ay_bias) / ay_scale
    az = (az - az_bias) / az_scale
    return [ax, ay, az]

def read_serial_port(serial_object):
    # Reads the serial port to get the IMU readings
    input_bytes = serial_object.readline()
    decoded_bytes = np.array(input_bytes.decode().replace('\r', '').replace('\n', '').split('\t'), dtype='float32')
    ay, ax, az, gy, gx, gz, mx, my, mz, T = decoded_bytes.tolist()
    return [ax, ay, az, gx, gy, gz, mx, my, mz]

def compute_angles_from_accelerometer(ax, ay, az):
    # Computes the arm angle using accelerometer
    theta_x = -((np.arctan2(az, ay) * (180 / np.pi)) + 90)
    theta_y = -((np.arctan2(-az, ax) * (180 / np.pi)) - 90)
    theta_z = np.arctan2(ay, ax) * (180 / np.pi)
    return [theta_x, theta_y, theta_z]

def arm_orientation_imu_9250(ser, dt, angles_t_1):
    # Inputs: ser, dt, previous_angles in degrees, sample_counter, num_samples_for_yaw
    # Outputs: current_degrees
    # theta_pitch -> Rotation about x (flexion/extension)
    # theta_roll -> Rotation about z (pronate/supinate)
    # theta_yaw -> Rotation about y (ulnar/radial)
    # Reading IMU data
    ax, ay, az, gx, gy, gz, mx, my, mz = read_serial_port(ser)
    # Calibrate the Accelerometer reading
    ax, ay, az = calibrate_accelerometer(ax, ay, az)
    # Correcting for magnetic distortion from hard and soft sources
    mx, my, mz = calibrate_magnetometer(mx, my, mz)
    # Computing the angle based on the accelerometer readings (Degrees)
    accelerometer_angles = np.array(compute_angles_from_accelerometer(ax, ay, az)).squeeze()
    # Computing the angle based on the Gyroscope readings (Degrees)
    gyro_readings = np.array([gx, gy, gz])
    gyro_angles = (gyro_readings * dt * (180 / np.pi))
    # Complementary filter for accurate pitch and roll (Degrees)
    G = 0.9
    A = 0.1
    angles_t = (angles_t_1 + gyro_angles) * G + accelerometer_angles * A

    # Defining angles for time steps t and t-1 in radians
    theta_pitch, theta_roll, theta_yaw = (angles_t * (np.pi / 180)).tolist()  # in radians
    # Yaw tilt compensation
    x_heading = mx * np.cos(theta_roll) + mz * (np.sin(theta_roll))
    y_heading = mx * (np.sin(theta_roll) * np.sin(theta_pitch)) + my * np.cos(theta_pitch) - mz * (
                np.cos(theta_roll) * np.sin(theta_pitch))
    theta_yaw = np.arctan2(y_heading, x_heading) * (180 / np.pi)  # in degrees

    # Low pass filtering the yaw data
    theta_yaw_t_1 = angles_t_1[2]
    theta_yaw_lpf = 0.1 * theta_yaw_t_1 + 0.9 * theta_yaw  # Yaw angle is in degrees
    theta_pitch, theta_roll, _ = angles_t
    return [theta_pitch, theta_roll, theta_yaw_lpf]


ser = initialize_serial_link('COM6', 115200)

data_stream_size = 100
x_vals = np.linspace(0,1,data_stream_size+1)[0:-1]
acc_history = gyro_history = sys_history = np.array(np.zeros([3, len(x_vals)]))
mag_history = np.zeros(len(x_vals))
axes_objects = [[],[],[],[],[],[],[],[],[],[]]
num_samples = 200
orientations = ['+x', '-x', '+y', '-y', '+z', '-z']
counter = 0
previous_millis = 0
gyro_previous = np.array([0, 0, 0])
sys_previous = np.array([0, 0, 0])
gz_history = []
num_meas = 700
magnetomer_readings = np.zeros([num_meas, 3])
yaw_resting = 0
num_samples_for_yaw = 50 # Avg first 20 samples to get a good measure of where the home yaw position is
sum_yaw = 0
yaw_cal = []
hand_configured = False
angles_t_1 = np.zeros(3,)
angles_t = np.zeros(3,)
yaw_sum = 0
yaw_sample_counter = 0
angles_t_1 = np.zeros(3,)

while True:
    try:
        current_millis = int(round(time.time() * 1000))
        if counter == 0:
            previous_millis = current_millis
        dt = (current_millis - previous_millis) / 1000
        previous_millis = current_millis
        counter += 1

        theta_pitch, theta_roll, theta_yaw = arm_orientation_imu_9250(ser, dt, angles_t_1)
        # Correcting for the resting yaw position by averaging the first num_samples_for_yaw measurements
        angles_t_1 = np.array([theta_pitch, theta_roll, theta_yaw])
        if counter < num_samples_for_yaw:
            yaw_sum += theta_yaw
        elif counter ==num_samples_for_yaw:
            yaw_resting = yaw_sum / num_samples_for_yaw
        theta_yaw -= yaw_resting


        R_shoulder_to_elbow = R.from_euler('xyz', [[theta_pitch, 0, theta_roll]], degrees=True).as_matrix().squeeze()
        R_shoulder_to_elbow_inv = np.linalg.inv(R_shoulder_to_elbow)

        wrist_pose_shoulder_frame = np.array([[0, 0, 1],
                                              [0, -1, 0],
                                              [1, 0, 0]])
        wrist_pose_shoulder_frame_inv = np.linalg.inv(wrist_pose_shoulder_frame)

        grasp_pose_shoulder_frame = np.array([[1, 0, 0],
                                              [0, 1, 0],
                                              [0, 0, 1]])

        desired_orientation = np.dot(np.dot(R_shoulder_to_elbow_inv, grasp_pose_shoulder_frame), wrist_pose_shoulder_frame_inv)
        theta1, theta2, theta3 = derive_motor_angles_v0(desired_orientation)
        print(theta1, theta2, theta3)
        angle_thresh = 3
        desired_motor_angles = np.array([theta1, theta2, theta3])
        if ((-angle_thresh <= desired_motor_angles) & (desired_motor_angles <= angle_thresh)).all():
            # Home the hand joints if all the joints are within a certain range [-angle_thresh, angle_thresh]
            home_command = 'h'
            ser.write(home_command.encode())
            print 'HOME'
            continue
        joint1, joint2, joint3 = orient_wrist(theta1, theta2, theta3).tolist()
        string_command = 'w %d %d %d' % (joint3, joint2, joint1)
        aperture_command = 'j 0 %d' % (compute_hand_aperture(50))
        # if counter %1==0:
        #     ser.write(string_command.encode())
        ser.write(string_command.encode())
    except:
        print("Keyboard Interrupt")
        break