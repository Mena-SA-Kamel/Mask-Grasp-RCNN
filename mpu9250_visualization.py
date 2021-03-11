import serial
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

def live_plotter(x_vec, acc_history, gyro_history, mag_history, sys_history, axes_objects, identifier='', pause_time=0.000001):
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
        line10, = ax.plot(x_vec, sys_z.squeeze(), '-o', alpha=0.8, label='System - Yaw - Z')

        # update plot label/title
        plt.ylabel('Acceleration')
        plt.title('RealSense IMU Output'.format(identifier))
        plt.legend()
        plt.show()

    line8.set_ydata(sys_x)
    line9.set_ydata(sys_y)
    line10.set_ydata(sys_z)
    plt.pause(pause_time)
    plt.ylim([-180, 180])

    # return line so we can update it again in the next iteration
    return [line1, line2, line3, line4, line5, line6, line7, line8, line9, line10]


ser = serial.Serial('COM6', 115200)
ser.flushInput()

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

mag_A_matrix = np.array([[0.9953,   -0.0146,    0.0201],
                         [-0.0146,    1.0135,    0.0253],
                         [0.0201,    0.0253,    0.9926]])
mag_b_vector = np.array([57.4634,   78.6249,    9.6638]).reshape([1, 3])

previous_millis = 0
gyro_previous = np.array([0, 0, 0])
sys_previous = np.array([0, 0, 0])
gz_history = []
num_meas = 700
magnetomer_readings = np.zeros([num_meas, 3])
yaw_history = 0
num_samples_for_yaw = 500 # Avg first 200 samples to get a good measure of where the home yaw position is
yaw_sum = 0
yaw_cal = []

while True:
    try:
        ser.write(b'r')
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
        sys_history[:, -1] = np.array([theta_pitch, theta_roll, theta_yaw])

        axes_objects = live_plotter(x_vals, acc_history, gyro_history, mag_history, sys_history, axes_objects)
        acc_history = np.append(acc_history[:, 1:], np.zeros([3, 1]), axis=-1)
        gyro_history = np.append(gyro_history[:, 1:], np.zeros([3, 1]), axis=-1)
        sys_history = np.append(sys_history[:, 1:], np.zeros([3, 1]), axis=-1)
        mag_history = np.append(mag_history[1:], 0)
        counter += 1
    except:
        print("Keyboard Interrupt")
        break