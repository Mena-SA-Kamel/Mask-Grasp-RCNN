import serial
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

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

    # after the figure, axis, and line are created, we only need to update the y-data
    # line1.set_ydata(acc_x)
    # line2.set_ydata(acc_y)
    # # line3.set_ydata(acc_z)
    # line4.set_ydata(gyro_x*(180/np.pi))
    # line5.set_ydata(gyro_y*(180/np.pi))
    line8.set_ydata(sys_x)
    line9.set_ydata(sys_y)
    line10.set_ydata(mag_history)
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
        ser.write(b'r')
        input_bytes = ser.readline()
        counter += 1
        #
        # if counter < 5:
        #     continue

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

        # Gathering data for calibration
        if counter<num_meas:
            if counter%10 ==0:
                print("Progress: ", (counter / num_meas ) * 100, "% DONE")
            magnetomer_readings[counter] = np.array([mx, my, mz])
        else:
            np.savetxt('magnetometer_calibration-Mar12-Calibration_with_case_imu_power_ON_new_home.txt', magnetomer_readings)
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(magnetomer_readings[:, 0], magnetomer_readings[:, 1], magnetomer_readings[:, 2])
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')
            ax.set_title("Magnetometer Calibration - Measurements in uT")
            import code;
            code.interact(local=dict(globals(), **locals()))
        time.sleep(0.1)

    except:
        print("Keyboard Interrupt")
        break