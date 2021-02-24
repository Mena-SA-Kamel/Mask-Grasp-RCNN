########################################################################################################################
# Mena SA Kamel
# Student Number: 251064703
# MESc Candidate, Robotics and Control
# Electrical and Computer Engineering, Western University
########################################################################################################################

import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from scipy.spatial.transform import Rotation as R

def live_plotter(x_vec, acc_history, gyro_history, system_history, axes_objects, identifier='', pause_time=0.1):
    acc_x, acc_y, acc_z = np.split(acc_history, indices_or_sections=3, axis=0)
    gyro_x, gyro_y, gyro_z = np.split(gyro_history, indices_or_sections=3, axis=0)
    system_x, system_y, system_z = np.split(system_history, indices_or_sections=3, axis=0)
    ax = axes_objects[0]
    line1, line2, line3, line4, line5, line6, line7, line8, line9 = axes_objects
    # line1, line2, line3= axes_objects
    #SOURCE: https://github.com/makerportal/pylive
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
        line7, = ax.plot(x_vec, system_x.squeeze(), '-o', alpha=0.8, label='Complementary Filter - X')
        line8, = ax.plot(x_vec, system_y.squeeze(), '-o', alpha=0.8, label='Complementary Filter - Y')
        line9, = ax.plot(x_vec, system_z.squeeze(), '-o', alpha=0.8, label='Complementary Filter - Z')
        # update plot label/title
        plt.ylabel('Theta (degrees)')
        plt.title('RealSense IMU Output'.format(identifier))
        plt.legend()
        plt.show()

    # after the figure, axis, and line are created, we only need to update the y-data
    # line1.set_ydata(acc_x)
    # line2.set_ydata(acc_y)
    # line3.set_ydata(acc_z)
    # line4.set_ydata(gyro_x)
    line5.set_ydata(gyro_y)
    # line6.set_ydata(gyro_z)
    line7.set_ydata(system_x)
    # line8.set_ydata(system_y)
    line9.set_ydata(system_z)
    # # adjust limits if new data goes beyond bounds
    # if np.min(acc_y) <= line1.axes.get_ylim()[0] or np.max(acc_y) >= line1.axes.get_ylim()[1]:
    #     plt.ylim([np.min(acc_y) - np.std(acc_y), np.max(acc_y) + np.std(acc_y)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)
    plt.ylim([-100, 100])

    # return line so we can update it again in the next iteration
    return [line1, line2, line3, line4, line5, line6, line7, line8, line9]

def gyro_data(gyro):
    return np.asarray([gyro.x, gyro.y, gyro.z])

def accel_data(accel):
    return np.asarray([accel.x, accel.y, accel.z])

def capture_frames(num_frames, x_vals, acc_history, gyro_history, system_history, axes_objects, frame_rate=15):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, frame_rate)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, frame_rate)
    config.enable_stream(rs.stream.accel)
    config.enable_stream(rs.stream.gyro)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    print("Depth Scale is: ", depth_scale)
    align_to = rs.stream.color
    align = rs.align(align_to)
    colorizer = rs.colorizer()
    frame_count = 0
    gyro_previous = np.array([0, 0, 0])
    gyro_y_data = []
    frame_number = 0
    num_frames = 2000
    t_sum = 0


    for i in list(range(frame_rate*5)):
        frames = pipeline.wait_for_frames()

    try:
        if num_frames != -1:
            condition = frame_count < num_frames
        else:
            condition = True
        while condition:
            frames = pipeline.wait_for_frames()

            # Aligning depth and rgb frames
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            aligned_color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frame or not aligned_color_frame:
                continue

            # Getting camera intrinsics
            depth_intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            color_intrinsics = aligned_color_frame.profile.as_video_stream_profile().intrinsics

            # Timing
            current_millis = int(round(time.time() * 1000))
            if frame_count == 0:
                previous_millis = current_millis
            dt = (current_millis - previous_millis)/1000
            previous_millis = current_millis

            # Getting IMU readings
            accel = accel_data(frames[2].as_motion_frame().get_motion_data())
            gyro = gyro_data(frames[3].as_motion_frame().get_motion_data())
            color_image = np.asanyarray(aligned_color_frame.get_data())
            color_image_to_display = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('RealSense Feed', color_image_to_display)

            ax,ay,az = np.split(accel/9.8, indices_or_sections=3)
            acc_theta_x = np.arctan2(ay, -az)*(180/np.pi)
            acc_theta_y = np.arctan2(-az, ax)*(180/np.pi)
            acc_theta_z = np.arctan2(ay, ax)*(180/np.pi)
            # acc_current = np.array([-np.atan2()])

            # gyro_current = [thetax, thetay, thetaz]
            gyro_current = gyro_previous + gyro*dt

            acc_history[:, -1] = np.array([-(acc_theta_x+90), acc_theta_y, -(acc_theta_z+90)]).squeeze()
            gyro_history[:, -1] = gyro_current * (180 / np.pi)

            # Correcting for Gyro drift in the Y axis
            m, b = [-0.17627199, 0.82105868]
            gyro_history[1, -1] = gyro_history[1, -1] - (m*t_sum + b)

            # complementary filter
            G = 0.90
            A = 0.1
            system_history[:, -1] = (system_history[:, -2] + (gyro*dt* (180 / np.pi)))*G + acc_history[:, -1]*A

            # Complementary filter does not work well with predicting yaw. Instead, replace those values with just
            # raw gyro data
            # Trying a high pass filter on the gyro = 1 - LPF
            lpf_gyro = 0.95*gyro_history[:, -2] + 0.05*(gyro_history[:, -1])
            # gyro_y_data.append(np.concatenate([gyro_history[:, -1], [dt]]))
            # hpf_gyro = 1 - lpf_gyro[1]


            system_history[1, -1] = gyro_history[1, -1]

            theta_x, theta_y, theta_z = np.split(system_history[:, -1], indices_or_sections=3)

            r = R.from_euler('zyx', [
                [theta_z, 0, 0],
                [0, theta_y, 0],
                [0, 0, theta_x]], degrees=True).as_matrix()
            rotation_matrix = np.dot(np.dot(r[0], r[1]), r[2])

            # This is the hard coded rotation as a -70 degree rotation about the x axis
            # array([[1., 0., 0.],
            #        [0., 0.34202014, 0.93969262],
            #        [0., -0.93969262, 0.34202014]])

            print(rotation_matrix, '\n')

            axes_objects = live_plotter(x_vals, acc_history, gyro_history, system_history, axes_objects)

            acc_history = np.append(acc_history[:, 1:], np.zeros([3, 1]), axis=-1)
            gyro_history = np.append(gyro_history[:, 1:], np.zeros([3, 1]), axis=-1)
            system_history = np.append(system_history[:, 1:], np.zeros([3, 1]), axis=-1)

            gyro_previous = gyro_current
            frame_number += 1
            t_sum += dt
            if frame_number == num_frames:
                data = np.array(gyro_y_data)
                np.savetxt('output_data_gyro.txt', data)
                break


            frame_count = frame_count + 1
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break
    finally:
        pipeline.stop()
        return depth_intrinsics, depth_scale

num_frames = 300
frame_rate = 60
noise_level = 0.025
data_stream_size = 100
x_vals = np.linspace(0,1,data_stream_size+1)[0:-1]
acc_x = np.zeros(len(x_vals))
acc_y = np.zeros(len(x_vals))
acc_z = np.zeros(len(x_vals))
gyro_x = np.zeros(len(x_vals))
gyro_y = np.zeros(len(x_vals))
gyro_z = np.zeros(len(x_vals))
system_x = np.zeros(len(x_vals))
system_y = np.zeros(len(x_vals))
system_z = np.zeros(len(x_vals))
axes_objects = [[], [], [], [], [], [], [], [], []]
# axes_objects = [[], [], []]

acc_history = np.array([acc_x, acc_y, acc_z])
gyro_history = np.array([gyro_x, gyro_y, gyro_z])
system_history = np.array([system_x, system_y, system_z])

capture_frames(-1, x_vals, acc_history, gyro_history, system_history, axes_objects, frame_rate)


# Correcting fot Gyro drift in the Y axis
# gyro_noise_data = np.loadtxt('output_data_gyro.txt')
# gx, gy, gz, dt = np.split(gyro_noise_data, indices_or_sections=4, axis=-1)
# time_stamps = np.cumsum(dt)
# m,b = np.polyfit(time_stamps, gy, 1)
#
# recon_gy = (m*time_stamps+b).reshape(-1, 1)
#
# fig, ax = plt.subplots()
# ax.plot(time_stamps, gx, label='Gyro - X')
# ax.plot(time_stamps, gy, label='Gyro - Y')
# ax.plot(time_stamps, gy - recon_gy, label='Gyro Reconstructed - Y')
# ax.plot(time_stamps, gz, label='Gyro - Z')
# ax.set_xlabel('Time (seconds)')
# ax.set_ylabel('Theta (degrees)')
# plt.title('Resting Gyro Readings')
# plt.legend()
# plt.show()
# import code; code.interact(local=dict(globals(), **locals()))
