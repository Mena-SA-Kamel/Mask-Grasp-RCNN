########################################################################################################################
# ECE 9516 - Topics in Autonomous Robotics - Final Project
# Mena SA Kamel
# Student Number: 251064703
# MESc Candidate, Robotics and Control
# Electrical and Computer Engineering, Western University
########################################################################################################################

import pyrealsense2 as rs
import numpy as np
import os
import cv2
from PIL import Image
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import time
# plt.style.use('ggplot')

def live_plotter(x_vec, acc_history, gyro_history, axes_objects, identifier='', pause_time=0.1):
    acc_x, acc_y, acc_z = np.split(acc_history, indices_or_sections=3, axis=0)
    gyro_x, gyro_y, gyro_z = np.split(gyro_history, indices_or_sections=3, axis=0)
    ax = axes_objects[0]
    line1, line2, line3, line4, line5, line6 = axes_objects
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
        # update plot label/title
        plt.ylabel('Acceleration (m/s^2)')
        plt.title('RealSense IMU Output'.format(identifier))
        plt.legend()
        plt.show()

    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(acc_x)
    line2.set_ydata(acc_y)
    line3.set_ydata(acc_z)
    line4.set_ydata(gyro_x)
    line5.set_ydata(gyro_y)
    line6.set_ydata(gyro_z)
    # # adjust limits if new data goes beyond bounds
    # if np.min(acc_y) <= line1.axes.get_ylim()[0] or np.max(acc_y) >= line1.axes.get_ylim()[1]:
    #     plt.ylim([np.min(acc_y) - np.std(acc_y), np.max(acc_y) + np.std(acc_y)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)
    plt.ylim([-20, 20])

    # return line so we can update it again in the next iteration
    return [line1, line2, line3, line4, line5, line6]

def gyro_data(gyro):
    return np.asarray([gyro.x, gyro.y, gyro.z])

def accel_data(accel):
    return np.asarray([accel.x, accel.y, accel.z])

def butterworth_filter(type, data, cutoff, fs, order=5):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff / nyquist_freq
    b, a = butter(order, normal_cutoff, btype=type, analog=False)
    y = lfilter(b, a, data)
    return y

def capture_frames(num_frames, x_vals, acc_history, gyro_history, axes_objects, frame_rate=15):
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
            time_step = current_millis - previous_millis
            previous_millis = current_millis

            # Getting IMU readings
            accel = accel_data(frames[2].as_motion_frame().get_motion_data())
            gyro = gyro_data(frames[3].as_motion_frame().get_motion_data())
            color_image = np.asanyarray(aligned_color_frame.get_data())
            color_image_to_display = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('RealSense Feed', color_image_to_display)

            acc_history[:, -1] = accel
            gyro_history[:, -1] = gyro

            axes_objects = live_plotter(x_vals, acc_history, gyro_history, axes_objects)

            acc_history = np.append(acc_history[:, 1:], np.zeros([3, 1]), axis=-1)
            gyro_history = np.append(gyro_history[:, 1:], np.zeros([3, 1]), axis=-1)


            # y_vec stores the readings of the signal

            # acc_data[-1] = accel[-1]
            # line1 = live_plotter(x_vals, acc_data, line1)
            # y_vals = np.append(y_vals[1:], 0.0)
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
axes_objects = [[], [], [], [], [], []]

acc_history = np.array([acc_x, acc_y, acc_z])
gyro_history = np.array([gyro_x, gyro_y, gyro_z])



capture_frames(-1, x_vals, acc_history, gyro_history, axes_objects, frame_rate)

# accelerometer_filtered = np.zeros(accelerometer_data.shape)
# for i in list(range(3)):
#     accelerometer_data[:, i] = accelerometer_data[:,i] - np.mean(accelerometer_data[:,i])
#     noise = np.logical_or((accelerometer_data[:, i] < -1 * noise_level), (accelerometer_data[:, i] > noise_level))
#     accelerometer_data[:, i] = accelerometer_data[:, i] * noise
#     max_unfiltered = np.max(accelerometer_data[:, i])
#     if np.sum(accelerometer_data[:, i]) == 0:
#         accelerometer_filtered[:, i] = accelerometer_data[:, i]
#         continue
#     # accelerometer_filtered[:,i] = butterworth_filter('low',accelerometer_data[:,i],1.5,15)
#     # accelerometer_filtered[:, i] = butterworth_filter('high',accelerometer_filtered[:,i],0.01,15)
#     accelerometer_filtered[:, i] = accelerometer_data[:, i]
#     max_filtered = np.max(accelerometer_filtered[:, i])
#     scale = max_unfiltered / max_filtered
#     accelerometer_filtered[:, i] = accelerometer_filtered[:, i] * scale
#
#
# # t = 1/frame_rate
# t = time_data/1000
# dx = np.zeros(num_frames)
# dy = np.zeros(num_frames)
# dz = np.zeros(num_frames)
# vx = 0; vy = 0; vz = 0
# i = 0
# for a in accelerometer_filtered:
#     dx[i] = (vx*t[i]) + 0.5*a[0]*t[i]**2
#     dy[i] = (vy*t[i]) + 0.5*a[1]*t[i]**2
#     dz[i] = (vz*t[i]) + 0.5*a[2]*t[i]**2
#     vx = vx + a[0] * t[i]
#     vy = vy + a[1] * t[i]
#     vz = vz + a[2] * t[i]
#     i = i + 1
#
# time = list(range(num_frames))
# x_acceleration = accelerometer_data[:,0]
# y_acceleration = accelerometer_data[:,1]
# z_acceleration = accelerometer_data[:,2]
# fig4,(ax1, ax2, ax3) = plt.subplots(3,1, sharex='col', sharey='row')
# ax1.plot(time, x_acceleration); ax1.set_title('x'); ax1.set_ylim(np.min(accelerometer_data)-0.05, np.max(accelerometer_data)+0.05)
# ax2.plot(time, y_acceleration);ax2.set_title('y'); ax2.set_ylim(np.min(accelerometer_data)-0.05, np.max(accelerometer_data)+0.05)
# ax3.plot(time, z_acceleration);ax3.set_title('z'); ax3.set_ylim(np.min(accelerometer_data)-0.05, np.max(accelerometer_data)+0.05)
# ax1.set_title('A unfiltered')
# fig4.show()
#
# from scipy.fftpack import fft
# N = num_frames
# T = 1/frame_rate
# x = np.linspace(0.0, N*T, N)
# x_acceleration_fft = fft(x_acceleration)
# x_acceleration_filtered_fft = fft(accelerometer_filtered[:,0])
# xf = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
#
# fig5,(ax1) = plt.subplots(1,1)
# ax1.plot(xf, 2.0/N * np.abs(x_acceleration_fft[0:int(N/2)]))
# ax1.grid()
# fig5.show()
#
# fig6,(ax1) = plt.subplots(1,1)
# ax1.plot(xf, 2.0/N * np.abs(x_acceleration_filtered_fft[0:int(N/2)]))
# ax1.grid()
# fig6.show()
#
# time = list(range(num_frames))
# accelerometer_data = accelerometer_filtered
# x_acceleration = accelerometer_data[:,0]
# y_acceleration = accelerometer_data[:,1]
# z_acceleration = accelerometer_data[:,2]
# fig,(ax1, ax2, ax3) = plt.subplots(3,1)
# # import code; code.interact(local=dict(globals(), **locals()))
#
# ax1.plot(time, x_acceleration); ax1.set_title('x'); ax1.set_ylim(np.min(accelerometer_data)-0.05, np.max(accelerometer_data)+0.05)
# ax2.plot(time, y_acceleration);ax2.set_title('y'); ax2.set_ylim(np.min(accelerometer_data)-0.05, np.max(accelerometer_data)+0.05)
# ax3.plot(time, z_acceleration);ax3.set_title('z'); ax3.set_ylim(np.min(accelerometer_data)-0.05, np.max(accelerometer_data)+0.05)
# ax1.set_title('A filtered')
# fig.show()
#
# x_gyro = gyroscope_data[:,0]
# y_gyro = gyroscope_data[:,1]
# z_gyro = gyroscope_data[:,2]
# fig2,(ax1, ax2, ax3) = plt.subplots(3,1)
# ax1.plot(time, x_gyro); ax1.set_title('x'); ax1.set_ylim(np.min(gyroscope_data)-0.05, np.max(gyroscope_data)+0.05)
# ax2.plot(time, y_gyro);ax2.set_title('y'); ax2.set_ylim(np.min(gyroscope_data)-0.05, np.max(gyroscope_data)+0.05)
# ax3.plot(time, z_gyro);ax3.set_title('z'); ax3.set_ylim(np.min(gyroscope_data)-0.05, np.max(gyroscope_data)+0.05)
# ax1.set_title('Gyroscope unfiltered')
# fig2.show()
#
# fig4,(ax1, ax2, ax3) = plt.subplots(3,1)
# d = np.array([dx, dy, dz])
# ax1.plot(time, dx); ax1.set_title('x'); ax1.set_ylim(np.min(d)-0.05, np.max(d)+0.05)
# ax2.plot(time, dy);ax2.set_title('y'); ax2.set_ylim(np.min(d)-0.05, np.max(d)+0.05)
# ax3.plot(time, dz);ax3.set_title('z'); ax3.set_ylim(np.min(d)-0.05, np.max(d)+0.05)
# ax1.set_title('distance')
# fig4.show()
#
# plt.show()