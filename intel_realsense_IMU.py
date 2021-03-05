########################################################################################################################
# Mena SA Kamel
# Student Number: 251064703
# MESc Candidate, Robotics and Control
# Electrical and Computer Engineering, Western University

# THIS SCRIPT COMPUTES THE INEL REALSENSE ORIENTATION FROM THE ONBOARD IMU
########################################################################################################################

import numpy as np

def gyro_data(gyro):
    return np.asarray([gyro.x, gyro.y, gyro.z])

def accel_data(accel):
    return np.asarray([accel.x, accel.y, accel.z])

def compute_angles_from_accelerometer(ax, ay, az):
    # This function computes the camera euler angles from the accelerometer readings: ax, ay, az
    theta_x = -((np.arctan2(ay, -az) * (180 / np.pi)) + 90)
    theta_z = -((np.arctan2(ay, ax) * (180 / np.pi)) + 90)
    theta_y = np.arctan2(-az, ax) * (180 / np.pi)
    return [theta_x, theta_y, theta_z]

def camera_orientation_realsense(frames, dt, angles_t_1):
    # theta_pitch -> Rotation about x
    # theta_roll -> Rotation about z
    # theta_yaw -> Rotation about y
    accel = accel_data(frames[2].as_motion_frame().get_motion_data())
    gyro = gyro_data(frames[3].as_motion_frame().get_motion_data())
    ax, ay, az = np.split(accel / 9.8, indices_or_sections=3)
    # Computing the angle based on the accelerometer readings (Degrees)
    accelerometer_angles = np.array(compute_angles_from_accelerometer(ax, ay, az)).squeeze()
    # Computing the angle based on the Gyroscope readings (Degrees)
    gyro_angles = (gyro * dt * (180 / np.pi))
    # Complementary filter for accurate pitch and roll (Degrees)
    G = 0.9
    A = 0.1
    angles_t = (angles_t_1 + gyro_angles) * G + accelerometer_angles * A
    theta_pitch, _, theta_roll = angles_t
    theta_yaw = angles_t_1[1] + gyro_angles[1]
    return [theta_pitch, theta_yaw, theta_roll]