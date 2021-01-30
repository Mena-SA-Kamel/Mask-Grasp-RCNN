import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytransform3d.rotations import *
import mrcnn.utils as utils
import mrcnn.model as modellib
from mask_grasp_rcnn import GraspMaskRCNNInferenceConfig, GraspMaskRCNNDataset
import datetime
import random
import time
import open3d as o3d
from matplotlib.path import Path
import serial
import time
import os

mouseX, mouseY = [0, 0]

def compute_real_box_size(intrinsics, aligned_depth_frame, rect):
    rect_camera_frame = get_camera_frame_box_coords(intrinsics, aligned_depth_frame, rect)
    # Need to find the width and height of the bounding box in the world/ camera frames
    # real_width is the average of the distance between P1-P2 and P0-P3
    # real_height is the average of the distance between P0-P1 and P2-P3
    #           w
    # P2 ---------------------- P1
    # ||                        ||
    # ||            .           || h
    # ||          (0,0,0)       ||
    # P3 ---------------------- P0
    P0 = rect_camera_frame[0]
    P1 = rect_camera_frame[1]
    P2 = rect_camera_frame[2]
    P3 = rect_camera_frame[3]

    real_width = np.mean([compute_distance(P1, P2), compute_distance(P0, P3)])
    real_height = np.mean([compute_distance(P0, P1), compute_distance(P2, P3)])
    # print('Grasp Width: ', real_width, 'Grasp Height: ', real_height)
    return [real_width, real_height]

def generate_pointcloud_from_rgbd(color_image, depth_image):
    # Creates an Open3D point cloud
    color_stream = o3d.geometry.Image(color_image)
    depth_stream = o3d.geometry.Image(depth_image)
    o3d_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_stream, depth_stream, depth_trunc=2.0)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(o3d_rgbd_image, o3d_intrinsics)
    return pcd

def generate_mask_from_polygon(image_shape, polygon_vertices):
    # This function returns a mask defined by some vertices
    ny, nx = image_shape[:2]
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T
    path = Path(polygon_vertices)
    grid = path.contains_points(points)
    grid = grid.reshape((ny, nx))
    return grid

def compute_distance(P1, P2):
    # This function computes the euclidean distance between two 3D points
    return np.sqrt((P1[0] - P2[0]) ** 2 + (P1[1] - P2[1]) ** 2 + (P1[2] - P2[2]) ** 2)

def generate_points_in_world_frame(real_width, real_height):
    # This function generates the points in the world frame of reference
    return np.array([[real_width/2, -real_height/2, 0],
                    [real_width/2, real_height/2, 0],
                    [-real_width/2, real_height/2, 0],
                    [-real_width/2, -real_height/2, 0]])

def de_project_point(intrinsics, depth_frame, point):
    # This function deprojects point from the image plane to the camera frame of reference using camera intrinsic
    # parameters
    x, y = point
    distance_to_point = depth_frame.as_depth_frame().get_distance(x, y)
    de_projected_point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], distance_to_point)
    return de_projected_point

def get_camera_frame_box_coords(intrinsics, depth_frame, box_vertices):
    # This function deprojects the vertex points that make up the grasp box from the image plane to the camera frame
    de_projected_points = []
    for point in box_vertices:
        de_projected_point = de_project_point(intrinsics, depth_frame, point)
        de_projected_points.append(de_projected_point)
    return np.array(de_projected_points)

def generate_random_color():
    r = int(random.random()*255)
    b = int(random.random()*255)
    g = int(random.random()*255)
    return (r, g, b)

def center_crop_frame(image, new_size):
    # This function center crops the image into a square of size w = h = new_size
    original_shape = image.shape
    diff_x = (original_shape[1] - new_size) // 2
    diff_y = (original_shape[0] - new_size) // 2
    new_image = image[diff_y:original_shape[0]-diff_y, diff_x:original_shape[1]-diff_x, :]
    return new_image

def resize_frame(image):
    min_dimension = np.min(image.shape[:2])
    max_dimension = np.max(image.shape[:2])
    diff = (max_dimension - min_dimension)//2
    square_image = image[:, diff:max_dimension-diff, :]
    square_image_resized = cv2.resize(square_image, dsize=(384, 384))
    return square_image_resized

def onMouse(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
       mouseX, mouseY = [x, y]

def generate_rgbd_image(color_image, depth_image, center_crop=True, square_size=384):
    depth_image = depth_image * (depth_image < 2000)
    depth_scaled = np.interp(depth_image, (depth_image.min(), depth_image.max()), (0, 1)) * 255
    rgbd_image = np.zeros([color_image.shape[0], color_image.shape[1], 4])
    rgbd_image[:, :, 0:3] = color_image[:, :, 0:3]
    rgbd_image[:, :, 3] = depth_scaled
    rgbd_image = rgbd_image.astype('uint8')
    if center_crop:
        rgbd_image = center_crop_frame(rgbd_image, square_size)
    return rgbd_image

def select_ROI(mouseX, mouseY, r):
    rois = r['rois']
    grasping_deltas = r['grasp_boxes']
    grasping_probs = r['grasp_probs']
    masks = r['masks']
    roi_scores = r['scores']
    selection_success = False
    if mouseY + mouseX != 0:
        print('x = %d, y = %d' % (mouseX, mouseY))
        # Here we need to find the detected BBOX that contains <mouseX, mouseY>. then pass those bbox coords to
        # the tracking loop
        y1, x1, y2, x2 = np.split(rois, indices_or_sections=4, axis=-1)
        y_condition = np.logical_and(y1 < mouseY, y2 > mouseY)
        x_condition = np.logical_and(x1 < mouseX, x2 > mouseX)
        selected_roi_ix = np.where(np.logical_and(x_condition, y_condition))[0]
        if selected_roi_ix.shape[0] > 0:
            selection_success = True
            selected_roi = rois[selected_roi_ix[0]]
            # Case when the coordinates of the cursor lies in two bounding boxes, if that is the case, we choose the
            # box with the smaller area
            if selected_roi_ix.shape[0] > 1:
                possible_rois = rois[selected_roi_ix]
                y1, x1, y2, x2 = np.split(possible_rois, indices_or_sections=4, axis=-1)
                box_areas = (y2 - y1) * (x2 - x1)
                selected_roi_ix = np.array([np.argmin(box_areas)])
                selected_roi = possible_rois[selected_roi_ix[0]]
            rois = selected_roi.reshape(1, 4)
            grasping_deltas = grasping_deltas[selected_roi_ix]
            grasping_probs = grasping_probs[selected_roi_ix]
            masks = masks[:,:,selected_roi_ix]
            roi_scores = roi_scores[selected_roi_ix]
            # print (masks.shape, roi_scores.shape, selected_roi_ix)

    return rois, grasping_deltas, grasping_probs, masks, roi_scores, selection_success

def evaluate_forward_kinematics(t1, t2, t3):
    # This function evaluates the forward kinematics of the robot at the specified joint angles
    inverse_kinematics = np.array([[((-np.cos(t1) * np.sin(t2) * np.cos(t3)) + (np.sin(t1) * np.sin(t3))),
                                    np.cos(t2) * np.cos(t1),
                                    (-(np.cos(t1) * np.sin(t2) * np.sin(t3)) - (np.sin(t1) * np.cos(t3)))],
                                   [((-np.sin(t1) * np.sin(t2) * np.cos(t3)) - (np.cos(t1) * np.sin(t3))),
                                    np.sin(t1) * np.cos(t2),
                                    ((-np.sin(t1) * np.sin(t2) * np.sin(t3)) + (np.cos(t1) * np.cos(t3)))],
                                   [np.cos(t2) * np.cos(t3), np.sin(t2), np.cos(t2) * np.sin(t3)]])
    return inverse_kinematics

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
    ur_home = 90
    fe_home = -90

    theta_1_corrected = theta_1 - ps_home
    theta_2_corrected = -1*(ur_home - theta_2)
    theta_3_corrected = theta_3 - fe_home

    theta_1_wrapped = dataset_object.wrap_angle_around_90(np.array([theta_1_corrected]))[0]
    theta_2_wrapped = dataset_object.wrap_angle_around_90(np.array([theta_2_corrected]))[0]
    theta_3_wrapped = dataset_object.wrap_angle_around_90(np.array([theta_3_corrected]))[0]
    return [theta_1_wrapped, theta_2_wrapped, theta_3_wrapped]

def orient_wrist(theta1, theta2, theta3):
    # This function takes the joint angles, theta1, theta2, theta3, for
    # pronate/supinate, ulnar/radial, and flexion/extension, respectively, and computes the 8 bit integers
    # needed to drive the hand

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
    return motor_command

def gyro_data(gyro):
    return np.asarray([gyro.x, gyro.y, gyro.z])

def accel_data(accel):
    return np.asarray([accel.x, accel.y, accel.z])

def euler_angles_from_accelerometer(accelerometer_output):
    # This function computes the camera euler angles from the accelerometer readings: ax, ay, az
    ax, ay, az = np.split(accelerometer_output / 9.8, indices_or_sections=3)
    theta_x_acc = np.arctan2(ay, -az) * (180 / np.pi)
    theta_y_acc = np.arctan2(-az, ax) * (180 / np.pi)
    theta_z_acc = np.arctan2(ay, ax) * (180 / np.pi)

    theta_x_acc = -(theta_x_acc + 90)
    theta_z_acc = -(theta_z_acc + 90)
    return [theta_x_acc, theta_y_acc, theta_z_acc]

def euler_angles_from_gyroscope(gyro, gyro_previous, dt, dt_sum):
    # This function computes the camera euler angles from the gyroscope readings
    gyro_current = gyro_previous + (gyro * dt) # Here gyro_previous needs to be in radians
    theta_x_gyro, theta_y_gyro, theta_z_gyro = gyro_current * (180 / np.pi)
    # m, b = [-0.17627199, 0.82105868]
    # correction_factor = (m*dt_sum) + b
    # theta_y_gyro = theta_y_gyro - correction_factor
    # print(theta_y_gyro, 'correction_factor: ', correction_factor)
    return [theta_x_gyro, theta_y_gyro, theta_z_gyro] # thetas are in degrees

def euler_angles_from_complementary_filter(theta_x_acc, theta_z_acc, theta_x_gyro, theta_y_gyro, theta_z_gyro, system):
    # complementary filter implementation
    G = 0.90
    A = 0.1
    theta_x = (theta_x_gyro * G) + (theta_x_acc * A)
    theta_z = (theta_z_gyro * G) + (theta_z_acc * A)
    theta_y = theta_y_gyro
    return [theta_x, theta_y, theta_z]

def compute_dt(previous_millis, frame_number):
    current_millis = int(round(time.time() * 1000))
    if frame_number == 0:
        previous_millis = current_millis
    dt = (current_millis - previous_millis) / 1000
    previous_millis = current_millis
    return [previous_millis, dt]

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# different resolutions of color and depth streams
image_width = 640
image_height = 480
fps = 15

config = rs.config()
config.enable_stream(rs.stream.depth, image_width, image_height, rs.format.z16, fps)
config.enable_stream(rs.stream.color, image_width, image_height, rs.format.rgb8, fps)
config.enable_stream(rs.stream.accel)
config.enable_stream(rs.stream.gyro)

# Starting serial link to robot arm
ser = serial.Serial('COM6', 115200, timeout=1)
ser.write(b'h')

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

mode = "mask_grasp_rcnn"
MODEL_DIR = "models"
mask_grasp_model_path = 'models/colab_result_id#1/MASK_GRASP_RCNN_MODEL.h5'
inference_config = GraspMaskRCNNInferenceConfig()
mask_grasp_model = modellib.MaskRCNN(mode="inference",
                           config=inference_config,
                           model_dir=MODEL_DIR, task=mode)
mask_grasp_model.load_weights(mask_grasp_model_path, by_name=True)

inference_config_track = GraspMaskRCNNInferenceConfig()
inference_config_track.USE_TRACKER_AS_ROI_SOURCE = True
mask_grasp_track_model = modellib.MaskRCNN(mode="inference",
                           config=inference_config_track,
                           model_dir=MODEL_DIR, task=mode)
mask_grasp_track_model.load_weights(mask_grasp_model_path, by_name=True)

dataset_object = GraspMaskRCNNDataset()

center_crop_size = inference_config.IMAGE_MAX_DIM
start_seconds = 0
current_seconds = 0
frame_count = 0
detection_period = 50 # Detecting using Mask-Grasp R-CNN every 120 frames, ie every 2 seconds when running at 60 fps
start_tracking = False
tracking_bbox = []
operating_mode = 'initialization'
num_tracked_frames = 0
num_detected_frames = 0
bbox_plot = []
location_history = []
delta_t = []
grasping_box_history = []
hand_configured = False
previous_millis = 0
gyro_previous = np.array([0, 0, 0])
system_previous = np.array([0, 0, 0])
dt_sum = 0

# Streaming loop
for i in list(range(20)):
    frames = pipeline.wait_for_frames()
cv2.namedWindow('MASK-GRASP RCNN OUTPUT', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('MASK-GRASP RCNN OUTPUT', onMouse)
try:
    while True:
        start_time = time.time()
        frames = pipeline.wait_for_frames()
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        intrinsics_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                                      [0, intrinsics.fy, intrinsics.ppy],
                                      [0, 0, 1]])
        accel = accel_data(frames[2].as_motion_frame().get_motion_data())
        gyro = gyro_data(frames[3].as_motion_frame().get_motion_data())

        # Defining the RealSense camera intrinsics as an Open3D object
        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=intrinsics.width, height=intrinsics.height,
                                                           fx=intrinsics.fx, fy=intrinsics.fy,
                                                           cx=intrinsics.ppx,
                                                           cy=intrinsics.ppy)

        hole_filling = rs.hole_filling_filter()
        aligned_depth_frame = hole_filling.process(aligned_depth_frame)
        if not aligned_depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        rgbd_image_resized = generate_rgbd_image(color_image, depth_image, center_crop=True, square_size=center_crop_size)
        rgb_image_resized = rgbd_image_resized[:, :, 0:3].astype('uint8')
        color_image_to_display = cv2.cvtColor(rgb_image_resized, cv2.COLOR_RGB2BGR)



        # If tracking did not start, then display detections of the network to the user
        if not start_tracking and num_detected_frames == 0:
            # Running Mask-Grasp R-CNN on a frame
            results = mask_grasp_model.detect([rgbd_image_resized], verbose=0, task=mode)[0]
            # Capture input from the user about which object to interact with. If selection is successful, then we
            # start tracking
            rois, grasping_deltas, grasping_probs, masks, roi_scores, start_tracking = select_ROI(mouseX, mouseY, results)
            masked_image, colors = dataset_object.get_mask_overlay(color_image_to_display, masks, roi_scores, threshold=0)
            tracking_bbox = rois
            mouseX, mouseY = [0, 0]
            print ('Detection mode, waiting for user to specify ROI for tracking')
            bbox_plot = tracking_bbox
            # Here, shape of tracking_bbox = [#rois, 4], form: [y1, x1, y2, x2]

        if start_tracking and num_tracked_frames == 0:
            bbox_plot = tracking_bbox
            y1, x1, y2, x2 = np.split(tracking_bbox, indices_or_sections=4, axis=-1)
            # Tracker takes bbox in the form [x, y, w, h]
            tracking_bbox = (x1, y1, x2 - x1, y2 - y1)
            # Initialize the tracker
            tracker = cv2.TrackerMedianFlow_create()
            # tracker = cv2.TrackerTLD_create()
            ok = tracker.init(color_image_to_display, tracking_bbox)
            operating_mode = 'tracking'
            print('Tracking mode, initialized')
            # Here, shape of tracking_bbox = [#rois, 4], form: [x1, y1, w, h]

        if operating_mode == 'tracking':
            ok, tracking_bbox = tracker.update(color_image_to_display)
            num_tracked_frames += 1
            x1, y1, w, h = tracking_bbox

            # Plotting results from tracker
            rect = cv2.boxPoints(((x1 + w / 2, y1 + h / 2), (w, h), 0))
            cv2.drawContours(color_image_to_display, [np.int0(rect)], 0, (0, 0, 255), 1)

            bbox_plot = np.array([[y1, x1, y1+h, x1+w]])
            if num_tracked_frames > detection_period:
                operating_mode = 'detection'
                num_tracked_frames = 0
            else:
                if num_tracked_frames == 1:
                    print('Tracking loop running')
                roi_for_detection = np.array([[[y1, x1, y1+h, x1+w]]])

                roi_for_detection_norm = utils.norm_boxes(roi_for_detection, inference_config_track.IMAGE_SHAPE[:2])
                results = mask_grasp_track_model.detect([rgbd_image_resized], verbose=0, task=mode, tracker_rois=roi_for_detection_norm)[0]
                rois, grasping_deltas, grasping_probs, masks, roi_scores, _ = select_ROI(mouseX, mouseY, results)

                if rois.shape[0] == 0:
                    # This handles the case when the regression fails, it predicts the ROI location using a constant
                    # velocity assumption, and a constant box size
                    t_minus_2_box = np.array(location_history[-2])
                    y1, x1, y2, x2 = t_minus_2_box
                    w = x2 - x1
                    h = y2 - y1
                    t_minus_2_loc = np.array([x1 + w / 2, y1 + h / 2])

                    t_minus_1_box = np.array(location_history[-1])
                    y1, x1, y2, x2 = t_minus_1_box
                    w = x2 - x1
                    h = y2 - y1
                    t_minus_1_loc = np.array([x1 + w / 2, y1 + h / 2])

                    delta_position = t_minus_1_loc - t_minus_2_loc
                    delta_time = delta_t[-1]

                    # We need to propagate the rois with the calculated velocity
                    next_position = t_minus_1_loc + (delta_position*delta_time)
                    x, y = next_position
                    roi_for_detection = np.array([[[y - h/2, x - w/2, y + h/2, x + w/2]]])

                    # Plotting prediction with constant velocity assumption
                    y1, x1, y2, x2 = roi_for_detection[0][0]
                    w = x2 - x1
                    h = y2 - y1
                    roi_vertices = cv2.boxPoints(((x1 + w / 2, y1 + h / 2), (w, h), 0))
                    cv2.drawContours(color_image_to_display, [np.int0(roi_vertices)], 0, (0, 255, 0), 2)

                    roi_for_detection_norm = utils.norm_boxes(roi_for_detection, inference_config_track.IMAGE_SHAPE[:2])
                    results = mask_grasp_track_model.detect([rgbd_image_resized], verbose=0, task=mode,
                                                            tracker_rois=roi_for_detection_norm)[0]
                    rois, grasping_deltas, grasping_probs, masks, roi_scores, _ = select_ROI(mouseX, mouseY, results)
                    # num_tracked_frames = 0
                    # tracking_bbox = rois
                    print ('Predicting with constant velocity model')

                masked_image, colors = dataset_object.get_mask_overlay(color_image_to_display, masks, roi_scores, threshold=0)
                bbox_plot = rois

        if operating_mode == 'detection':
            results = mask_grasp_model.detect([rgbd_image_resized], verbose=0, task=mode)[0]
            rois, grasping_deltas, grasping_probs, masks, roi_scores, _ = select_ROI(mouseX, mouseY, results)
            masked_image, colors = dataset_object.get_mask_overlay(color_image_to_display, masks, roi_scores, threshold=0)
            operating_mode = 'tracking'
            num_tracked_frames = 0
            num_detected_frames += 1
            print('Detection mode, frame number: ', num_detected_frames)
            # We need to reinstantiate the tracker with an updated ROI as predicted by Mask-Grasp R-CNN
            # Need to calculate IOU between rois and tracking bbox
            # First thing we need to convert tracking_bbox [x, y, w, h] -> [y1, x1, y2, x2]
            x1, y1, w, h = tracking_bbox
            x2 = x1 + w
            y2 = y1 + h
            tracking_bbox = np.array([y1, x1, y2, x2])
            tracking_bbox_area = w * h
            roi_areas = (rois[:, 2] - rois[:, 0]) * (rois[:, 3] - rois[:, 1])
            try:
                iou = utils.compute_iou(tracking_bbox, rois, tracking_bbox_area, roi_areas)
                tracking_bbox = rois[np.argmax(iou)]
                bbox_plot = tracking_bbox.reshape(-1, 4)
            except:
                print("ERROR: Encountered error when computing IOU")
                import code;
                code.interact(local=dict(globals(), **locals()))
            # Here, shape of tracking_bbox = [#rois, 4], form: [y1, x1, y2, x2]

        # Plotting bounding boxes
        for j, roi in enumerate(bbox_plot):
            color = (0, 255, 0)
            y1, x1, y2, x2 = np.split(roi, indices_or_sections=4, axis=-1)
            w = x2 - x1
            h = y2 - y1
            roi_vertices = cv2.boxPoints(((x1 + w / 2, y1 + h / 2), (w, h), 0))
            cv2.drawContours(color_image_to_display, [np.int0(roi_vertices)], 0, (0, 0, 0), 2)
            normalized_roi = utils.norm_boxes(roi, inference_config.IMAGE_SHAPE[:2])
            y1, x1, y2, x2 = np.split(normalized_roi, indices_or_sections=4, axis=-1)
            w = x2 - x1
            h = y2 - y1
            ROI_shape = np.array([h, w])
            pooled_feature_stride = np.array(ROI_shape / inference_config.GRASP_POOL_SIZE)
            grasping_anchors = utils.generate_grasping_anchors(inference_config.GRASP_ANCHOR_SIZE,
                                                               inference_config.GRASP_ANCHOR_RATIOS,
                                                               [inference_config.GRASP_POOL_SIZE, inference_config.GRASP_POOL_SIZE],
                                                               pooled_feature_stride,
                                                               1,
                                                               inference_config.GRASP_ANCHOR_ANGLES,
                                                               normalized_roi,
                                                               inference_config)
            post_nms_predictions, top_box_probabilities, pre_nms_predictions, pre_nms_scores = dataset_object.refine_results(
                        grasping_probs[j], grasping_deltas[j],
                        grasping_anchors, inference_config, filter_mode='top_k', k=1, nms=False)
            color = (255, 0, 0)
            for i, rect in enumerate(pre_nms_predictions):
                five_dim_box = rect
                rect = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
                color_image_to_display = cv2.drawContours(color_image_to_display, [np.int0(rect)], 0, (0,0,0), 2)
                color_image_to_display = cv2.drawContours(color_image_to_display, [np.int0(rect[:2])], 0, color, 2)
                color_image_to_display = cv2.drawContours(color_image_to_display, [np.int0(rect[2:])], 0, color, 2)
                if start_tracking:
                    # Computing time between frames
                    previous_millis, dt = compute_dt(previous_millis, frame_count)
                    # Getting camera pose from accelerometer
                    theta_x_acc, theta_y_acc, theta_z_acc = euler_angles_from_accelerometer(accel)
                    #Getting camera pose from gyroscope
                    print ('dt: ', dt, 'dt_sum', dt_sum)
                    theta_x_gyro, theta_y_gyro, theta_z_gyro = euler_angles_from_gyroscope(gyro, gyro_previous, dt, dt_sum)

                    gyro_previous = np.array([theta_x_gyro, theta_y_gyro, theta_z_gyro])*(np.pi/180)
                    print('\ngyro_current: ', theta_x_gyro, theta_y_gyro, theta_z_gyro)

                    # complementary filter implementation
                    G = 0.90
                    A = 0.1
                    theta_x = (system_previous[0] + gyro[0]*dt*(180 / np.pi)) * G + (theta_x_acc * A)
                    theta_z = (system_previous[2] + gyro[2]*dt*(180 / np.pi)) * G + (theta_z_acc * A)
                    theta_y = theta_y_gyro
                    system_previous = np.array([theta_x, theta_y, theta_z])
                    print('\nComplementary filter: ', theta_x, theta_y, theta_z)

                    # Need to crop out the point cloud at the oriented rectangle bounds
                    # Create a mask to specify the region to extract the surface normals from. Based on Lenz et al.,
                    # the approach vector is estimated as the surface normal calculated at the point of minumum depth in
                    # the central one third (horizontally) of the rectangle

                    x, y, w, h, theta = five_dim_box
                    # Undoing the center cropping of the image
                    x = int(x + (image_width - center_crop_size) // 2)
                    y = int(y + (image_height - center_crop_size) // 2)

                    rect_vertices = cv2.boxPoints(((x, y), (w, h), theta))
                    extraction_mask_vertices = cv2.boxPoints(((x, y), (w, h/3), theta))
                    grasp_box_mask = generate_mask_from_polygon([image_height, image_width, 3], extraction_mask_vertices)

                    # Masking the color and depth frames with the grasp_box_mask
                    masked_color = color_image * np.repeat(grasp_box_mask[...,None],3,axis=2)
                    masked_depth = depth_image * grasp_box_mask

                    # Generating a point cloud by open3D for the grasping region
                    pcd = generate_pointcloud_from_rgbd(masked_color, masked_depth)
                    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=50))
                    pcd.orient_normals_towards_camera_location()
                    normals = np.array(pcd.normals)
                    points = np.array(pcd.points)

                    # Finding the point with the minumum depth
                    min_depth_ix = np.argmin(points[:, -1])
                    # approach_vector = normals[min_depth_ix]
                    approach_vector = np.mean(normals, axis=0)
                    min_depth_point = points[min_depth_ix]

                    # Approach vector needs to be pointing in the opposite direction
                    approach_vector = -1 * approach_vector

                    # Getting the XYZ coordinates of the grasping box center <x, y>
                    box_center = de_project_point(intrinsics, aligned_depth_frame, [x, y])

                    #### Visualization
                    # Generating a point cloud by open3D
                    pcd_full_image = generate_pointcloud_from_rgbd(color_image, depth_image)
                    points = np.array(pcd_full_image.points)

                    # Plotting approach vector on the point cloud
                    approach_vector_points = [box_center, box_center + (approach_vector*0.2)]
                    lines = [[0, 1]]
                    colors = [[1, 0, 0] for i in range(len(lines))]
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(approach_vector_points)
                    line_set.lines = o3d.utility.Vector2iVector(lines)
                    line_set.colors = o3d.utility.Vector3dVector(colors)

                    theta = dataset_object.wrap_angle_around_90(np.array([theta]))[0]
                    # print('Angle in the image plane: ', theta, 'degrees')
                    theta = theta * (np.pi / 180) # network outputs positive angles in bottom right quadrant

                    vz = approach_vector
                    vx = np.array([1, 0, 0])
                    # need to get the projection of vx onto the approach vector (normal)
                    proj_n_vx = (np.dot(vx, approach_vector)/(np.linalg.norm(approach_vector)**2)) * approach_vector
                    vx = vx - proj_n_vx
                    vy = np.cross(vz, vx)
                    vx = vx.reshape([3, 1])
                    vy = vy.reshape([3, 1])
                    vz = vz.reshape([3, 1])
                    V = np.concatenate([vx, vy, vz], axis=-1)
                    rotation_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                                           [np.sin(theta),  np.cos(theta), 0],
                                           [0            ,              0, 1]])

                    approach_vector_orientation = np.dot(V, rotation_z)

                    # print('Approach vector orientation relative to camera coordinates: \n', approach_vector_orientation)

                    real_width, real_height = compute_real_box_size(intrinsics, aligned_depth_frame, rect_vertices)
                    box_vert_obj_frame = generate_points_in_world_frame(real_width, real_height)

                    # Rotation about the Z axis of the surface normal
                    r_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])

                    rotated_points = np.dot(r_z, box_vert_obj_frame.T).T
                    HTM_camera_to_object = np.concatenate([approach_vector_orientation, np.expand_dims(box_center, -1)], axis=-1)
                    HTM_camera_to_object = np.concatenate([HTM_camera_to_object, np.array([[0, 0, 0, 1]])], axis=0)
                    rotated_points = np.concatenate([rotated_points, np.ones((4, 1))], axis=-1)
                    rotated_points_camera_frame = np.dot(HTM_camera_to_object, rotated_points.T)

                    # Plotting the grasp box on the point cloud
                    # grasp_box_points = rotated_points_camera_frame.T[:,:3].tolist()
                    rect_camera_frame = get_camera_frame_box_coords(intrinsics, aligned_depth_frame, rect_vertices)
                    grasp_box_points = rect_camera_frame.tolist()
                    lines = [[0, 1], [2, 3]]
                    colors = [[0, 0, 1] for i in range(len(lines))]
                    grasp_box = o3d.geometry.LineSet()
                    grasp_box.points = o3d.utility.Vector3dVector(grasp_box_points)
                    grasp_box.lines = o3d.utility.Vector2iVector(lines)
                    grasp_box.colors = o3d.utility.Vector3dVector(colors)

                    lines = [[1, 2], [3, 0]]
                    colors = [[0, 0, 0] for i in range(len(lines))]
                    grasp_box_2 = o3d.geometry.LineSet()
                    grasp_box_2.points = o3d.utility.Vector3dVector(grasp_box_points)
                    grasp_box_2.lines = o3d.utility.Vector2iVector(lines)
                    grasp_box_2.colors = o3d.utility.Vector3dVector(colors)

                    object_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=box_center)
                    object_frame.rotate(approach_vector_orientation)

                    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=box_center)

                    # o3d.visualization.draw_geometries([pcd_full_image, line_set, grasp_box_2, grasp_box, object_frame, camera_frame])

                    # theta1 : pronate/supinate
                    # theta2 : ulnar/radial
                    # theta3 : flexion/extension
                    camera_angle = -70 * (np.pi / 180)
                    rotation_x = np.array([[1,                    0,                     0],
                                           [0, np.cos(camera_angle), -np.sin(camera_angle)],
                                           [0, np.sin(camera_angle), np.cos(camera_angle)]])
                    r_x_inverse = np.linalg.inv(rotation_x)
                    # vector = np.dot(approach_vector_orientation, rotation_x)

                    vector = np.dot(np.dot(rotation_x, V), rotation_z)
                    calibration_matrix = np.array([[0, 0, 1],
                                                   [0, -1, 0],
                                                   [1, 0, 0]])
                    vector_new = np.dot(vector, calibration_matrix)
                    # vector = np.dot(rotation_x, V)

                    theta1, theta2, theta3 = derive_motor_angles_v0(vector_new)
                    joint1, joint2, joint3 = orient_wrist(theta1, theta2, theta3).tolist()
                    # configure_hand(real_width, real_height, joint3, joint2, joint1)
                    string_command = 'w %d %d %d' % (joint3, joint2, joint1)
                    aperture_command = 'j 0 %d' % (compute_hand_aperture(real_width*1000))

                    # if not hand_configured:
                    #     ser.write(aperture_command.encode())
                    #     time.sleep(3)
                    #     ser.write(b'j 3 800')
                    #     hand_configured = True
                    #
                    # time.sleep(3)
                    # ser.write(string_command.encode())
                    # import code;
                    #
                    # code.interact(local=dict(globals(), **locals()))
                    frame_count += 1
                    dt_sum += dt

        images = color_image_to_display

        if not start_tracking and num_detected_frames == 0:
            images = masked_image
        cv2.imshow('MASK-GRASP RCNN OUTPUT', images)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        end_time = time.time()

        # Only keep track of the time and location history when the objects are tracked
        if operating_mode == 'tracking' and bbox_plot.shape[0] != 0:
            location_history.append(bbox_plot[0])
            delta_t.append(end_time - start_time)
            grasping_box_history.append(pre_nms_predictions[0])

finally:
    pipeline.stop()

# # Running Mask-Grasp R-CNN on a frame
# results = mask_grasp_model.detect([rgbd_image_resized], verbose=0, task = mode)[0]
#
# # Capture input from the user about which object to interact with
# rois, grasping_deltas, grasping_probs, masks, roi_scores = select_ROI(mouseX, mouseY, results)
#
# masked_image, colors = dataset_object.get_mask_overlay(color_image_to_display, masks, roi_scores, threshold=0)
#
# if mouseY + mouseX != 0:
#     mouseX, mouseY = [0, 0]
#     # Plotting the selected ROI
#     for roi in rois:
#         y1, x1, y2, x2 = np.split(roi, indices_or_sections=4, axis=-1)
#         w = x2 - x1
#         h = y2 - y1
#         rect = cv2.boxPoints(((x1+ w/2, y1 + h/2), (w, h), 0))
#         cv2.drawContours(masked_image, [np.int0(rect)], 0, (0, 0, 0), 1)
#     # Since we are only using a single object tracker, we only initiate the tracker if there is 1 ROI selected
#     if rois.shape[0] == 1:
#         y1, x1, y2, x2 = np.split(rois, indices_or_sections=4, axis=-1)
#         # Tracker takes bbox in the form [x, y, w, h]
#         bbox_to_track = (x1, y1, x2 - x1, y2 - y1)
#         ok = tracker.init(color_image_to_display, bbox_to_track)
#
# # Processing output from Mask Grasp R-CNN and plotting the results for the ROIs
# for j, rect in enumerate(rois):
#     color = (np.array(colors[j])*255).astype('uint8')
#     color = (int(color[0]), int(color[1]), int(color[2]))
#     expanded_rect_normalized = utils.norm_boxes(rect, inference_config.IMAGE_SHAPE[:2])
#     y1, x1, y2, x2 = expanded_rect_normalized
#     w = abs(x2 - x1)
#     h = abs(y2 - y1)
#     ROI_shape = np.array([h, w])
#     pooled_feature_stride = np.array(ROI_shape/inference_config.GRASP_POOL_SIZE)#.astype('uint8')
#     grasping_anchors = utils.generate_grasping_anchors(inference_config.GRASP_ANCHOR_SIZE,
#                                                        inference_config.GRASP_ANCHOR_RATIOS,
#                                                        [inference_config.GRASP_POOL_SIZE, inference_config.GRASP_POOL_SIZE],
#                                                        pooled_feature_stride,
#                                                        1,
#                                                        inference_config.GRASP_ANCHOR_ANGLES,
#                                                        expanded_rect_normalized,
#                                                        inference_config)
#     post_nms_predictions, top_box_probabilities, pre_nms_predictions, pre_nms_scores = dataset_object.refine_results(
#         grasping_probs[j], grasping_deltas[j],
#         grasping_anchors, inference_config, filter_mode='prob', nms=False)
#     for i, rect in enumerate(pre_nms_predictions):
#         rect = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
#         grasp_rectangles_image = cv2.drawContours(masked_image, [np.int0(rect)], 0, (0,0,0), 1)
#         grasp_rectangles_image = cv2.drawContours(grasp_rectangles_image, [np.int0(rect[:2])], 0, color, 1)
#         grasp_rectangles_image = cv2.drawContours(grasp_rectangles_image, [np.int0(rect[2:])], 0, color, 1)
# if frame_count == 0:
#     currentDT = datetime.datetime.now()
#     start_seconds = str(currentDT).split(' ')[1].split(':')[2]
#     start_minute = str(currentDT).split(' ')[1].split(':')[1]
#     start_seconds = int(float(start_seconds))
#     start_minute = int(float(start_minute))
# else:
#     currentDT = datetime.datetime.now()
#     current_seconds = str(currentDT).split(' ')[1].split(':')[2]
#     current_seconds = int(float(current_seconds))
# print(start_seconds, current_seconds)
# if (start_seconds == current_seconds):
#     current_minute = str(currentDT).split(' ')[1].split(':')[1]
#     current_minute = int(float(current_minute))
#     if (start_minute != current_minute):
#         print('####################\n')
#         print('frames per minute:', frame_count)
#         print('####################\n')
#         # break
# frame_count = frame_count + 1
# # Press esc or 'q' to close the image window