########################################################################################################################
# Mena SA Kamel
# Student Number: 251064703
# MESc Candidate, Robotics and Control
# Electrical and Computer Engineering, Western University
########################################################################################################################

import pyrealsense2 as rs
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import mrcnn.utils as utils
import mrcnn.model as modellib
from mask_grasp_rcnn import GraspMaskRCNNInferenceConfig, GraspMaskRCNNDataset
import open3d as o3d
from matplotlib.path import Path
import serial
import time
import os
import intel_realsense_IMU
import zmq
import msgpack
import threading

def onMouse(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
       mouseX, mouseY = [x, y]

def center_crop_frame(image, new_size):
    # This function center crops the image into a square of size w = h = new_size
    original_shape = image.shape
    diff_x = (original_shape[1] - new_size) // 2
    diff_y = (original_shape[0] - new_size) // 2
    new_image = image[diff_y:original_shape[0]-diff_y, diff_x:original_shape[1]-diff_x, :]
    return new_image

def generate_rgbd_image(color_image, depth_image, center_crop=True, square_size=384):
    # Constructs an RGBD image out of an RGB and depth image that are aligned
    depth_image = depth_image * (depth_image < 2000)
    depth_scaled = np.interp(depth_image, (depth_image.min(), depth_image.max()), (0, 1)) * 255
    rgbd_image = np.zeros([color_image.shape[0], color_image.shape[1], 4])
    rgbd_image[:, :, 0:3] = color_image[:, :, 0:3]
    rgbd_image[:, :, 3] = depth_scaled
    rgbd_image = rgbd_image.astype('uint8')
    if center_crop:
        rgbd_image = center_crop_frame(rgbd_image, square_size)
    return rgbd_image

def compute_dt(previous_millis, frame_number, zero_timer=False):
    # This function computes the time difference between two consecutive frames
    current_millis = int(round(time.time() * 1000))
    if frame_number == 0 or zero_timer:
        previous_millis = current_millis
    dt = (current_millis - previous_millis) / 1000
    previous_millis = current_millis
    return [previous_millis, dt]

def arm_orientation_imu_9250(serial_object, command='r'):
    # This function performs the handshake between Python and the Robot arm. Returns the arm angles.
    serial_object.write(command.encode())
    input_bytes = serial_object.readline()
    while input_bytes == b'':
        serial_object.write(command.encode())
        input_bytes = serial_object.readline()
    decoded_bytes = np.array(input_bytes.decode().replace('\r', '').replace('\n', '').split('\t'), dtype='float32')
    theta_pitch, theta_roll, theta_yaw = decoded_bytes.tolist()
    return ([theta_pitch, theta_roll, theta_yaw])

def select_ROI(mouseX, mouseY, r):
    # Uses the X, Y coordinates that the user selected to fetch the underlying detection
    rois = r['rois']
    grasping_deltas = r['grasp_boxes']
    grasping_probs = r['grasp_probs']
    masks = r['masks']
    roi_scores = r['scores']
    image_size = np.shape(masks[:, :, 0])
    mouseX = mouseX
    # mouseY = image_size[0] - mouseY
    selection_success = False

    if rois.shape[0] > 0 and (mouseY + mouseX) != 0:
        # y1, x1, y2, x2
        box_centers_x = ((rois[:, 1] + rois[:, 3])/2).reshape(-1, 1)
        box_centers_y = ((rois[:, 0] + rois[:, 2])/2).reshape(-1, 1)
        w = np.abs(rois[:, 1] - rois[:, 3]).reshape(-1, 1)
        h = np.abs(rois[:, 0] - rois[:, 2]).reshape(-1, 1)
        delta = 1.5

        thresholds = (np.sqrt((w/2)**2 + (h/2)**2) * delta).squeeze()
        box_centers = np.concatenate([box_centers_x, box_centers_y], axis=-1)
        gaze_point = np.array([mouseX, mouseY])
        gaze_point_repeats = np.tile(gaze_point, [np.shape(box_centers)[0], 1])
        distances = np.sqrt((gaze_point_repeats[:, 0] - box_centers[:, 0]) ** 2 + (gaze_point_repeats[:,1] - box_centers[:, 1]) ** 2)

        selected_ROI_ix = np.argmin(distances)
        # if distances[selected_ROI_ix] < thresholds[selected_ROI_ix]:
        if True:
            rois = rois[selected_ROI_ix].reshape(-1, 4)
            grasping_deltas = np.expand_dims(grasping_deltas[selected_ROI_ix], axis=0)
            grasping_probs = np.expand_dims(grasping_probs[selected_ROI_ix], axis=0)
            masks = np.expand_dims(masks[:, :, selected_ROI_ix], axis=-1)
            roi_scores = np.expand_dims(roi_scores[selected_ROI_ix], axis=0)
            selection_success = True

    return rois, grasping_deltas, grasping_probs, masks, roi_scores, selection_success

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

def generate_pointcloud_from_rgbd(color_image, depth_image):
    # Creates an Open3D point cloud
    color_stream = o3d.geometry.Image(color_image)
    depth_stream = o3d.geometry.Image(depth_image)
    o3d_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_stream, depth_stream, depth_trunc=2.0)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(o3d_rgbd_image, o3d_intrinsics)
    return pcd

def compute_approach_vector(grasping_box):
    # This function computes the approach vector for a given grasping box in 5-dimensional coordinates
    # [x, y, w, h, theta].
    # Returns the camera frame coordinates of the
    x, y, w, h, theta = grasping_box
    # Cropping the point cloud at the central 1/3 of the grasping box
    extraction_mask_vertices = cv2.boxPoints(((x, y), (w, h / 3), theta))
    grasp_box_mask = generate_mask_from_polygon([image_height, image_width, 3], extraction_mask_vertices)
    # Masking the color and depth frames with the grasp_box_mask
    masked_color = color_image * np.repeat(grasp_box_mask[..., None], 3, axis=2)
    masked_depth = depth_image * grasp_box_mask
    # Generating a point cloud by open3D for the grasping region as defined by the newly cropped color
    # and depth channels
    pcd = generate_pointcloud_from_rgbd(masked_color, masked_depth)
    # Estimating surface normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=50))
    # Pointing the surface normala towards the camera
    pcd.orient_normals_towards_camera_location()
    normals = np.array(pcd.normals)
    points = np.array(pcd.points)
    # Finding the point with the minumum depth - point with lowest distance in the Z direction
    min_depth_ix = np.argmin(points[:, -1])
    # approach_vector = normals[min_depth_ix]
    # Defining approach vector as the average normal vector of the central 1/3 of the grasping box
    approach_vector = np.mean(normals, axis=0)
    min_depth_point = points[min_depth_ix]
    # Approach vector needs to be pointing in the opposite direction - ie into the object because that is
    # how we plan on approaching it
    approach_vector = -1 * approach_vector
    return approach_vector

def de_project_point(intrinsics, depth_frame, point):
    # This function deprojects point from the image plane to the camera frame of reference using camera intrinsic
    # parameters
    x, y = point
    distance_to_point = depth_frame.as_depth_frame().get_distance(x, y)
    de_projected_point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], distance_to_point)
    return de_projected_point

def compute_distance_3D(P1, P2):
    # This function computes the euclidean distance between two 3D points
    return np.sqrt((P1[0] - P2[0]) ** 2 + (P1[1] - P2[1]) ** 2 + (P1[2] - P2[2]) ** 2)

def compute_distance_2D(p1, p2):
    # This function computes the euclidean distance between two 2D points
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def get_camera_frame_box_coords(intrinsics, depth_frame, box_vertices):
    # This function deprojects the vertex points that make up the grasp box from the image plane to the camera frame
    de_projected_points = []
    for point in box_vertices:
        de_projected_point = de_project_point(intrinsics, depth_frame, point)
        de_projected_points.append(de_projected_point)
    return np.array(de_projected_points)

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

    real_width = np.mean([compute_distance_3D(P1, P2), compute_distance_3D(P0, P3)])
    real_height = np.mean([compute_distance_3D(P0, P1), compute_distance_3D(P2, P3)])
    # print('Grasp Width: ', real_width, 'Grasp Height: ', real_height)
    return [real_width, real_height]

def generate_points_in_world_frame(real_width, real_height):
    # This function generates the points in the world frame of reference
    return np.array([[real_width/2, -real_height/2, 0],
                    [real_width/2, real_height/2, 0],
                    [-real_width/2, real_height/2, 0],
                    [-real_width/2, -real_height/2, 0]])

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
        sign = np.sign(c2_i)
        if sign == 0:
            sign = 1
        c2_i = sign*np.maximum(0.0001, np.abs(c2_i)) # Gimbal lock case - Need to look into this
        s1 = -b / c2_i
        c1 = e / c2_i
        s3 = -i / c2_i
        c3 = g / c2_i
        theta_1 = np.arctan2(s1, c1) / (np.pi / 180)
        theta_2 = np.arctan2(s2, c2_i) / (np.pi / 180)
        theta_3 = np.arctan2(s3, c3) / (np.pi / 180)
        joint_combinations[j,:] = np.array([theta_1,theta_2, theta_3])
        join_sum.append(np.sum(np.abs(np.array([theta_1,theta_2, theta_3]))))
    # theta_1, theta_2, theta_3 = joint_combinations[np.argmin(join_sum)] # Choosing the angle combo with the least
                                                                        # deviation required
    theta_1, theta_2, theta_3 = joint_combinations[0]  # Choosing the angle combo with the least
    return [theta_1, theta_2, theta_3]

def get_joint_home_values_8bits():
    FE_MIN = 260
    FE_MAX = 540
    FE_HOME = 380

    UR_MIN = 320
    UR_MAX = 490
    UR_HOME = 440

    SP_MIN = 120
    SP_MAX = 480
    SP_HOME = 340

    FE_HOME_8_bits = ((FE_HOME - FE_MIN) / (FE_MAX - FE_MIN))*255
    UR_HOME_8_bits = ((UR_HOME - UR_MIN) / (UR_MAX - UR_MIN))*255
    SP_HOME_8_bits = ((SP_HOME - SP_MIN) / (SP_MAX - SP_MIN))*255

    return ([SP_HOME_8_bits, UR_HOME_8_bits, FE_HOME_8_bits])

def orient_wrist(theta1, theta2, theta3):
    # This function takes the joint angles, theta1, theta2, theta3, for
    # pronate/supinate, ulnar/radial, and flexion/extension, respectively, and computes the 8 bit integers
    # needed to drive the hand

    # Joint center positions as an 8 bit integer
    ps_home, ur_home, fe_home = get_joint_home_values_8bits()

    # Defining the physical joint limits
    pronate_supinate_limit = [-90, 80]
    ulnar_radial_limit = [-8, 15]
    flexion_extension_limit = [-47, 62]

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

    if np.sign(theta3) == 1: # flexion
        # joint3 = np.interp(theta3, (0, flexion_extension_limit[1]), (fe_home, 255))
        joint3 = np.interp(theta3, (0, flexion_extension_limit[1]), (fe_home, 0))
    else:
        # joint3 = np.interp(theta3, (flexion_extension_limit[0], 0), (0, fe_home))
        joint3 = np.interp(theta3, (flexion_extension_limit[0], 0), (255, fe_home))

    joint_output = np.array([joint1, joint2, joint3]).astype('uint8')
    return joint_output

def compute_grasp_type(grasp_box_height):
    # Computes the grasp type based on grasp box height
    # grasp_type = 0 -> OneFPinch
    # grasp_type = 1 -> TwoFPinch
    # grasp_type = 2 -> Power
    if 0 < grasp_box_height <= 50:
        grasp_type = 0
    elif 50 < grasp_box_height <= 100:
        grasp_type = 1
    else:
        grasp_type = 2
    return grasp_type

def compute_hand_aperture(grasp_box_width):
    # This function computes the motor command required to acheive an aperture of size grasp_box_width
    coef_path = 'aperture_mapping.txt'
    if os.path.exists(coef_path):
        coeffs = np.loadtxt(coef_path)
    else:
        # motor_commands = np.arange(0, 1100, 100)
        # aperture_size = np.array([90, 88, 75, 60, 40, 30, 18, 10, 8, 5.9, 5.4])
        motor_commands = np.array(
            [0, 20, 40, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 255])
        aperture_size = np.array([83, 83, 83, 75, 65, 61, 49, 44, 30, 23, 14, 7, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        coeffs = np.polynomial.polynomial.polyfit(aperture_size, motor_commands, 8)
        np.savetxt(coef_path, coeffs)
    return np.polynomial.polynomial.polyval(grasp_box_width, coeffs)

def plot_selected_box(image, boxes, box_index):
    color = np.array([[0, 255, 0]])
    color = np.repeat(color, boxes.shape[0], axis=0)

    for j, five_dim_box in enumerate(boxes):
        col = tuple([int(x) for x in color[j]])
        rect = cv2.boxPoints(
            ((five_dim_box[0], five_dim_box[1]), (five_dim_box[2], five_dim_box[3]), five_dim_box[4]))
        image = cv2.drawContours(image, [np.int0(rect)], 0, (0, 0, 0), 2)
        image = cv2.drawContours(image, [np.int0(rect[:2])], 0, col, 2)
        image = cv2.drawContours(image, [np.int0(rect[2:])], 0, col, 2)

    color[box_index] = [0, 0, 255]
    five_dim_box = boxes[box_index]
    col = tuple([int(x) for x in color[box_index]])
    rect = cv2.boxPoints(
        ((five_dim_box[0], five_dim_box[1]), (five_dim_box[2], five_dim_box[3]), five_dim_box[4]))
    image = cv2.drawContours(image, [np.int0(rect)], 0, (0, 0, 0), 2)
    image = cv2.drawContours(image, [np.int0(rect[:2])], 0, col, 2)
    image = cv2.drawContours(image, [np.int0(rect[2:])], 0, col, 2)
    return image

def preshape_hand(real_width, real_height, ser):
    grasp_width = real_width * 1000
    grasp_width = np.minimum(grasp_width, 83)
    aperture_command = compute_hand_aperture(grasp_width)
    aperture_command = np.minimum(aperture_command, 220)
    aperture_command = np.maximum(aperture_command, 0)
    grasp_type = compute_grasp_type(real_height * 1000)
    string_command = 'g %d %d' % (grasp_type, int(aperture_command))
    ser.write(string_command.encode())

def fetch_gaze_vector(subscriber, avg_gaze, terminate, rvec, tvec, realsense_intrinsics_matrix, image_width, image_height,
                      center_crop_size):
    while not terminate[0]:
        topic, payload = subscriber.recv_multipart()
        message = msgpack.loads(payload)
        gaze_point_3d = message[b'gaze_point_3d']
        avg_gaze[:] = get_gaze_points_in_realsense_frame(gaze_point_3d, rvec, tvec,
                                                         realsense_intrinsics_matrix,
                                                         image_width, center_crop_size,
                                                         image_height)

def initialize_pupil_tracker():
    # Establishing connection to eye tracker
    ctx = zmq.Context()
    # The REQ talks to Pupil remote and receives the session unique IPC SUB PORT
    pupil_remote = ctx.socket(zmq.REQ)
    ip = 'localhost'  # If you talk to a different machine use its IP.
    port = 50020  # The port defaults to 50020. Set in Pupil Capture GUI.
    pupil_remote.connect(f'tcp://{ip}:{port}')
    # Request 'SUB_PORT' for reading data
    pupil_remote.send_string('SUB_PORT')
    sub_port = pupil_remote.recv_string()
    # Request 'PUB_PORT' for writing dataYour location
    pupil_remote.send_string('PUB_PORT')
    pub_port = pupil_remote.recv_string()
    subscriber = ctx.socket(zmq.SUB)
    subscriber.connect(f'tcp://{ip}:{sub_port}')
    subscriber.subscribe('gaze.')  # receive all gaze messages
    return subscriber

def initialize_realsense(image_width, image_height, fps):
    # Create an Intel RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, image_width, image_height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, image_width, image_height, rs.format.rgb8, fps)
    config.enable_stream(rs.stream.accel)
    config.enable_stream(rs.stream.gyro)
    profile = pipeline.start(config)
    # Create an align object: rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)
    colorizer = rs.colorizer()
    return [pipeline, profile, align, colorizer]

def fetch_realsense_frame(pipeline, align, aligned_depth_frame, color_frame, intrinsics, center_crop_size,
                          rgbd_image_resized, terminate):
    while not terminate[0]:
        # Gets a color and depth image
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame_object = aligned_frames.get_depth_frame()
        color_frame[0] = aligned_frames.get_color_frame()
        # Hole filling to get a clean depth image
        hole_filling = rs.hole_filling_filter()
        aligned_depth_frame[0] = hole_filling.process(aligned_depth_frame_object)
        intrinsics[0] = color_frame[0].profile.as_video_stream_profile().intrinsics
        color_image, depth_image = get_images_from_frames(color_frame, aligned_depth_frame)
        rgbd_image_resized[0] = generate_rgbd_image(color_image, depth_image, center_crop=True,
                                                    square_size=center_crop_size)

def get_images_from_frames(color_frame, aligned_depth_frame):
    color_image = np.asanyarray(color_frame[0].get_data())
    depth_image = np.asanyarray(aligned_depth_frame[0].get_data())
    return [color_image, depth_image]

def get_gaze_points_in_realsense_frame(avg_gaze, rvec, tvec, realsense_intrinsics_matrix,
                                       image_width, center_crop_size, image_height):
    gaze_points = np.array(avg_gaze).reshape(-1, 1, 3)
    gaze_points_realsense_image, jacobian = cv2.projectPoints(gaze_points, rvec, tvec,
                                                              realsense_intrinsics_matrix, None)
    gaze_x_realsense, gaze_y_realsense = gaze_points_realsense_image.squeeze().astype('uint16')
    gaze_x_realsense = int(gaze_x_realsense - (image_width - center_crop_size) / 2)
    gaze_y_realsense = int(gaze_y_realsense - (image_height - center_crop_size) / 2)
    return [gaze_x_realsense, gaze_y_realsense]

def display_gaze_on_image(image, x, y, color=(0,0,255)):
    image = cv2.circle(image, (x, y), 20, color, 3)
    image = cv2.circle(image, (x, y), 2, color, 2)
    return image

def resize_image(image, resize_factor):
    new_shape = tuple(resize_factor * np.shape(image)[:2])
    return cv2.resize(image, new_shape, interpolation=cv2.INTER_AREA)

def plot_to_UI(rgbd_image_resized, avg_gaze, mask_grasp_results, window_resize_factor, dataset_object, terminate):
    cv2.namedWindow('MASK-GRASP RCNN OUTPUT', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('MASK-GRASP RCNN OUTPUT', onMouse)
    while not terminate[0]:
        try:
            bgr_image = cv2.cvtColor(rgbd_image_resized[0].astype('uint8'), cv2.COLOR_RGB2BGR)
            gaze_x_realsense, gaze_y_realsense = avg_gaze
            image_to_display = bgr_image
            if mask_grasp_results != [None]:
                # Capture input from user about which object to interact with
                selected_ROI = select_ROI(gaze_x_realsense, gaze_y_realsense, mask_grasp_results[0])
                rois, grasping_deltas, grasping_probs, masks, roi_scores, selection_flag = selected_ROI
                masked_image, colors = dataset_object.get_mask_overlay(bgr_image, masks, roi_scores, threshold=0)
                image_to_display = masked_image
            image_with_gaze = display_gaze_on_image(image_to_display, gaze_x_realsense, gaze_y_realsense)
            resized_color_image_to_display = resize_image(image_with_gaze, window_resize_factor)

            display_output = resized_color_image_to_display
            cv2.imshow('MASK-GRASP RCNN OUTPUT', display_output)
            key = cv2.waitKey(100)

            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                terminate[0] = True
                break
        except:
            continue

def main():
    # Defining variable to store the gaze location
    avg_gaze = [0, 0]

    # Defining variables for Intel RealSense Feed
    image_width = 640
    image_height = 480
    fps = 15
    aligned_depth_frame = [None]
    color_frame = [None]
    intrinsics = [None]
    terminate = [False]
    display_output = np.array([image_height, image_width, 1]).astype('uint8')

    # Defining calibration parameters between RealSense and Pupil Trackers
    M_t = np.array([[ 0.99891844, -0.0402869,   0.02321457, -0.08542229],
 [ 0.03949054,  0.99864817,  0.03379828, -0.06768186],
 [-0.02454482, -0.03284497,  0.99915903,  0.015576012]])
    tvec = M_t[:, -1]
    rvec, jacobian = cv2.Rodrigues(M_t[:, :3])
    realsense_intrinsics_matrix = np.array([[609.87304688, 0., 332.6171875],
                                            [0., 608.84387207, 248.34165955],
                                            [0., 0., 1.]])

    # Defining Display parameters
    window_resize_factor = np.array([2, 2])

    # Defining variables for Mask-Grasp R-CNN
    mode = "mask_grasp_rcnn"
    MODEL_DIR = "models"
    mask_grasp_model_path = 'models/colab_result_id#1/MASK_GRASP_RCNN_MODEL.h5'
    # Loading the Mask-Grasp R-CNN Model
    inference_config = GraspMaskRCNNInferenceConfig()
    mask_grasp_model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR, task=mode)
    mask_grasp_model.load_weights(mask_grasp_model_path, by_name=True)
    dataset_object = GraspMaskRCNNDataset()
    center_crop_size = inference_config.IMAGE_MAX_DIM
    rgbd_image_resized = [None]
    mask_grasp_results = [None]

    # Initializing the eye tracker and storing the subscriber object to fetch gaze values as they update
    # Thread T1 - Running the eye tracker
    subscriber = initialize_pupil_tracker()
    t1 = threading.Thread(target=fetch_gaze_vector,
                          args=(subscriber, avg_gaze, terminate, rvec, tvec, realsense_intrinsics_matrix, image_width,
                                image_height, center_crop_size))
    t1.start()

    # Thread T2 - Running Intel RealSense camera
    pipeline, profile, align, colorizer = initialize_realsense(image_width, image_height, fps)
    t2 = threading.Thread(target=fetch_realsense_frame,
                          args=(pipeline, align, aligned_depth_frame, color_frame, intrinsics, center_crop_size,
                                rgbd_image_resized, terminate))
    t2.start()

    # Thread T3 - Plotting the results to UI
    t3 = threading.Thread(target=plot_to_UI,
                          args=(rgbd_image_resized, avg_gaze, mask_grasp_results, window_resize_factor, dataset_object,
                                terminate))
    t3.start()

    while not terminate[0]:
        try:
            rgbd_image_temp = rgbd_image_resized[0]
            mask_grasp_results[0] = mask_grasp_model.detect([rgbd_image_temp], verbose=0, task=mode)[0]  # slow step

        except:
            continue
main()