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

def select_ROI(mouseX, mouseY, r, resize_factor):
    # Uses the X, Y coordinates that the user selected to fetch the underlying detection
    mouseX = mouseX/resize_factor[1]
    mouseY = mouseY/resize_factor[0]
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

def compute_distance(P1, P2):
    # This function computes the euclidean distance between two 3D points
    return np.sqrt((P1[0] - P2[0]) ** 2 + (P1[1] - P2[1]) ** 2 + (P1[2] - P2[2]) ** 2)

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

    real_width = np.mean([compute_distance(P1, P2), compute_distance(P0, P3)])
    real_height = np.mean([compute_distance(P0, P1), compute_distance(P2, P3)])
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

# def derive_motor_angles_v0(orientation_matrix):
#     orientation_matrix = np.around(orientation_matrix, decimals=2)
#
#     b = orientation_matrix[0, 1]
#     e = orientation_matrix[1, 1]
#     i = orientation_matrix[2, 2]
#     g = orientation_matrix[2, 0]
#     h = orientation_matrix[2, 1]
#
#     s2 = h
#     c2 = np.sqrt(1 - s2 ** 2) # c2 is either a positive or negative angle. We need to try those angle combinations,
#                                 # and choose the angle combination with the lowest sum of angles
#     angle_signs = [1, -1]
#     join_sum = []
#     joint_combinations = np.zeros((2,3))
#     for j, angle_sign in enumerate(angle_signs):
#         c2_i = c2 * angle_sign
#         if c2_i == 0:
#             print("Potentially Gimbal lock")
#         c2_i = np.sign(c2_i)*np.maximum(0.0001, np.abs(c2_i)) # Gimbal lock case - Need to look into this
#         s1 = e / c2_i
#         c1 = b / c2_i
#         s3 = i / c2_i
#         c3 = g / c2_i
#         theta_1 = np.arctan2(s1, c1) / (np.pi / 180)
#         theta_2 = np.arctan2(s2, c2_i) / (np.pi / 180)
#         theta_3 = np.arctan2(s3, c3) / (np.pi / 180)
#         joint_combinations[j,:] = np.array([theta_1,theta_2, theta_3])
#         join_sum.append(np.sum(np.abs(np.array([theta_1,theta_2, theta_3]))))
#     # theta_1, theta_2, theta_3 = joint_combinations[np.argmin(join_sum)] # Choosing the angle combo with the least
#                                                                         # deviation required
#     theta_1, theta_2, theta_3 = joint_combinations[0]  # Choosing the angle combo with the least
#     theta_1_wrapped = dataset_object.wrap_angle_around_90(np.array([theta_1]))[0]
#     theta_2_wrapped = dataset_object.wrap_angle_around_90(np.array([theta_2]))[0]
#     theta_3_wrapped = dataset_object.wrap_angle_around_90(np.array([theta_3]))[0]
#     return [theta_1_wrapped, theta_2_wrapped, theta_3_wrapped]

def orient_wrist(theta1, theta2, theta3):
    # This function takes the joint angles, theta1, theta2, theta3, for
    # pronate/supinate, ulnar/radial, and flexion/extension, respectively, and computes the 8 bit integers
    # needed to drive the hand

    # Joint center positions as an 8 bit integer
    ps_home = 155 #102
    ur_home = 180
    fe_home = 150

    # Defining the physical joint limits
    pronate_supinate_limit = [-95, 90]
    ulnar_radial_limit = [-13, 21]
    # flexion_extension_limit = [-42, 30]
    flexion_extension_limit = [-35, 55]

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

# Variables that store where the subject pointed
mouseX, mouseY = [0, 0]

# Defining variables for Intel RealSense Feed
image_width = 640
image_height = 480
fps = 15
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

# Starting serial link to prosthetic arm
baud_rate = 115200
time_out = 0.05
ser = serial.Serial('COM6', baud_rate, timeout=time_out)
ser.write(b'h') # Homing the arm at the beginning of execution
# Loading the Mask-Grasp R-CNN Model
mode = "mask_grasp_rcnn"
MODEL_DIR = "models"
mask_grasp_model_path = 'models/colab_result_id#1/MASK_GRASP_RCNN_MODEL.h5'
inference_config = GraspMaskRCNNInferenceConfig()
mask_grasp_model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR, task=mode)
mask_grasp_model.load_weights(mask_grasp_model_path, by_name=True)
dataset_object = GraspMaskRCNNDataset()
center_crop_size = inference_config.IMAGE_MAX_DIM

# Defining variables for robot execution loop
frame_count = 0 # Stores the number of frames that have been run so far
cam_angles_t_1 = np.zeros(3,) # Stores the camera angles in frame t-1
num_samples_for_yaw = 50 # Number of frames to average to get the home heading angle
yaw_sum = 0 # Stores the cumulative sum of yaw angles for the first num_samples_for_yaw frames
previous_millis = 0 # Stores the time the last update was made
selection_flag = False # Specifies if the user selected an object or not
grasp_prob_thresh = 0.5
display_counter = 0
button_press_counter = 0
initiate_grasp = False
open_hand = False
aperture_command = 0
hand_preshaped = False
button_pressed = False
grasp_config_correct = False
window_resize_factor = np.array([2, 2])
theta1_t, theta2_t, theta3_t = [0,0,0]
fe_home = -18 #-28
d_theta = 0 ##stores the change in theta between frames
prev_thetas = np.array([theta1_t, theta2_t, theta3_t])
error_msg = ""

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


        # Defining the RealSense camera intrinsics as an Open3D object
        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=intrinsics.width, height=intrinsics.height,
                                                           fx=intrinsics.fx, fy=intrinsics.fy,
                                                           cx=intrinsics.ppx,
                                                           cy=intrinsics.ppy)
        # Hole filling to get a clean depth image
        hole_filling = rs.hole_filling_filter()
        aligned_depth_frame = hole_filling.process(aligned_depth_frame)

        if not aligned_depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        rgbd_image_resized = generate_rgbd_image(color_image, depth_image, center_crop=True,
                                                 square_size=center_crop_size)
        rgb_image_resized = rgbd_image_resized[:, :, 0:3].astype('uint8')
        color_image_to_display = cv2.cvtColor(rgb_image_resized, cv2.COLOR_RGB2BGR)
        new_shape = tuple(window_resize_factor * np.shape(color_image_to_display)[:2])
        resized_color_image_to_display = cv2.resize(color_image_to_display, new_shape, interpolation=cv2.INTER_AREA)

        # Computing time between frames
        previous_millis, dt = compute_dt(previous_millis, frame_count)
        if dt > 2:
            previous_millis, dt = compute_dt(previous_millis, frame_count, zero_timer=True)

        # Getting Camera Orientation [theta_pitch, theta_yaw, theta_roll]
        cam_angles_t = intel_realsense_IMU.camera_orientation_realsense(frames, dt, cam_angles_t_1)
        cam_angles_t_1 = np.array(cam_angles_t)
        # Getting Arm pose [theta_pitch, theta_roll, theta_yaw]
        if button_pressed and hand_preshaped:
            # c -> Read IMU and increment finger positions
            arm_angles_t = np.array(arm_orientation_imu_9250(ser, 'c'))
        elif open_hand and hand_preshaped:
            # o -> Read IMU and decrement finger positions
            arm_angles_t = np.array(arm_orientation_imu_9250(ser, 'o'))
        else:
            arm_angles_t = np.array(arm_orientation_imu_9250(ser))

        if frame_count < num_samples_for_yaw:
            yaw_sum += arm_angles_t[2]
            frame_count += 1
            print("WARNING: ZEROING MAGNETOMETER!...")
            continue
        elif frame_count == num_samples_for_yaw:
            yaw_resting = yaw_sum / (num_samples_for_yaw)
            print('DONE CALIBRATION')
        else:
            arm_angles_t[2] -= yaw_resting

        cam_pitch, cam_yaw, cam_roll = cam_angles_t
        arm_pitch, arm_roll, arm_yaw = arm_angles_t

        if not selection_flag:
            # Running Mask-Grasp R-CNN on the frame
            mask_grasp_results = mask_grasp_model.detect([rgbd_image_resized], verbose=0, task=mode)[0]
            # Capture input from user about which object to interact with
            selected_ROI = select_ROI(mouseX, mouseY, mask_grasp_results, window_resize_factor)
            rois, grasping_deltas, grasping_probs, masks, roi_scores, selection_flag = selected_ROI
            masked_image, colors = dataset_object.get_mask_overlay(color_image_to_display, masks, roi_scores, threshold=0)
            image_to_display = masked_image
            new_shape = tuple(window_resize_factor * np.shape(image_to_display)[:2])
            resized_image_to_display = cv2.resize(image_to_display, new_shape, interpolation=cv2.INTER_AREA)
            state = "roi_selection"
        else:
            if state == "roi_selection":
                color_image_with_grasp = cv2.cvtColor(rgb_image_resized, cv2.COLOR_RGB2BGR)
                color = (0, 255, 0)
                roi = rois[0]
                y1, x1, y2, x2 = np.split(roi, indices_or_sections=4, axis=-1)
                w = x2 - x1
                h = y2 - y1
                roi_vertices = cv2.boxPoints(((x1 + w / 2, y1 + h / 2), (w, h), 0))
                cv2.drawContours(color_image_with_grasp, [np.int0(roi_vertices)], 0, (0, 0, 0), 2)
                normalized_roi = utils.norm_boxes(roi, inference_config.IMAGE_SHAPE[:2])
                y1, x1, y2, x2 = np.split(normalized_roi, indices_or_sections=4, axis=-1)
                w = x2 - x1
                h = y2 - y1
                ROI_shape = np.array([h, w])
                pooled_feature_stride = np.array(ROI_shape / inference_config.GRASP_POOL_SIZE)
                grasping_anchors = utils.generate_grasping_anchors(inference_config.GRASP_ANCHOR_SIZE,
                                                                   inference_config.GRASP_ANCHOR_RATIOS,
                                                                   [inference_config.GRASP_POOL_SIZE,
                                                                    inference_config.GRASP_POOL_SIZE],
                                                                   pooled_feature_stride,
                                                                   1,
                                                                   inference_config.GRASP_ANCHOR_ANGLES,
                                                                   normalized_roi,
                                                                   inference_config)

                # Choosing grasp box that has the highest probability
                refined_results = dataset_object.refine_results(grasping_probs[0], grasping_deltas[0],
                                                                grasping_anchors, inference_config, filter_mode='top_k',
                                                                k=grasping_anchors.shape[0], nms=False)

                post_nms_predictions, top_box_probabilities, pre_nms_predictions, pre_nms_scores = refined_results
                # Selecting the grasp boxes with a grasp probability higher than grasp_prob_thresh
                filtered_predictions = post_nms_predictions[top_box_probabilities > grasp_prob_thresh]
                if filtered_predictions.shape[0] ==0:
                    filtered_predictions = post_nms_predictions[0].reshape((1,5))
                post_nms_predictions = filtered_predictions

                # Storing the high probability grasp orientations in Direction Cosine Matrix Format (DCM)
                potential_grasps = np.zeros((post_nms_predictions.shape[0], 3, 3))

                color_image_with_grasp = plot_selected_box(color_image_with_grasp, post_nms_predictions, 0)

                for k, five_dim_box in enumerate(post_nms_predictions):
                    # print("ARM Angles (theta_pitch, theta_roll, theta_yaw): ", arm_angles_t)
                    # print("CAMERA Angles (theta_pitch, theta_yaw, theta_roll): ", cam_angles_t)
                    # Need to crop out the point cloud at the oriented rectangle bounds
                    # Create a mask to specify the region to extract the surface normals from. Based on Lenz et al.,
                    # the approach vector is estimated as the surface normal calculated at the point of minumum depth in
                    # the central one third (horizontally) of the rectangle
                    x, y, w, h, theta = five_dim_box
                    # Undoing the center cropping of the image
                    x = int(x + (image_width - center_crop_size) // 2)
                    y = int(y + (image_height - center_crop_size) // 2)
                    # Getting grasping box vertices in the image plane
                    rect_vertices = cv2.boxPoints(((x, y), (w, h), theta))
                    # Calculating the approach vector for the grasp box
                    approach_vector = compute_approach_vector([x, y, w, h, theta])
                    # Getting the XYZ coordinates of the grasping box center <x, y>
                    box_center = de_project_point(intrinsics, aligned_depth_frame, [x, y])
                    # Wrapping angles so they are in [-90, 90] range
                    theta = dataset_object.wrap_angle_around_90(np.array([theta]))[0]
                    theta = theta * (np.pi / 180)  # network outputs positive angles in bottom right quadrant
                    # Computing the grasp box size in camera coordinates, size in meters
                    real_width, real_height = compute_real_box_size(intrinsics, aligned_depth_frame, rect_vertices)
                    box_vert_obj_frame = generate_points_in_world_frame(real_width, real_height)
                    # visualize_wrist_in_camera_frame(color_image, depth_image, box_center, approach_vector, intrinsics,
                    #                                 aligned_depth_frame, rect_vertices, approach_vector_orientation)

                    # Computing the desired wrist orientation in the camera frame
                    V = compute_wrist_orientation(approach_vector, theta)
                    rotation_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                                           [np.sin(theta), np.cos(theta), 0],
                                           [0, 0, 1]])
                    approach_vector_orientation = np.dot(V, rotation_z)

                    # Defining rotation matrix to go from the shoulder to camera frame


                    R_shoulder_camera = R.from_euler('xyz', [[cam_pitch, 0, cam_roll]],
                                                     degrees=True).as_matrix().squeeze()
                    grasp_orientation_shoulder = np.dot(R_shoulder_camera, approach_vector_orientation)
                    potential_grasps[k] = grasp_orientation_shoulder

                joint_deviations = np.zeros(potential_grasps.shape[0])
                joint_angles = np.zeros((potential_grasps.shape[0], 3))
                for i, grasp_orientation_shoulder in enumerate(potential_grasps):
                    grasp_orientation_shoulder_2 = grasp_orientation_shoulder.copy()
                    grasp_orientation_shoulder_2[:, :2] = grasp_orientation_shoulder_2[:, :2] * -1  # Equivalent graspbox
                    equivalent_grasps = [grasp_orientation_shoulder, grasp_orientation_shoulder_2]
                    equivalent_joint_deviations = np.zeros(2)
                    equivalent_joint_angles = np.zeros((2, 3))
                    for j, equivalent_grasp in enumerate(equivalent_grasps):
                        theta1_t, theta2_t, theta3_t = derive_motor_angles_v0(equivalent_grasp)
                        equivalent_joint_deviations[j] = np.sum(np.abs(np.array([theta1_t, theta2_t, theta3_t])))
                        equivalent_joint_angles[j] = np.array([theta1_t, theta2_t, theta3_t])
                    theta1_t, theta2_t, theta3_t = equivalent_joint_angles[np.argmin(equivalent_joint_deviations)]
                    joint_deviations[i] = np.sum(np.abs(np.array([theta1_t, theta2_t, theta3_t])))
                    joint_angles[i] = np.array([theta1_t, theta2_t, theta3_t])
                    potential_grasps[i] = equivalent_grasps[np.argmin(equivalent_joint_deviations)]

                grasp_box_index = np.argmin(joint_deviations)

                state = "display"
                color_image_with_grasp = plot_selected_box(color_image_with_grasp, post_nms_predictions,
                                                           grasp_box_index)
                new_shape = tuple(window_resize_factor * np.shape(color_image_with_grasp)[:2])
                resized_color_image_with_grasp = cv2.resize(color_image_with_grasp, new_shape,
                                                            interpolation=cv2.INTER_AREA)

            if state == "display" and grasp_config_correct:
                # theta1 : pronate/supinate
                # theta2 : ulnar/radial
                # theta3 : extension/flexion
                # print('arm_pitch, arm_roll, arm_yaw: ', arm_pitch, arm_roll, arm_yaw)
                arm_pitch, arm_roll, arm_yaw = arm_angles_t
                R_shoulder_arm = R.from_euler('xyz', [[arm_pitch, arm_yaw, arm_roll]],
                                              degrees=True).as_matrix().squeeze()
                R_shoulder_arm_inv = np.linalg.inv(R_shoulder_arm)
                theta1_t, theta2_t, theta3_t = derive_motor_angles_v0(np.dot(R_shoulder_arm_inv, potential_grasps[grasp_box_index]))
                theta3_t = theta3_t - fe_home
                d_theta = np.sum(np.abs(np.array([theta1_t, theta2_t, theta3_t]) - prev_thetas))

                if d_theta > 100 and display_counter !=0:
                    theta1_t, theta2_t, theta3_t = prev_thetas
                    error_msg = "CAUTION: delta theta was larger than limit"
                else:
                    error_msg = ""
                prev_thetas = np.array([theta1_t, theta2_t, theta3_t])


                joint1, joint2, joint3 = orient_wrist(theta1_t, theta2_t, theta3_t).tolist()

                print('ps', theta1_t, 'ur', theta2_t, 'ef', theta3_t)
                if initiate_grasp and not hand_preshaped:
                    preshape_hand(real_width, real_height, ser)
                    hand_preshaped = True
                if not initiate_grasp:
                    string_command = 'w %d %d %d' % (joint3, joint2, joint1)
                    ser.write(string_command.encode())
                display_counter += 1

            resized_image_to_display = images = np.hstack((resized_color_image_to_display, resized_color_image_with_grasp))
        frame_count += 1
        mouseX, mouseY = [0, 0]

        font = cv2.FONT_HERSHEY_SIMPLEX
        cam_orientation_text = "CAMERA ORIENTATION - Pitch : %d, Roll: %d, Yaw: %d" % (cam_pitch, cam_roll, cam_yaw)
        cam_orientation_location = (10, resized_image_to_display.shape[0] - 60)
        cv2.putText(resized_image_to_display, cam_orientation_text, cam_orientation_location, font, 0.5, (0, 0, 255))

        arm_orientation_text = "ARM ORIENTATION - Pitch : %d, Roll: %d, Yaw: %d" % (arm_pitch, arm_roll, arm_yaw)
        arm_orientation_location = (10, resized_image_to_display.shape[0] - 40)
        cv2.putText(resized_image_to_display, arm_orientation_text, arm_orientation_location, font, 0.5, (0, 0, 255))

        desired_arm_angles_text = "DESIRED ARM ANGLES - PS : %d, UR: %d, EF: %d" % (theta1_t, theta2_t, theta3_t)
        desired_arm_angles_location = (10, resized_image_to_display.shape[0] - 20)
        cv2.putText(resized_image_to_display, desired_arm_angles_text, desired_arm_angles_location, font, 0.5, (0, 0, 255))

        desired_arm_angles_text = "D_THETA: %d %s " % (d_theta, error_msg)
        desired_arm_angles_location = (10, resized_image_to_display.shape[0] - 80)
        cv2.putText(resized_image_to_display, desired_arm_angles_text, desired_arm_angles_location, font, 0.5,
                    (0, 0, 255))

        cv2.imshow('MASK-GRASP RCNN OUTPUT', resized_image_to_display)
        key = cv2.waitKey(1)


        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        if key & 0xFF == ord('h'):
            time.sleep(1)
            ser.write(b'h')
            time.sleep(1)
            break
        if key & 0xFF == ord('n'): # Grasp another object if n is pressed
            time.sleep(1)
            ser.write(b'h')
            time.sleep(1)
            hand_preshaped = False
            button_pressed = False
            open_hand = False
            grasp_config_correct = False
            selection_flag = False
            initiate_grasp = False
            display_counter = 0
            theta1_t, theta2_t, theta3_t = [0, 0, 0]
        if key == 13:
            grasp_config_correct = True
        if key == 8: # If backspace is pressed then open the hand
            open_hand = True
        else:
            open_hand = False
        if key == 32: # If space bar is pressed, capture state
            initiate_grasp = True
            button_pressed = True
        else:
            button_pressed = False

        if key & 0xFF == ord('a'): # For fine tuning the home extension position (fe_home)
            fe_home += 1
            print(fe_home)

        if key & 0xFF == ord('s'):
            fe_home -= 1
            print(fe_home)

        end_time = time.time()

finally:
    pipeline.stop()

