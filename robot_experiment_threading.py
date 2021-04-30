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
import time
import os

import threading
from helpers.thread_2 import thread2
from helpers.thread_1 import *
from helpers.grasp_selector import *


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

def get_images_from_frames(color_frame, aligned_depth_frame):
    color_image = np.asanyarray(color_frame[0].get_data())
    depth_image = np.asanyarray(aligned_depth_frame[0].get_data())
    return [color_image, depth_image]

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

def preshape_hand(real_width, real_height, ser):
    grasp_width = real_width * 1000
    grasp_width = np.minimum(grasp_width, 83)
    aperture_command = compute_hand_aperture(grasp_width)
    aperture_command = np.minimum(aperture_command, 220)
    aperture_command = np.maximum(aperture_command, 0)
    grasp_type = compute_grasp_type(real_height * 1000)
    string_command = 'g %d %d' % (grasp_type, int(aperture_command))
    ser.write(string_command.encode())

def extract_top_grasp_boxes(selected_roi, inference_config, grasp_prob_thresh, dataset_object):
    roi, grasping_deltas, grasping_probs, masks, roi_score, _ = selected_roi[0]
    normalized_roi = utils.norm_boxes(roi[0], inference_config.IMAGE_SHAPE[:2])
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
    if filtered_predictions.shape[0] == 0:
        # If no grasp box is above the threshold, get the highest probability box, regardless of the threshold
        filtered_predictions = post_nms_predictions[0].reshape((1, 5))
    return filtered_predictions

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

def main():

    # Defining variables for Intel RealSense Feed
    image_width = 640
    image_height = 480
    fps = 15

    # Defining variable to store the gaze location
    avg_gaze = [0, 0]
    aligned_depth_frame = [None]
    color_frame = [None]
    intrinsics = [None]
    terminate = [False]
    realsense_orientation = [None]

    # Defining calibration parameters between RealSense and Pupil Trackers
    M_t = np.array([[ 0.99891844, -0.0402869,   0.02321457, -0.08542229],
                    [ 0.03949054,  0.99864817,  0.03379828, -0.06768186],
                    [-0.02454482, -0.03284497,  0.99915903,  0.015576012]])
    tvec = M_t[:, -1]
    rvec, jacobian = cv2.Rodrigues(M_t[:, :3])
    realsense_intrinsics_matrix = np.array([[609.87304688, 0., 332.6171875],
                                            [0., 608.84387207, 248.34165955],
                                            [0., 0., 1.]])
    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=image_width, height=image_height,
                                                       fx=realsense_intrinsics_matrix[0, 0],
                                                       fy=realsense_intrinsics_matrix[1, 1],
                                                       cx=realsense_intrinsics_matrix[0, 2],
                                                       cy=realsense_intrinsics_matrix[1, 2])

    pipeline, profile, align, colorizer = initialize_realsense(image_width, image_height, fps)

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
    top_grasp_boxes = [None]
    grasp_box_index = [None]

    # Robot execution variables
    selected_roi = [None]
    initiate_grasp = [False]
    grasp_prob_thresh = 0.5

    # Initializing the eye tracker and storing the subscriber object to fetch gaze values as they update
    # Thread T1 - Running the eye tracker
    subscriber = initialize_pupil_tracker()
    t1 = threading.Thread(target=thread1,
                          args=(subscriber, avg_gaze, terminate, rvec, tvec, realsense_intrinsics_matrix, image_width,
                                image_height, center_crop_size))
    t1.start()

    # Thread T2 - Running Intel RealSense camera + UI
    t2 = threading.Thread(target=thread2,
                          args=(pipeline, profile, align, colorizer, image_width, image_height, fps, selected_roi,
                                realsense_orientation, center_crop_size,
                                color_frame, aligned_depth_frame, rgbd_image_resized, intrinsics, dataset_object,
                                mask_grasp_results, initiate_grasp, avg_gaze, top_grasp_boxes, grasp_box_index,
                                terminate))
    t2.start()

    while not terminate[0]:
        try:
            if not initiate_grasp[0]:
                rgbd_image_temp = rgbd_image_resized[0]
                mask_grasp_results[0] = mask_grasp_model.detect([rgbd_image_temp], verbose=0, task=mode)[0]  # slow step
                top_grasp_boxes[0] = extract_top_grasp_boxes(selected_roi, inference_config, grasp_prob_thresh,
                                                          dataset_object)
            else:
                cam_pitch, _, cam_roll = realsense_orientation[0]
                grasp_DCM, grasp_box_index[0] = select_grasp_box(realsense_orientation, top_grasp_boxes[0], image_width,
                                                                 image_height, center_crop_size, color_frame,
                                                                 aligned_depth_frame, o3d_intrinsics, dataset_object,
                                                                 intrinsics)


        except:
          continue
main()