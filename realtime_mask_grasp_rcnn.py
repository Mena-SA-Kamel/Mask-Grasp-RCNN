# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
from mrcnn import visualize
import matplotlib.pyplot as plt
import mrcnn.utils as utils
import os
import mrcnn.model as modellib
from mask_grasp_rcnn import GraspMaskRCNNInferenceConfig, GraspMaskRCNNDataset
import datetime
import matplotlib.patches as patches
import random


def generate_random_color():
    r = int(random.random()*255)
    b = int(random.random()*255)
    g = int(random.random()*255)
    return (r, g, b)

def resize_frame(image):
    min_dimension = np.min(image.shape[:2])
    max_dimension = np.max(image.shape[:2])

    diff = (max_dimension - min_dimension)//2
    square_image = image[:, diff:max_dimension-diff, :]
    square_image_resized = cv2.resize(square_image, dsize=(384, 384))
    return square_image_resized

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 6)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)
colorizer = rs.colorizer()
frame_count = 0
mode = "mask_grasp_rcnn"

inference_config = GraspMaskRCNNInferenceConfig()

MODEL_DIR = "models"
# mask_grasp_model_path = 'models/colab_result_id#1/mask_rcnn_grasp_and_mask_0152.h5'
# mask_grasp_model_path = 'models/colab_result_id#1/mask_rcnn_grasp_and_mask_0040.h5'
mask_grasp_model_path = 'models/colab_result_id#1/mask_rcnn_grasp_and_mask_0288.h5'


mask_grasp_model = modellib.MaskRCNN(mode="inference",
                           config=inference_config,
                           model_dir=MODEL_DIR, task=mode)

mask_grasp_model.load_weights(mask_grasp_model_path, by_name=True)
start_seconds = 0
current_seconds = 0

dataset_object = GraspMaskRCNNDataset()

for i in list(range(20)):
    frames = pipeline.wait_for_frames()
# Streaming loop
try:
    while True:
        plot_boxes = False
        frames = pipeline.wait_for_frames()
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        hole_filling = rs.hole_filling_filter()  # hole filling - hole filling
        aligned_depth_frame = hole_filling.process(aligned_depth_frame)
        # depth_color_frame = colorizer.colorize(aligned_depth_frame)

        if not aligned_depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(aligned_depth_frame.get_data())

        depth_image = depth_image * (depth_image < 2000)
        depth_scaled = np.interp(depth_image, (depth_image.min(), depth_image.max()), (0, 1)) * 255
        # depth_scaled = cv2.equalizeHist(depth_scaled)

        rgbd_image = np.zeros([color_image.shape[0], color_image.shape[1], 4])
        rgbd_image[:, :, 0:3] = color_image[:, :, 0:3]
        rgbd_image[:, :, 3] = depth_scaled
        rgbd_image = rgbd_image.astype('uint8')

        rgbd_image_resized = resize_frame(rgbd_image)
        results = mask_grasp_model.detect([rgbd_image_resized], verbose=0, task = mode)
        r = results[0]

        # post_nms_predictions, pre_nms_predictions = dataset_object.refine_results(r, mask_grasp_model.anchors, model.config)
        color_image_to_display = depth_3_channel = cv2.cvtColor(rgbd_image_resized[:, :, 0:3].astype('uint8'),
                                                                cv2.COLOR_RGB2BGR)
        masked_image, colors = dataset_object.get_mask_overlay(color_image_to_display, r['masks'], r['scores'], threshold=0)

        grasping_deltas = r['grasp_boxes']
        grasping_probs = r['grasp_probs']
        fig, ax = plt.subplots(1)
        ax.imshow(color_image_to_display)

        if len(r['rois']) > 0:
            plot_boxes=True
        for j, rect in enumerate(r['rois']):
            # color = generate_random_color()
            color = (np.array(colors[j])*255).astype('uint8')
            color = (int(color[0]), int(color[1]), int(color[2]))
            # centroid_color = np.array((int(centroid_color[0]), int(centroid_color[1]), int(centroid_color[2])))

            expanded_rect = utils.expand_roi_by_percent(rect, percentage=inference_config.GRASP_ROI_EXPAND_FACTOR,
                                                        image_shape=inference_config.IMAGE_SHAPE[:2])

            expanded_rect_normalized = utils.norm_boxes(expanded_rect, inference_config.IMAGE_SHAPE[:2])
            y1, x1, y2, x2 = expanded_rect_normalized
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            ROI_shape = np.array([h, w])
            pooled_feature_stride = np.array(ROI_shape/inference_config.GRASP_POOL_SIZE)#.astype('uint8')
            grasping_anchors = utils.generate_grasping_anchors(inference_config.GRASP_ANCHOR_SIZE,
                                                               inference_config.GRASP_ANCHOR_RATIOS,
                                                               [inference_config.GRASP_POOL_SIZE, inference_config.GRASP_POOL_SIZE],
                                                               pooled_feature_stride,
                                                               1,
                                                               inference_config.GRASP_ANCHOR_ANGLES,
                                                               expanded_rect_normalized)
            #
            # post_nms_predictions, pre_nms_predictions = dataset_object.refine_results(probabilities=grasping_probs[j], deltas=grasping_deltas[j],anchors=grasping_anchors, config=inference_config)
            post_nms_predictions, top_box_probabilities, pre_nms_predictions, pre_nms_scores = dataset_object.refine_results(
                grasping_probs[j], grasping_deltas[j], grasping_anchors, inference_config, filter_mode='prob')
            for i, rect in enumerate(pre_nms_predictions):
                # rect = dataset_object.bbox_convert_to_four_vertices([rect])
                rect = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
                grasp_rectangles_image = cv2.drawContours(color_image_to_display, [np.int0(rect)], 0, color, 2)

        depth_3_channel = cv2.cvtColor(rgbd_image_resized[:, :, 3].astype('uint8'), cv2.COLOR_GRAY2BGR)
        if plot_boxes:
            # images = np.hstack((color_image_to_display, depth_3_channel, masked_image, grasp_rectangles_image))
            images = np.hstack((masked_image, grasp_rectangles_image))
        else:
            # images = np.hstack((color_image_to_display, depth_3_channel, masked_image))
            images = np.hstack((masked_image, color_image_to_display))

        # images = np.hstack((color_image))
        cv2.namedWindow('MASK-GRASP RCNN OUTPUT', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('MASK-GRASP RCNN OUTPUT', images)
        key = cv2.waitKey(1)

        if frame_count == 0:
            currentDT = datetime.datetime.now()
            start_seconds = str(currentDT).split(' ')[1].split(':')[2]
            start_minute = str(currentDT).split(' ')[1].split(':')[1]
            start_seconds = int(float(start_seconds))
            start_minute = int(float(start_minute))
        else:
            currentDT = datetime.datetime.now()
            current_seconds = str(currentDT).split(' ')[1].split(':')[2]
            current_seconds = int(float(current_seconds))
        print(start_seconds, current_seconds)
        if (start_seconds == current_seconds):
            current_minute = str(currentDT).split(' ')[1].split(':')[1]
            current_minute = int(float(current_minute))
            if (start_minute != current_minute):
                print ('####################\n')
                print ('frames per minute:', frame_count)
                print('####################\n')
                # break
        frame_count = frame_count + 1
        # Press esc or 'q' to close the image window

        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()