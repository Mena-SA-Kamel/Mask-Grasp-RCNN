import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
import mrcnn.utils as utils
import mrcnn.model as modellib
from mask_grasp_rcnn import GraspMaskRCNNInferenceConfig, GraspMaskRCNNDataset
import datetime
import random
import time

mouseX, mouseY = [0, 0]

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

def onMouse(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
       mouseX, mouseY = [x, y]

def generate_rgbd_image(color_image, depth_image):
    depth_image = depth_image * (depth_image < 2000)
    depth_scaled = np.interp(depth_image, (depth_image.min(), depth_image.max()), (0, 1)) * 255
    rgbd_image = np.zeros([color_image.shape[0], color_image.shape[1], 4])
    rgbd_image[:, :, 0:3] = color_image[:, :, 0:3]
    rgbd_image[:, :, 3] = depth_scaled
    rgbd_image = rgbd_image.astype('uint8')
    # Frame is center cropped to 384x384
    rgbd_image_resized = resize_frame(rgbd_image)
    return rgbd_image_resized

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
            print (masks.shape, roi_scores.shape, selected_roi_ix)

    return rois, grasping_deltas, grasping_probs, masks, roi_scores, selection_success

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 60)

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
        hole_filling = rs.hole_filling_filter()
        aligned_depth_frame = hole_filling.process(aligned_depth_frame)
        if not aligned_depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        rgbd_image_resized = generate_rgbd_image(color_image, depth_image)
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
                    delta_time = delta_t[-2]
                    velocity = delta_position/ delta_time

                    # We need to propagate the rois with the calculated velocity
                    next_position = t_minus_1_loc + (velocity*delta_t[-1])
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
                rect = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
                color_image_to_display = cv2.drawContours(color_image_to_display, [np.int0(rect)], 0, (0,0,0), 2)
                color_image_to_display = cv2.drawContours(color_image_to_display, [np.int0(rect[:2])], 0, color, 2)
                color_image_to_display = cv2.drawContours(color_image_to_display, [np.int0(rect[2:])], 0, color, 2)

        images = color_image_to_display

        if not start_tracking and num_detected_frames == 0:
            images = masked_image
        cv2.imshow('MASK-GRASP RCNN OUTPUT', images)
        frame_count += 1
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