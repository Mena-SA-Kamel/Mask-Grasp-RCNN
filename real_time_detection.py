## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
from mrcnn import visualize
import matplotlib.pyplot as plt
import os
import mrcnn.model as modellib
from object_vs_background import InferenceConfig
import datetime

def get_mask_overlay(image, masks, scores, threshold = 0.90):
    num_masks = masks.shape[-1]
    colors = visualize.random_colors(num_masks)
    for i in list(range(num_masks)):
        if (scores[i] < threshold):
            continue
        mask = masks[:,:,i]
        image = visualize.apply_mask(image, mask, colors[i])
    image = np.array(image, dtype='uint8')
    return image
    # import code;
    # code.interact(local=dict(globals(), **locals()))

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)

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

inference_config = InferenceConfig()
MODEL_DIR = "models"
model_path = os.path.join(MODEL_DIR, "mask_rcnn_object_vs_background_SAMS-20-epochs.h5")
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

model.load_weights(model_path, by_name=True)
start_seconds = 0
current_seconds = 0

for i in list(range(20)):
    frames = pipeline.wait_for_frames()
# Streaming loop
try:
    while True:
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
        # import code;
        # code.interact(local=dict(globals(), **locals()))
        depth_image = depth_image * (depth_image < 2000)
        depth_scaled = ((depth_image / float(np.max( depth_image ))) * 255).astype('uint8')
        # depth_scaled = cv2.equalizeHist(depth_scaled)

        rgbd_image = np.zeros([color_image.shape[0], color_image.shape[1], 4])
        rgbd_image[:, :, 0:3] = color_image
        rgbd_image[:, :, 3] = depth_scaled
        results = model.detect([rgbd_image], verbose=0)
        r = results[0]
        masked_image = get_mask_overlay(rgbd_image[:,:,0:3], r['masks'], r['scores'], threshold=0.9)
        depth_3_channel = cv2.cvtColor(depth_scaled,cv2.COLOR_GRAY2BGR)
        images = np.hstack((color_image, depth_3_channel, masked_image))
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', images)
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
                break
        frame_count = frame_count + 1
        # Press esc or 'q' to close the image window

        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()