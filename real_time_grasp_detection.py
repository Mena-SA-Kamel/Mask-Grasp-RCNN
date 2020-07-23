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
from grasping_points import InferenceConfig, GraspingPointsDataset
import datetime
import matplotlib.patches as patches

def resize_frame(image):
    min_dimension = np.min(image.shape[:2])
    max_dimension = np.max(image.shape[:2])

    diff = (max_dimension - min_dimension)//2
    square_image = image[:, diff:max_dimension-diff, :]
    square_image_resized = cv2.resize(square_image, dsize=(320, 320))
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

inference_config = InferenceConfig()
MODEL_DIR = "models"
model_path = os.path.join(MODEL_DIR, 'colab_result_id#1',"train_#11c.h5")
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=inference_config, task="grasping_points")

model.load_weights(model_path, by_name=True)
start_seconds = 0
current_seconds = 0

dataset_object = GraspingPointsDataset()

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

        depth_image = depth_image * (depth_image < 2000)
        depth_scaled = np.interp(depth_image, (depth_image.min(), depth_image.max()), (0, 1)) * 255
        # depth_scaled = cv2.equalizeHist(depth_scaled)

        rgbd_image = np.zeros([color_image.shape[0], color_image.shape[1], 3])
        rgbd_image[:, :, 0:2] = color_image[:, :, 0:2]
        rgbd_image[:, :, 2] = depth_scaled
        rgbd_image = rgbd_image.astype('uint8')

        rgbd_image_resized = resize_frame(rgbd_image)
        results = model.detect([rgbd_image_resized], verbose=0, task = "grasping_points")
        r = results[0]
        post_nms_predictions, pre_nms_predictions = dataset_object.refine_results(r, model.anchors, model.config)

        fig, ax = plt.subplots()
        ax.imshow(rgbd_image_resized)
        for i, rect2 in enumerate(pre_nms_predictions):
            rect2 = dataset_object.bbox_convert_to_four_vertices([rect2])
            p2 = patches.Polygon(rect2[0], linewidth=2,edgecolor=dataset_object.generate_random_color(),facecolor='none')
            ax.add_patch(p2)
            ax.set_title('Boxes post non-maximum supression')
        plt.show()

        # depth_3_channel = cv2.cvtColor(depth_scaled,cv2.COLOR_GRAY2BGR)
        images = np.hstack((color_image))
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
                # break
        frame_count = frame_count + 1
        # Press esc or 'q' to close the image window

        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()