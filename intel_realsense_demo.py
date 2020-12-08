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

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)

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


# Streaming loop
for i in list(range(20)):
    frames = pipeline.wait_for_frames()
cv2.namedWindow('MASK-GRASP RCNN OUTPUT', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('MASK-GRASP RCNN OUTPUT', onMouse)
try:
    while True:
        start_time = time.time()
        frames = pipeline.wait_for_frames()
        colorized = colorizer.process(frames)
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
        images = color_image_to_display

        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        x, y = [200, 200]
        distance_to_point = aligned_depth_frame.as_depth_frame().get_distance(x, y)
        point1 = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], distance_to_point)

        # Calculating distance between two points
        # point2 = rs.rs2_deproject_pixel_to_point(color_intrin, [x, y], vdist)
        # # print str(point1)+str(point2)
        #
        # dist = math.sqrt(
        #     math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2) + math.pow(
        #         point1[2] - point2[2], 2))

        import code;

        code.interact(local=dict(globals(), **locals()))
        # :
        # pc = rs.pointcloud();
        # pc.map_to(color_frame);
        # pointcloud = pc.calculate(depth_frame);
        # pointcloud.export_to_ply("1.ply", color_frame);
        # cloud = PyntCloud.from_file("1.ply");
        # cloud.plot()

        #
        # ply = rs.save_to_ply("1.ply")
        #
        # # Set options to the desired values
        # # In this example we'll generate a textual PLY with normals (mesh is already created by default)
        # ply.set_option(rs.save_to_ply.option_ply_binary, False)
        # ply.set_option(rs.save_to_ply.option_ply_normals, True)
        #
        # print("Saving to 1.ply...")
        # # Apply the processing block to the frameset which contains the depth frame and the texture
        # ply.process(colorized)
        # print("Done")

        cv2.imshow('MASK-GRASP RCNN OUTPUT', images)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pipeline.stop()