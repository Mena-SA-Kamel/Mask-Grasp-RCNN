import pyrealsense2 as rs
import numpy as np
import cv2
import intel_realsense_IMU
import time

def onMouse(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
       mouseX, mouseY = [x, y]

def get_images_from_frames(color_frame, aligned_depth_frame):
    color_image = np.asanyarray(color_frame[0].get_data())
    depth_image = np.asanyarray(aligned_depth_frame[0].get_data())
    return [color_image, depth_image]

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

def display_gaze_on_image(image, x, y, color=(0,0,255)):
    image = cv2.circle(image, (x, y), 20, color, 3)
    image = cv2.circle(image, (x, y), 2, color, 2)
    return image

def resize_image(image, resize_factor):
    new_shape = tuple(resize_factor * np.shape(image)[:2])
    return cv2.resize(image, new_shape, interpolation=cv2.INTER_AREA)

def thread2(pipeline, profile, align, colorizer, image_width, image_height, fps, selected_roi, realsense_orientation,
            center_crop_size, color_frame, aligned_depth_frame, rgbd_image_resized, intrinsics, dataset_object,
            mask_grasp_results, initiate_grasp, avg_gaze, terminate):

    display_output = np.array([image_height, image_width, 1]).astype('uint8')
    # Defining Display parameters
    window_resize_factor = np.array([2, 2])

    cv2.namedWindow('MASK-GRASP RCNN OUTPUT', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('MASK-GRASP RCNN OUTPUT', onMouse)
    font = cv2.FONT_HERSHEY_SIMPLEX
    previous_millis = 0
    frame_count = 0
    cam_angles_t_1 = np.zeros(3, )

    # Streaming loop

    while not terminate[0]:
        try:
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

            previous_millis, dt = compute_dt(previous_millis, frame_count)
            if dt > 2:
                previous_millis, dt = compute_dt(previous_millis, frame_count, zero_timer=True)
            cam_angles_t = intel_realsense_IMU.camera_orientation_realsense(frames, dt, cam_angles_t_1)
            realsense_orientation[0] = cam_angles_t
            cam_angles_t_1 = np.array(cam_angles_t)
            frame_count += 1

            bgr_image = cv2.cvtColor(rgbd_image_resized[0].astype('uint8'), cv2.COLOR_RGB2BGR)
            gaze_x_realsense, gaze_y_realsense = avg_gaze
            image_to_display = bgr_image
            if mask_grasp_results != [None] and not initiate_grasp[0]:
                # Capture input from user about which object to interact with
                selected_roi[0] = select_ROI(gaze_x_realsense, gaze_y_realsense, mask_grasp_results[0])
                rois, grasping_deltas, grasping_probs, masks, roi_scores, selection_flag = selected_roi[0]
                masked_image, colors = dataset_object.get_mask_overlay(bgr_image, masks, roi_scores, threshold=0)
                image_to_display = masked_image
            image_with_gaze = display_gaze_on_image(image_to_display, gaze_x_realsense, gaze_y_realsense)
            resized_color_image_to_display = resize_image(image_with_gaze, window_resize_factor)
            display_output = resized_color_image_to_display
            cam_pitch, cam_roll, cam_yaw = realsense_orientation[0]
            cam_orientation_text = "CAMERA ORIENTATION - Pitch : %d, Roll: %d, Yaw: %d" % (cam_pitch, cam_yaw, cam_roll)
            cam_orientation_location = (10, display_output.shape[0] - 60)
            cv2.putText(display_output, cam_orientation_text, cam_orientation_location, font, 0.5, (0, 0, 255))
            key = cv2.waitKey(10)
            cv2.imshow('MASK-GRASP RCNN OUTPUT', display_output)

            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                terminate[0] = True
                break
            # 'Enter' pressed means initiate grasp for now
            if key == 13:
                initiate_grasp[0] = True
        except:
            continue