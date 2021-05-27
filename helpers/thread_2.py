import pyrealsense2 as rs
import numpy as np
import cv2
import intel_realsense_IMU
import time
import helpers.experiment_planner as experiment_planner
import helpers.logger as logger
import winsound


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

    # image_shape = np.shape(image)
    # # CV2 reference is the bottom left corner, therefore y = image_height - y
    # y = image_shape[0] - y
    image = cv2.circle(image, (x, y), 20, color, 3)
    image = cv2.circle(image, (x, y), 2, color, 2)
    return image

def resize_image(image, resize_factor):
    new_shape = tuple(resize_factor * np.shape(image)[:2])
    return cv2.resize(image, new_shape, interpolation=cv2.INTER_AREA)

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

def current_time():
    return int(round(time.time() * 1000))

def thread2(pipeline, profile, align, colorizer, image_width, image_height, fps, selected_roi, realsense_orientation,
            arm_orientation, error_msg, center_crop_size, color_frame, aligned_depth_frame,
            rgbd_image_resized, intrinsics, dataset_object, mask_grasp_results, avg_gaze,
            top_grasp_boxes, grasp_box_index, UI_operations, terminate):

    display_output = np.array([image_height, image_width, 1]).astype('uint8')
    # Defining Display parameters
    window_resize_factor = np.array([2, 2])

    cv2.namedWindow('MASK-GRASP RCNN OUTPUT', cv2.WINDOW_AUTOSIZE)
    # cv2.namedWindow('MASK-GRASP RCNN OUTPUT', cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty('MASK-GRASP RCNN OUTPUT', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback('MASK-GRASP RCNN OUTPUT', onMouse)
    font = cv2.FONT_HERSHEY_SIMPLEX
    previous_millis = 0
    frame_count = 0
    cam_angles_t_1 = np.zeros(3, )
    image_to_display = np.zeros((image_height, image_width, 3))

    ########### Experiment Logging
    experiment_ID = 0
    num_baskets = 10
    objects = experiment_planner.load_data("helpers/all_objects.txt")
    data = []
    for i in range(num_baskets):
        basket_contents = experiment_planner.generate_experiment_basket(objects)
        trial_name = "Experiment_" + str(experiment_ID) + "_Basket_" + str(i)
        data.append(logger.log_new_trial(basket_contents, trial_name))
        # data[i][basket_contents[0]]['t_start']
        logger.write_to_json(data[i])
    object_iterator = 0
    basket_iterator = 0
    num_objects_in_basket = data[0]["num_objects"]
    basket_objects = data[0]["objects"]
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    t_start = t_complete = t_select = t_close = 0
    ###############################


    # Streaming loop
    while not terminate[0]:
        try:
            confirm_selection, move_Wrist, open_hand, close_hand, home_arm = UI_operations
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
            if mask_grasp_results != [None] and not confirm_selection:
                image_to_display = bgr_image # Displaying the current frame
                # Capture input from user about which object to interact with
                selected_roi[0] = select_ROI(gaze_x_realsense, gaze_y_realsense, mask_grasp_results[0])
                rois, grasping_deltas, grasping_probs, masks, roi_scores, selection_flag = selected_roi[0]
                masked_image, colors = dataset_object.get_mask_overlay(bgr_image, masks, roi_scores, threshold=0)
                image_to_display = masked_image
            elif confirm_selection:
                image_to_display = plot_selected_box(image_to_display, top_grasp_boxes[0], grasp_box_index[0])
            image_with_gaze = display_gaze_on_image(image_to_display.copy(), gaze_x_realsense, gaze_y_realsense)
            resized_color_image_to_display = resize_image(image_with_gaze, window_resize_factor)
            display_output = resized_color_image_to_display
            cam_pitch, cam_roll, cam_yaw = realsense_orientation[0]
            cam_orientation_text = "CAMERA ORIENTATION - Pitch : %d, Roll: %d, Yaw: %d" % (cam_pitch, cam_yaw, cam_roll)
            cam_orientation_location = (10, display_output.shape[0] - 60)
            cv2.putText(display_output, cam_orientation_text, cam_orientation_location, font, 0.5, (0, 0, 255))

            arm_pitch, arm_roll, arm_yaw = arm_orientation[0]
            arm_orientation_text = "ARM ORIENTATION - Pitch : %d, Roll: %d, Yaw: %d" % (arm_pitch, arm_roll, arm_yaw)
            arm_orientation_location = (10, display_output.shape[0] - 40)
            cv2.putText(display_output, arm_orientation_text, arm_orientation_location, font, 0.5, (0, 0, 255))

            desired_arm_angles_text = "EEROR: %s " % (error_msg)
            desired_arm_angles_location = (10, display_output.shape[0] - 80)
            cv2.putText(display_output, desired_arm_angles_text, desired_arm_angles_location, font, 0.5,
                        (0, 0, 255))

            key = cv2.waitKey(10)


            #confirm_selection, initiate_grasp, open_hand, close_hand, home_arm
            # Press q to quit
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                terminate[0] = True
                break

            if key & 0xFF == ord('h'):
                UI_operations[4] = True # home_arm

            # 'Enter' pressed means confirm object selection and grasp for now
            if key == 13:
                UI_operations[0] = True #confirm_selection
                object_type = basket_objects[object_iterator]
                t_select = current_time()
                data[basket_iterator][object_type]["t_select"] = t_select
                data[basket_iterator][object_type]["t_OSS"] = t_select - t_start

            if key & 0xFF == ord('n'): # Press n to go back to select another object
                UI_operations[0] = False #confirm_selection
                UI_operations[1] = False #confirm_selection
                UI_operations[4] = True  # home_arm

                object_type = basket_objects[object_iterator]
                t_complete = current_time()
                data[basket_iterator][object_type]["t_completion"] = t_complete

                object_iterator += 1
                if object_iterator == num_objects_in_basket:
                    logger.write_to_json(data[basket_iterator])
                    object_iterator = 0
                    basket_iterator += 1
                    num_objects_in_basket = data[basket_iterator]["num_objects"]
                    basket_objects = data[basket_iterator]["objects"]

                if basket_iterator == num_baskets:
                    terminate[0] = True
                    cv2.destroyAllWindows()
                    break

            if key & 0xFF == ord('b'): # Press b to begin timer
                object_type = basket_objects[object_iterator]
                t_start = current_time()
                data[basket_iterator][object_type]["t_start"] = t_start
                # when this is clicked, need to display the object type to the subject
                object_type_text = "GRAB: %s " % (object_type)
                object_type_location = (10, display_output.shape[0] - 100)
                cv2.putText(display_output, object_type_text, object_type_location, font, 0.7,
                            (0, 0, 255))
                winsound.Beep(frequency, duration)
                t_complete = t_select = t_close = 0

            if key & 0xFF == ord('o'): # Press o to go back to reselect object
                UI_operations[0] = False #confirm_selection
                UI_operations[1] = False #initiate_grasp
                UI_operations[4] = True  # home_arm
                object_type = basket_objects[object_iterator]
                t_select = current_time()
                data[basket_iterator][object_type]["NOSC"] += 1
                data[basket_iterator][object_type]["t_select"] = t_select
                data[basket_iterator][object_type]["t_OSS"] = t_select - t_start

            if key == 32:  # If space bar is pressed, initiate grasp - disable in experiment #1
                UI_operations[1] = True #orient_wrist
                UI_operations[4] = False  # home_arm


            if key == 9: # If tab is pressed then close the hand
                UI_operations[3] = True  # close_hand
                UI_operations[1] = False  # orient_wrist
            else:
                UI_operations[3] = False  # close_hand


            if key == 8:  # If backspace is pressed then open the hand
                UI_operations[2] = True #open_hand
            else:
                UI_operations[2] = False #open_hand

            cv2.imshow('MASK-GRASP RCNN OUTPUT', display_output)
        except:
            continue