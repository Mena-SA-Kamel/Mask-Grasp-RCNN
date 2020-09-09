import numpy as np
import os
import mrcnn.utils as utils
import glob
import matplotlib.pyplot as plt
import skimage.io
import cv2
import math
import matplotlib
import imageio
import matplotlib.patches as patches
import matplotlib as mpl


IMAGE_MIN_DIM = 640
IMAGE_MAX_DIM = 640
IMAGE_RESIZE_MODE = "square"
IMAGE_MIN_SCALE = 0

def load_image(image_path):
    image = skimage.io.imread(image_path)
    return image

def load_jacquard_gt_boxes(positive_rectangles_path):
    bounding_boxes = []
    with open(positive_rectangles_path) as f:
        positive_rectangles = f.readlines()

    for line in positive_rectangles:

        bbox_5_dim = line.strip('\n').split(';')
        bbox_5_dim = np.array(bbox_5_dim).astype(np.float)
        x, y, theta, w, h = bbox_5_dim
        # Need to put a constraint on the aspect ratio and box sizes
        # Setting max aspect ratio to 20 and minimum to be 1/20
        aspect_ratio = w / h
        if (aspect_ratio < 1 / 20.0 or aspect_ratio > 20.0):
            continue
        # Minimum dimension allowed is 2 pixels
        if (h < 5 or w < 5):
            continue
        # # Theta re-adjustments:
        # if theta > 90:
        #     theta = theta - 180
        # elif theta < -90:
        #     theta = theta + 180
        bounding_boxes.append([x, y, w, h, theta])
    return np.array(bounding_boxes)

def get_five_dimensional_box(coordinates):
    x_coordinates = coordinates[:, 0]
    y_coordinates = coordinates[:, 1]
    gripper_orientation = coordinates[:2, :]
    x = ((np.max(x_coordinates) - np.min(x_coordinates)) / 2) + np.min(x_coordinates)
    y = ((np.max(y_coordinates) - np.min(y_coordinates)) / 2) + np.min(y_coordinates)

    # Gripper opening distance
    w = np.abs(np.linalg.norm(gripper_orientation[0] - gripper_orientation[1]))

    # Gripper width - Cross product betwee
    # h = np.cross(p2-p1, p1-p3)/norm(p2-p1)
    p1 = gripper_orientation[0]
    p2 = gripper_orientation[1]
    p3 = coordinates[2, :]
    #### ERROR HERE ####
    if (np.linalg.norm(p2 - p1) == 0):
        print("ZERO DIVISION")
        import code;
        code.interact(local=dict(globals(), **locals()))
    h = np.abs(np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1))

    deltaY = gripper_orientation[0][1] - gripper_orientation[1][1]
    deltaX = gripper_orientation[0][0] - gripper_orientation[1][0]
    theta = -1 * np.arctan2(deltaY, deltaX) * 180 / math.pi
    if theta > 90:
        theta = theta - 180
    elif theta < -90:
        theta = theta + 180
    return [x, y, w, h, theta]

def bbox_convert_to_four_vertices(bbox_5_dimension):
    rotated_points = []
    for i in range(len(bbox_5_dimension)):
        x, y, w, h, theta = bbox_5_dimension[i]
        original_points = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        original_points = np.array([np.float32(original_points)])

        translational_matrix = np.array([[1, 0, x - (w / 2)],
                                         [0, 1, y - (h / 2)]])

        original_points_translated = cv2.transform(original_points, translational_matrix)
        scale = 1
        angle = theta
        if np.sign(theta) == 1:
            angle = angle + 90
            if angle > 180:
                angle = angle - 360
        else:
            angle = angle + 90

        rotational_matrix = cv2.getRotationMatrix2D((x, y), angle + 90, scale)
        original_points_rotated = cv2.transform(original_points_translated, rotational_matrix)
        rotated_points.append(original_points_rotated[0])

    return np.array(rotated_points)

def load_bounding_boxes(box_paths):
    bbox_5_dimensional = load_jacquard_gt_boxes(box_paths)
    bounding_box_vertices = bbox_convert_to_four_vertices(bbox_5_dimensional)

    return bounding_box_vertices, bbox_5_dimensional

def bbox_convert_to_five_dimension(bounding_box_vertices):
    bbox_5_dimensional = []
    for bounding_box in bounding_box_vertices:
        x, y, w, h, theta = get_five_dimensional_box(bounding_box)
        bbox_5_dimensional.append([x, y, w, h, theta])
    bbox_5_dimensional = np.array(bbox_5_dimensional)
    return bbox_5_dimensional

def resize_grasp_box(window, grasp_box, original_shape):

    x, y, w, h, theta = np.split(grasp_box, indices_or_sections=5, axis=-1)

    y1, x1, y2, x2 = window
    new_image_width = x2 - x1
    new_image_height = y2 - y1

    x_ratio = x / original_shape[1]
    y_ratio = y / original_shape[0]
    w_ratio = w / original_shape[1]
    h_ratio = h / original_shape[0]

    x_new = x_ratio * new_image_width
    y_new = y_ratio * new_image_height
    w_new = w_ratio * new_image_width
    h_new = h_ratio * new_image_height

    resized_boxes = np.concatenate([x_new, y_new, w_new, h_new, theta], axis=-1)
    return resized_boxes



dataset_path = 'D:/Datasets/Jacquard-Dataset'
target_path = 'D:/Datasets/jacquard_dataset_resized_new'



folder_names = ['train_set', 'val_set', 'test_set']
subfolder_names = ['depth', 'rgb', 'grasp_rectangles']
# if not os.path.exists(target_path):
#     os.makedirs(target_path)
#     for folder_name in folder_names:
#         os.makedirs(os.path.join(target_path, folder_name))
#         for subfolder in subfolder_names:
#             os.makedirs(os.path.join(target_path, folder_name, subfolder))

for dataset in os.listdir(target_path):
    subfolder_path = os.path.join(target_path, dataset)
    image_list = glob.glob(subfolder_path + '/**/rgb/*.png', recursive=True)
    id = 0

    for image in image_list:
        object_id = image.split('\\')[-1].split('_')[1]
        image_id = image.split('\\')[-1].replace('_RGB.png', '_grasps.txt')
        source_grasp_rectangles = os.path.join(dataset_path, object_id, image_id)
        source_image = os.path.join(dataset_path, object_id, image.split('\\')[-1])
        target_grasp_rectangles_path = os.path.join(target_path, dataset, 'grasp_rectangles_new', image_id)

        image = (plt.imread(source_image) * 255).astype('uint8')
        original_shape = image.shape

        resized_image, window, scale, padding, crop = utils.resize_image(
            image,
            min_dim=IMAGE_MIN_DIM,
            min_scale=IMAGE_MIN_SCALE,
            max_dim=IMAGE_MAX_DIM,
            mode=IMAGE_RESIZE_MODE)

        bbox_vertices, bbox_5_dimensional = load_bounding_boxes(source_grasp_rectangles)
        # bbox_vertices, bbox_5_dimensional = load_bounding_boxes(positive_grasp_points)
        bbox_resized = resize_grasp_box(window, bbox_5_dimensional, original_shape)

        #
        # fig, ax1 = plt.subplots()
        # ax1.imshow(resized_image);
        #
        # for grasp_box in bbox_resized:
        #     x, y, w, h, theta = grasp_box
        #
        #     # import code;
        #     #
        #     # code.interact(local=dict(globals(), **locals()))
        #     w_new = np.minimum(w, h)
        #     h_new = np.maximum(w, h)
        #     x1 = x - w/2
        #     y1 = y - h/2
        #     theta %=360
        #     p = patches.Rectangle((x1, y1), w, h, angle=0, edgecolor=(1, 1, 0),
        #                           linewidth=0.5, facecolor='none')
        #     t2 = mpl.transforms.Affine2D().rotate_deg_around(x, y, theta) + ax1.transData
        #     p.set_transform(t2)
        #     ax1.add_patch(p)

        bbox_file = open(target_grasp_rectangles_path, "a")
        for row in bbox_resized:
            box_str = ''
            x, y, w, h, theta = row
            box_str = str(x) + ';' + str(y) + ';'+ str(theta) + ';'+ str(w) + ';'+ str(h) + '\n'
            bbox_file.write(box_str)
        bbox_file.close()
        #
        # depth_image = depth_image.astype('uint8')
        #
        # imageio.imwrite(target_rgb_path, rgb_image)
        # imageio.imwrite(target_depth_path, depth_image)
