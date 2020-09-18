import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from math import pi

def validate_box(bbox, image_shape):
    # checks if the bbox_vertices are all contained in the image boundaries
    bbox_vertices = bbox_convert_to_four_vertices([bbox])[0]
    x_vals = bbox_vertices[:, 0]
    y_vals = bbox_vertices[:, 1]
    invalid_x = np.where(np.logical_or(x_vals < 0, x_vals > image_shape[1]))[0]
    invalid_y = np.where(np.logical_or(y_vals < 0, y_vals > image_shape[0]))[0]

    return not(len(invalid_x) + len(invalid_y) > 0)

def deg2rad(angle):
    pi_over_180 = pi /180
    return angle * pi_over_180

def rotate_bboxes(bboxes, rotation_angle, image_shape, class_ids, scale=1):
    bbox_new = []
    invalid_indices = []
    img_center_y, img_center_x = np.array(image_shape[:2]) / 2
    rotational_matrix = cv2.getRotationMatrix2D((img_center_x, img_center_y), rotation_angle, scale)

    for i in range(bboxes.shape[0]):
        # import code;
        # code.interact(local=dict(globals(), **locals()))
        x, y, w, h, theta = bboxes[i]
        center_location = bboxes[i,:2]
        center_location = np.append(center_location, 1)
        new_x, new_y = np.dot(center_location , rotational_matrix.T)

        # new_theta = deg2rad(theta) + deg2rad(1*rotation_angle)
        # angle = np.arctan2(np.sin(new_theta), np.cos(new_theta))
        # angle /= (pi /180)

        transformed_bbox = [new_x, new_y, w, h, theta + rotation_angle]

        if validate_box(transformed_bbox, image_shape):
            bbox_new.append(transformed_bbox)
        else:
            invalid_indices.append(i)
    class_ids = np.delete(class_ids, invalid_indices)
    return np.array(bbox_new), class_ids

def translate_bbox(bboxes, dx, dy, image_shape, class_ids):
    bbox_new = []
    invalid_indices = []
    for i in range(bboxes.shape[0]):
        x, y, w, h, theta = bboxes[i]
        transformed_bbox = [x + dx, y + dy, w, h, theta]
        if validate_box(transformed_bbox, image_shape):
            bbox_new.append(transformed_bbox)
        else:
            invalid_indices.append(i)
    class_ids = np.delete(class_ids, invalid_indices)
    return np.array(bbox_new), class_ids

def flip_bbox(bboxes, flip_code, image_shape):
    bbox_new = []
    # flipcode = 0: flip vertically
    # flipcode > 0: flip horizontally
    # flipcode < 0: flip vertically and horizontally
    for i in range(bboxes.shape[0]):
        x, y, w, h, theta = bboxes[i]
        x = x - (image_shape[1]/2)
        y = y - (image_shape[0]/2)
        if flip_code == 0:
            y = -y
        elif flip_code == 1:
            x = -x
        x = x + (image_shape[1] / 2)
        y = y + (image_shape[0] / 2)
        theta = -theta
        transformed_bbox = [x, y, w, h, theta]
        bbox_new.append(transformed_bbox)
    return np.array(bbox_new)

def rotate_image(image, rotation_angle, scale=1):
    center_y, center_x = np.array(image.shape[:2]) / 2
    rotational_matrix = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, scale)
    rotated_image = cv2.warpAffine(image, rotational_matrix, (image.shape[1], image.shape[0]))
    return rotated_image

def translate_image(image, dx, dy):
    translation_matrix = np.float32([[1, 0, dx],
                                     [0, 1, dy]])
    translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    return translated_image

def flip_image(image, flip_code):
    return cv2.flip(image, flip_code)

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


#
# # bbox = [x,y,w,h,theta]
# bbox = np.array([[310, 120, 80, 130, 0],
#                 [175, 150, 130, 80, 0],
#                 [120, 320, 80, 130, -10]])
# image = plt.imread('texture_source.png')
#
# angles = np.arange(180, -180, -15)
# translations = np.arange(-140, 140, 20)
#
# flips = np.array([0, 1])
#
# # translate_bbox(bbox, 20, 20, image.shape)
# for rotation_angle in angles:
#     scale = 1
#     image_transformed = rotate_image(image, rotation_angle)
#     fig, ax = plt.subplots(1)
#     ax.imshow(image_transformed)
#     bbox_transformed = rotate_bboxes(bbox, rotation_angle, image.shape)
#     for i in range(bbox_transformed.shape[0]):
#         rect = bbox_convert_to_four_vertices([bbox_transformed[i]])
#         p = patches.Polygon(rect[0], linewidth=2,edgecolor='r',facecolor='none')
#         ax.add_patch(p)
#     plt.show(block=False)
#
# for i in range(10):
#     dx = random.choice(translations)
#     dy = random.choice(translations)
#     image_transformed = translate_image(image, dx, dy)
#     fig, ax = plt.subplots(1)
#     ax.imshow(image_transformed)
#     bbox_transformed = translate_bbox(bbox, dx, dy, image.shape)
#     for j in range(bbox_transformed.shape[0]):
#         rect = bbox_convert_to_four_vertices([bbox_transformed[j]])
#         p = patches.Polygon(rect[0], linewidth=2, edgecolor='r', facecolor='none')
#         ax.add_patch(p)
#     plt.show(block=False)
#
# for flip_type in flips:
#     image_transformed = flip_image(image, flip_type)
#     bbox_transformed = flip_bbox(bbox, flip_type, image.shape)
#     fig, ax = plt.subplots(1)
#     ax.imshow(image_transformed)
#     for j in range(bbox_transformed.shape[0]):
#         rect = bbox_convert_to_four_vertices([bbox_transformed[j]])
#         p = patches.Polygon(rect[0], linewidth=2, edgecolor='r', facecolor='none')
#         ax.add_patch(p)
#     plt.show(block=False)
#
#
# import code; code.interact(local=dict(globals(), **locals()))
