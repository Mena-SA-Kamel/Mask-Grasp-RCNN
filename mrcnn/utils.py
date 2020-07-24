"""
Mask R-CNN
Common utility functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import sys
import os
import logging
import math
import random
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy
import skimage.color
import skimage.io
import skimage.transform
import urllib.request
import shutil
import warnings
import cv2
from distutils.version import LooseVersion
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# URL from which to download the latest COCO trained weights
COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"


############################################################
#  Bounding Boxes
############################################################

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2, mode=''):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    if mode == 'grasping_points':
        # need anchor vertices to be in the form y1, x1, y2, x2
        xs = boxes1[:, 0]
        ys = boxes1[:, 1]
        ws = boxes1[:, 2]
        hs = boxes1[:, 3]
        thetas = boxes1[:, 4]
        anchor_areas = ws * hs

        anchor_vertices = np.zeros((boxes1.shape[0], 4))
        anchor_vertices[:, 0] = ys - hs / 2
        anchor_vertices[:, 1] = xs - ws / 2
        anchor_vertices[:, 2] = ys + hs / 2
        anchor_vertices[:, 3] = xs + ws / 2

        overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
        i = 1
        for i in range(overlaps.shape[1]):
            box2 = boxes2[i]
            # need gt vertices to be in the form y1, x1, y2, x2
            x, y, w, h, theta = box2
            gt_box_vertices = [y - h / 2, x - w / 2, y + h / 2, x + w / 2]
            gt_box_area = w * h
            # computing angle-related IoU (Based on Learning a Rotation Invariant Detector with Rotatable Bounding Box)
            iou = compute_iou(gt_box_vertices, anchor_vertices, gt_box_area, anchor_areas)
            angle_differences = np.cos(np.radians(thetas) - np.radians(theta))
            angle_differences[angle_differences < 0] = 0
            arIoU = iou * angle_differences
            # arIoU = np.interp(arIoU, (arIoU.min(), arIoU.max()), (0, 1))
            overlaps[:, i] = arIoU

        #
        #     image = np.zeros((384, 384))
        #     fig, ax = plt.subplots(1, figsize=(10, 10))
        #     ax.imshow(image)
        #
        #     for rect in boxes1[np.where(arIoU>0.3)[0]]:
        #         rect = bbox_convert_to_four_vertices([rect])[0]
        #         p = patches.Polygon(rect, linewidth=1,edgecolor='c',facecolor='none')
        #         ax.add_patch(p)
        #
        #     # plt.savefig(os.path.join('Grasping_anchors','P'+str(level+2)+ 'center_anchors.png'))
        #     gt_box = bbox_convert_to_four_vertices([box2])[0]
        #     p = patches.Polygon(gt_box, linewidth=1.5, edgecolor='b', facecolor='none')
        #     ax.add_patch(p)
        #     plt.show(block=False)
        #     if i > 20:
        #         break
        #     i = i + 1
        # import code;
        # code.interact(local=dict(globals(), **locals()))

    else:
        # Change area calculation for mode = 'grasping_points'
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
        # Each cell contains the IoU value.
        overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
        for i in range(overlaps.shape[1]):
            box2 = boxes2[i]
            overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps

def compute_grasping_training_anchors(gt_box, anchors, config):
    overlaps = np.zeros(anchors.shape[0])
    anchor_indices = np.arange(anchors.shape[0]).reshape(anchors.shape[0], 1)
    anchors = np.append(anchors, anchor_indices, axis=1)

    x, y, w, h, theta = gt_box
    gt_area = h * w
    angle_threshold = 30

    # anchors_out_of_boundary = anchors[np.logical_not(matching_anchors_index1)]
    # Negative Anchor if the distance between the centers are greater than 10% of the max distance (what if object is at a corner?)
    euclidean_distance = np.sqrt((anchors[:, 0] - x) ** 2 + (anchors[:, 1] - y) ** 2)
    euclidean_distance = (euclidean_distance / np.max(euclidean_distance))
    euclidean_distance = np.interp(euclidean_distance, (euclidean_distance.min(), euclidean_distance.max()), (0, 1))
    anchor_match_score = 1 - euclidean_distance
    overlaps[anchor_match_score < 0.9] = -1

    # print('Anchors before filtering: ', str(anchors.shape[0]))

    # Filtering based on center position

    gt_bbox_vertices = bbox_convert_to_four_vertices([gt_box])
    gt_bbox_vertices = gt_bbox_vertices[0]
    vertices_path = Path(gt_bbox_vertices)
    anchor_vertices = anchors[:, 0:2]
    matching_anchors_index1 = vertices_path.contains_points(anchor_vertices)
    anchors_step1 = anchors[matching_anchors_index1]
    if len(anchors_step1) == 0:
        anchors_step1 = anchors[np.where(euclidean_distance == 0)]
    # print('Anchors left after step 1 (Center Position): ', str(anchors_step1.shape[0]))

    # Filtering based on area / scale
    scales, ratios = np.meshgrid(config.RPN_ANCHOR_SCALES, config.RPN_ANCHOR_RATIOS)
    anchor_areas = anchors_step1[:, 2] * anchors_step1[:, 3]
    available_areas = np.unique(anchor_areas)
    matching_area = available_areas[0]

    min_diff = abs(matching_area - gt_area)
    for area in available_areas:
        difference = abs(area - gt_area)
        if difference < min_diff:
            min_diff = difference
            matching_area = area

    matching_anchors_index2 = (anchor_areas == matching_area)
    anchors_step2 = anchors_step1[matching_anchors_index2]
    # print('Anchors left after step 2 (Scale): ', str(anchors_step2.shape[0]))

    # Filtering based on angle
    anchor_angles = anchors_step2[:,4]
    angle_ranges = np.array([theta - angle_threshold, theta + angle_threshold])
    allowable_angles = np.array([angle_ranges - 180,
                                 angle_ranges,
                                 angle_ranges + 180])
    angle_values = np.unique(anchor_angles)

    # Finds the anchor angles that most closely match the gt_bbox theta value
    for angles in allowable_angles:
        angle_index = np.where(np.logical_and(angle_values >= angles[0], angle_values <= angles[1]))
        if not np.array(angle_index).size == 0:
            break
    matching_anchors_index3 = np.isin(anchor_angles, angle_values[angle_index])
    anchors_step3 = anchors_step2[matching_anchors_index3]
    if anchors_step3.shape[0] == 0:
        print('Anchors all cancelled')
        import code;
        code.interact(local=dict(globals(), **locals()))
    # print('Anchors left after step 3 (angle): ', str(anchors_step3.shape[0]))

    # Filtering based on euclidean distance
    anchor_x = anchors_step3[:,0]
    anchor_y = anchors_step3[:,1]
    euclidean_distance = np.sqrt((anchor_x - x) ** 2 + (anchor_y - y) ** 2)
    minimum_distance = np.min(euclidean_distance)
    anchors_step4 = anchors_step3[np.where(euclidean_distance == minimum_distance)]
    # print('Anchors left after step 4 (distance): ', str(anchors_step4.shape[0]))

    # Filtering based on aspect ratio
    gt_aspect_ratio = gt_area/h**2
    anchor_aspect_ratios = matching_area / anchors_step4[:,3]**2

    index = 0
    min_diff = abs(anchor_aspect_ratios[0] - gt_aspect_ratio)
    for i in list(range(len(anchor_aspect_ratios))):
        difference = abs(anchor_aspect_ratios[i] - gt_aspect_ratio)
        if difference < min_diff:
            min_diff = difference
            index = i

    final_anchor = anchors_step4[index]
    overlaps[int(final_anchor[-1])] = 1

    # image = np.zeros(config.IMAGE_SHAPE)
    # fig, ax = plt.subplots(1, figsize=(10, 10))
    # ax.imshow(image)
    # for i, rect in enumerate(anchors_step3):
    #     rect = bbox_convert_to_four_vertices([rect[0:5]])
    #     p = patches.Polygon(rect[0], linewidth=1,edgecolor='c',facecolor='none')
    #     ax.add_patch(p)
    # for i, rect in enumerate(anchors[overlaps == -1]):
    #     rect = bbox_convert_to_four_vertices([rect[0:5]])
    #     p = patches.Polygon(rect[0], linewidth=1,edgecolor='r',facecolor='none')
    #     ax.add_patch(p)
    # # plt.savefig(os.path.join('Grasping_anchors','P'+str(level+2)+ 'center_anchors.png'))
    # p = patches.Polygon(gt_bbox_vertices, linewidth=1.5, edgecolor='b', facecolor='none')
    # ax.add_patch(p)
    # matching_anchor_vertices = bbox_convert_to_four_vertices([final_anchor[0:5]])
    # p = patches.Polygon(matching_anchor_vertices[0], linewidth=1.5, edgecolor='k', facecolor='none')
    # ax.add_patch(p)
    # plt.show()
    return overlaps

def compute_grasping_training_anchors_new(gt_box, anchors, config):
    matches = np.zeros(anchors.shape[0])

    # need gt vertices to be in the form y1, x1, y2, x2
    x, y, w, h, theta = gt_box.copy()
    gt_box_vertices = [y - h/2, x - w/2, y + h/2, x + w/2]
    gt_box_area = gt_box[2] * gt_box[3]

    # need anchor vertices to be in the form y1, x1, y2, x2
    xs = anchors[:, 0]
    ys = anchors[:, 1]
    ws = anchors[:, 2]
    hs = anchors[:, 3]
    thetas = anchors[:, 4]
    anchor_areas = ws * hs

    anchor_vertices = np.zeros((anchors.shape[0], 4))
    anchor_vertices[:, 0] = ys - hs/2
    anchor_vertices[:, 1] = xs - ws/2
    anchor_vertices[:, 2] = ys + hs/2
    anchor_vertices[:, 3] = xs + ws/2

    # computing angle-related IoU (Based on Learning a Rotation Invariant Detector with Rotatable Bounding Box)
    iou = compute_iou(gt_box_vertices, anchor_vertices, gt_box_area, anchor_areas)
    angle_differences = np.cos(np.radians(thetas) - np.radians(theta))
    arIoU = iou*angle_differences
    import code;
    code.interact(local=dict(globals(), **locals()))

    # final_anchors_ix = np.where(arIoU > 0)[0]

    # final_anchors_ix = np.argsort(arIoU)[::-1][:25]
    final_anchors_ix = np.argmax(arIoU)
    final_anchors = anchors[final_anchors_ix]

    # Visulaizing anchors
    image = np.zeros(config.IMAGE_SHAPE)
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    for i, rect in enumerate([final_anchors]):
        rect = bbox_convert_to_four_vertices([rect])
        p = patches.Polygon(rect[0], linewidth=1,edgecolor='c',facecolor='none')
        ax.add_patch(p)

    matching_anchor_vertices = bbox_convert_to_four_vertices([gt_box])
    p = patches.Polygon(matching_anchor_vertices[0], linewidth=1.5, edgecolor='k', facecolor='none')
    ax.add_patch(p)
    plt.show()
    return matches

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


def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """
    
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps


def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def apply_box_deltas(boxes, deltas, mode='', num_angles=0):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)]. Note that (y2, x2) is outside the box.
    deltas: [N, (dy, dx, log(dh), log(dw))] # Modify this desription
    """
    if mode == 'grasping_points':
        boxes = boxes.astype(np.float32)
        center_x = boxes[:, 0]
        center_y = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]
        theta = boxes[:, 4]

        x = center_x + (deltas[:, 0] * width)
        y = center_y + (deltas[:, 1] * height)
        w = width * np.exp(deltas[:, 2])
        h = height * np.exp(deltas[:, 3])
        angle = theta + (deltas[:, 4] * (180/num_angles))
        return np.stack([x, y, w, h, angle], axis=1)

    else:
        boxes = boxes.astype(np.float32)
        # Convert to y, x, h, w
        height = boxes[:, 2] - boxes[:, 0]
        width = boxes[:, 3] - boxes[:, 1]
        center_y = boxes[:, 0] + 0.5 * height
        center_x = boxes[:, 1] + 0.5 * width
        # Apply deltas
        center_y += deltas[:, 0] * height
        center_x += deltas[:, 1] * width
        height *= np.exp(deltas[:, 2])
        width *= np.exp(deltas[:, 3])
        # Convert back to y1, x1, y2, x2
        y1 = center_y - 0.5 * height
        x1 = center_x - 0.5 * width
        y2 = y1 + height
        x2 = x1 + width
        return np.stack([y1, x1, y2, x2], axis=1)


def box_refinement_graph(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.math.log(gt_height / height)
    dw = tf.math.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result


def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]. (y2, x2) is
    assumed to be outside the box.
    """
    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = np.log(gt_height / height)
    dw = np.log(gt_width / width)

    return np.stack([dy, dx, dh, dw], axis=1)


############################################################
#  Dataset
############################################################

class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return

        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        logging.warning("You are using the default load_mask(), maybe you need to define your own one.")
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids


def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)),
                       preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop

def crop_bbox(window, bbox_vertices, original_shape, target_shape):
    """ Coorects the bounding boxes after center cropping the image.
    """
    # window
    y1, x1, y2, x2 = window
    cropped_bbox_vertices = []
    for box in bbox_vertices:
        adjusted_box = np.copy(box)
        adjusted_box[:, 0] = box[:, 0] - x1 # width
        adjusted_box[:, 1] = box[:, 1] - y1
        if any(adjusted_box[:, 0] > target_shape[1]) or any(adjusted_box[:, 1] > target_shape[0]):
            continue
        if any(adjusted_box[:, 0] < 0) or any(adjusted_box[:, 1] < 0):
            continue
        cropped_bbox_vertices.append(adjusted_box)
    cropped_bbox_vertices = np.array(cropped_bbox_vertices)
    return cropped_bbox_vertices

def resize_bbox(window, bbox_vertices, original_shape):
    """ Resizes the bounding_boxes specified by bbox_vertices using the affine transformation
    window: (y1, x1, y2, x2).
    """
    width = original_shape[1]
    height = original_shape[0]
    original_points = np.float32([[0,0], [0, height - 1], [width - 1, 0] ])

    # window
    y1, x1, y2, x2 = window
    # final_points = np.float32([[y1, x1], [y2-1, x1], [y1, x2-1]])
    final_points = np.float32([[x1, y1], [x1, y2-1], [x2-1, y1]])
    transformation_matrix = cv2.getAffineTransform(original_points, final_points)
    resized_bbox_vertices = []
    for box in bbox_vertices:
        box = np.array([np.float32(box)])
        resized_bbox_vertices.extend(cv2.transform(box, transformation_matrix).astype(int))
    resized_bbox_vertices = np.array(resized_bbox_vertices)
    return resized_bbox_vertices


def resize_mask(mask, scale, padding, crop=None):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y:y + h, x:x + w]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to reduce memory load.
    Mini-masks can be resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        # Pick slice and cast to bool in case load_mask() returned wrong dtype
        m = mask[:, :, i].astype(bool)
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        # Resize with bilinear interpolation
        m = resize(m, mini_shape)
        mini_mask[:, :, i] = np.around(m).astype(np.bool)
    return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().

    See inspect_data.ipynb notebook for more details.
    """
    mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        h = y2 - y1
        w = x2 - x1
        # Resize with bilinear interpolation
        m = resize(m, (h, w))
        mask[y1:y2, x1:x2, i] = np.around(m).astype(np.bool)
    return mask


# TODO: Build and use this function to reduce code duplication
def mold_mask(mask, config):
    pass


def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network to a format similar
    to its original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = resize(mask, (y2 - y1, x2 - x1))
    mask = np.where(mask >= threshold, 1, 0).astype(np.bool)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.bool)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask


############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride, mode='', angles=[]):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    if mode == 'grasping_points':
        boxes = generate_grasping_anchors(scales, ratios, shape, feature_stride, anchor_stride, angles)
    else:
        # Get all combinations of scales and ratios
        scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
        scales = scales.flatten()
        ratios = ratios.flatten()

        # Enumerate heights and widths from scales and ratios
        heights = scales / np.sqrt(ratios)
        widths = scales * np.sqrt(ratios)

        # Enumerate shifts in feature space
        shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
        shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
        shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

        # Enumerate combinations of shifts, widths, and heights
        box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
        box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

        # Reshape to get a list of (y, x) and a list of (h, w)
        box_centers = np.stack(
            [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
        box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

        # Convert to corner coordinates (y1, x1, y2, x2)
        boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                                box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_grasping_anchors(scales, ratios, shape, feature_stride, anchor_stride, thetas):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Main goal - Get bbox in the form {x, y, w, h, theta}
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride

    box_sizes = np.stack([heights, widths], axis = 1)

    boxes = np.array(np.meshgrid(shifts_x, shifts_y, ratios, thetas)).T.reshape(-1, 4)
    final_boxes = np.zeros((boxes.shape[0], 5))
    j = 0
    for i in ratios:
        final_boxes[boxes[:, 2] == i, 2:4] = box_sizes[j]
        j += 1
    final_boxes[:,0:2] = boxes[:,0:2]
    final_boxes[:,-1] = boxes[:,-1]

    return final_boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride, image_shape, mode='', angles=[]):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        if mode == 'grasping_points':
            # Using only anchors for feature level #4
            if i == 2:
                anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                                feature_strides[i], anchor_stride, mode, angles))
            else:
                continue
        else:
            anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                            feature_strides[i], anchor_stride, mode, angles))

    anchors = np.concatenate(anchors, axis=0)

    # anchors_filtered = anchors[valid_anchors_mask]
    # fig, ax = plt.subplots(1, figsize=(10, 10))
    # ax.imshow(np.zeros((500, 500)))
    # for i, rect2 in enumerate(anchors_filtered):
    #     rect2 = bbox_convert_to_four_vertices([rect2])
    #     p = patches.Polygon(rect2[0], linewidth=1,edgecolor='r',facecolor='none')
    #     ax.add_patch(p)
    # plt.show(block = False)
    #
    # fig, ax = plt.subplots(1, figsize=(10, 10))
    # ax.imshow(np.zeros((500, 500)))
    #
    # for i, rect2 in enumerate(anchors):
    #     rect2 = bbox_convert_to_four_vertices([rect2])
    #     p = patches.Polygon(rect2[0], linewidth=1, edgecolor='r', facecolor='none')
    #     ax.add_patch(p)
    # plt.show(block=False)
    # import code;
    # code.interact(local=dict(globals(), **locals()))
    return anchors

############################################################
#  Miscellaneous
############################################################

def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]


def compute_matches(gt_boxes, gt_class_ids, gt_masks,
                    pred_boxes, pred_class_ids, pred_scores, pred_masks,
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps


def compute_ap(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps


def compute_ap_range(gt_box, gt_class_id, gt_mask,
                     pred_box, pred_class_id, pred_score, pred_mask,
                     iou_thresholds=None, verbose=1):
    """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)
    
    # Compute AP over range of IoU thresholds
    AP = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls, overlaps =\
            compute_ap(gt_box, gt_class_id, gt_mask,
                        pred_box, pred_class_id, pred_score, pred_mask,
                        iou_threshold=iou_threshold)
        if verbose:
            print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))
        AP.append(ap)
    AP = np.array(AP).mean()
    if verbose:
        print("AP @{:.2f}-{:.2f}:\t {:.3f}".format(
            iou_thresholds[0], iou_thresholds[-1], AP))
    return AP


def compute_recall(pred_boxes, gt_boxes, iou):
    """Compute the recall at the given IoU threshold. It's an indication
    of how many GT boxes were found by the given prediction boxes.

    pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    gt_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    """
    # Measure overlaps
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)
    positive_ids = np.where(iou_max >= iou)[0]
    matched_gt_boxes = iou_argmax[positive_ids]

    recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]
    return recall, positive_ids


# ## Batch Slicing
# Some custom layers support a batch size of 1 only, and require a lot of work
# to support batches greater than 1. This function slices an input tensor
# across the batch dimension and feeds batches of size 1. Effectively,
# an easy way to support batches > 1 quickly with little code modification.
# In the long run, it's more efficient to modify the code to support large
# batches and getting rid of this function. Consider this a temporary solution
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result


def download_trained_weights(coco_model_path, verbose=1):
    """Download COCO trained weights from Releases.

    coco_model_path: local path of COCO trained weights
    """
    if verbose > 0:
        print("Downloading pretrained model to " + coco_model_path + " ...")
    with urllib.request.urlopen(COCO_MODEL_URL) as resp, open(coco_model_path, 'wb') as out:
        shutil.copyfileobj(resp, out)
    if verbose > 0:
        print("... done downloading pretrained model!")


def norm_boxes(boxes, shape, mode=''):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = shape
    if mode == 'grasping_points':
        # normalize angles relative to 90 degrees
        scale = np.array([w - 1, h - 1, w - 1, h - 1, 90])
        return np.divide(boxes, scale).astype(np.float32)
    else:
        scale = np.array([h - 1, w - 1, h - 1, w - 1])
        shift = np.array([0, 0, 1, 1])
        return np.divide((boxes - shift), scale).astype(np.float32)


def denorm_boxes(boxes, shape, mode=''):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = shape
    if mode == 'grasping_points':
        scale = np.array([w - 1, h - 1, w - 1, h - 1, 90])
        # return np.around(np.multiply(boxes, scale)).astype(np.float32)
        return (np.multiply(boxes, scale)).astype(np.float32)
    else:
        scale = np.array([h - 1, w - 1, h - 1, w - 1])
        shift = np.array([0, 0, 1, 1])
        return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)
