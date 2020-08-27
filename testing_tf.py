from math import pi

import keras.backend as K
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import mrcnn.utils as utils
import matplotlib.patches as patches

tf.enable_eager_execution()
# loss = np.arange(60)
# loss = loss.reshape((1, 12, 5))
# bbox_loss = K.variable(value=loss)
# bbox_loss = K.mean(bbox_loss, axis=1, keepdims=True)
# # tf.gather_nd(bbox_loss, negative_indices)
#
# test = np.arange(60)
# test = test.reshape((60, 1))
# test = K.variable(value=test)
# import code; code.interact(local=dict(globals(), **locals()))
# test = K.sum(test) / 256.0
#
# target_bbox = np.zeros(200).reshape((1, 40, 5))
# target_bbox = K.variable(value=target_bbox)
#
#
# num rows = batch size
# num cols = number of anchors
# location_bool = np.array([[True, False, False, True],
#                           [False, False, False, True],
#                           [False, True, True, True],
#                           [False, False, True, False],
#                           [False, False, True, False],
#                           [True, False, False, True]])
#
# data = np.ones(location_bool.shape)
#
# location_tensor = K.variable(value=location_bool)
#
# mean_regression_loss = np.ones((3, 5))
# negative_indices = np.array([[0,0],
#                             [1,1],
#                             [2,2]])
#
#
# mean_regression_loss = K.variable(value=mean_regression_loss)
# negative_indices = K.variable(value=negative_indices)
# negative_indices = K.cast(negative_indices, tf.int32)

# inplace_array = tf.gather_nd(mean_regression_loss, negative_indices)
# inplace_array = KL.Lambda(lambda x: x * 0.0)(inplace_array)
#
# bbox_loss = tf.tensor_scatter_nd_update(mean_regression_loss, negative_indices, inplace_array)
#
# classification_loss = np.random.rand(1, 3, 5)
# classification_loss = K.variable(value=classification_loss)
#
#
# target_class = np.array([[1,-1,-1,-1, 1],
#                          [-1,1,-1,-1, 1],
#                          [1,-1,-1,-1, -1]])
#
# target_class = np.reshape(target_class, classification_loss.shape)
#
# negative_class_mask = K.cast(K.equal(target_class, -1), tf.int32)
# positive_class_mask = K.cast(K.equal(target_class, 1), tf.int32)
#
# negative_indices = tf.where(K.equal(negative_class_mask, 1))
# positive_indices = tf.where(K.equal(positive_class_mask, 1))
#
# N = tf.count_nonzero(positive_class_mask, axis = -1)
# N = tf.reduce_sum(N, axis=-1)
# N = K.cast(N, tf.int32)
#
# values = tf.gather_nd(classification_loss, negative_indices)
#
# negative_class_losses = tf.sparse.SparseTensor(negative_indices, values, classification_loss.shape)
# negative_class_losses = tf.sparse.to_dense(negative_class_losses)
#
# BATCH_SIZE = 1
# TRAIN_ROIS_PER_IMAGE = 3
# GRASP_ANCHORS_PER_ROI = 5
# top_negative_losses_sum = K.variable(value=0)
# # for i in range(BATCH_SIZE):# batch size
# #     import code;
# #
# #     code.interact(local=dict(globals(), **locals()))
# #     for j in range(TRAIN_ROIS_PER_IMAGE):
# #         num_negatives = K.minimum(N[i][j] * 2, GRASP_ANCHORS_PER_ROI - N[i][j])
# #         num_negatives = tf.dtypes.cast(num_negatives, tf.int32)
# #         top_loss_per_row = tf.nn.top_k(negative_class_losses[i][j], num_negatives, sorted=True).values
# #         top_negative_losses_sum = tf.add(top_negative_losses_sum, K.sum(top_loss_per_row))
# #         print(top_loss_per_row)
# # print(top_negative_losses_sum)
# for i in range(BATCH_SIZE):
#     num_negatives = K.minimum(N[i] * 2, (GRASP_ANCHORS_PER_ROI * TRAIN_ROIS_PER_IMAGE) - N[i])
#     num_negatives = tf.dtypes.cast(num_negatives, tf.int32)
#     top_loss_per_row = tf.nn.top_k(K.flatten(negative_class_losses[i]), num_negatives, sorted=True).values
#     top_negative_losses_sum = tf.add(top_negative_losses_sum, K.sum(top_loss_per_row))
#
# import code;
# code.interact(local=dict(globals(), **locals()))
# total_positive_loss = K.sum(tf.gather_nd(classification_loss, positive_indices))



def generate_grasping_anchors_graph(inputs):
    # Main goal - Get bbox in the form {x, y, w, h, thetas
    stride_y, stride_x, y1, x1, y2, x2 = tf.split(inputs, num_or_size_splits=6)

    # To be replaced with config reference
    GRASP_POOL_SIZE = 7
    GRASP_ANCHOR_RATIOS = [1]
    GRASP_ANCHOR_ANGLES = [-67.5, -22.5, 22.5, 67.5]
    GRASP_ANCHOR_SIZE = [48]
    GRASP_ANCHOR_STRIDE = 1
    GRASP_ANCHORS_PER_ROI = GRASP_POOL_SIZE * GRASP_POOL_SIZE * len(GRASP_ANCHOR_RATIOS) * len(GRASP_ANCHOR_ANGLES) * len(GRASP_ANCHOR_SIZE)
    # GRASP_POOL_SIZE = K.variable(value=GRASP_POOL_SIZE)
    # GRASP_ANCHOR_RATIOS = K.variable(value=GRASP_ANCHOR_RATIOS)
    # GRASP_ANCHOR_ANGLES = K.variable(value=GRASP_ANCHOR_ANGLES)
    # GRASP_ANCHOR_SIZE = K.variable(value=GRASP_ANCHOR_SIZE)

    roi_heights = y2 - y1
    roi_widths = x2 - x1

    # feature_map_shape = [config.GRASP_POOL_SIZE, config.GRASP_POOL_SIZE]
    feature_map_shape = [GRASP_POOL_SIZE, GRASP_POOL_SIZE]

    # Adaptive anchor sizes with overlap factor
    overlap_factor = 1.5

    anchor_width = (roi_widths / feature_map_shape[1]) * overlap_factor
    anchor_height = (roi_heights / feature_map_shape[0]) * overlap_factor

    # Enumerate shifts in feature space
    shifts_y = tf.cast(tf.range(0, feature_map_shape[0], delta=GRASP_ANCHOR_STRIDE), dtype=tf.float32)
    shifts_y = (shifts_y*stride_y) + stride_y/2 + y1

    shifts_x = tf.cast(tf.range(0, feature_map_shape[1], delta=GRASP_ANCHOR_STRIDE), dtype=tf.float32)
    shifts_x = (shifts_x*stride_x) + stride_x / 2 + x1

    # Creating anchors
    boxes = tf.reshape(tf.transpose(tf.meshgrid(shifts_x, shifts_y, GRASP_ANCHOR_ANGLES)), (-1, 3))
    box_sizes = tf.concat([anchor_height, anchor_width / 2], axis=-1)
    box_sizes = tf.expand_dims(box_sizes, axis=-1)

    anchor_heights, anchor_widths = tf.split(K.repeat(box_sizes, n=GRASP_ANCHORS_PER_ROI), num_or_size_splits=2, axis=0)
    anchor_x, anchor_y, anchor_thetas = tf.split(boxes, num_or_size_splits=3, axis=-1)
    anchors = tf.concat([anchor_x, anchor_y, anchor_widths[0], anchor_heights[0], anchor_thetas], axis=-1)
    return anchors

def compute_grasp_deltas(box, gt_box):
    # Need to create a mask to keep valid refinements, ie: where the grasp box is not all zeros
    box_flattened = tf.reshape(box, [-1, 5])
    gt_box_flattened = tf.reshape(gt_box, [-1, 5])

    refinements_to_remove = tf.cast(tf.reduce_sum(gt_box_flattened, axis=-1), tf.bool)
    refinements_to_remove = tf.logical_not(refinements_to_remove)
    invalid_locations = tf.where(refinements_to_remove)
    refinements_replace = tf.zeros([tf.shape(invalid_locations)[0], 5])


    a_center_x = box_flattened[:, 0]
    a_center_y = box_flattened[:, 1]
    a_width = box_flattened[:, 2]
    a_height = box_flattened[:, 3]
    a_theta = box_flattened[:, 4]

    gt_center_x = gt_box_flattened[:, 0]
    gt_center_y = gt_box_flattened[:, 1]
    gt_width = gt_box_flattened[:, 2]
    gt_height = gt_box_flattened[:, 3]
    gt_theta = gt_box_flattened[:, 4]

    dx = (gt_center_x - a_center_x) / a_width
    dy = (gt_center_y - a_center_y) / a_height
    dw = tf.math.log(gt_width / a_width)
    dh = tf.math.log(gt_height / a_height)
    GRASP_ANCHOR_ANGLES = 4
    dtheta = (gt_theta - a_theta) / (180 / GRASP_ANCHOR_ANGLES)

    result = tf.stack([dx, dy, dw, dh, dtheta], axis=1)
    # Replacing refinements of zero boxes with zeros
    final_result = tf.tensor_scatter_nd_update(result, invalid_locations, refinements_replace)
    final_result_reshaped = tf.reshape(final_result, tf.shape(gt_box))
    return final_result_reshaped

def tf_deg2rad(angle):
    pi_over_180 = pi / 180
    return angle * pi_over_180

def grasp_box_refinement_graph(box, gt_box):
    # Need to create a mask to keep valid refinements, ie: where the grasp box is not all zeros
    box_flattened = tf.reshape(box, [-1, 5])
    gt_box_flattened = tf.reshape(gt_box, [-1, 5])

    refinements_to_remove = tf.cast(tf.reduce_sum(gt_box_flattened, axis=-1), tf.bool)
    refinements_to_remove = tf.logical_not(refinements_to_remove)
    invalid_locations = tf.where(refinements_to_remove)
    refinements_replace = tf.zeros([tf.shape(invalid_locations)[0], 5])

    a_center_x = box_flattened[:, 0]
    a_center_y = box_flattened[:, 1]
    a_width = box_flattened[:, 2]
    a_height = box_flattened[:, 3]
    a_theta = box_flattened[:, 4]

    gt_center_x = gt_box_flattened[:, 0]
    gt_center_y = gt_box_flattened[:, 1]
    gt_width = gt_box_flattened[:, 2]
    gt_height = gt_box_flattened[:, 3]
    gt_theta = gt_box_flattened[:, 4]

    dx = (gt_center_x - a_center_x) / a_width
    dy = (gt_center_y - a_center_y) / a_height
    dw = tf.math.log(gt_width / a_width)
    dh = tf.math.log(gt_height / a_height)
    dtheta = (gt_theta - a_theta) / (180 / 4)

    result = tf.stack([dx, dy, dw, dh, dtheta], axis=1)
    # Replacing refinements of zero boxes with zeros
    final_result = tf.tensor_scatter_nd_update(result, invalid_locations, refinements_replace)
    final_result_reshaped = tf.reshape(final_result, tf.shape(gt_box))
    return final_result_reshaped

def grasping_overlaps_graph_new(grasping_anchors, grasping_boxes):
    # boxes1:  grasping_anchors (proposals)
    # boxes2: final_roi_gt_grasp_boxes (gt)

    grasp_anchors = tf.reshape(grasping_anchors, [-1, 5])
    gt_grasp_boxes = tf.reshape(grasping_boxes, [-1, 5])

    # grasp_anchors, gt_grasp_boxes = tf.split(inputs, num_or_size_splits=2, axis=-1)
    b1_anchors = grasp_anchors
    b2_grasp_boxes = gt_grasp_boxes

    non_zero_grasp_boxes = tf.cast(tf.reduce_sum(tf.abs(b2_grasp_boxes), axis=1), tf.bool)

    # b1_anchors = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
    #                         [1, 1, tf.shape(boxes2)[0]]), [-1, 5])
    # b2_grasp_boxes = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])

    # Need to represent b1 and b2 in the form y1, x1, y2, x2
    b1_x, b1_y, b1_w, b1_h, b1_theta = tf.split(b1_anchors, num_or_size_splits=5, axis=-1)
    b2_x, b2_y, b2_w, b2_h, b2_theta = tf.split(b2_grasp_boxes, num_or_size_splits=5, axis=-1)

    b1_y1, b1_x1, b1_y2, b1_x2  = [b1_y - (b1_h/2), b1_x - (b1_w/2), b1_y + (b1_h/2), b1_x + (b1_w/2)]
    b2_y1, b2_x1, b2_y2, b2_x2 = [b2_y - (b2_h/2), b2_x - (b2_w/2), b2_y + (b2_h/2), b2_x + (b2_w/2)]

    # Compute intersection
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)

    # Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection

    # Compute IOU
    iou = intersection / union

    # Compute delta theta between boxes1 and boxes2 to get the ARIoU
    angle_differences = tf.cos(tf_deg2rad(b1_theta) - tf_deg2rad(b2_theta))
    angle_differences = tf.maximum(angle_differences, 0)
    arIoU = iou * angle_differences
    arIoU = arIoU * tf.expand_dims(tf.cast(non_zero_grasp_boxes, dtype=tf.float32), axis=-1)
    arIoU = tf.expand_dims(tf.reshape(arIoU, tf.shape(grasping_anchors)[:2]), axis=-1)
    return arIoU



def grasping_overlaps_graph(inputs):
    # boxes1:  grasping_anchors (proposals)
    # boxes2: final_roi_gt_grasp_boxes (gt)

    grasp_anchors, gt_grasp_boxes = tf.split(inputs, num_or_size_splits=2, axis=-1)
    b1_anchors = grasp_anchors
    b2_grasp_boxes = gt_grasp_boxes

    non_zero_grasp_boxes = tf.cast(tf.reduce_sum(tf.abs(b2_grasp_boxes), axis=1), tf.bool)

    # b1_anchors = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
    #                         [1, 1, tf.shape(boxes2)[0]]), [-1, 5])
    # b2_grasp_boxes = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])

    # Need to represent b1 and b2 in the form y1, x1, y2, x2
    b1_x, b1_y, b1_w, b1_h, b1_theta = tf.split(b1_anchors, num_or_size_splits=5, axis=-1)
    b2_x, b2_y, b2_w, b2_h, b2_theta = tf.split(b2_grasp_boxes, num_or_size_splits=5, axis=-1)

    b1_y1, b1_x1, b1_y2, b1_x2  = [b1_y - (b1_h/2), b1_x - (b1_w/2), b1_y + (b1_h/2), b1_x + (b1_w/2)]
    b2_y1, b2_x1, b2_y2, b2_x2 = [b2_y - (b2_h/2), b2_x - (b2_w/2), b2_y + (b2_h/2), b2_x + (b2_w/2)]

    # Compute intersection
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)

    # Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection

    # Compute IOU
    iou = intersection / union

    # Compute delta theta between boxes1 and boxes2 to get the ARIoU

    angle_differences = tf.cos(tf_deg2rad(b1_theta) - tf_deg2rad(b2_theta))
    angle_differences = tf.maximum(angle_differences, 0)
    arIoU = iou * angle_differences
    arIoU = arIoU * tf.expand_dims(tf.cast(non_zero_grasp_boxes, dtype=tf.float32), axis=-1)
    return arIoU


positive_rois = [[ 70, 50, 313, 330],
                 [ 90, 128, 275, 360]]

roi_gt_grasp_boxes = np.array([[[215, 272.5, 14.86606875, 28.31952463,-42.27368901],
                                     [240.5, 140.5,  36.05551275,   4.10478145,-70.55996517],
                                     [267, 253, 15.62049935, 29.06437174,-39.80557109],
                                     [0, 0, 0, 0, 0],
                                     [159.5, 205.5,  37.01351105,  18.42570404,-51.58194466],
                                     [280, 268.5,  12.80624847,  18.58467766,-38.65980825],
                                     [242, 141,  37.48332963,   3.52156549, -43.91907581],
                                     [242, 306,  11.66190379,  18.86484437,-30.96375653],
                                     [0, 0, 0, 0, 0],
                                     [153, 138.5,   4.47213595,   9.39148551,63.43494882]],
                               [[215, 272.5, 14.86606875, 28.31952463, -42.27368901],
                                [240.5, 140.5, 36.05551275, 4.10478145, -70.55996517],
                                [267, 253, 15.62049935, 29.06437174, -39.80557109],
                                [0, 0, 0, 0, 0],
                                [159.5, 205.5, 37.01351105, 18.42570404, -51.58194466],
                                [280, 268.5, 12.80624847, 18.58467766, -38.65980825],
                                [242, 141, 37.48332963, 3.52156549, -43.91907581],
                                [242, 306, 11.66190379, 18.86484437, -30.96375653],
                                [0, 0, 0, 0, 0],
                                [159, 138.5, 4.47213595, 9.39148551, 63.43494882]]
                               ])

# roi_gt_grasp_boxes = np.ones((2, 7, 5)) + 5
# roi_gt_grasp_boxes[0,:, 2] = np.zeros((1, 7))
# roi_gt_grasp_boxes[0, 1, :] = np.zeros((1, 5))
# roi_gt_grasp_boxes[1, :, 3] = np.zeros((1, 7))
# roi_gt_grasp_boxes[1, 3, :] = np.zeros((1, 5))

dx_s = np.array([1, 1]).reshape((2, 1))
dy_s = np.array([3, 3]).reshape((2, 1))

roi_gt_grasp_boxes = K.variable(value=roi_gt_grasp_boxes)
dx_s = K.variable(value=dx_s)
dy_s = K.variable(value=dy_s)
positive_rois = K.variable(value=positive_rois)

dx_s = tf.tile(dx_s, [1, tf.shape(roi_gt_grasp_boxes[:, :, 0])[1]])
dy_s = tf.tile(dy_s, [1, tf.shape(roi_gt_grasp_boxes[:, :, 1])[1]])

gt_grasp_box_x = tf.expand_dims(roi_gt_grasp_boxes[:, :, 0] - dx_s, axis=-1)
gt_grasp_box_y = tf.expand_dims(roi_gt_grasp_boxes[:, :, 1] - dy_s, axis=-1)
gt_grasp_box_w = tf.expand_dims(roi_gt_grasp_boxes[:, :, 2], axis=-1)
gt_grasp_box_h = tf.expand_dims(roi_gt_grasp_boxes[:, :, 3], axis=-1)
gt_grasp_box_theta = tf.expand_dims(roi_gt_grasp_boxes[:, :, 4], axis=-1)

# Gt grasp boxes specified in the reference frame of each proposal
referenced_grasp_boxes = tf.concat([gt_grasp_box_x, gt_grasp_box_y, gt_grasp_box_w, gt_grasp_box_h,
                                    gt_grasp_box_theta], axis=-1)

proposal_y1 = positive_rois[:, 0]
proposal_x1 = positive_rois[:, 1]
proposal_y2 = positive_rois[:, 2]
proposal_x2 = positive_rois[:, 3]
proposal_widths = tf.expand_dims(proposal_x2 - proposal_x1, axis=-1)
proposal_heights = tf.expand_dims(proposal_y2 - proposal_y1, axis=-1)

sensitivity = 0.7
radius = (((gt_grasp_box_w/2)**2 + (gt_grasp_box_h/2)**2)**0.5) * sensitivity
proposal_y1_reshaped = K.repeat(tf.expand_dims(proposal_y1, axis=-1), n=radius.shape[1])
proposal_x1_reshaped = K.repeat(tf.expand_dims(proposal_x1, axis=-1), n=radius.shape[1])
proposal_y2_reshaped = K.repeat(tf.expand_dims(proposal_y2, axis=-1), n=radius.shape[1])
proposal_x2_reshaped = K.repeat(tf.expand_dims(proposal_x2, axis=-1), n=radius.shape[1])

valid_grasp_boxes = tf.less(gt_grasp_box_x + radius, proposal_x2_reshaped)
valid_grasp_boxes = tf.logical_and(valid_grasp_boxes, tf.greater(gt_grasp_box_x - radius, proposal_x1_reshaped))
valid_grasp_boxes = tf.logical_and(valid_grasp_boxes, tf.less(gt_grasp_box_y + radius, proposal_y2_reshaped))
valid_grasp_boxes = tf.logical_and(valid_grasp_boxes, tf.greater(gt_grasp_box_y - radius, proposal_y1_reshaped))

# sample_arr = [True, False]
# valid_grasp_boxes = np.random.choice(sample_arr, size=14).reshape((2,7,1))
# valid_grasp_boxes = K.variable(value=valid_grasp_boxes)
final_roi_gt_grasp_boxes = referenced_grasp_boxes * tf.cast(valid_grasp_boxes, dtype='float32')

GRASP_POOL_SIZE = 7
pooled_feature_stride = tf.concat([proposal_heights, proposal_widths], axis = -1) /GRASP_POOL_SIZE
pooled_feature_stride = tf.cast(pooled_feature_stride, tf.float32)

grasping_anchors = tf.map_fn(generate_grasping_anchors_graph,
          tf.concat([pooled_feature_stride, positive_rois], axis=1))


final_roi_gt_grasp_class = np.ones(10)
final_roi_gt_grasp_class[3] = 0
final_roi_gt_grasp_class[8] = 0
final_roi_gt_grasp_class = final_roi_gt_grasp_class.reshape((1,10,1))

final_roi_gt_grasp_boxes = K.variable(value=final_roi_gt_grasp_boxes)
final_roi_gt_grasp_class = K.variable(value=final_roi_gt_grasp_class)
positive_rois = K.variable(value=positive_rois)

boxes1 = grasping_anchors
boxes2 = final_roi_gt_grasp_boxes

# Reshaping to get equal sized tensors
grasping_anchors_reshaped = tf.tile(boxes1, [1, tf.shape(boxes2)[1], 1])
grasping_boxes_reshaped = tf.tile(K.expand_dims(boxes2, axis=2), [1, 1,tf.shape(boxes1)[1], 1])
grasping_boxes_reshaped = tf.reshape(grasping_boxes_reshaped, tf.shape(grasping_anchors_reshaped))


# grasp_overlaps = tf.map_fn(fn=grasping_overlaps_graph,
#           elems=tf.concat([grasping_anchors_reshaped, grasping_boxes_reshaped], axis=-1))


grasp_overlaps = grasping_overlaps_graph_new(grasping_anchors_reshaped, grasping_boxes_reshaped)



# grasp_overlaps_reshaped has the shape [num_instances, num_grasp_instances, num_anchors]
grasp_overlaps_reshaped = tf.reshape(tf.squeeze(grasp_overlaps), [grasp_overlaps.shape[0], boxes2.shape[1], boxes1.shape[1]])

ARIOU_NEG_THRESHOLD = 0.01
ARIOU_POS_THRESHOLD = 0.1

# grasp_anchor_match - this is 1 for positive anchors, -1 for negative anchors
# grasp_box_match - this specifies the id of the grasp box for each positive anchor, -1 for negative anchors
# grasp_anchor_match = tf.ones(grasping_anchors.shape[:2])*-1
# grasp_box_match = tf.ones(grasping_anchors.shape[:2])*-1
grasp_anchor_match = tf.ones(tf.shape(grasping_anchors)[:2])*-1
grasp_box_match = tf.ones(tf.shape(grasping_anchors)[:2])*-1

# Specifies the max overlap for each GT box: shape [num of ROIs, num of GT grasp boxes]
grasp_roi_iou_max = tf.reduce_max(grasp_overlaps_reshaped, axis=-1)

# Specifies the max overlap anchor for each GT box: shape [num of ROIs, num of GT grasp boxes].
# Unmatched boxes match to anchor #0
non_zero_grasp_boxes = tf.cast(tf.reduce_sum(tf.abs(grasp_overlaps_reshaped), axis=-1), tf.bool)
non_zero_grasp_box_indices = tf.where(tf.cast(tf.reduce_sum(tf.abs(grasp_overlaps_reshaped), axis=-1), tf.bool))

grasp_roi_iou_argmax = tf.expand_dims(tf.argmax(grasp_overlaps_reshaped, axis=-1), axis=-1)

# Creating an ROI index variable that specifies the ROI that each GT grasp box belongs to
grasp_roi_ix = tf.range(0, tf.shape(grasp_roi_iou_argmax)[0])
grasp_roi_ix = tf.tile(tf.expand_dims(grasp_roi_ix, axis=-1), [1, tf.shape(grasp_roi_iou_argmax)[1]])
grasp_roi_ix = tf.expand_dims(tf.cast(grasp_roi_ix, dtype=tf.int64), axis=-1)

# Creating a grasp box index variable
grasp_box_ix = tf.tile(tf.range(tf.shape(grasp_overlaps_reshaped)[1]), [tf.shape(grasp_overlaps_reshaped)[0]])
grasp_box_ix = tf.cast(tf.reshape(grasp_box_ix, grasp_roi_iou_argmax.shape), dtype=tf.int64)
grasp_box_ix = tf.gather_nd(grasp_box_ix, non_zero_grasp_box_indices)

# Creating a tensor that specifies the indices of the positive anchors [ROI number, anchor id].
# Number of indices == number of GT grasp boxes
positive_grasp_anchors_ix = tf.concat([grasp_roi_ix, grasp_roi_iou_argmax], axis=-1)
positive_grasp_anchors_ix = tf.reshape(positive_grasp_anchors_ix, [-1, 2])
ones_tensor_1 = tf.ones(tf.shape(positive_grasp_anchors_ix)[0])

# PROBLEM: zero padded GT boxes will also match to anchor at index 0, need to set those anchors to negative
grasp_anchor_match = tf.tensor_scatter_nd_update(grasp_anchor_match, positive_grasp_anchors_ix, ones_tensor_1)

# grasp_box_match specifies the GT grasp box index an anchor matched to
updates_grasp_box = tf.cast(tf.reshape(grasp_box_ix, [-1]), dtype=tf.float32)
grasp_box_match = tf.tensor_scatter_nd_update(grasp_box_match, positive_grasp_anchors_ix, updates_grasp_box)

# Keeping boxes that have an overlap higher than ARIOU_POS_THRESHOLD
anchors_to_keep = tf.where(grasp_overlaps_reshaped > ARIOU_POS_THRESHOLD)
roi_ix, grasp_box_ix, anchor_ix = tf.split(anchors_to_keep, num_or_size_splits=3, axis=-1)
anchors_to_keep_ix = tf.concat([roi_ix, anchor_ix], axis=-1)

ones_tensor_2 = tf.ones(tf.shape(anchors_to_keep_ix)[0])
grasp_anchor_match = tf.tensor_scatter_nd_update(grasp_anchor_match, anchors_to_keep_ix, ones_tensor_2)

updates_grasp_box = tf.cast(tf.reshape(grasp_box_ix, [-1]), dtype=tf.float32)
grasp_box_match = tf.tensor_scatter_nd_update(grasp_box_match, anchors_to_keep_ix, updates_grasp_box)

# Creating mask for positive anchors. True for positive anchors False for negative anchors
positive_anchors_mask = grasp_anchor_match > 0

# Compute deltas, shape of grasp_bbox : [number of ROIs, num of positive grasp instances, 5]
grasp_bbox = tf.zeros(grasping_anchors.shape)

positive_grasp_box_locations = tf.where(positive_anchors_mask)
grasp_box_ix = tf.gather_nd(grasp_box_match, positive_grasp_box_locations)
grasp_box_ix = tf.cast(grasp_box_ix, dtype=tf.int64)

# Format of anchor_assignment is [instance #, anchor_id, grasp_box_id]
anchor_assignment = tf.concat([positive_grasp_box_locations, tf.reshape(grasp_box_ix, [-1, 1])], axis=-1)
roi_ix, anchor_ix, grasp_box_ix = tf.split(anchor_assignment, num_or_size_splits=3, axis=-1)
updates = tf.gather_nd(final_roi_gt_grasp_boxes, tf.concat([roi_ix, grasp_box_ix], axis=-1))

# gt_grasp_boxes_filtered is a tensor that specifies the gt grasp box that matches with each tensor.
# Shape is [num of rois, num of anchors, 5]
gt_grasp_boxes_filtered = tf.tensor_scatter_nd_update(grasp_bbox, tf.concat([roi_ix, anchor_ix], axis=-1), updates)

grasp_deltas = grasp_box_refinement_graph(grasping_anchors, gt_grasp_boxes_filtered)

import code; code.interact(local=dict(globals(), **locals()))

GRASP_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2, 1])
grasp_deltas /= GRASP_BBOX_STD_DEV

# grasp_anchor_match : Marks the positive anchors with 1 and negative anchors with -1
# grasp_box_match : Specifies the id of the grasp box the positive anchor matched to, -1 for negative anchors
# grasp_deltas : Specifies the refinements [dx, dy, dw, dh, dtheta] to go from the anchor to the gt grasp box
# grasping_anchors : The adaptive sized anchors for each ROI, covering the 7x7 pooled feature space

# Visualizing Boxes and Anchors and Anchor selection strategy
grasp_anchors_np = grasping_anchors.numpy()
gt_boxes_np = positive_rois.numpy()
gt_grasp_points_np = final_roi_gt_grasp_boxes.numpy()
grasp_overlaps_reshaped_np = grasp_overlaps_reshaped.numpy()

image = np.zeros((384, 384))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
ax1.imshow(image)
ax2.imshow(image)


for i in range(grasp_anchors_np.shape[1]):
    rect_1 = utils.bbox_convert_to_four_vertices([grasp_anchors_np[0, i]])[0]
    p = patches.Polygon(rect_1, linewidth=0.25, edgecolor='c', facecolor='none')
    ax1.add_patch(p)

    rect_2 = utils.bbox_convert_to_four_vertices([grasp_anchors_np[1, i]])[0]
    p = patches.Polygon(rect_2, linewidth=0.25, edgecolor='c', facecolor='none')
    ax2.add_patch(p)

for i in range(gt_boxes_np.shape[0]):
    y1, x1, y2, x2 = gt_boxes_np[i]
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    p = patches.Rectangle((x1, y1), w, h, facecolor=None, fill=False, color='k', linewidth=2)
    fig.axes[i].add_patch(p)
    for j in gt_grasp_points_np[i]:
        rect = utils.bbox_convert_to_four_vertices([j])[0]
        q = patches.Polygon(rect, linewidth=1.5, edgecolor='r', facecolor='none')
        fig.axes[i].add_patch(q)
    # [number_of_anchors, number of gt_grasp_boxes]
    anchor_overlaps = grasp_overlaps_reshaped_np[i]

    # grasp_anchors_np[i, np.where(anchor_overlaps > 0.01)[1]]
    for matching_anchors in grasp_anchors_np[i, np.where(grasp_anchor_match[i] > 0)][0]:
        rect = utils.bbox_convert_to_four_vertices([matching_anchors])[0]
        p = patches.Polygon(rect, linewidth=2,edgecolor='g',facecolor='none')
        fig.axes[i].add_patch(p)

    GRASP_ANCHOR_ANGLES = [-67.5, -22.5, 22.5, 67.5]
    # Choose positive_anchors
    anchors = grasp_anchors_np[i]
    positive_anchors_mask = (grasp_anchor_match[i]>0).numpy()
    deltas = grasp_deltas[i] * GRASP_BBOX_STD_DEV
    refined_anchors = utils.apply_box_deltas(anchors[positive_anchors_mask], deltas[positive_anchors_mask], 'mask_grasp_rcnn', len(GRASP_ANCHOR_ANGLES))
    for refined_anchor in refined_anchors:
        rect = utils.bbox_convert_to_four_vertices([refined_anchor])[0]
        p = patches.Polygon(rect, linewidth=0.5,edgecolor='2',facecolor='none')
        fig.axes[i].add_patch(p)



plt.show(block=False)
import code;


code.interact(local=dict(globals(), **locals()))

# updates = np.ones((4,1))
# positive_indices = np.array([2, 5, 7, 6])
# positive_indices = np.reshape(positive_indices, (-1, 1))
#
# updates = K.variable(value=updates)
# positive_indices = K.variable(value=positive_indices)

# positive_indices_mask = tf.scatter_nd(positive_indices, updates, [10, 1])
#
#
# #
# # top_loss_ix = tf.nn.top_k(total_loss, 2, sorted=True, name="rpn_top_losses").values
# #
# # filtered_target_bbox = tf.gather(total_loss_tensor, valid_anchor_indices, axis = 1)
# # filtered_target_bbox = tf.squeeze(filtered_target_bbox)
