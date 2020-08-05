import tensorflow as tf
import keras.backend as K
import keras.layers as KL
import numpy as np
from mrcnn import utils

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
#
# # inplace_array = tf.gather_nd(mean_regression_loss, negative_indices)
# # inplace_array = KL.Lambda(lambda x: x * 0.0)(inplace_array)
# #
# # bbox_loss = tf.tensor_scatter_nd_update(mean_regression_loss, negative_indices, inplace_array)
#
# classification_loss = np.random.rand(3, 5)
# classification_loss = K.variable(value=classification_loss)
#
#
# target_class = np.array([[1,-1,-1,-1, 1],
#                          [-1,1,-1,-1, 1],
#                          [1,-1,-1,-1, -1]])
#
# negative_class_mask = K.cast(K.equal(target_class, -1), tf.int32)
# positive_class_mask = K.cast(K.equal(target_class, 1), tf.int32)
#
# negative_indices = tf.where(K.equal(negative_class_mask, 1))
# positive_indices = tf.where(K.equal(positive_class_mask, 1))
#
# N = tf.count_nonzero(positive_class_mask, axis = 1)
# N = K.cast(N, tf.int32)
#
# values = tf.gather_nd(classification_loss, negative_indices)
#
# negative_class_losses = tf.sparse.SparseTensor(negative_indices, values, classification_loss.shape)
# negative_class_losses = tf.sparse.to_dense(negative_class_losses)
#
# top_losses_sum = K.variable(value=0)
# for i in range(3):# batch size
#     top_loss_values = tf.nn.top_k(negative_class_losses[i], N[i]*2, sorted=True, name="rpn_top_negative_losses").values
#     top_losses_sum = tf.add(top_losses_sum, K.sum(top_loss_values))
#     print(top_loss_values)
# print(top_losses_sum)
#
# total_positive_loss = K.sum(tf.gather_nd(classification_loss, positive_indices))

positive_rois = np.arange(0,8).reshape((2, 4))*10

roi_gt_grasp_boxes = np.ones((2, 7, 5)) + 5
roi_gt_grasp_boxes[0,:, 2] = np.zeros((1, 7))
roi_gt_grasp_boxes[0, 1, :] = np.zeros((1, 5))
roi_gt_grasp_boxes[1, :, 3] = np.zeros((1, 7))
roi_gt_grasp_boxes[1, 3, :] = np.zeros((1, 5))

dx_s = np.arange(1,3).reshape((2, 1))
dy_s = np.arange(7,9).reshape((2, 1))

roi_gt_grasp_boxes = K.variable(value=roi_gt_grasp_boxes)
dx_s = K.variable(value=dx_s)
dy_s = K.variable(value=dy_s)
positive_rois = K.variable(value=positive_rois)

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

radius = ((gt_grasp_box_w/2)**2 + (gt_grasp_box_h/2)**2)**0.5
proposal_y1_reshaped = K.repeat(tf.expand_dims(proposal_y1, axis=-1), n=radius.shape[1])
proposal_x1_reshaped = K.repeat(tf.expand_dims(proposal_x1, axis=-1), n=radius.shape[1])
proposal_y2_reshaped = K.repeat(tf.expand_dims(proposal_y2, axis=-1), n=radius.shape[1])
proposal_x2_reshaped = K.repeat(tf.expand_dims(proposal_x2, axis=-1), n=radius.shape[1])

valid_grasp_boxes = tf.less(gt_grasp_box_x + radius, proposal_x2_reshaped)
valid_grasp_boxes = tf.logical_and(valid_grasp_boxes, tf.greater(gt_grasp_box_x - radius, proposal_x1_reshaped))
valid_grasp_boxes = tf.logical_and(valid_grasp_boxes, tf.less(gt_grasp_box_y + radius, proposal_y2_reshaped))
valid_grasp_boxes = tf.logical_and(valid_grasp_boxes, tf.greater(gt_grasp_box_y - radius, proposal_y1_reshaped))

sample_arr = [True, False]
valid_grasp_boxes = np.random.choice(sample_arr, size=14).reshape((2,7,1))
valid_grasp_boxes = K.variable(value=valid_grasp_boxes)

final_roi_gt_grasp_boxes = referenced_grasp_boxes * tf.cast(valid_grasp_boxes, dtype='float32')

GRASP_POOL_SIZE = 7
GRASP_ANCHOR_RATIOS = [1]
GRASP_ANCHOR_ANGLES = [-67.5, -22.5, 22.5, 67.5]
GRASP_ANCHOR_SIZE = [48]


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

    anchor_heights, anchor_widths = tf.split(K.repeat(box_sizes, n=GRASP_ANCHORS_PER_ROI), num_or_size_splits=2, axis=-1)
    anchor_x, anchor_y, anchor_thetas = tf.split(boxes, num_or_size_splits=3, axis=-1)
    anchors = tf.concat([anchor_x, anchor_y, anchor_widths, anchor_heights, anchor_thetas], axis=-1)
    return anchors

pooled_feature_stride = tf.concat([proposal_heights, proposal_widths], axis = -1) /GRASP_POOL_SIZE
pooled_feature_stride = tf.cast(pooled_feature_stride, tf.float32)



final_anchors = tf.map_fn(generate_grasping_anchors_graph,
          tf.concat([pooled_feature_stride, positive_rois], axis=1))
import code;

code.interact(local=dict(globals(), **locals()))



#
# top_loss_ix = tf.nn.top_k(total_loss, 2, sorted=True, name="rpn_top_losses").values
#
# filtered_target_bbox = tf.gather(total_loss_tensor, valid_anchor_indices, axis = 1)
# filtered_target_bbox = tf.squeeze(filtered_target_bbox)
