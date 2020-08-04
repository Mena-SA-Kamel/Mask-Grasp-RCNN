import tensorflow as tf
import keras.backend as K
import keras.layers as KL
import numpy as np

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

positive_rois = np.arange(0,8).reshape((2, 4))

roi_gt_grasp_boxes = np.ones((2, 7, 5))
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
radius = ((gt_grasp_box_w/2)**2 + (gt_grasp_box_h/2)**2)**0.5
proposal_y1_reshaped = K.repeat(tf.expand_dims(proposal_y1, axis=-1), n=radius.shape[1])
proposal_x1_reshaped = K.repeat(tf.expand_dims(proposal_x1, axis=-1), n=radius.shape[1])
proposal_y2_reshaped = K.repeat(tf.expand_dims(proposal_y2, axis=-1), n=radius.shape[1])
proposal_x2_reshaped = K.repeat(tf.expand_dims(proposal_x2, axis=-1), n=radius.shape[1])

import code; code.interact(local=dict(globals(), **locals()))
valid_grasp_boxes = tf.less(gt_grasp_box_x + radius, proposal_x2_reshaped)
valid_grasp_boxes = tf.logical_and(valid_grasp_boxes, tf.greater(gt_grasp_box_x - radius, proposal_x1_reshaped))
valid_grasp_boxes = tf.logical_and(valid_grasp_boxes, tf.less(gt_grasp_box_y + radius, proposal_y2_reshaped))
valid_grasp_boxes = tf.logical_and(valid_grasp_boxes, tf.greater(gt_grasp_box_y - radius, proposal_y1_reshaped))

import code; code.interact(local=dict(globals(), **locals()))


non_zero_grasp_boxes = tf.cast(tf.reduce_sum(tf.abs(gt_grasp_boxes), axis=2), tf.bool)


#
# top_loss_ix = tf.nn.top_k(total_loss, 2, sorted=True, name="rpn_top_losses").values
#
# filtered_target_bbox = tf.gather(total_loss_tensor, valid_anchor_indices, axis = 1)
# filtered_target_bbox = tf.squeeze(filtered_target_bbox)
