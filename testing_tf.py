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

mean_regression_loss = np.ones((3, 5))
negative_indices = np.array([[0,0],
                            [1,1],
                            [2,2]])


mean_regression_loss = K.variable(value=mean_regression_loss)
negative_indices = K.variable(value=negative_indices)
negative_indices = K.cast(negative_indices, tf.int32)

# inplace_array = tf.gather_nd(mean_regression_loss, negative_indices)
# inplace_array = KL.Lambda(lambda x: x * 0.0)(inplace_array)
#
# bbox_loss = tf.tensor_scatter_nd_update(mean_regression_loss, negative_indices, inplace_array)
valid_anchor_mask = np.array([True, False, True, True])

valid_anchor_mask_tensor = K.variable(value=valid_anchor_mask)

valid_anchor_indices = tf.where(valid_anchor_mask)
target_bbox = np.random.rand(3, 4, 5)
target_bbox = K.variable(value=target_bbox)
import code; code.interact(local=dict(globals(), **locals()))
target_bbox = tf.gather(target_bbox, valid_anchor_indices, axis = 1)
#
# top_loss_ix = tf.nn.top_k(total_loss, 2, sorted=True, name="rpn_top_losses").values
#
# filtered_target_bbox = tf.gather(total_loss_tensor, valid_anchor_indices, axis = 1)
# filtered_target_bbox = tf.squeeze(filtered_target_bbox)
