import tensorflow as tf
import keras.backend as K
import numpy as np

tf.enable_eager_execution()
loss = np.arange(60)
loss = loss.reshape((1, 12, 5))
bbox_loss = K.variable(value=loss)
bbox_loss = K.mean(bbox_loss, axis=1, keepdims=True)
# tf.gather_nd(bbox_loss, negative_indices)

test = np.arange(60)
test = test.reshape((60, 1))
test = K.variable(value=test)
import code; code.interact(local=dict(globals(), **locals()))
test = K.sum(test) / 256.0

target_bbox = np.zeros(200).reshape((1, 40, 5))
target_bbox = K.variable(value=target_bbox)