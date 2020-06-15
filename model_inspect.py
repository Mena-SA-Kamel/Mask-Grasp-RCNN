import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from object_vs_background import InferenceConfig
from object_vs_background import ObjectVsBackgroundDataset

# Directory to save logs and trained model
MODEL_DIR = "models"
MASKRCNN_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_object_vs_background_0020.h5")
config = InferenceConfig()
config.display()
DEVICE = "/gpu:0"
TEST_MODE = "inference"

# Loading validation set
validating_dataset = ObjectVsBackgroundDataset()
validating_dataset.load_dataset('val_set', dataset_dir='wisdom_dataset')
validating_dataset.prepare()

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR,
                              config=config)

weights_path = MASKRCNN_MODEL_PATH

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

image_ids = random.choices(validating_dataset.image_ids, k=1)
for image_id in image_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(validating_dataset, config, image_id, use_mini_mask=False)

# Validating RPN
# Generate RPN trainig targets
# target_rpn_match is 1 for positive anchors, -1 for negative anchors
# and 0 for neutral anchors.
backbone_shapes = modellib.compute_backbone_shapes(config, config.IMAGE_SHAPE)
anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                          config.RPN_ANCHOR_RATIOS,
                                          backbone_shapes,
                                          config.BACKBONE_STRIDES,
                                          config.RPN_ANCHOR_STRIDE,
                                          config.IMAGE_SHAPE)
target_rpn_match, target_rpn_bbox = modellib.build_rpn_targets(
    image.shape, anchors, gt_class_id, gt_bbox, model.config)
log("target_rpn_match", target_rpn_match)
log("target_rpn_bbox", target_rpn_bbox)

positive_anchor_ix = np.where(target_rpn_match[:] == 1)[0]
negative_anchor_ix = np.where(target_rpn_match[:] == -1)[0]
neutral_anchor_ix = np.where(target_rpn_match[:] == 0)[0]
positive_anchors = anchors[positive_anchor_ix]
negative_anchors = anchors[negative_anchor_ix]
neutral_anchors = anchors[neutral_anchor_ix]
log("positive_anchors", positive_anchors)
log("negative_anchors", negative_anchors)
log("neutral anchors", neutral_anchors)

# Apply refinement deltas to positive anchors
refined_anchors = utils.apply_box_deltas(
    positive_anchors,
    target_rpn_bbox[:positive_anchors.shape[0]] * model.config.RPN_BBOX_STD_DEV)
log("refined_anchors", refined_anchors, )

# Display positive anchors before refinement (dotted) and
# after refinement (solid).
visualize.draw_boxes(image, boxes=positive_anchors, refined_boxes=refined_anchors)

# Run RPN sub-graph
pillar = model.keras_model.get_layer("ROI").output  # node to start searching from

# TF 1.4 and 1.9 introduce new versions of NMS. Search for all names to support TF 1.3~1.10
nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression:0")
if nms_node is None:
    nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV2:0")
if nms_node is None: #TF 1.9-1.10
    nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV3:0")

rpn = model.run_graph([image], [
    ("rpn_class", model.keras_model.get_layer("rpn_class").output),
    ("pre_nms_anchors", model.ancestor(pillar, "ROI/pre_nms_anchors:0")),
    ("refined_anchors", model.ancestor(pillar, "ROI/refined_anchors:0")),
    ("refined_anchors_clipped", model.ancestor(pillar, "ROI/refined_anchors_clipped:0")),
    ("post_nms_anchor_ix", nms_node),
    ("proposals", model.keras_model.get_layer("ROI").output),
])

# Show top anchors by score (before refinement)
limit = 100
sorted_anchor_ids = np.argsort(rpn['rpn_class'][:,:,1].flatten())[::-1]
# visualize.draw_boxes(image[:,:,0:3], boxes=model.anchors[sorted_anchor_ids[:limit]])


# Show top anchors with refinement. Then with clipping to image boundaries
limit = 50
pre_nms_anchors = utils.denorm_boxes(rpn["pre_nms_anchors"][0], image.shape[:2])
refined_anchors = utils.denorm_boxes(rpn["refined_anchors"][0], image.shape[:2])
refined_anchors_clipped = utils.denorm_boxes(rpn["refined_anchors_clipped"][0], image.shape[:2])
# visualize.draw_boxes(image[:,:,0:3], boxes=pre_nms_anchors[:limit],
#                      refined_boxes=refined_anchors[:limit])

# Show refined anchors after non-max suppression
limit = 50
ixs = rpn["post_nms_anchor_ix"][:limit]
visualize.draw_boxes(image, refined_boxes=refined_anchors_clipped[ixs])

# Show final proposals
# These are the same as the previous step (refined anchors
# after NMS) but with coordinates normalized to [0, 1] range.
limit = 50
# Convert back to image coordinates for display
h, w = config.IMAGE_SHAPE[:2]
proposals = rpn['proposals'][0, :limit] * np.array([h, w, h, w])
# visualize.draw_boxes(image[:,:,0:3], refined_boxes=proposals)
#
# # Get input and output to classifier and mask heads.
# mrcnn = model.run_graph([image], [
#     ("proposals", model.keras_model.get_layer("ROI").output),
#     ("probs", model.keras_model.get_layer("mrcnn_class").output),
#     ("deltas", model.keras_model.get_layer("mrcnn_bbox").output),
#     ("masks", model.keras_model.get_layer("mrcnn_mask").output),
#     ("detections", model.keras_model.get_layer("mrcnn_detection").output),
# ])
#
# # Get detection class IDs. Trim zero padding.
# det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
# det_count = np.where(det_class_ids == 0)[0][0]
# det_class_ids = det_class_ids[:det_count]
# detections = mrcnn['detections'][0, :det_count]
#
# print("{} detections: {}".format(
#     det_count, np.array(validating_dataset.class_names)[det_class_ids]))
#
# captions = ["{} {:.3f}".format(validating_dataset.class_names[int(c)], s) if c > 0 else ""
#             for c, s in zip(detections[:, 4], detections[:, 5])]
# visualize.draw_boxes(
#     image,
#     refined_boxes=utils.denorm_boxes(detections[:, :4], image.shape[:2]),
#     visibilities=[2] * len(detections),
#     captions=captions, title="Detections")


# Get activations of a few sample layers
# import code;
#
# code.interact(local=dict(globals(), **locals()))
layer_names=[layer.name for layer in model.keras_model.layers]



activations = model.run_graph([image], [
    ("input_image",        tf.identity(model.keras_model.get_layer("input_image").output)),
    ("res2b_branch2b",          model.keras_model.get_layer("res2b_branch2b").output),
    ("res3c_branch2b",          model.keras_model.get_layer("res3c_branch2b").output),
    ("bn4a_branch2a",          model.keras_model.get_layer("bn4a_branch2a").output),
    ("res5b_branch2c",          model.keras_model.get_layer("res5b_branch2c").output),
    ("rpn_bbox",           model.keras_model.get_layer("rpn_bbox").output),
    ("roi",                model.keras_model.get_layer("ROI").output),
])

# Input image (normalized)
_ = plt.imshow(modellib.unmold_image(activations["input_image"][0],config))
plt.show()


layer_activations = activations["res2b_branch2b"]
layer_shape = layer_activations.shape
l = layer_shape[1]
depth = layer_shape[-1]
save_dir = os.path.join('logs', 'res2b_branch2b')
os.makedirs(save_dir)
plt.imshow(image[:,:,0:3].astype('uint8'))
plt.savefig(os.path.join(save_dir, 'image'))
for i in list(range(depth)):
    plt.imshow(layer_activations[0,:,:,i].reshape(l,l), cmap='jet')
    plt.savefig(os.path.join(save_dir, str(i)))