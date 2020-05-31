import mrcnn
from mrcnn.config import Config
from mrcnn.utils import Dataset
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.utils as utils
import mrcnn.model as modellib
from mrcnn.model import log

import random
import numpy as np
import os
import glob
from shutil import copyfile
import matplotlib.pyplot as plt
import skimage.io
from PIL import Image
import open3d as o3d
import math
import matplotlib.patches as patches
import cv2

# Extending the mrcnn.config class to update the default variables
class GraspingPointsConfig(Config):
    NAME = 'grasping_points'
    BACKBONE = "resnet50"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1 # Object and background classes
    IMAGE_MIN_DIM = 288
    IMAGE_MAX_DIM = 384
    # IMAGE_MIN_DIM = 480
    # IMAGE_MAX_DIM = 640
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128) # To modify based on image size
    TRAIN_ROIS_PER_IMAGE = 200 # Default (Paper uses 512)
    RPN_NMS_THRESHOLD = 0.7 # Default Value. Update to increase the number of proposals out of the RPN
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 20
    IMAGE_CHANNEL_COUNT = 4 # For RGB-D images of 4 channels
    # MEAN_PIXEL = np.array([134.6, 125.7, 119.0, 147.6]) # Added a 4th channel. Modify the mean of the pixel depth
    MEAN_PIXEL = np.array([112.7, 112.1, 113.5, 123.5]) # Added a 4th channel. Modify the mean of the pixel depth
    MAX_GT_INSTANCES = 50
    RPN_GRASP_ANGLES = [-60, -30, 0, 30, 60]
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2, 1])

class InferenceConfig(GraspingPointsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class GraspingPointsDataset(Dataset):

    def load_dataset(self, type = 'train_set', dataset_dir = 'cornell_grasping_dataset'):
        self.add_class(GraspingPointsConfig().NAME, 1, "graspable_location")
        dataset_path = os.path.join(dataset_dir, type)
        image_list = glob.glob(dataset_path + '/**/rgb/*.png', recursive=True)
        random.shuffle(image_list)

        id = 0
        for image in image_list:
            rgb_path = image
            depth_path = image.replace('rgb', 'depth')
            positive_grasp_points = image.replace('r.png', 'cpos.txt').replace('rgb', 'grasp_rectangles')
            negative_grasp_points = image.replace('r.png', 'cneg.txt').replace('rgb', 'grasp_rectangles')
            self.add_image("grasping_points", image_id = id, path = rgb_path,
                           depth_path = depth_path, positive_points = positive_grasp_points,
                           negative_points = negative_grasp_points)
            id = id + 1

    def load_image(self, image_id):
        image = skimage.io.imread(self.image_info[image_id]['path'])
        depth = skimage.io.imread(self.image_info[image_id]['depth_path'])
        if depth.dtype.type == np.uint16:
            depth = self.rescale_depth_image(depth)
        rgbd_image = np.zeros([image.shape[0], image.shape[1], 4])
        rgbd_image[:, :, 0:3] = image
        rgbd_image[:, :, 3] = depth
        rgbd_image = np.array(rgbd_image).astype('uint8')
        return rgbd_image

    def rescale_depth_image(self, depth_image):
        # Rescales the depth images to occupy the range 0 - 255
        # input: numpy array of depth image
        # output: returns a 0-255 integer depth numpy array
        original_range = np.max(depth_image) - np.min(depth_image)
        scaling_factor = 255 / original_range
        img_rescale = depth_image * scaling_factor
        img_rescale_int = img_rescale.astype('uint8')
        return img_rescale_int

    def display_image(self, image):
        # Displays the image
        # input: 2D numpy array
        plt.imshow(image)
        plt.show()

    def split_dataset(self, dataset_paths, train, val):
        # Shuffles and splits a list (dataset_paths) into training, validation and testing sets based on the ratio
        # train:val:test(1 - (train + val))
        # input:
        # dataset_paths - list of image paths in the dataset
        # train - fraction to be dedicated to training set (0-1)
        # val - fraction to be dedicated to validation set (0-1)
        # output:
        # [train_set, val_set, test_set]
        random.shuffle(dataset_paths)
        random.shuffle(dataset_paths)
        num = len(dataset_paths)
        train_set = dataset_paths[0: int(train * num)]
        val_set = dataset_paths[int(train * num) : int((train + val) * num)]
        test_set = dataset_paths[int((train + val) * num) : ]
        return train_set, val_set, test_set

    def convert_to_depth_image(self, pcd_path):
        point_cloud = o3d.io.read_point_cloud(pcd_path, format='pcd')
        xyz_load = np.asarray(point_cloud.points)
        depth_values = xyz_load[:, 2]
        depth_values_reshaped = np.interp(depth_values, (depth_values.min(), depth_values.max()), (0, 1))

        with open(pcd_path) as f:
            points = f.readlines()

        img = np.zeros([480, 640], dtype=np.uint8)
        i = 0
        for point in points[10:]:
            index = int(point.split(' ')[-1].replace('\n', ''))
            row = int(np.floor(index / 640))
            col = int(index % 640 + 1)
            img[row, col] = depth_values_reshaped[i] * 255
            i += 1
        return img

    def construct_dataset(self, train = 0.70, val = 0.15, dataset_dir = '../../../Datasets/Cornell-Dataset/Raw-Dataset'):
        dataset_folder = 'cornell_grasping_dataset'
        folder_names = ['train_set', 'val_set', 'test_set']
        subfolder_names = ['depth', 'rgb', 'grasp_rectangles']
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
            for folder_name in folder_names:
                os.makedirs(os.path.join(dataset_folder, folder_name))
                for subfolder in subfolder_names:
                    os.makedirs(os.path.join(dataset_folder, folder_name, subfolder))

        image_paths = glob.glob(dataset_dir + '/**/*.png', recursive=True)

        train_set, val_set, test_set = self.split_dataset(image_paths, train, val)

        i = 0
        for dataset_paths in [train_set, val_set, test_set]:
            for image_path in dataset_paths:
                source_rgb_path = image_path
                source_depth_path = image_path.replace('r.png', '.txt')
                source_positive_rect_path = image_path.replace('r.png', 'cpos.txt')
                source_negative_rect_path = image_path.replace('r.png', 'cneg.txt')

                target_rgb_path = os.path.join(dataset_folder, folder_names[i],'rgb', source_rgb_path.split('\\')[-1])
                target_depth_path = os.path.join(dataset_folder, folder_names[i],'depth', source_rgb_path.split('\\')[-1])
                target_pos_rect_path = os.path.join(dataset_folder, folder_names[i], 'grasp_rectangles', source_positive_rect_path.split('\\')[-1])
                target_neg_rect_path = os.path.join(dataset_folder, folder_names[i], 'grasp_rectangles', source_negative_rect_path.split('\\')[-1])

                depth_image = Image.fromarray(self.convert_to_depth_image(source_depth_path))
                depth_image.save(target_depth_path)

                copyfile(source_rgb_path, target_rgb_path)
                copyfile(source_positive_rect_path, target_pos_rect_path)
                copyfile(source_negative_rect_path, target_neg_rect_path)
            i = i + 1

    def load_ground_truth_bbox_files(self, image_id):
        positive_rectangles_path = self.image_info[image_id]['positive_points']
        negative_rectangles_path = self.image_info[image_id]['negative_points']

        with open(positive_rectangles_path) as f:
            positive_rectangles = f.readlines()
        with open(negative_rectangles_path) as f:
            negative_rectangles = f.readlines()

        bounding_boxes_formatted = []
        vertices = []
        class_ids = []

        class_id = 0
        for bounding_box_class in [negative_rectangles, positive_rectangles]:
            i = 0
            for bounding_box in bounding_box_class:
                try:
                    x = int(float(bounding_box.split(' ')[0]))
                    y = int(float(bounding_box.split(' ')[1]))
                except:
                    print("ERROR : ValueError: cannot convert float NaN to integer")
                    import code;
                    code.interact(local=dict(globals(), **locals()))
                vertices.append([x, y])
                i += 1
                if i % 4 == 0:
                    bounding_boxes_formatted.append(vertices)
                    class_ids.append(class_id)
                    vertices = []
            class_id += 1
        return np.array(bounding_boxes_formatted), class_ids

    def get_five_dimensional_box(self, coordinates):
        x_coordinates = coordinates[:, 0]
        y_coordinates = coordinates[:, 1]
        gripper_orientation = coordinates[:2, :]

        # x = int((np.max(x_coordinates) - np.min(x_coordinates)) / 2) + np.min(x_coordinates)
        # y = int((np.max(y_coordinates) - np.min(y_coordinates)) / 2) + np.min(y_coordinates)
        x = ((np.max(x_coordinates) - np.min(x_coordinates)) / 2) + np.min(x_coordinates)
        y = ((np.max(y_coordinates) - np.min(y_coordinates)) / 2) + np.min(y_coordinates)

        # Gripper opening distance
        w = np.abs(np.linalg.norm(gripper_orientation[0] - gripper_orientation[1]))

        # Gripper width - Cross product betwee
        # h = np.cross(p2-p1, p1-p3)/norm(p2-p1)
        p1 = gripper_orientation[0]
        p2 = gripper_orientation[1]
        p3 = coordinates[2, :]
        h = np.abs(np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1))

        deltaY = gripper_orientation[0][1] - gripper_orientation[1][1]
        deltaX = gripper_orientation[0][0] - gripper_orientation[1][0]
        theta = -1 * np.arctan2(deltaY, deltaX) * 180 / math.pi
        if theta < -90:
            theta = theta + 180
        return [x, y, w, h, theta]

    def visualize_bbox(self, image_id, bounding_box, class_id, bbox_5_dimensional, rgbd_image = []):
        x, y, w, h, theta = bbox_5_dimensional
        if rgbd_image == []:
            rgbd_image = self.load_image(image_id)
        fig2, ax2 = plt.subplots(1)
        ax2.imshow(rgbd_image)
        ax2.set_title('Visualizing rectangles')
        colors = ['r', 'g']
        rect = patches.Polygon(np.array(bounding_box), linewidth=1, edgecolor=colors[class_id], facecolor='none')
        ax2.plot(bounding_box[2:, 0], bounding_box[2:, 1], linewidth=1, color='b')
        ax2.scatter([x], [y], linewidth=1.25, color='k')
        title = ('x = %d, y = %d, w = %d, h = %d, theta = %d' % (x, y, w, h, theta))
        ax2.add_patch(rect)
        ax2.set_title(title)
        plt.show(block=False)

    def bbox_convert_to_four_vertices(self, bbox_5_dimension):
        x, y, w, h, theta = bbox_5_dimension
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

        return original_points_rotated


    def bbox_convert_to_five_dimension(self, bounding_box_vertices, image_id = 0):
        bbox_5_dimensional = []
        for bounding_box in bounding_box_vertices:
            x, y, w, h, theta = self.get_five_dimensional_box(bounding_box)
            bbox_5_dimensional.append([x, y, w, h, theta])
            # self.visualize_bbox(image_id, bounding_box, 1, [x, y, w, h, theta])
        bbox_5_dimensional = np.array(bbox_5_dimensional)
        return bbox_5_dimensional


    def load_bounding_boxes(self, image_id):
        # bounding boxes here have a shape of N x 4 x 2, consisting of four vertices per rectangle given N rectangles
        bounding_box_vertices, class_ids = self.load_ground_truth_bbox_files(image_id)
        bbox_5_dimensional = self.bbox_convert_to_five_dimension(bounding_box_vertices)

        class_ids = np.array(class_ids)
        bounding_box_vertices = np.array(bounding_box_vertices)
        p = np.random.permutation(len(class_ids))
        return bounding_box_vertices[p], bbox_5_dimensional[p], class_ids[p].astype('uint8')


# SETUP ##
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Directory to save logs and trained model
MODEL_DIR = "models"
MASKRCNN_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_object_vs_background_0020.h5")
config = InferenceConfig()
config.display()
DEVICE = "/gpu:0"
TEST_MODE = "inference"
mode = "grasping_points"

training_dataset = GraspingPointsDataset()
# training_dataset.construct_dataset()
training_dataset.load_dataset()
training_dataset.prepare()

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config, task="grasping_points")
    import code;

    code.interact(local=dict(globals(), **locals()))
weights_path = MASKRCNN_MODEL_PATH

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

image_ids = random.choices(training_dataset.image_ids, k=1)
for image_id in image_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(training_dataset, config, image_id, use_mini_mask=False, mode='grasping_points')
    for i in list(range(len(gt_class_id))):
        bounding_box = training_dataset.bbox_convert_to_four_vertices(gt_bbox[i])
        # training_dataset.visualize_bbox(image_id, bounding_box[0], gt_class_id[i], gt_bbox[i], rgbd_image=image)

normalized_anchors = model.get_anchors(config.IMAGE_SHAPE, mode='grasping_points', angles=config.RPN_GRASP_ANGLES)

# Generate Anchors
mode= 'grasping_points'
backbone_shapes = modellib.compute_backbone_shapes(config, config.IMAGE_SHAPE)
anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                          config.RPN_ANCHOR_RATIOS,
                                          backbone_shapes,
                                          config.BACKBONE_STRIDES,
                                          config.RPN_ANCHOR_STRIDE,
                                          mode,
                                          config.RPN_GRASP_ANGLES)

target_rpn_match, target_rpn_bbox, anchor_data = modellib.build_rpn_targets(
    image.shape, model.anchors, gt_class_id, gt_bbox, model.config, mode)

log("target_rpn_match", target_rpn_match)
log("target_rpn_bbox", target_rpn_bbox)

positive_anchor_ix = np.where(target_rpn_match[:] == 1)[0]
negative_anchor_ix = np.where(target_rpn_match[:] == -1)[0]
neutral_anchor_ix = np.where(target_rpn_match[:] == 0)[0]
positive_anchors = model.anchors[positive_anchor_ix]
negative_anchors = model.anchors[negative_anchor_ix]
neutral_anchors = model.anchors[neutral_anchor_ix]
log("positive_anchors", positive_anchors)
log("negative_anchors", negative_anchors)
log("neutral anchors", neutral_anchors)

# Apply refinement deltas to positive anchors
positive_anchor_indices = anchor_data[anchor_data[:,1] == 1][:,0]
positive_anchors = anchors[positive_anchor_indices]

gt_boxes = gt_bbox[gt_class_id == 1]
deltas = target_rpn_bbox[:positive_anchors.shape[0]]* np.append(model.config.RPN_BBOX_STD_DEV, [1])
refined_anchors = utils.apply_box_deltas(positive_anchors, deltas, mode)

# Display positive anchors before refinement (dotted) and
# after refinement (solid).
visualize.draw_boxes(image, boxes=positive_anchors, refined_boxes=refined_anchors, mode=mode)








####################################### VISUALIZING ANCHORS ############################################################
# num_levels = len(backbone_shapes)
# anchors_per_cell = len(config.RPN_ANCHOR_RATIOS) * len(config.RPN_GRASP_ANGLES)
# anchors_per_level = []
# for l in range(num_levels):
#     num_cells = backbone_shapes[l][0] * backbone_shapes[l][1]
#     anchors_per_level.append(anchors_per_cell * num_cells // config.RPN_ANCHOR_STRIDE**2)
#     print("Anchors in Level {}: {}".format(l, anchors_per_level[l]))
#
# ## Visualize anchors of one cell at the center of the feature map of a specific level
# # Load and draw random image
# image_id = np.random.choice(training_dataset.image_ids, 1)[0]
# image, image_meta, _, _, _ = modellib.load_image_gt(training_dataset, config, image_id)
# levels = len(backbone_shapes)
#
# for level in range(levels):
#     colors = visualize.random_colors(levels)
#     level_start = sum(anchors_per_level[:level]) # sum of anchors of previous levels
#     level_anchors = anchors[level_start:level_start+anchors_per_level[level]]
#     level_indexes = np.unique(level_anchors[:, 0])
#     num_indexes = len(level_indexes)
#     center_index = int(level_indexes[int(num_indexes/2)])
#     anchors_to_show = level_anchors[(level_anchors[:, 0] == [center_index]) & (level_anchors[:, 1] == [center_index])]
#     fig, ax = plt.subplots(1, figsize=(10, 10))
#     ax.imshow(image)
#     for i, rect in enumerate(anchors_to_show):
#         rect = training_dataset.bbox_convert_to_four_vertices(rect)
#         p = patches.Polygon(rect[0], linewidth=1,edgecolor='r',facecolor='none')
#         ax.add_patch(p)
#     plt.savefig(os.path.join('Grasping_anchors','P'+str(level+2)+ 'center_anchors.png'))
# plt.show(block=False)
