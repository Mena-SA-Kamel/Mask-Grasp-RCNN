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
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import skimage.io
from PIL import Image
import open3d as o3d

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
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256) # To modify based on image size
    TRAIN_ROIS_PER_IMAGE = 200 # Default (Paper uses 512)
    RPN_NMS_THRESHOLD = 0.7 # Default Value. Update to increase the number of proposals out of the RPN
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 20
    IMAGE_CHANNEL_COUNT = 4 # For RGB-D images of 4 channels
    # MEAN_PIXEL = np.array([134.6, 125.7, 119.0, 147.6]) # Added a 4th channel. Modify the mean of the pixel depth
    MEAN_PIXEL = np.array([112.7, 112.1, 113.5, 123.5]) # Added a 4th channel. Modify the mean of the pixel depth
    MAX_GT_INSTANCES = 50

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
            self.add_image("object_vs_background", image_id = id, path = rgb_path,
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

# SETUP ##
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

training_dataset = GraspingPointsDataset()
# training_dataset.construct_dataset()
training_dataset.load_dataset()