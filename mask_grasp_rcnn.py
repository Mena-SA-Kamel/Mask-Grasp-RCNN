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
import matplotlib.patches as patches
import matplotlib as mpl
import skimage.io
from PIL import Image
import image_augmentation
import cv2
import math
from math import pi
from random import randrange
from PIL import Image, ImageEnhance
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon


# from grasping_points import GraspingInferenceConfig, GraspingPointsDataset

# Extending the mrcnn.config class to update the default variables
class GraspMaskRCNNConfig(Config):
    NAME = 'grasp_and_mask'
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
    # MEAN_PIXEL = np.array([112.7, 112.1, 113.5, 123.5]) # Added a 4th channel. Modify the mean of the pixel depth
    # MEAN_PIXEL = np.array([122.6, 113.4 , 118.2, 135.8]) # SAMS dataset
    # MEAN_PIXEL = np.array([181.6, 180.0, 180.9, 224.3]) # Jacquard_dataset dataset
    # MEAN_PIXEL = np.array([212.3, 204.8, 205.8, 146.5]) # multi_grasp dataset
    MEAN_PIXEL = np.array([199.2, 184.1, 185.8, 116.9]) # cornell dataset

    GRASP_MEAN_PIXEL = np.array([181.6, 180.0, 224.3]) # Jacquard_dataset dataset
    GRASP_ANCHOR_RATIOS = [1]
    GRASP_ANCHOR_ANGLES = [-67.5, -22.5, 22.5, 67.5]
    GRASP_ANCHOR_RATIOS = [1]  # To modify based on image size
    GRASP_POOL_SIZE = 7
    GRASP_ANCHOR_SIZE = [48]
    GRASP_ANCHORS_PER_ROI = GRASP_POOL_SIZE * GRASP_POOL_SIZE * len(GRASP_ANCHOR_RATIOS) * len(GRASP_ANCHOR_ANGLES) * len(GRASP_ANCHOR_SIZE)
    GRASP_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2, 1])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    GRASP_ANCHOR_STRIDE = 1
    GRASP_ROI_EXPAND_FACTOR = 0.4
    USE_EXPANDED_ROIS = False
    MAX_GT_INSTANCES = 50
    TRAIN_ROIS_PER_IMAGE = 200
    NUM_GRASP_BOXES_PER_INSTANCE = 128
    ARIOU_NEG_THRESHOLD = 0.01
    ARIOU_POS_THRESHOLD = 0.1
    # LEARNING_RATE = 0.00002
    LEARNING_RATE = 0.0001 # was 0.00001
    # WEIGHT_DECAY = 0.000003
    WEIGHT_DECAY = 0.00001
    ROI_BOX_SENSITIVITY = 0 # 0 means just checks if the gt_grasp_box center <x,y> is in the RoI

    GRADIENT_CLIP_NORM = 5.0
    LEARNING_MOMENTUM = 0.9
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.,
        "grasp_loss": 1.,
    }
    GRASP_LOSS_BETA = 2
    GRADIENT_CLIP_VALUE = 0.5 # was 0.1
    CYCLIC_LR = False
    MODEL_SAVE_PERIOD = 8
    TRAIN_BN = False
    TRAIN_GRASP_BN = False
    NUM_AUGMENTATIONS = 5
    ANGLE_BETA_FACTOR = 10
    GRASP_MIN_CONFIDENCE = 0.9

class GraspMaskRCNNInferenceConfig(GraspMaskRCNNConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    TRAIN_GRASP_BN = False
    GRASP_MIN_CONFIDENCE = 0.9

class GraspMaskRCNNDataset(Dataset):

    def load_dataset(self, type = 'train_set', dataset_dir = 'New Graspable Objects Dataset', augmentation=False):
        self.add_class(GraspMaskRCNNConfig().NAME, 1, "object")
        dataset_path = os.path.join(dataset_dir, type)
        image_list = glob.glob(dataset_path + '/**/rgb/*.png', recursive=True)
        if image_list == []:
            image_list = glob.glob(dataset_path + '/**/rgb/*.jpg', recursive=True)
        random.shuffle(image_list)
        id = 0

        for image in image_list:
            if 'jacquard_dataset' in dataset_dir:
                rgb_path = image
                depth_path = image.replace('rgb', 'depth')
                positive_grasp_points = image.replace('_RGB.png', '_grasps.txt').replace('rgb', 'grasp_rectangles_new')
                label_path = image.replace('_RGB.png', '_mask.png').replace('rgb', 'mask')
                self.add_image("grasp_and_mask", image_id=id, path=rgb_path,
                                depth_path=depth_path, label_path=label_path, positive_points=positive_grasp_points, augmentation = [])
                ## To do: Enable augmentation
                if augmentation:
                    # Augmentation will create hypothetical paths that load_image_gt() uses to construct the different
                    # image variants
                    # image_name = image.split('\\')[-1].strip('_RGB.png')
                    for i in range(GraspMaskRCNNConfig().NUM_AUGMENTATIONS):
                        self.add_image("grasp_and_mask", image_id=id, path=rgb_path,
                                       depth_path=depth_path, label_path=label_path, positive_points=positive_grasp_points,
                                       augmentation = self.generate_augmentations())
                        id = id + 1
            elif 'multi_grasp' in dataset_dir:
                rgb_path = image
                depth_path = image.replace('rgb', 'depth').replace('table_', '').replace('floor_', '').replace('.jpg', '.png')
                label_path = image.replace('rgb', 'label').replace('table_', '').replace('floor_', '')
                positive_grasp_points = image.replace('rgb', 'grasp_rectangles').replace('.jpg', '_annotations.txt').replace('grasp_rectangles_', 'rgb_')
                self.add_image("grasp_and_mask", image_id=id, path=rgb_path,
                               depth_path=depth_path, label_path=label_path, positive_points=positive_grasp_points,
                               augmentation=[])
                ## To do: Enable augmentation
                if augmentation:
                    # Augmentation will create hypothetical paths that load_image_gt() uses to construct the different
                    # image variants
                    # image_name = image.split('\\')[-1].strip('_RGB.png')
                    for i in range(GraspMaskRCNNConfig().NUM_AUGMENTATIONS):
                        self.add_image("grasp_and_mask", image_id=id, path=rgb_path,
                                       depth_path=depth_path, label_path=label_path,
                                       positive_points=positive_grasp_points,
                                       augmentation=self.generate_augmentations())
                        id = id + 1
            elif 'cornell' in dataset_dir:
                rgb_path = image
                depth_path = image.replace('rgb', 'depth').replace('table_', '').replace('floor_', '')
                label_path = image.replace('rgb', 'label').replace('table_', '').replace('floor_', '')
                positive_grasp_points = image.replace('r.png', 'cpos.txt').replace('rgb', 'grasp_rectangles')
                self.add_image("grasp_and_mask", image_id=id, path=rgb_path,
                               depth_path=depth_path, label_path=label_path, positive_points=positive_grasp_points,
                               augmentation=[])
                ## To do: Enable augmentation
                if augmentation:
                    # Augmentation will create hypothetical paths that load_image_gt() uses to construct the different
                    # image variants
                    # image_name = image.split('\\')[-1].strip('_RGB.png')
                    for i in range(GraspMaskRCNNConfig().NUM_AUGMENTATIONS):
                        self.add_image("grasp_and_mask", image_id=id, path=rgb_path,
                                       depth_path=depth_path, label_path=label_path,
                                       positive_points=positive_grasp_points,
                                       augmentation=self.generate_augmentations())
                        id = id + 1
            else:
                rgb_path = image
                depth_path = image.replace('rgb', 'depth').replace('table_', '').replace('floor_', '')
                label_path = image.replace('rgb', 'label').replace('table_', '').replace('floor_', '')
                self.add_image("grasp_and_mask", image_id = id, path = rgb_path,
                               label_path = label_path, depth_path = depth_path)
            id = id + 1

    def generate_augmentations(self):
        augmentation_types = ['angle', 'dx', 'dy', 'flip', 'contrast', 'noise']
        # augmentation_types = ['dx', 'dy', 'contrast', 'noise',  'flip']
        augmentations = random.sample(list(augmentation_types), 3)
        augmentations_list = np.zeros(6)
        augmentations_list[3] = 2

        if 'angle' in augmentations:
            angle = randrange(0, 30)
            augmentations_list[0] = angle

        if 'dx' in augmentations:
            dx = randrange(-50, 50)
            augmentations_list[1] = dx

        if 'dy' in augmentations:
            dy = randrange(0, 50)
            augmentations_list[2] = dy

        if 'flip' in augmentations:
            flip = randrange(0, 3) # 0 -> vertical flip, 1 -> horizontal flip, 2 -> no flip
            augmentations_list[3] = flip

        if 'contrast' in augmentations:
            contrast = random.uniform(0.5, 2)
            augmentations_list[4] = contrast

        if 'noise' in augmentations: # Gaussian noise std
            noise = randrange(5, 15)
            augmentations_list[5] = noise

        return augmentations_list

    def load_mask(self, image_id, augmentation=[]):
        mask_path = self.image_info[image_id]['label_path']
        mask_processed = self.process_label_image(mask_path)
        mask, class_ids = self.reshape_label_image(mask_processed)
        if len(augmentation) != 0:
            for i in range(mask.shape[-1]):
                mask[:,:, i] = self.apply_augmentation_to_mask(mask[:,:, i], augmentation)
                mask[np.where(mask[:,:, i] != 0)] = 1
        return mask, np.array(class_ids)

    def load_image(self, image_id, augmentation=[], image_type='rgd'):
        image_path = self.image_info[image_id]['path']
        image = skimage.io.imread(image_path)
        try:
            depth = skimage.io.imread(self.image_info[image_id]['depth_path'])
        except:
            print("ERROR : SyntaxError: not an FLI/FLC file")
            import code;
            code.interact(local=dict(globals(), **locals()))

        depth = np.interp(depth, (depth.min(), depth.max()), (0, 1)) * 255
        # if depth.dtype.type == np.uint16:
        #     depth = self.rescale_depth_image(depth)
        # rgbd_image = np.zeros([image.shape[0], image.shape[1], 4])
        # RG_D image based on literature
        if image_type == 'rgd':
            rgbd_image = np.zeros([image.shape[0], image.shape[1], 3])
            rgbd_image[:, :, 0:2] = image[:, :, 0:2]
            rgbd_image[:, :, 2] = depth
        elif image_type == 'rgb':
            rgbd_image = np.zeros([image.shape[0], image.shape[1], 3])
            rgbd_image[:, :, 0:3] = image[:, :, 0:3]
        elif image_type == 'rgbd':
            rgbd_image = np.zeros([image.shape[0], image.shape[1], 4])
            rgbd_image[:, :, 0:3] = image[:, :, 0:3]
            rgbd_image[:, :, 3] = depth
        rgbd_image = np.array(rgbd_image).astype('uint8')
        if len(augmentation) != 0:
            rgbd_image = self.apply_augmentation_to_image(rgbd_image, augmentation)
        return rgbd_image

    def apply_augmentation_to_mask(self, mask, augmentation):
        angle, dx, dy, flip_code, contrast, noise = augmentation
        mask = mask.astype('uint8')
        rotated_mask = image_augmentation.rotate_image(mask, angle, scale=1)
        rotated_translated_mask = image_augmentation.translate_image(rotated_mask, dx, dy)
        if flip_code != 2:
            rotated_translated_mask = image_augmentation.flip_image(rotated_translated_mask, int(flip_code))

        return rotated_translated_mask


    def apply_augmentation_to_image(self, image, augmentation):
        # ['angle', 'dx', 'dy', 'flip', 'contrast', 'noise']
        angle, dx, dy, flip_code, contrast, noise = augmentation
        rgbd_image = image_augmentation.rotate_image(image, angle, scale=1)
        rgbd_image = image_augmentation.translate_image(rgbd_image, dx, dy)
        if flip_code != 2:
            rgbd_image = image_augmentation.flip_image(rgbd_image, int(flip_code))

        # Contrast augmentation
        if contrast != 0:
            pillow_image = Image.fromarray(rgbd_image)
            enhancer = ImageEnhance.Contrast(pillow_image)
            rgbd_image = np.array(enhancer.enhance(contrast))

        # Additive Gaussian noise
        if noise != 0:
            mean = 0.0
            std = noise
            rgbd_image = rgbd_image + np.random.normal(mean, std, rgbd_image.shape)
            rgbd_image = np.clip(rgbd_image, 0, 255)
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

    def load_image_from_file(self, image_path):
        # Loads the image into a matplotlib.Image object
        # input: image path
        # output: numpy array for image
        img = mpimg.imread(image_path)
        return img

    def load_label_from_file(self, mask_path):
        # Reads the label image and scales it up so each instance has an integer label rather than a float
        # input: mask path
        # output: numpy array for mask
        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask = mask.astype(int)
        return mask

    def display_image(self, image):
        # Displays the image
        # input: 2D numpy array
        plt.imshow(image)
        plt.show()

    def reshape_label_image(self, mask):
        # For an i*j image, this function converts mask to an i*j*N boolean 3D matrix, where N = # of object instances
        # in the image
        # input: numpy array of mask
        # output: i*j*N boolean mask
        object_instances = np.unique(mask)
        mask_reshaped = np.zeros([mask.shape[0], mask.shape[1], len(object_instances)-1])
        class_ids = []
        j = 0
        for i in object_instances[1:]:
            mask_reshaped[:,:,j] = (mask==i)
            class_id = 1
            if (len(np.unique(mask==i)) == 1):
                class_id = 0
            class_ids.append(class_id)
            j = j + 1
        return mask_reshaped, class_ids

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

    def construct_OCID_dataset(self, train = 0.70, val = 0.15, dataset_dir = '../../../Datasets/OCID-dataset'):
        dataset_paths = glob.glob(dataset_dir + '/**/rgb/*.png', recursive=True)
        dataset_folder = 'ocid_dataset'
        folder_names = ['train_set', 'val_set', 'test_set']
        subfolder_names = ['depth', 'label', 'rgb']
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
            for folder_name in folder_names:
                os.makedirs(os.path.join(dataset_folder, folder_name))
                for subfolder in subfolder_names:
                    os.makedirs(os.path.join(dataset_folder, folder_name, subfolder))

        train_set, val_set, test_set = self.split_dataset(dataset_paths, train, val)
        i = 0
        for dataset_paths in [train_set, val_set, test_set]:
            for image_path in dataset_paths:
                image_path_split = image_path.split('\\')
                image_name = image_path_split[-1]

                for subfolder_name in subfolder_names:
                    image_path = os.path.join(*image_path_split[0:-2], subfolder_name, image_name)
                    target_path = os.path.join(dataset_folder, folder_names[i], subfolder_name, image_name)
                    if (subfolder_name == 'label'):
                        location = image_path.split('\\')[2]
                        target_path = target_path.replace(image_name, (location + '_' + image_name))
                    copyfile(image_path, target_path)
            i = i + 1

    def construct_WISDOM_dataset(self, train = 0.70, val = 0.15, dataset_dir = '../../../Datasets/wisdom-full-single/wisdom/wisdom-real'):
        dataset_folder = 'wisdom_dataset'
        folder_names = ['train_set', 'val_set', 'test_set']
        subfolder_names = ['depth', 'label', 'rgb']
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
            for folder_name in folder_names:
                os.makedirs(os.path.join(dataset_folder, folder_name))
                for subfolder in subfolder_names:
                    os.makedirs(os.path.join(dataset_folder, folder_name, subfolder))

        resolutions = os.listdir(dataset_dir)
        training_images = []
        testing_images = []
        validating_images = []
        resolutions_count = 0

        for resolution in resolutions:
            rgb_images_path = os.path.join(dataset_dir, resolution, 'color_ims')
            depth_images_path = os.path.join(dataset_dir, resolution, 'depth_ims')
            label_images_path = os.path.join(dataset_dir, resolution, 'modal_segmasks')

            image_names = np.array(os.listdir(rgb_images_path))

            train_set, val_set, test_set = self.split_dataset(image_names, train, val)
            i = 0
            for dataset_paths in [train_set, val_set, test_set]:
                for image_name in dataset_paths:
                    for subfolder_name in subfolder_names:
                        subfolder_path = vars()[(subfolder_name + '_images_path')]

                        image_path = os.path.join(subfolder_path, image_name)

                        # Converting depth to sngle channel
                        if subfolder_name == 'depth':
                            img = skimage.io.imread(image_path)
                            img = img[:,:,2]
                            img = Image.fromarray(img)
                            img.save(image_path)

                        if resolutions_count == 1:
                            image_name_new = 'image_' + str(int(image_name.split('_')[1].split('.')[0]) + 400).zfill(6) + '.png'
                            target_path = os.path.join(dataset_folder, folder_names[i], subfolder_name, image_name_new)
                        else:
                            target_path = os.path.join(dataset_folder, folder_names[i], subfolder_name, image_name)
                        copyfile(image_path, target_path)

                i = i + 1
            resolutions_count = resolutions_count + 1

    def construct_SAMS_dataset(self, train=0.70, val=0.15, dataset_dir='../../../Datasets/SAMS-Dataset'):
        dataset_paths = glob.glob(dataset_dir + '/**/rgb/*.png', recursive=True)
        dataset_folder = 'sams_dataset'
        folder_names = ['train_set', 'val_set', 'test_set']
        subfolder_names = ['depth', 'label', 'rgb']
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
            for folder_name in folder_names:
                os.makedirs(os.path.join(dataset_folder, folder_name))
                for subfolder in subfolder_names:
                    os.makedirs(os.path.join(dataset_folder, folder_name, subfolder))

        train_set, val_set, test_set = self.split_dataset(dataset_paths, train, val)
        i = 0
        for dataset_paths in [train_set, val_set, test_set]:
            for image_path in dataset_paths:
                image_path_split = image_path.split('\\')
                image_name = image_path_split[-1]

                for subfolder_name in subfolder_names:
                    image_path = os.path.join(*image_path_split[0:-2], subfolder_name, image_name)
                    target_path = os.path.join(dataset_folder, folder_names[i], subfolder_name, image_name)
                    copyfile(image_path, target_path)
            i = i + 1




    def construct_dataset(self, train = 0.70, val = 0.15, dataset_dir = '../../../Datasets/OCID-dataset'):
        # Iterates recursively through the OCID dataset folder structure, moves all rgb, label and depth images into
        # the following folders: "ocid-dataset/rgb", "ocid-dataset/label", "ocid-dataset/depth". Images are shuffled
        # and split into 'train_set', 'val_set', 'test_set' sub folders based on "train" and "val" ratios

        if 'OCID' in dataset_dir:
            self.construct_OCID_dataset(train, val, dataset_dir)
        elif 'wisdom' in dataset_dir:
            self.construct_WISDOM_dataset(train, val, dataset_dir)
        elif 'SAMS' in dataset_dir:
            self.construct_SAMS_dataset(train, val, dataset_dir)

    def process_label_image(self, label_path):
        # Removes the table and background from the label images and saves the output image at target_path
        # input:
        # label_paths - path of the original label images
        label = self.load_label_from_file(label_path)
        if ('floor' in label_path) or ('table' in label_path):
            image_name = label_path.split('\\')[-1]
            location = image_name.split('_')[0]
            if (location == 'floor'):
                label_new = label * (label > 1)
            else:
                label_new = label * (label > 2)
            label = label_new
        return label

    def visualize_dataset(self, dataset_type = 'train_set', num_images = 3):
        # Randomly selects num_images images from the OCID datasets and displays them in the following order:
        # RGB | DEPTH | LABEL
        # input: num_images - number of images to display
        dataset_path = os.path.join('ocid_dataset', dataset_type)
        image_list = glob.glob(dataset_path + '/**/rgb/*.png', recursive=True)
        images_to_show = random.choices(image_list, k = num_images)
        columns = 3
        rows = num_images
        fig, axes = plt.subplots(nrows=rows, ncols=columns)
        i = 0
        for image in images_to_show:
            image_name = image.split('\\')[-1]
            rgb = self.load_image_from_file(image)
            depth = self.load_image_from_file(image.replace('rgb','depth'))
            depth = self.rescale_depth_image(depth)
            label_options = ['floor_', 'table_']
            label_path = image.replace('rgb', 'label')
            for option in label_options:
                temp_path = label_path.replace(image_name, option + image_name)
                if os.path.exists(temp_path):
                    label_path = temp_path
            label = self.load_label_from_file(label_path)
            if (num_images > 1):
                axes[i, 0].imshow(rgb)
                axes[i, 1].imshow(depth)
                axes[i, 2].imshow(label)
            else:
                axes[0].imshow(rgb)
                axes[1].imshow(depth)
                axes[2].imshow(label)
            i = i + 1
        fig.show()

    def visualize_masks_bbox(self, image_ids):
        for image_id in image_ids:
            image = self.load_image(image_id)
            mask, class_ids = self.load_mask(image_id)
            summary = str(len(class_ids)) + ' objects\n' + self.image_info[image_id]['label_path']
            visualize.display_top_masks(image, mask, class_ids, self.class_names, limit=1,
                                        image_info=summary)
            bbox = utils.extract_bboxes(mask)
            visualize.display_instances(image[:, :, 0:3], bbox, mask, class_ids, self.class_names)

    def get_channel_means(self):
        depth_sum = 0
        red_sum = 0
        green_sum = 0
        blue_sum = 0
        for id in self.image_ids:
            image = self.load_image(id, image_type='rgbd')
            mean_red = np.mean(image[:, :, 0])
            red_sum = red_sum + mean_red
            mean_green = np.mean(image[:, :, 1])
            green_sum = green_sum + mean_green
            mean_blue = np.mean(image[:, :, 2])
            blue_sum = blue_sum + mean_blue
            mean_depth = np.mean(image[:, :, 3])
            depth_sum = depth_sum + mean_depth

        red_channel_mean = red_sum / len(self.image_ids)
        blue_channel_mean = green_sum / len(self.image_ids)
        green_channel_mean = blue_sum / len(self.image_ids)
        depth_channel_mean = depth_sum / len(self.image_ids)
        return red_channel_mean, green_channel_mean, blue_channel_mean, depth_channel_mean

    def get_mask_overlay(self, image, masks, scores, threshold=0.90):
        num_masks = masks.shape[-1]
        colors = visualize.random_colors(num_masks)
        masked_image = np.copy(image)
        for i in list(range(num_masks)):
            if (scores[i] < threshold):
                continue
            mask = masks[:, :, i]
            masked_image = visualize.apply_mask(masked_image, mask, colors[i])
        masked_image = np.array(masked_image, dtype='uint8')
        return masked_image, colors

    def load_jacquard_gt_boxes(self, image_id):
        positive_rectangles_path = self.image_info[image_id]['positive_points']
        class_ids = []
        bounding_boxes = []
        with open(positive_rectangles_path) as f:
            positive_rectangles = f.readlines()

        for line in positive_rectangles:

            bbox_5_dim = line.strip('\n').split(';')
            bbox_5_dim = np.array(bbox_5_dim).astype(np.float)
            x, y, theta, w, h = bbox_5_dim
            # Need to put a constraint on the aspect ratio and box sizes
            # Setting max aspect ratio to 20 and minimum to be 1/20
            aspect_ratio = w / h
            if (aspect_ratio < 1/20.0 or aspect_ratio > 20.0):
                continue
            # Minimum dimension allowed is 5 pixels
            if (h < 5 or w < 5):
                continue

            # # Theta re-adjustments:
            # if theta > 90:
            #     theta  = theta - 180
            # elif theta < -90:
            #     theta = theta + 180
            theta %= 360
            bounding_boxes.append([x, y, w, h, theta])
            class_ids.append(1)
        return np.array(bounding_boxes), np.array(class_ids)

    def apply_augmentation_to_box(self, boxes, class_ids, image_id, augmentation):
        # ['angle', 'dx', 'dy', 'flip']
        angle, dx, dy, flip_code, _, _ = augmentation
        image_shape = plt.imread(self.image_info[image_id]['path']).shape[:2]
        transformed_boxes, class_ids = image_augmentation.rotate_bboxes(boxes, angle, image_shape, class_ids)
        transformed_boxes, class_ids = image_augmentation.translate_bbox(transformed_boxes, dx, dy, image_shape, class_ids)
        if flip_code != 2:
            transformed_boxes = image_augmentation.flip_bbox(transformed_boxes, int(flip_code), image_shape)

        return transformed_boxes, class_ids

    def load_ground_truth_bbox_files(self, image_id, include_negatives=False):
        positive_rectangles_path = self.image_info[image_id]['positive_points']

        boxes = []
        with open(positive_rectangles_path) as f:
            positive_rectangles = f.readlines()
            boxes.append(positive_rectangles)
            class_id = 1

        if include_negatives:
            negative_rectangles_path = self.image_info[image_id]['negative_points']
            with open(negative_rectangles_path) as f:
                negative_rectangles = f.readlines()
            boxes.append(negative_rectangles)
            class_id = 0

        bounding_boxes_formatted = []
        vertices = []
        class_ids = []
        bad_rectangle = False

        for bounding_box_class in boxes:
            i = 0
            for bounding_box in bounding_box_class:
                try:
                    x = int(float(bounding_box.split(' ')[0]))
                    y = int(float(bounding_box.split(' ')[1]))
                    vertices.append([x, y])
                except:
                    bad_rectangle = True
                i += 1
                if i % 4 == 0:
                    if not bad_rectangle:
                        bounding_boxes_formatted.append(vertices)
                        class_ids.append(class_id)
                    vertices = []
                    bad_rectangle = False
            class_id += 1
        return np.array(bounding_boxes_formatted), class_ids

    def get_five_dimensional_box(self, coordinates):
        if np.sum(coordinates == 0):
            return [0,0,0,0,0]

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
        #### ERROR HERE ####
        if (np.linalg.norm(p2 - p1) == 0):
            print("ZERO DIVISION")
            import code;
            code.interact(local=dict(globals(), **locals()))
        h = np.abs(np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1))

        deltaY = gripper_orientation[0][1] - gripper_orientation[1][1]
        deltaX = gripper_orientation[0][0] - gripper_orientation[1][0]
        theta = -1 * np.arctan2(deltaY, deltaX) * 180 / math.pi
        if theta > 90:
            theta = theta - 180
        elif theta < -90:
            theta = theta + 180
        return [x, y, w, h, theta]

    def bbox_convert_to_five_dimension(self, bounding_box_vertices, image_id=0):
        bbox_5_dimensional = []
        for object_instance in bounding_box_vertices:
            instance_bboxes = []
            for bounding_box in object_instance:
                x, y, w, h, theta = self.get_five_dimensional_box(bounding_box)
                instance_bboxes.append([x, y, w, h, theta])
                # self.visualize_bbox(image_id, bounding_box, 1, [x, y, w, h, theta])
            bbox_5_dimensional.append(instance_bboxes)
        bbox_5_dimensional = np.array(bbox_5_dimensional)
        return bbox_5_dimensional

    def load_bounding_boxes(self, image_id, augmentation=[], target_num_boxes=256):
        # bounding boxes here have a shape of N x 4 x 2, consisting of four vertices per rectangle given N rectangles
        # loading jacquard style bboxes. NOTE: class_ids will all be 1 since jacquard only has positive boxes

        if 'jacquard' in self.image_info[image_id]['path']:
            bbox_5_dimensional, class_ids = self.load_jacquard_gt_boxes(image_id)
            if len(augmentation) != 0:
                bbox_5_dimensional, class_ids = self.apply_augmentation_to_box(bbox_5_dimensional, class_ids, image_id, augmentation)
            # bounding_box_vertices = self.bbox_convert_to_four_vertices(bbox_5_dimensional)
        else:
            bounding_box_vertices, class_ids = self.load_ground_truth_bbox_files(image_id)
            image_path = self.image_info[image_id]['path']
            if ('cornell' in image_path) or ('multi_grasp' in image_path):
                bbox_5_dimensional = []
                for box in bounding_box_vertices:
                    center_location, box_size, theta = cv2.minAreaRect(box)
                    # The first two coordinates of a rectangle define the line
                    # representing the orientation of the gripper plate
                    x1, y1 = box[0]
                    x2, y2 = box[1]
                    x3, y3 = box[2]
                    theta_radians = math.atan2(y2 - y1, x2 - x1)
                    theta_degree = self.rad2deg(theta_radians)
                    width = ((y2 - y1)**2 + (x2 - x1)**2)**0.5
                    height = ((y3 - y2)**2 + (x3 - x2)**2)**0.5
                    x, y = center_location
                    bbox_5_dimensional.append([x, y, width, height, theta_degree])
                bbox_5_dimensional = np.array(bbox_5_dimensional)

                # bbox_5_dimensional = self.bbox_convert_to_five_dimension([bounding_box_vertices])
                # bbox_5_dimensional = bbox_5_dimensional[0]
            else:
                bbox_5_dimensional = self.bbox_convert_to_five_dimension(bounding_box_vertices)
            if len(augmentation) != 0:
                bbox_5_dimensional, class_ids = self.apply_augmentation_to_box(bbox_5_dimensional, class_ids, image_id, augmentation)
            # bounding_box_vertices = self.bbox_convert_to_four_vertices(bbox_5_dimensional)

        class_ids = np.array(class_ids)
        # bounding_box_vertices = np.array(bounding_box_vertices)

        num_boxes = bbox_5_dimensional.shape[0]
        extra = target_num_boxes - num_boxes
        if extra > 0:
            zero_pad_box = np.zeros((extra, ) + bbox_5_dimensional[0].shape)
            # bounding_box_vertices = np.append(bounding_box_vertices, zero_pad_box, axis=0)
            # was originally a tensor of zeros, now changed it to -1
            zero_pad_class_ids = np.ones(extra)*-1
            class_ids = np.append(class_ids, zero_pad_class_ids)
            zero_pad_5_dimensional = np.zeros((extra,) + bbox_5_dimensional[0].shape)
            bbox_5_dimensional = np.append(bbox_5_dimensional, zero_pad_5_dimensional, axis=0)
        else:
            extra_ix_to_remove = np.random.permutation(abs(extra))
            # bounding_box_vertices = np.delete(bounding_box_vertices, extra_ix_to_remove, axis=0)
            class_ids = np.delete(class_ids, extra_ix_to_remove)
            bbox_5_dimensional = np.delete(bbox_5_dimensional, extra_ix_to_remove, axis=0)

        # shuffling the rectangles
        p = np.random.permutation(len(class_ids))

        # Reshaping boxes to have a shape [num_instances, num_grasp_boxes, ..]
        # bounding_box_vertices = np.array([bounding_box_vertices[p]])
        bbox_5_dimensional = np.array([bbox_5_dimensional[p]])
        class_ids = np.array([class_ids[p].astype('int8')])
        class_ids = np.expand_dims(class_ids, axis=-1)

        return bbox_5_dimensional, class_ids

    def bbox_convert_to_four_vertices(self, bbox_5_dimension):
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
            # import code;
            # code.interact(local=dict(globals(), **locals()))
            # if np.sign(theta) == 1:
            #     angle = angle + 90
            #     if angle > 180:
            #         angle = angle - 360
            # else:
            #     angle = angle + 90

            rotational_matrix = cv2.getRotationMatrix2D((x, y), (angle), scale)
            # rotational_matrix = cv2.getRotationMatrix2D((x, y), (angle + 90), scale)
            original_points_rotated = cv2.transform(original_points_translated, rotational_matrix)
            rotated_points.append(original_points_rotated[0])

        return np.array(rotated_points)

    def deg2rad(self, angle):
        pi_over_180 = pi / 180
        return angle * pi_over_180

    def rad2deg(self, angle):
        pi_over_180 = pi / 180
        return angle / pi_over_180

    def shapely_polygon(self, five_dim_box):
        x, y, w, h, theta = five_dim_box
        p = Polygon([(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)])
        return translate(rotate(p, theta), x, y)

    def compute_jacquard_index_and_angle_difference(self, top_1_box, gt_boxes):
        gt_boxes = gt_boxes[0]
        gt_x, gt_y, gt_w, gt_h, gt_theta = np.split(gt_boxes, indices_or_sections = 5, axis=-1)

        gt_box_vertices = np.zeros((gt_boxes.shape[0], 4))
        gt_box_vertices[:, 0] = np.squeeze(gt_y - gt_h / 2)
        gt_box_vertices[:, 1] = np.squeeze(gt_x - gt_w / 2)
        gt_box_vertices[:, 2] = np.squeeze(gt_y + gt_h / 2)
        gt_box_vertices[:, 3] = np.squeeze(gt_x + gt_w / 2)

        x, y, w, h, theta = top_1_box[0]
        iou = np.zeros(gt_boxes.shape[0])
        top_1_box_shapely = self.shapely_polygon(top_1_box[0])

        for i, gt_box in enumerate(gt_boxes):

            gt_box_shapely = self.shapely_polygon(gt_box)
            intersection_area = top_1_box_shapely.intersection(gt_box_shapely).area
            if intersection_area:
                iou[i] = intersection_area / top_1_box_shapely.union(gt_box_shapely).area

        # iou = utils.compute_iou(top_1_vertices, gt_box_vertices, top_1_box_area, np.squeeze(gt_box_areas))
        jacquard_condition = np.reshape(iou >= 0.25, [-1, 1])

        top_1_box_theta = np.tile(np.array(theta), (gt_boxes.shape[0], 1))
        x_minus_y = self.deg2rad(gt_theta) - self.deg2rad(top_1_box_theta)
        angle_difference = np.arctan2(np.sin(x_minus_y), np.cos(x_minus_y))
        angle_difference = np.abs(self.rad2deg(angle_difference))

        angle_condition = angle_difference <= 30

        matches = np.logical_and(jacquard_condition, angle_condition)
        # print ('num_overlaps: ', np.count_nonzero(jacquard_condition == True), 'num_angle_matches: ', np.count_nonzero(angle_condition == True))

        return np.any(matches)

    def compute_validation_metrics(self, grasp_predictions, grasp_probabilities, gt_grasp_boxes, detection_thresholds):
        gt_boxes  = np.reshape(gt_grasp_boxes, [-1, 5])
        gt_boxes = gt_boxes[np.where(np.sum(gt_boxes, axis=-1) != 0)]

        # # need to check all combinations of gt_boxes and the grasp boxes
        # gt_boxes_tiled = np.tile(gt_boxes, (grasp_predictions.shape[0], 1))
        # grasp_predictions_tiled = np.tile(grasp_predictions, (gt_boxes.shape[0], 1))
        gt_x, gt_y, gt_w, gt_h, gt_theta = np.split(gt_boxes, indices_or_sections=5, axis=-1)
        true_positives = np.zeros(detection_thresholds.shape)
        false_negatives = np.zeros(detection_thresholds.shape)
        false_positives = np.zeros(detection_thresholds.shape)

        for j, prediction in enumerate(grasp_predictions):
            match = False
            matches = np.zeros((gt_boxes.shape[0], 1), dtype=bool)
            prediction_probability = grasp_probabilities[j]

            x, y, w, h, theta = prediction
            prediction_theta = np.tile(np.array(theta), (gt_boxes.shape[0], 1))
            x_minus_y = self.deg2rad(gt_theta) - self.deg2rad(prediction_theta)
            angle_difference = np.arctan2(np.sin(x_minus_y), np.cos(x_minus_y))
            angle_difference = np.abs(self.rad2deg(angle_difference))
            angle_condition = angle_difference <= 30

            iou = np.zeros(gt_boxes.shape[0])
            prediction_shapely = self.shapely_polygon(prediction)
            for i, gt_box in enumerate(gt_boxes):
                gt_box_shapely = self.shapely_polygon(gt_box)
                intersection_area = prediction_shapely.intersection(gt_box_shapely).area
                if intersection_area:
                    iou[i] = intersection_area / prediction_shapely.union(gt_box_shapely).area
                    match = iou[i] >= 0.25 and angle_condition[i]
                    if match:
                        break
            # jacquard_condition = np.reshape(iou >= 0.25, [-1, 1])

            # matches = np.logical_and(jacquard_condition, angle_condition)

            prediction_probability = np.tile(prediction_probability, detection_thresholds.shape)
            # if np.any(matches):
            if match:
                false_negatives[np.where(prediction_probability < detection_thresholds)] += 1
                true_positives[np.where(prediction_probability >= detection_thresholds)] += 1
                # if prediction_probability < detection_thresholds:
                #     false_negatives += 1
                # else:
                #     true_positives += 1
            else:
                false_positives[np.where(prediction_probability >= detection_thresholds)] += 1
                # if prediction_probability >= detection_thresholds:
                #     false_positives += 1
        return [true_positives, false_negatives, false_positives]


    def orient_box_nms(self, boxes, scores, config):
        scores = scores[:, 1]
        sorting_ix = np.argsort(scores)[::-1]
        filtered_boxes = boxes[sorting_ix]
        filtered_scores = scores[sorting_ix]

        xs = filtered_boxes[:, 0]
        ys = filtered_boxes[:, 1]
        ws = filtered_boxes[:, 2]
        hs = filtered_boxes[:, 3]
        thetas = filtered_boxes[:, 4]
        box_areas = ws * hs

        box_vertices = np.zeros((box_areas.shape[0], 4))
        box_vertices[:, 0] = ys - hs / 2
        box_vertices[:, 1] = xs - ws / 2
        box_vertices[:, 2] = ys + hs / 2
        box_vertices[:, 3] = xs + ws / 2

        ix = 0
        while ix < filtered_boxes.shape[0]:
            max_score_box = filtered_boxes[ix]
            x, y, w, h, theta = max_score_box
            gt_box_vertices = [y - h / 2, x - w / 2, y + h / 2, x + w / 2]
            gt_box_area = w * h

            iou = utils.compute_iou(gt_box_vertices, box_vertices, gt_box_area, box_areas)
            angle_differences = np.cos(np.radians(thetas) - np.radians(theta))
            angle_differences[angle_differences < 0] = 0
            arIoU = iou * angle_differences

            # Find all boxes that have a high IoU
            ix_to_remove = np.where(np.delete(arIoU, ix) > config.DETECTION_NMS_THRESHOLD)[0]
            filtered_boxes = np.delete(filtered_boxes, ix_to_remove, axis=0)
            filtered_scores = np.delete(filtered_scores, ix_to_remove)
            box_areas = np.delete(box_areas, ix_to_remove)
            box_vertices = np.delete(box_vertices, ix_to_remove, axis=0)
            thetas = np.delete(thetas, ix_to_remove)
            ix += 1
        # print(boxes.shape, filtered_boxes.shape)
        return [filtered_boxes, filtered_scores, boxes, scores]

    def refine_results(self, probabilities, deltas, anchors, config, filter_mode='prob', k=1):
        '''
        Applies a NMS refinement algorithm on the proposals output by the RPN
        # '''
        deltas = deltas * config.GRASP_BBOX_STD_DEV
        mode = 'mask_grasp_rcnn'


        all_boxes = utils.apply_box_deltas(anchors, deltas, mode,
                                           len(config.GRASP_ANCHOR_ANGLES))
        all_boxes[:, -1] /= 360
        all_boxes = utils.denorm_boxes(all_boxes, config.IMAGE_SHAPE[:2], mode='grasping_points')
        # all_boxes[:,-1] = all_boxes[:,-1]%360

        # Filter out boxes with center coordinates out of the image
        # radius = ((0.5 * all_boxes[:, 2]) ** 2 + (0.5 * all_boxes[:, 3]) ** 2) ** 0.5
        #
        # invalid_x = np.where(all_boxes[:, 0] + radius > config.IMAGE_SHAPE[1])[0]
        # all_boxes = np.delete(all_boxes, invalid_x, axis = 0)
        # probabilities = np.delete(probabilities, invalid_x, axis = 0)
        # radius = np.delete(radius, invalid_x)
        #
        # invalid_x = np.where(all_boxes[:, 0] - radius < 0)[0]
        # all_boxes = np.delete(all_boxes, invalid_x, axis=0)
        # probabilities = np.delete(probabilities, invalid_x, axis=0)
        # radius = np.delete(radius, invalid_x)
        #
        # invalid_y = np.where(all_boxes[:, 1] + radius > config.IMAGE_SHAPE[0])[0]
        # all_boxes = np.delete(all_boxes, invalid_y, axis = 0)
        # probabilities = np.delete(probabilities, invalid_y, axis = 0)
        # radius = np.delete(radius, invalid_y)
        #
        # invalid_y = np.where(all_boxes[:, 1] - radius < 0)[0]
        # all_boxes = np.delete(all_boxes, invalid_y, axis=0)
        # probabilities = np.delete(probabilities, invalid_y, axis=0)

        if filter_mode == 'top_k':
            sorting_ix = np.argsort(probabilities[:, 1])[::-1][:k]
            top_boxes = all_boxes[sorting_ix]
            top_box_probabilities = probabilities[sorting_ix]
        elif filter_mode == 'prob':
            top_boxes = all_boxes[probabilities[:,1] > config.GRASP_MIN_CONFIDENCE]
            top_box_probabilities = probabilities[probabilities[:,1] > config.GRASP_MIN_CONFIDENCE]
        else:
            top_boxes = all_boxes
            top_box_probabilities = probabilities

        top_boxes, top_box_probabilities, pre_nms_boxes, pre_nms_scores = self.orient_box_nms(top_boxes, top_box_probabilities, config)

        return top_boxes, top_box_probabilities, pre_nms_boxes, pre_nms_scores

    def generate_random_color(self):
        r = random.random()#*255
        b = random.random()#*255
        g = random.random()#*255
        return (r, g, b)

# SETUP ##

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

training_dataset = GraspMaskRCNNDataset()
# training_dataset.construct_dataset(dataset_dir = '../../../Datasets/SAMS-Dataset')
# training_dataset.load_dataset(dataset_dir='../../../Datasets/jacquard_dataset_resized_new', augmentation=True)
training_dataset.load_dataset(dataset_dir='cornell_grasping_dataset', augmentation=True)
# training_dataset.load_dataset(dataset_dir='../../../Datasets/jacquard_dataset_resized_new')
training_dataset.prepare()

validating_dataset = GraspMaskRCNNDataset()
# validating_dataset.load_dataset(dataset_dir='../../../Datasets/jacquard_dataset_resized_new', type='val_set', augmentation=True)
# validating_dataset.load_dataset(dataset_dir='../../../Datasets/jacquard_dataset_resized_new', type='val_set')
validating_dataset.load_dataset(dataset_dir='cornell_grasping_dataset', type='val_set')
# validating_dataset.load_dataset(dataset_dir='cornell_grasping_dataset', type='val_set')
validating_dataset.prepare()

testing_dataset = GraspMaskRCNNDataset()
# testing_dataset.load_dataset(dataset_dir='../../../Datasets/jacquard_dataset_resized_new', type='test_set', augmentation=True)
# testing_dataset.load_dataset(dataset_dir='../../../Datasets/multi_grasp_dataset_new', type='test_set')
testing_dataset.load_dataset(dataset_dir='cornell_grasping_dataset', type='test_set')
testing_dataset.prepare()

config = GraspMaskRCNNConfig()
# channel_means = np.array(testing_dataset.get_channel_means())
# import code; code.interact(local=dict(globals(), **locals()))
# config.MEAN_PIXEL = np.around(channel_means, decimals = 1)
# config.display()
inference_config = GraspMaskRCNNInferenceConfig()
# grasping_inference_config = GraspingInferenceConfig()
MODEL_DIR = "models"
mode = "mask_grasp_rcnn"
#
# # # # #### TRAINING #####
# # COCO_MODEL_PATH = os.path.join("models", "mask_rcnn_object_vs_background_100_heads_50_all.h5")
# # COCO_MODEL_PATH = os.path.join("models", "mask_rcnn_object_vs_background_HYBRID-50_head_50_all.h5")
# # COCO_MODEL_PATH = 'models/Good_models/Training_SAMS_dataset_LR-same-div-2-HYBRID-weights.h5'
# # COCO_MODEL_PATH = os.path.join("models", "Good_models", "Training_SAMS_dataset_LR-same-div-2-HYBRID-weights", "SAMS_DATASET_TRAINING_REFERENCE.h5")
# COCO_MODEL_PATH = 'models/colab_result_id#1/mask_rcnn_grasp_and_mask_0288.h5'
# # COCO_MODEL_PATH = os.path.join("models", "mask_rcnn_coco.h5")
# model = modellib.MaskRCNN(mode="training", config=config,
#                              model_dir=MODEL_DIR, task='mask_grasp_rcnn')
#
# # model.load_weights(COCO_MODEL_PATH, by_name=True,
# #                       exclude=["conv1", "mrcnn_class_logits", "mrcnn_bbox_fc",
# #                                "mrcnn_bbox", "mrcnn_mask"])
# model.load_weights(COCO_MODEL_PATH, by_name=True)
# #
# # model.train(training_dataset, validating_dataset,load_bounding_boxes
# #                learning_rate=config.LEARNING_RATE,
# #                epochs=200,
# #                layers=r"(conv1)|(rpn\_.*)|(fpn\_.*)|(mrcnn\_.*)|(grasp\_.*)",
# #                task=mode)
# #
#
# model.train(training_dataset, validating_dataset,
#                learning_rate=config.LEARNING_RATE,
#                epochs=200,
#                layers=r"(grasp\_.*)",
#                task=mode)
#
# #
# # model.train(training_dataset, validating_dataset,
# #                learning_rate=config.LEARNING_RATE/10,
# #                epochs=400,
# #                layers="all",
# #                task=mode)
#
#
# # model.train(training_dataset, validating_dataset,
# #                 learning_rate=config.LEARNING_RATE/10,
# #                 epochs=400,
# #                 layers="all",
# #                 task=mode)
#
# model_path = os.path.join(MODEL_DIR, "mask_grasp_rcnn.h5")
# model.keras_model.save_weights(model_path)

# ##### TESTING #####

# mrcnn_model_path = 'models/Good_models/Training_SAMS_dataset_LR-div-5-div-10-HYBRID-weights/mask_rcnn_object_vs_background_0051.h5'
# mask_grasp_model_path = 'models/grasp_and_mask20200905T1322/mask_rcnn_grasp_and_mask_0400.h5'
# mask_grasp_model_path = 'models/mask_grasp_rcnn_attempt#1b/mask_rcnn_grasp_and_mask_0108.h5'
# mask_grasp_model_path = 'models/colab_result_id#1/mask_rcnn_grasp_and_mask_0344.h5'
# mask_grasp_model_path = 'models/colab_result_id#1/attempt#26c_weights.h5'
mask_grasp_model_path = 'models/colab_result_id#1/mask_rcnn_grasp_and_mask_0288.h5'


mask_grasp_model = modellib.MaskRCNN(mode="inference",
                           config=inference_config,
                           model_dir=MODEL_DIR, task=mode)
mask_grasp_model.load_weights(mask_grasp_model_path, by_name=True)

# grasping_model_path = os.path.join(MODEL_DIR, 'colab_result_id#1',"train_#12.h5")
# grasping_model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
#                               config=inference_config, task="grasping_points")
# grasping_model.load_weights(grasping_model_path, by_name=True)

dataset = testing_dataset
image_ids = random.choices(dataset.image_ids, k=30)
for image_id in image_ids:
     original_image, image_meta, gt_class_id, gt_bbox, gt_mask, gt_grasp_boxes, gt_grasp_id =\
         modellib.load_image_gt(dataset, inference_config,
                                image_id, use_mini_mask=True, image_type='rgbd', mode='mask_grasp_rcnn')

     results = mask_grasp_model.detect([original_image], verbose=0, task=mode)
     r = results[0]
     if r['rois'].shape[0] > 0:
         mask_image, _ = testing_dataset.get_mask_overlay(original_image[:, :, 0:3], r['masks'], r['scores'], 0)
         grasping_deltas = r['grasp_boxes']
         grasping_probs = r['grasp_probs']

         fig, axs = plt.subplots(2, 3, figsize=(15, 5))
         axs[0, 0].imshow(original_image.astype(np.uint8))
         axs[0, 1].imshow(mask_image.astype(np.uint8))
         axs[0, 2].imshow(original_image.astype(np.uint8))
         axs[1, 0].imshow(original_image[:, :, :3].astype(np.uint8))
         axs[1, 1].imshow(original_image[:, :, :3].astype(np.uint8))
         axs[1, 2].imshow(original_image[:, :, :3].astype(np.uint8))

         axs[0, 0].set_title('Original Image')
         axs[0, 1].set_title('Masks and ROIs')
         axs[0, 2].set_title('Grasping anchors')
         axs[1, 0].set_title('GT grasp boxes')
         axs[1, 1].set_title('Boxes pre non-maximum supression')
         axs[1, 2].set_title('Boxes post non-maximum supression')
         bboxes = r['rois']

         top_image_rois = np.reshape(np.array([]), [-1, 5])
         top_image_roi_probabilities = np.reshape(np.array([]), [-1])

         for j, rect in enumerate(r['rois']):
             color = dataset.generate_random_color()
             if config.USE_EXPANDED_ROIS:
                 rect = utils.expand_roi_by_percent(rect, percentage=config.GRASP_ROI_EXPAND_FACTOR,
                                                              image_shape=config.IMAGE_SHAPE[:2])

             y1, x1, y2, x2 = rect
             p = patches.Rectangle((x1, y1), (x2 - x1), (y2 - y1), angle=0, edgecolor=dataset.generate_random_color(),
                                   linewidth=2, facecolor='none')
             axs[0, 0].add_patch(p)
             expanded_rect_normalized = utils.norm_boxes(rect, config.IMAGE_SHAPE[:2])

             y1, x1, y2, x2 = expanded_rect_normalized
             w = abs(x2 - x1)
             h = abs(y2 - y1)
             ROI_shape = np.array([h, w])
             pooled_feature_stride = np.array(ROI_shape/config.GRASP_POOL_SIZE)#.astype('uint8')

             grasping_anchors = utils.generate_grasping_anchors(config.GRASP_ANCHOR_SIZE,
                                                       config.GRASP_ANCHOR_RATIOS,
                                                       [config.GRASP_POOL_SIZE, config.GRASP_POOL_SIZE],
                                                       pooled_feature_stride,
                                                       1,
                                                       config.GRASP_ANCHOR_ANGLES,
                                                       expanded_rect_normalized)

             post_nms_predictions, top_box_probabilities, pre_nms_predictions, pre_nms_scores = dataset.refine_results(grasping_probs[j], grasping_deltas[j],
                                                                                grasping_anchors, config, filter_mode='top_k', k=10)

             top_image_rois = np.append(top_image_rois, pre_nms_predictions, axis=0)
             top_image_roi_probabilities = np.append(top_image_roi_probabilities, pre_nms_scores, axis=0)

             for i, rect in enumerate(pre_nms_predictions):
                 x, y, w, h, theta = rect
                 x1 = x - w / 2
                 y1 = y - h / 2
                 theta %= 360
                 p = patches.Rectangle((x1, y1), w, h, angle=0, edgecolor=color,
                                       linewidth=1, facecolor='none')
                 t2 = mpl.transforms.Affine2D().rotate_deg_around(x, y, theta) + axs[1, 1].transData
                 p.set_transform(t2)
                 axs[1, 1].add_patch(p)


             for i, rect2 in enumerate(post_nms_predictions):
                 x, y, w, h, theta = rect2
                 x1 = x - w / 2
                 y1 = y - h / 2
                 theta %= 360
                 p2 = patches.Rectangle((x1, y1), w, h, angle=0, edgecolor=color,
                                       linewidth=1, facecolor='none')
                 t2 = mpl.transforms.Affine2D().rotate_deg_around(x, y, theta) + axs[1, 2].transData
                 p2.set_transform(t2)
                 axs[1, 2].add_patch(p2)

             for i, rect3 in enumerate(utils.denorm_boxes(grasping_anchors, config.IMAGE_SHAPE[:2], 'grasping_points')):
                 x, y, w, h, theta = rect3
                 x1 = x - w / 2
                 y1 = y - h / 2
                 theta %= 360
                 p3 = patches.Rectangle((x1, y1), w, h, angle=0, edgecolor=color,
                                        linewidth=0.5, facecolor='none')
                 t2 = mpl.transforms.Affine2D().rotate_deg_around(x, y, theta) + axs[0, 2].transData
                 p3.set_transform(t2)
                 axs[0, 2].add_patch(p3)

         for roi_instance in gt_grasp_boxes:
             for i, rect4 in enumerate(roi_instance):
                 if np.sum(rect4) == 0:
                     continue
                 x, y, w, h, theta = rect4
                 x1 = x - w / 2
                 y1 = y - h / 2
                 theta %= 360
                 p4 = patches.Rectangle((x1, y1), w, h, angle=0, edgecolor=(0, 1, 0),
                                        linewidth=0.5, facecolor='none')
                 t2 = mpl.transforms.Affine2D().rotate_deg_around(x, y, theta) + axs[1, 0].transData
                 p4.set_transform(t2)
                 axs[1, 0].add_patch(p4)

         fig.suptitle('Image path : ' + dataset.image_info[image_id]['path'] +
                      '\nAugmentations : [\'angle\', \'dx\', \'dy\', \'flip\', \'contrast\', \'Gaussian STD\'] => ' +
                      str(dataset.image_info[image_id]['augmentation']))
         plt.show(block=False)

import code;
code.interact(local=dict(globals(), **locals()))

#### Computing Grasp Detection Accuracy #####
#
# # mrcnn_model_path = 'models/Good_models/Training_SAMS_dataset_LR-div-5-div-10-HYBRID-weights/mask_rcnn_object_vs_background_0051.h5'
# # mask_grasp_model_path = 'models/grasp_and_mask20200905T1322/mask_rcnn_grasp_and_mask_0400.h5'
# # mask_grasp_model_path = 'models/mask_grasp_rcnn_attempt#1b/mask_rcnn_grasp_and_mask_0108.h5'
# # mask_grasp_model_path = 'models/colab_result_id#1/mask_rcnn_grasp_and_mask_0288.h5'
# mask_grasp_model_path = 'models/colab_result_id#1/mask_rcnn_grasp_and_mask_0288.h5'
#
# mask_grasp_model = modellib.MaskRCNN(mode="inference",
#                            config=inference_config,
#                            model_dir=MODEL_DIR, task=mode)
# mask_grasp_model.load_weights(mask_grasp_model_path, by_name=True)
#
# dataset = testing_dataset
# counter = 0
# num_positive_samples = 0
# for image_id in dataset.image_ids:
#      try:
#          original_image, image_meta, gt_class_id, gt_bbox, gt_mask, gt_grasp_boxes, gt_grasp_id =\
#              modellib.load_image_gt(dataset, inference_config,
#                                     image_id, use_mini_mask=True, image_type='rgbd', mode='mask_grasp_rcnn')
#      except:
#          continue
#
#      results = mask_grasp_model.detect([original_image], verbose=0, task=mode)
#      r = results[0]
#      if r['rois'].shape[0] > 0:
#          grasping_deltas = r['grasp_boxes']
#          grasping_probs = r['grasp_probs']
#          bboxes = r['rois']
#
#          top_image_rois = np.reshape(np.array([]), [-1, 5])
#          top_image_roi_probabilities = np.reshape(np.array([]), [-1])
#
#          for j, rect in enumerate(r['rois']):
#              color = dataset.generate_random_color()
#              if config.USE_EXPANDED_ROIS:
#                  rect = utils.expand_roi_by_percent(rect, percentage=config.GRASP_ROI_EXPAND_FACTOR,
#                                                     image_shape=config.IMAGE_SHAPE[:2])
#              expanded_rect_normalized = utils.norm_boxes(rect, config.IMAGE_SHAPE[:2])
#
#              y1, x1, y2, x2 = expanded_rect_normalized
#              w = abs(x2 - x1)
#              h = abs(y2 - y1)
#              ROI_shape = np.array([h, w])
#              pooled_feature_stride = np.array(ROI_shape/config.GRASP_POOL_SIZE)#.astype('uint8')
#
#              grasping_anchors = utils.generate_grasping_anchors(config.GRASP_ANCHOR_SIZE,
#                                                        config.GRASP_ANCHOR_RATIOS,
#                                                        [config.GRASP_POOL_SIZE, config.GRASP_POOL_SIZE],
#                                                        pooled_feature_stride,
#                                                        1,
#                                                        config.GRASP_ANCHOR_ANGLES,
#                                                        expanded_rect_normalized)
#              post_nms_predictions, top_box_probabilities, pre_nms_predictions, pre_nms_scores = dataset.refine_results(grasping_probs[j], grasping_deltas[j],
#                                                                                 grasping_anchors, config, filter_mode='top_k', k=1)
#              top_image_rois = np.append(top_image_rois, pre_nms_predictions, axis=0)
#              top_image_roi_probabilities = np.append(top_image_roi_probabilities, pre_nms_scores, axis=0)
#
#      # top_box = np.reshape(top_image_rois[np.argmax(top_image_roi_probabilities)], [-1, 5])
#      # top_prob = np.max(top_image_roi_probabilities)
#
#      counter += 1
#      true_positive = False
#
#      for top_box in top_image_rois:
#          true_positive = dataset.compute_jacquard_index_and_angle_difference([top_box], gt_grasp_boxes)
#          if true_positive:
#              num_positive_samples += 1
#              break
#      print(num_positive_samples / counter)
#      if not true_positive:
#          fig, ax = plt.subplots()
#          ax.imshow(original_image[:, :, :3])
#          for i, rect in enumerate(gt_grasp_boxes[0]):
#              x, y, w, h, theta = rect
#              x1 = x - w / 2
#              y1 = y - h / 2
#              theta %= 360
#              p = patches.Rectangle((x1, y1), w, h, angle=0, edgecolor=(1, 0, 1),
#                                    linewidth=1, facecolor='none')
#              t2 = mpl.transforms.Affine2D().rotate_deg_around(x, y, theta) + ax.transData
#              p.set_transform(t2)
#              ax.add_patch(p)
#          for i, rect in enumerate(top_image_rois):
#              x, y, w, h, theta = rect
#              x1 = x - w / 2
#              y1 = y - h / 2
#              theta %= 360
#              p = patches.Rectangle((x1, y1), w, h, angle=0, edgecolor=(0, 0, 0),
#                                    linewidth=1, facecolor='none')
#              t2 = mpl.transforms.Affine2D().rotate_deg_around(x, y, theta) + ax.transData
#              p.set_transform(t2)
#              ax.add_patch(p)
#
#          plt.show()
#
#
#
# import code;
#
# code.interact(local=dict(globals(), **locals()))

#### Miss Rate as a function of False Positives Per Image (FPPI) #####
#
# mask_grasp_model_path = 'models/colab_result_id#1/mask_rcnn_grasp_and_mask_0288.h5'
# mask_grasp_model = modellib.MaskRCNN(mode="inference",
#                            config=inference_config,
#                            model_dir=MODEL_DIR, task=mode)
# mask_grasp_model.load_weights(mask_grasp_model_path, by_name=True)
#
# dataset = testing_dataset
# counter = 0
# num_positive_samples = 0
# detection_thresholds = np.arange(-0.01, 1.1, 0.1)
# false_neg = np.zeros(detection_thresholds.shape)
# true_pos = np.zeros(detection_thresholds.shape)
# false_pos = np.zeros(detection_thresholds.shape)
# MR_list = np.zeros(detection_thresholds.shape)
# FPPI_list = np.zeros(detection_thresholds.shape)
# for image_id in np.random.choice(dataset.image_ids, 50):
#      try:
#          original_image, image_meta, gt_class_id, gt_bbox, gt_mask, gt_grasp_boxes, gt_grasp_id =\
#              modellib.load_image_gt(dataset, inference_config,
#                                     image_id, use_mini_mask=True, image_type='rgbd', mode='mask_grasp_rcnn')
#      except:
#          continue
#
#      results = mask_grasp_model.detect([original_image], verbose=0, task=mode)
#      r = results[0]
#      if r['rois'].shape[0] > 0:
#          grasping_deltas = r['grasp_boxes']
#          grasping_probs = r['grasp_probs']
#          bboxes = r['rois']
#
#          top_image_rois = np.reshape(np.array([]), [-1, 5])
#          top_image_roi_probabilities = np.reshape(np.array([]), [-1])
#
#          for j, rect in enumerate(r['rois']):
#              color = dataset.generate_random_color()
#              if config.USE_EXPANDED_ROIS:
#                  rect = utils.expand_roi_by_percent(rect, percentage=config.GRASP_ROI_EXPAND_FACTOR,
#                                                     image_shape=config.IMAGE_SHAPE[:2])
#              rect_normalized = utils.norm_boxes(rect, config.IMAGE_SHAPE[:2])
#
#              y1, x1, y2, x2 = rect_normalized
#              w = abs(x2 - x1)
#              h = abs(y2 - y1)
#              ROI_shape = np.array([h, w])
#              pooled_feature_stride = np.array(ROI_shape/config.GRASP_POOL_SIZE)#.astype('uint8')
#
#              grasping_anchors = utils.generate_grasping_anchors(config.GRASP_ANCHOR_SIZE,
#                                                        config.GRASP_ANCHOR_RATIOS,
#                                                        [config.GRASP_POOL_SIZE, config.GRASP_POOL_SIZE],
#                                                        pooled_feature_stride,
#                                                        1,
#                                                        config.GRASP_ANCHOR_ANGLES,
#                                                        rect_normalized)
#              post_nms_predictions, top_box_probabilities, pre_nms_predictions, pre_nms_scores = dataset.refine_results(grasping_probs[j], grasping_deltas[j],
#                                                                                 grasping_anchors, config, filter_mode='None')
#
#
#              top_image_rois = np.append(top_image_rois, pre_nms_predictions, axis=0)
#              top_image_roi_probabilities = np.append(top_image_roi_probabilities, pre_nms_scores, axis=0)
#
#      counter += 1
#      print (counter)
#      tp, fn, fp = dataset.compute_validation_metrics(top_image_rois, top_image_roi_probabilities, gt_grasp_boxes, detection_thresholds)
#
#      false_neg += fn
#      true_pos += tp
#      false_pos += fp
#      MR = fn / (fn + tp)
#      MR_list += MR
#
# MR_list /= counter
# # MR_list = false_neg / (false_neg + true_pos)
# FPPI_list = false_pos /counter
# fig, ax = plt.subplots()
# ax.set_xscale('log')
# ax.plot(FPPI_list, MR_list)
# ax.set_ylim([0, 1])
# plt.xlabel('false positives per image')
# plt.ylabel('miss rate')
# plt.grid(linestyle='--')
#
# plt.show(block=False)
#
# import code;
#
# code.interact(local=dict(globals(), **locals()))






































































     # grasp_boxes_subset = random.choices(gt_grasp_boxes, k=5)
     # for i, rect in enumerate(grasp_boxes_subset):
     #     rect = validating_dataset.bbox_convert_to_four_vertices([rect])
     #     p = patches.Polygon(rect[0], linewidth=1,edigecolor='g',facecolor='none')
     #     ax1.add_patch(p)
     # ax1.set_title(validating_dataset.image_info[image_id]['path'])
     #
     # mask_image = testing_dataset.get_mask_overlay(original_image[:,:,0:3], gt_mask, [1], 0.96)
     # ax2.imshow(mask_image)
     # plt.show()
     #
     # results = mrcnn_model.detect([original_image], verbose=1)
     # r = results[0]
     # mask_image = testing_dataset.get_mask_overlay(original_image[:, :, 0:3], r['masks'], r['scores'], 0.96)
     # bounding_boxes = r['rois']
     #
     # plt.imshow(mask_image);plt.show()
     #
     #
     # regions_to_analyze = []
     # for box in bounding_boxes:
     #     y1, x1, y2, x2 = box
     #     center_x = (x1 + x2) // 2
     #     center_y = (y1 + y2) // 2
     #     width = abs(x2 - x1) * 1.3
     #     height = abs(y2 - y1) * 1.3
     #
     #     x1 = int(center_x - (width//2))
     #     y1 = int(center_y - (height//2))
     #     x2 = int(center_x + (width//2))
     #     y2 = int(center_y + (height//2))
     #     # p = patches.Rectangle((x1, y1), width, height, linewidth=2,
     #     #                       alpha=0.7, linestyle="dashed",
     #     #                       edgecolor='b', facecolor='none')
     #     # ax.add_patch(p)
     #     # regions_to_analyze.append([y1, x1, y2, x2])
     #     image_crop = original_image[y1:y2, x1:x2, :]
     #
     #     input_image = np.zeros([image_crop.shape[0], image_crop.shape[1], 3])
     #     input_image[:, :, 0:2] = image_crop[:,:,0:2]
     #     input_image[:, :, 2] = image_crop[:,:,3]
     #     input_image = input_image.astype('uint8')
     #     #
     #     # ax.imshow(input_image)
     #     # plt.show(block=False)
     #     #
     #
     #     grasping_results = grasping_model.detect([input_image], verbose=1, task='grasping_points')
     #     r = grasping_results[0]
     #     post_nms_predictions, pre_nms_predictions = testing_dataset.refine_results(r, grasping_model.anchors, config)
     #
     #     fig, ax = plt.subplots()
     #     ax.imshow(input_image)
     #     for i, rect2 in enumerate(pre_nms_predictions):
     #         rect2 = testing_dataset.bbox_convert_to_four_vertices([rect2])
     #         p2 = patches.Polygon(rect2[0], linewidth=2, edgecolor=testing_dataset.generate_random_color(),
     #                              facecolor='none')
     #         ax.add_patch(p2)
     #         ax.set_title('Boxes post non-maximum supression')
     #     plt.show()