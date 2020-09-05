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
import skimage.io
from PIL import Image
import image_augmentation
import cv2
import math


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
    MEAN_PIXEL = np.array([181.6, 180.0, 180.9, 224.3]) # Jacquard_dataset dataset
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
    GRASP_ROI_EXPAND_FACTOR = 0.3
    USE_EXPANDED_ROIS = True
    MAX_GT_INSTANCES = 50
    TRAIN_ROIS_PER_IMAGE = 200
    NUM_GRASP_BOXES_PER_INSTANCE = 128
    NUM_GRASP_BOXES_PER_INSTANCE = 128
    ARIOU_NEG_THRESHOLD = 0.01
    ARIOU_POS_THRESHOLD = 0.1
    # LEARNING_RATE = 0.00002
    LEARNING_RATE = 0.0002 # was 0.00001
    # WEIGHT_DECAY = 0.000003
    WEIGHT_DECAY = 0.0001

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
    GRASP_LOSS_BETA = 10
    GRADIENT_CLIP_VALUE = 0.5 # was 0.1
    # TRAIN_BN=True


class GraspMaskRCNNInferenceConfig(GraspMaskRCNNConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class GraspMaskRCNNDataset(Dataset):

    def load_dataset(self, type = 'train_set', dataset_dir = 'New Graspable Objects Dataset', augmentation=False):
        self.add_class(GraspMaskRCNNConfig().NAME, 1, "object")
        dataset_path = os.path.join(dataset_dir, type)
        image_list = glob.glob(dataset_path + '/**/rgb/*.png', recursive=True)
        random.shuffle(image_list)
        id = 0
        for image in image_list:
            if 'jacquard_dataset' in dataset_dir:
                rgb_path = image
                depth_path = image.replace('rgb', 'depth')
                positive_grasp_points = image.replace('_RGB.png', '_grasps.txt').replace('rgb', 'grasp_rectangles')
                label_path = image.replace('_RGB.png', '_mask.png').replace('rgb', 'mask')
                self.add_image("grasp_and_mask", image_id=id, path=rgb_path,
                                depth_path=depth_path, label_path=label_path, positive_points=positive_grasp_points, augmentation = [])
                ## To do: Enable augmentation
            else:
                rgb_path = image
                depth_path = image.replace('rgb', 'depth').replace('table_', '').replace('floor_', '')
                label_path = image.replace('rgb', 'label').replace('table_', '').replace('floor_', '')
                self.add_image("grasp_and_mask", image_id = id, path = rgb_path,
                               label_path = label_path, depth_path = depth_path)
            id = id + 1

    def load_mask(self, image_id):
        mask_path = self.image_info[image_id]['label_path']
        mask_processed = self.process_label_image(mask_path)
        mask, class_ids = self.reshape_label_image(mask_processed)
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
            image = self.load_image(id)
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
        return masked_image

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

            # Theta re-adjustments:
            if theta > 90:
                theta  = theta - 180
            elif theta < -90:
                theta = theta + 180
            bounding_boxes.append([x, y, w, h, theta])
            class_ids.append(1)
        return np.array(bounding_boxes), np.array(class_ids)

    def apply_augmentation_to_box(self, boxes, class_ids, image_id, augmentation):
        # ['angle', 'dx', 'dy', 'flip']
        angle, dx, dy, flip_code = augmentation
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
            bounding_box_vertices = self.bbox_convert_to_four_vertices(bbox_5_dimensional)
        else:
            bounding_box_vertices, class_ids = self.load_ground_truth_bbox_files(image_id)
            bbox_5_dimensional = self.bbox_convert_to_five_dimension(bounding_box_vertices)
            if len(augmentation) != 0:
                bbox_5_dimensional, class_ids = self.apply_augmentation_to_box(bbox_5_dimensional, class_ids, image_id, augmentation)
            bounding_box_vertices = self.bbox_convert_to_four_vertices(bbox_5_dimensional)

        class_ids = np.array(class_ids)
        bounding_box_vertices = np.array(bounding_box_vertices)

        num_boxes = bounding_box_vertices.shape[0]
        extra = target_num_boxes - num_boxes
        if extra > 0:
            zero_pad_box = np.zeros((extra, ) + bounding_box_vertices[0].shape)
            bounding_box_vertices = np.append(bounding_box_vertices, zero_pad_box, axis=0)
            # was originally a tensor of zeros, now changed it to -1
            zero_pad_class_ids = np.ones(extra)*-1
            class_ids = np.append(class_ids, zero_pad_class_ids)
            zero_pad_5_dimensional = np.zeros((extra,) + bbox_5_dimensional[0].shape)
            bbox_5_dimensional = np.append(bbox_5_dimensional, zero_pad_5_dimensional, axis=0)
        else:
            extra_ix_to_remove = np.random.permutation(abs(extra))
            bounding_box_vertices = np.delete(bounding_box_vertices, extra_ix_to_remove, axis=0)
            class_ids = np.delete(class_ids, extra_ix_to_remove)
            bbox_5_dimensional = np.delete(bbox_5_dimensional, extra_ix_to_remove, axis=0)

        # shuffling the rectangles
        p = np.random.permutation(len(class_ids))

        # Reshaping boxes to have a shape [num_instances, num_grasp_boxes, ..]
        bounding_box_vertices = np.array([bounding_box_vertices[p]])
        bbox_5_dimensional = np.array([bbox_5_dimensional[p]])
        class_ids = np.array([class_ids[p].astype('int8')])
        class_ids = np.expand_dims(class_ids, axis=-1)

        return bounding_box_vertices, bbox_5_dimensional, class_ids

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
            if np.sign(theta) == 1:
                angle = angle + 90
                if angle > 180:
                    angle = angle - 360
            else:
                angle = angle + 90

            # rotational_matrix = cv2.getRotationMatrix2D((x, y), (angle + 90), scale)
            rotational_matrix = cv2.getRotationMatrix2D((x, y), (angle + 90), scale)
            original_points_rotated = cv2.transform(original_points_translated, rotational_matrix)
            rotated_points.append(original_points_rotated[0])

        return np.array(rotated_points)

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
        print(boxes.shape, filtered_boxes.shape)
        return [filtered_boxes, filtered_scores, boxes, scores]

    def refine_results(self, probabilities, deltas, anchors, config):
        '''
        Applies a NMS refinement algorithm on the proposals output by the RPN
        # '''
        # import code;
        # code.interact(local=dict(globals(), **locals()))
        deltas = deltas * config.GRASP_BBOX_STD_DEV
        mode = 'mask_grasp_rcnn'
        all_boxes = utils.apply_box_deltas(anchors, deltas, mode,
                                           len(config.GRASP_ANCHOR_ANGLES))

        all_boxes = utils.denorm_boxes(all_boxes, config.IMAGE_SHAPE[:2], mode='grasping_points')
        all_boxes[:,-1] = all_boxes[:,-1]%360

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

        sorting_ix = np.argsort(probabilities[:, 1])[::-1][:10]

        # top_boxes = all_boxes[probabilities[:,1] > config.DETECTION_MIN_CONFIDENCE]
        # top_box_probabilities = probabilities[probabilities[:,1] > config.DETECTION_MIN_CONFIDENCE]
        top_boxes = all_boxes[probabilities[:,1] > 0.80]
        top_box_probabilities = probabilities[probabilities[:,1] > 0.80]
        # top_boxes = all_boxes[sorting_ix]
        # top_box_probabilities = probabilities[sorting_ix]
        top_boxes, top_box_probabilities, pre_nms_boxes, pre_nms_scores = self.orient_box_nms(top_boxes, top_box_probabilities, config)

        return top_boxes, pre_nms_boxes

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
training_dataset.load_dataset(dataset_dir='../../../Datasets/jacquard_dataset_resized_new')
training_dataset.prepare()

validating_dataset = GraspMaskRCNNDataset()
validating_dataset.load_dataset(dataset_dir='../../../Datasets/jacquard_dataset_resized_new', type='val_set')
validating_dataset.prepare()

testing_dataset = GraspMaskRCNNDataset()
testing_dataset.load_dataset(dataset_dir='../../../Datasets/jacquard_dataset_resized_new', type='test_set')
testing_dataset.prepare()

config = GraspMaskRCNNConfig()
# channel_means = np.array(training_dataset.get_channel_means())
# config.MEAN_PIXEL = np.around(channel_means, decimals = 1)
# config.display()
inference_config = GraspMaskRCNNInferenceConfig()
# grasping_inference_config = GraspingInferenceConfig()
MODEL_DIR = "models"
mode = "mask_grasp_rcnn"

#### TRAINING #####
# # COCO_MODEL_PATH = os.path.join("models", "mask_rcnn_object_vs_background_100_heads_50_all.h5")
# # COCO_MODEL_PATH = os.path.join("models", "mask_rcnn_object_vs_background_HYBRID-50_head_50_all.h5")
# # COCO_MODEL_PATH = 'models/Good_models/Training_SAMS_dataset_LR-same-div-2-HYBRID-weights.h5'
# COCO_MODEL_PATH = os.path.join("models", "Good_models", "Training_SAMS_dataset_LR-same-div-2-HYBRID-weights", "SAMS_DATASET_TRAINING_REFERENCE.h5")
# # COCO_MODEL_PATH = os.path.join("models", "mask_rcnn_coco.h5")
# model = modellib.MaskRCNN(mode="training", config=config,
#                              model_dir=MODEL_DIR, task='mask_grasp_rcnn')
#
#
#
# # model.load_weights(COCO_MODEL_PATH, by_name=True,
# #                       exclude=["conv1", "mrcnn_class_logits", "mrcnn_bbox_fc",
# #                                "mrcnn_bbox", "mrcnn_mask"])
# model.load_weights(COCO_MODEL_PATH, by_name=True)
#
# # model.train(training_dataset, validating_dataset,
# #                learning_rate=config.LEARNING_RATE,
# #                epochs=200,
# #                layers=r"(conv1)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(grasp\_.*)",
# #                task=mode)
#
# model.train(training_dataset, validating_dataset,
#                learning_rate=config.LEARNING_RATE,
#                epochs=400,
#                layers=r"(grasp\_.*)",
#                task=mode)
#
#
# # model.train(training_dataset, validating_dataset,
# #                 learning_rate=config.LEARNING_RATE/10,
# #                 epochs=300,
# #                 layers="all",
# #                 task=mode)
#
# model_path = os.path.join(MODEL_DIR, "mask_grasp_rcnn.h5")
# model.keras_model.save_weights(model_path)

##### TESTING #####

# mrcnn_model_path = 'models/Good_models/Training_SAMS_dataset_LR-div-5-div-10-HYBRID-weights/mask_rcnn_object_vs_background_0051.h5'
# mask_grasp_model_path = 'models/mask_grasp_rcnn_attempt#1b/mask_rcnn_grasp_and_mask_0200.h5'
# mask_grasp_model_path = 'models/mask_grasp_rcnn_attempt#1b/mask_rcnn_grasp_and_mask_0108.h5'
mask_grasp_model_path = 'models/colab_result_id#1/mask_rcnn_grasp_and_mask_0500.h5'


mask_grasp_model = modellib.MaskRCNN(mode="inference",
                           config=inference_config,
                           model_dir=MODEL_DIR, task=mode)
mask_grasp_model.load_weights(mask_grasp_model_path, by_name=True)

# grasping_model_path = os.path.join(MODEL_DIR, 'colab_result_id#1',"train_#12.h5")
# grasping_model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
#                               config=inference_config, task="grasping_points")
# grasping_model.load_weights(grasping_model_path, by_name=True)

dataset = validating_dataset
image_ids = random.choices(dataset.image_ids, k=15)

for image_id in image_ids:
     original_image, image_meta, gt_class_id, gt_bbox, gt_mask, gt_grasp_boxes, gt_grasp_id =\
         modellib.load_image_gt(dataset, inference_config,
                                image_id, use_mini_mask=True, image_type='rgbd', mode='mask_grasp_rcnn')

     results = mask_grasp_model.detect([original_image], verbose=1, task=mode)
     r = results[0]
     if r['rois'].shape[0] > 0:
         # visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
         #                            dataset.class_names, r['scores'], show_bbox=True, thresh = 0)
         mask_image = testing_dataset.get_mask_overlay(original_image[:, :, 0:3], r['masks'], r['scores'], 0)
         grasping_deltas = r['grasp_boxes']
         grasping_probs = r['grasp_probs']

         fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))
         ax1.imshow(mask_image)

         bboxes = r['rois']

         for bbox in bboxes:
             expanded_rect = utils.expand_roi_by_percent(bbox, percentage=config.GRASP_ROI_EXPAND_FACTOR,
                                                         image_shape=config.IMAGE_SHAPE[:2])
             y1, x1, y2, x2 = expanded_rect
             p = patches.Rectangle((x1, y1), (x2-x1), (y2-y1), angle=0, edgecolor=dataset.generate_random_color(), linewidth=2, facecolor='none')
             ax1.add_patch(p)
             break



         activations = mask_grasp_model.run_graph([original_image], [
             ("grasp_conv2", mask_grasp_model.keras_model.get_layer("grasp_conv2").output),  # for resnet100
             ("grasp_conv3", mask_grasp_model.keras_model.get_layer("grasp_conv3").output),
             ("grasp_conv4", mask_grasp_model.keras_model.get_layer("grasp_conv4").output),
             ("roi_align_grasp", mask_grasp_model.keras_model.get_layer("roi_align_grasp").output),
         ])

         ax2.imshow(activations["roi_align_grasp"][0][0][:, :, 0]);

         plt.show(block=False)

import code;

code.interact(local=dict(globals(), **locals()))















