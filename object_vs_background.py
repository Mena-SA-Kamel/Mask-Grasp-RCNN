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

from grasping_points import GraspingInferenceConfig, GraspingPointsDataset

# Extending the mrcnn.config class to update the default variables
class ObjectVsBackgroundConfig(Config):
    NAME = 'object_vs_background'
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
    MEAN_PIXEL = np.array([122.6, 113.4 , 118.2, 135.8]) # SAMS dataset
    MAX_GT_INSTANCES = 50

class InferenceConfig(ObjectVsBackgroundConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class ObjectVsBackgroundDataset(Dataset):

    def load_dataset(self, type = 'train_set', dataset_dir = 'New Graspable Objects Dataset'):
        self.add_class(ObjectVsBackgroundConfig().NAME, 1, "object")
        dataset_path = os.path.join(dataset_dir, type)
        image_list = glob.glob(dataset_path + '/**/label/*.png', recursive=True)
        random.shuffle(image_list)
        id = 0
        for image in image_list:
            rgb_path = image.replace('label', 'rgb').replace('table_', '').replace('floor_', '')
            depth_path = image.replace('label', 'depth').replace('table_', '').replace('floor_', '')
            label_path = image

            # if 'ocid_dataset' in dataset_dir:
            #     rgb_path = rgb_path.replace('table_', '').replace('floor_', '')
            #     depth_path = depth_path.replace('table_', '').replace('floor_', '')
            self.add_image("object_vs_background", image_id = id, path = rgb_path,
                           label_path = label_path, depth_path = depth_path)
            id = id + 1

    def load_mask(self, image_id):
        mask_path = self.image_info[image_id]['label_path']
        mask_original = self.load_label_from_file(mask_path)
        mask_processed = self.process_label_image(mask_path)
        mask, class_ids = self.reshape_label_image(mask_processed)
        return mask, np.array(class_ids)

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

    # def get_mask_overlay(self, image, masks):
    #     num_masks = masks.shape[-1]
    #     colors = visualize.random_colors(num_masks)
    #     for i in list(range(num_masks)):
    #         mask = masks[:, :, i]
    #         image = visualize.apply_mask(image, mask, colors[i])
    #     image = image.astype('uint8')
    #     return image

    def get_mask_overlay(self, image, masks, scores, threshold=0.90):
        num_masks = masks.shape[-1]
        colors = visualize.random_colors(num_masks)
        masked_image = np.copy(image)
        for i in list(range(num_masks)):
            if (scores[i] < threshold):
                continue
            mask = masks[:, :, i]
            masked_image = visualize.apply_mask(masked_image, mask, colors[i])
        masked_image = np.array(image, dtype='uint8')
        return masked_image

# SETUP ##

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

training_dataset = ObjectVsBackgroundDataset()
# training_dataset.construct_dataset(dataset_dir = '../../../Datasets/SAMS-Dataset')
training_dataset.load_dataset('train_set', dataset_dir='sams_dataset')
training_dataset.prepare()

validating_dataset = ObjectVsBackgroundDataset()
validating_dataset.load_dataset('val_set', dataset_dir='sams_dataset')
validating_dataset.prepare()

testing_dataset = ObjectVsBackgroundDataset()
testing_dataset.load_dataset('test_set', dataset_dir='sams_dataset')
testing_dataset.prepare()

config = ObjectVsBackgroundConfig()
# channel_means = np.array(training_dataset.get_channel_means())
# config.MEAN_PIXEL = np.around(channel_means, decimals = 1)
# config.display()
inference_config = InferenceConfig()
grasping_inference_config = GraspingInferenceConfig()

#
# # # ##### TRAINING #####
# #
# MODEL_DIR = "models"
# # COCO_MODEL_PATH = os.path.join("models", "mask_rcnn_object_vs_background_100_heads_50_all.h5")
# COCO_MODEL_PATH = os.path.join("models", "mask_rcnn_object_vs_background_HYBRID-50_head_50_all.h5")
# model = modellib.MaskRCNN(mode="training", config=config,
#                              model_dir=MODEL_DIR)
#
# # model.load_weights(COCO_MODEL_PATH, by_name=True,
# #                       exclude=["conv1", "mrcnn_class_logits", "mrcnn_bbox_fc",
# #                                "mrcnn_bbox", "mrcnn_mask"])
# model.load_weights(COCO_MODEL_PATH, by_name=True)
#
# model.train(training_dataset, validating_dataset,
#                learning_rate=config.LEARNING_RATE,
#                epochs=50,
#                layers=r"(conv1)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)")
#
#
# # model.train(training_dataset, validating_dataset,
# #                 learning_rate=config.LEARNING_RATE/10,
# #                 epochs=250,
# #                 layers="all")
#
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_object_vs_background_HYBRID-Weights_SAMS-50_head_50_all.h5")
# model.keras_model.save_weights(model_path)

# # # # ##### TESTING #####

MODEL_DIR = "models"
model_path = 'models/Good_models/Training_SAMS_dataset_LR-div-5-div-10-HYBRID-weights/mask_rcnn_object_vs_background_0051.h5'
model = modellib.MaskRCNN(mode="inference",
                           config=inference_config,
                           model_dir=MODEL_DIR)
model.load_weights(model_path, by_name=True)

grasping_model_path = os.path.join(MODEL_DIR, 'colab_result_id#1',"train_#11c.h5")
grasping_model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=grasping_inference_config, task="grasping_points")
grasping_model.load_weights(grasping_model_path, by_name=True)
grasping_dataset_object = GraspingPointsDataset()

dataset = testing_dataset
image_ids = random.choices(dataset.image_ids, k=15)
for image_id in image_ids:
     original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
         modellib.load_image_gt(dataset, inference_config,
                                image_id, use_mini_mask=False)
     # visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
     #                        dataset.class_names, figsize=(8, 8))

     print(dataset.image_info[image_id]['label_path'])
     results = model.detect([original_image], verbose=1)
     r = results[0]
     image = testing_dataset.get_mask_overlay(original_image[:,:,0:3], r['masks'], r['scores'], 0.96)

     bounding_boxes = r['rois']

     regions_to_analyze = []
     for box in bounding_boxes:
         y1, x1, y2, x2 = box
         center_x = (x1 + x2) // 2
         center_y = (y1 + y2) // 2
         width = abs(x2 - x1) * 1.3
         height = abs(y2 - y1) * 1.3

         x1 = int(center_x - (width//2))
         y1 = int(center_y - (height//2))
         x2 = int(center_x + (width//2))
         y2 = int(center_y + (height//2))
         # p = patches.Rectangle((x1, y1), width, height, linewidth=2,
         #                       alpha=0.7, linestyle="dashed",
         #                       edgecolor='b', facecolor='none')
         # ax.add_patch(p)
         # regions_to_analyze.append([y1, x1, y2, x2])
         image_crop = original_image[y1:y2, x1:x2, :]

         input_image = np.zeros([image_crop.shape[0], image_crop.shape[1], 3])
         input_image[:, :, 0:2] = image_crop[:,:,0:2]
         input_image[:, :, 2] = image_crop[:,:,3]
         input_image = input_image.astype('uint8')
         #
         # ax.imshow(input_image)
         # plt.show(block=False)
         #

         grasping_results = grasping_model.detect([input_image], verbose=1, task='grasping_points')
         r = grasping_results[0]
         post_nms_predictions, pre_nms_predictions = grasping_dataset_object.refine_results(r, grasping_model.anchors, grasping_model.config)

         fig, ax = plt.subplots()
         ax.imshow(input_image)
         for i, rect2 in enumerate(pre_nms_predictions):
             rect2 = grasping_dataset_object.bbox_convert_to_four_vertices([rect2])
             p2 = patches.Polygon(rect2[0], linewidth=2, edgecolor=grasping_dataset_object.generate_random_color(),
                                  facecolor='none')
             ax.add_patch(p2)
             ax.set_title('Boxes post non-maximum supression')
         plt.show()


     # plt.show()
     # visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
     #                            dataset.class_names, r['scores'],show_bbox=True, thresh = 0.95)