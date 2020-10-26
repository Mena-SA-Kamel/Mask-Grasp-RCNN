import numpy as np
import os
import mrcnn.utils as utils
import glob
import matplotlib.pyplot as plt
import skimage.io
import cv2
import math
import matplotlib
import imageio
from PIL import Image


IMAGE_MIN_DIM = 640
IMAGE_MAX_DIM = 640
IMAGE_RESIZE_MODE = "square"
IMAGE_MIN_SCALE = 0

dataset_path = 'cornell_grasping_dataset'
source_path = 'cornell_grasping_dataset'


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def generate_cornell_mask(image_path, gb_mapping_file_path):
    try:
        background_images_path = os.path.join('cornell_grasping_dataset', 'background_images')
        gb_mapping_file_path = os.path.join(background_images_path, 'backgroundMapping.txt')
        with open(gb_mapping_file_path) as f:
            bg_image_mapping = f.read()
        image_name = image_path.split('\\')[-1]
        background_image_name = bg_image_mapping.split(image_name)[1].split('\np')[0].strip(' ').replace('_', '').replace('\n', '')
        background_path = os.path.join(background_images_path, background_image_name)

        background_image = skimage.io.imread(background_path)
        image = skimage.io.imread(image_path)

        image_gray = rgb2gray(image)
        background_image_gray = rgb2gray(background_image)

        dilated_img = cv2.dilate(image_gray, np.ones((7, 7), np.uint8))
        diff_img = 255 - cv2.absdiff(image_gray.astype('uint8'), dilated_img.astype('uint8'))
        image_gray_new = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        dilated_img = cv2.dilate(background_image_gray, np.ones((7, 7), np.uint8))
        diff_img = 255 - cv2.absdiff(background_image_gray.astype('uint8'), dilated_img.astype('uint8'))
        background_image_gray_new = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                                  dtype=cv2.CV_8UC1)

        preliminary_mask = cv2.absdiff(background_image_gray_new, image_gray_new)
        preliminary_mask_ennhanced = cv2.blur(preliminary_mask, (5, 5))

        mask = np.zeros(image_gray.shape)
        # Specifying relevant bounds in the image
        mask[180:480, 105:513] = (preliminary_mask_ennhanced > 20)[180:480, 105:513]
    except:
        import code;
        code.interact(local=dict(globals(), **locals()))
    return mask

gb_mapping_file_path = 'cornell_grasping_dataset/background_images/backgroundMapping.txt'

folder_names = ['train_set', 'val_set', 'test_set']
for dataset_type in folder_names:
    image_folder = os.path.join(dataset_path, dataset_type, 'rgb')
    image_names = os.listdir(image_folder)
    for image_name in image_names:
        image_path = os.path.join(image_folder, image_name)
        mask = generate_cornell_mask(image_path, gb_mapping_file_path)
        mask = mask.astype('uint8')
        target_path = os.path.join(dataset_path, dataset_type, 'mask', image_name)
        imageio.imwrite(target_path, mask)

