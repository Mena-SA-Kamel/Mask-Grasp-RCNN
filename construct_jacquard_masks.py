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

dataset_path = 'D:/Datasets/jacquard_dataset_resized_new'
source_path = 'D:/Datasets/Jacquard-Dataset'

folder_names = ['train_set', 'val_set', 'test_set']
for dataset_type in folder_names:
    image_folder = os.path.join(dataset_path, dataset_type, 'rgb')
    image_names = os.listdir(image_folder)
    for image_name in image_names:
        mask_image_name = image_name.replace('_RGB', '_mask')
        mask_path = os.path.join(source_path, image_name.split('_')[1], image_name.replace('_RGB', '_mask'))

        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask = mask.astype(int)
        mask = skimage.color.gray2rgb(mask)

        resized_mask, window, scale, padding, crop = utils.resize_image(
            mask,
            min_dim=IMAGE_MIN_DIM,
            min_scale=IMAGE_MIN_SCALE,
            max_dim=IMAGE_MAX_DIM,
            mode=IMAGE_RESIZE_MODE)
        resized_mask = skimage.color.rgb2gray(resized_mask)
        resized_mask = np.interp(resized_mask, (resized_mask.min(), resized_mask.max()), (0, 1))*255
        resized_mask = resized_mask.astype('uint8')
        # import code;
        #
        # code.interact(local=dict(globals(), **locals()))

        target_path = os.path.join(dataset_path, dataset_type, 'mask', mask_image_name)
        imageio.imwrite(target_path, resized_mask)