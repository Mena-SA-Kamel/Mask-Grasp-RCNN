from mrcnn.config import Config
from mrcnn.utils import Dataset
import mrcnn.utils as utils
import mrcnn.model as modellib
from mrcnn import visualize

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
from random import randrange
import image_augmentation

os.environ["PATH"] += os.pathsep + 'D:/Software/graphviz-2.38/release/bin'

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
    # MEAN_PIXEL = np.array([112.7, 112.1, 113.5, 123.5]) # Added a 4th channel. Modify the mean of the pixel depth
    MEAN_PIXEL = np.array([181.2, 179.9, 180.7, 229.4]) # Jacquard Dataset image mean, based on 1000 random images
    MAX_GT_INSTANCES = 1000
    RPN_GRASP_ANGLES = [-60, -30, 0, 30, 60]
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2, 1])
    RPN_TRAIN_ANCHORS_PER_IMAGE = 1000
    RPN_OHEM_NUM_SAMPLES = 256
    LEARNING_RATE = 0.005
    LEARNING_MOMENTUM = 0.9
    NUM_AUGMENTATIONS = 15

class InferenceConfig(GraspingPointsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class GraspingPointsDataset(Dataset):

    def load_dataset(self, type='train_set', dataset_dir='cornell_grasping_dataset', augmentation=False):
        self.add_class(GraspingPointsConfig().NAME, 1, "graspable_location")
        dataset_path = os.path.join(dataset_dir, type)
        image_list = glob.glob(dataset_path + '/**/rgb/*.png', recursive=True)
        random.shuffle(image_list)

        id = 0
        for image in image_list:
            if 'jacquard_dataset' in dataset_dir:
                rgb_path = image
                depth_path = image.replace('rgb', 'depth')
                positive_grasp_points = image.replace('_RGB.png', '_grasps.txt').replace('rgb', 'grasp_rectangles')

                self.add_image("grasping_points", image_id=id, path=rgb_path,
                                depth_path=depth_path, positive_points=positive_grasp_points, augmentation = [])

                if augmentation:
                    # Augmentation will create hypothetical paths that load_image_gt() uses to construct the different
                    # image variants
                    image_name = image.split('\\')[-1].strip('_RGB.png')
                    for i in range(config.NUM_AUGMENTATIONS):
                        self.add_image("grasping_points", image_id=id, path=rgb_path,
                                       depth_path=depth_path, positive_points=positive_grasp_points,
                                       augmentation = self.generate_augmentations())
                        id = id + 1

            elif 'cornell_grasping_dataset' in dataset_dir:
                rgb_path = image
                depth_path = image.replace('rgb', 'depth')
                positive_grasp_points = image.replace('r.png', 'cpos.txt').replace('rgb', 'grasp_rectangles')
                negative_grasp_points = image.replace('r.png', 'cneg.txt').replace('rgb', 'grasp_rectangles')
                self.add_image("grasping_points", image_id = id, path = rgb_path,
                               depth_path = depth_path, positive_points = positive_grasp_points,
                               negative_points = negative_grasp_points)
            id = id + 1

    def generate_augmentations(self):
        augmentation_types = ['angle', 'dx', 'dy', 'flip']
        augmentations = random.sample(list(augmentation_types), 2)
        augmentations_list = np.zeros(4)

        if 'angle' in augmentations:
            angle = randrange(-30, 30)
            augmentations_list[0] = angle

        if 'dx' in augmentations:
            dx = randrange(0, 100)
            augmentations_list[1] = dx

        if 'dy' in augmentations:
            dy = randrange(0, 100)
            augmentations_list[2] = dy

        if 'flip' in augmentations:
            flip = randrange(0, 3) # 0 -> vertical flip, 1 -> horizontal flip, 2 -> no flip
            augmentations_list[3] = flip

        return augmentations_list

    def apply_augmentation_to_image(self, image, augmentation):
        # ['angle', 'dx', 'dy', 'flip']
        angle, dx, dy, flip_code = augmentation

        rgbd_image = image_augmentation.rotate_image(image, angle, scale=1)
        rgbd_image = image_augmentation.translate_image(rgbd_image, dx, dy)
        if flip_code != 2:
            rgbd_image = image_augmentation.flip_image(rgbd_image, int(flip_code))
        return rgbd_image

    def apply_augmentation_to_box(self, boxes, class_ids, image_id, augmentation):
        # ['angle', 'dx', 'dy', 'flip']
        angle, dx, dy, flip_code = augmentation
        image_shape = plt.imread(self.image_info[image_id]['path']).shape[:2]
        transformed_boxes, class_ids = image_augmentation.rotate_bboxes(boxes, angle, image_shape, class_ids)
        transformed_boxes, class_ids = image_augmentation.translate_bbox(transformed_boxes, dx, dy, image_shape, class_ids)
        if flip_code != 2:
            transformed_boxes = image_augmentation.flip_bbox(transformed_boxes, int(flip_code), image_shape)

        return transformed_boxes, class_ids

    def load_image(self, image_id, augmentation=[]):
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
        rgbd_image = np.zeros([image.shape[0], image.shape[1], 4])
        rgbd_image[:, :, 0:3] = image
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
        # img_rescale_int = img_rescale.astype('uint8')
        img_rescale_int = img_rescale
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

    def construct_jacquard_dataset(self, train = 0.75, val = 0.15, dataset_dir = '../../../Datasets/Jacquard-Dataset'):
        dataset_folder = dataset_dir.replace('Jacquard-Dataset', 'Jacquard-Dataset-Formatted')
        folder_names = ['train_set', 'val_set', 'test_set']
        subfolder_names = ['depth', 'rgb', 'grasp_rectangles']
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
            for folder_name in folder_names:
                os.makedirs(os.path.join(dataset_folder, folder_name))
                for subfolder in subfolder_names:
                    os.makedirs(os.path.join(dataset_folder, folder_name, subfolder))
        image_paths = glob.glob(dataset_dir + '/**/*_RGB.png', recursive=True)
        train_set, val_set, test_set = self.split_dataset(image_paths, train, val)

        i = 0
        for dataset_paths in [train_set, val_set, test_set]:
            for image_path in dataset_paths:
                source_rgb_path = image_path
                source_depth_path = image_path.replace('_RGB.png', '_stereo_depth.tiff')
                source_rect_path = image_path.replace('_RGB.png', '_grasps.txt')

                target_rgb_path = os.path.join(dataset_folder, folder_names[i], 'rgb', source_rgb_path.split('\\')[-1])
                target_depth_path = os.path.join(dataset_folder, folder_names[i], 'depth',
                                                 source_depth_path.split('\\')[-1])
                target_rect_path = os.path.join(dataset_folder, folder_names[i], 'grasp_rectangles',
                                                    source_rect_path.split('\\')[-1])

                copyfile(source_rgb_path, target_rgb_path)
                copyfile(source_depth_path, target_depth_path)
                copyfile(source_rect_path, target_rect_path)
            i = i + 1

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
            # Minimum dimension allowed is 2 pixels
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
        bad_rectangle = False

        class_id = 0
        for bounding_box_class in [negative_rectangles, positive_rectangles]:
            i = 0
            for bounding_box in bounding_box_class:
                try:
                    x = int(float(bounding_box.split(' ')[0]))
                    y = int(float(bounding_box.split(' ')[1]))
                    vertices.append([x, y])
                except:
                    # print("ERROR : ValueError: cannot convert float NaN to integer")
                    # import code;
                    # code.interact(local=dict(globals(), **locals()))
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

            rotational_matrix = cv2.getRotationMatrix2D((x, y), angle + 90, scale)
            original_points_rotated = cv2.transform(original_points_translated, rotational_matrix)
            rotated_points.append(original_points_rotated[0])

        return np.array(rotated_points)


    def bbox_convert_to_five_dimension(self, bounding_box_vertices, image_id=0):
        bbox_5_dimensional = []
        for bounding_box in bounding_box_vertices:
            x, y, w, h, theta = self.get_five_dimensional_box(bounding_box)
            bbox_5_dimensional.append([x, y, w, h, theta])
            # self.visualize_bbox(image_id, bounding_box, 1, [x, y, w, h, theta])
        bbox_5_dimensional = np.array(bbox_5_dimensional)
        return bbox_5_dimensional


    def load_bounding_boxes(self, image_id, augmentation=[]):
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

        class_ids = np.array(class_ids)
        bounding_box_vertices = np.array(bounding_box_vertices)

        # shuffling the rectangles
        p = np.random.permutation(len(class_ids))
        return bounding_box_vertices[p], bbox_5_dimensional[p], class_ids[p].astype('uint8')

    def clip_oriented_boxes_to_boundary(self, boxes, window):
        """
        boxes: [N, (x, y, w, h, theta)]
        window: [4] in the form y1, x1, y2, x2
        """
        # Split
        x = boxes[:, 0]
        y = boxes[:, 1]
        wx1, wy1, wx2, wy2 = window

        # Clip
        x = np.maximum(np.minimum(x, wx2), wx1)
        y = np.maximum(np.minimum(y, wy2), wy1)

        boxes[:, 0] = x
        boxes[:, 1] = y
        return boxes

    def orient_box_nms(self, boxes, scores):
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
            ix_to_remove = np.where(np.delete(arIoU, ix) > 0.3)[0]
            filtered_boxes = np.delete(filtered_boxes, ix_to_remove, axis=0)
            filtered_scores = np.delete(filtered_scores, ix_to_remove)
            box_areas = np.delete(box_areas, ix_to_remove)
            box_vertices = np.delete(box_vertices, ix_to_remove, axis=0)
            thetas = np.delete(thetas, ix_to_remove)
            ix += 1
        print(boxes.shape, filtered_boxes.shape)
        return [filtered_boxes, filtered_scores, boxes, scores]


    def refine_results(self, results, anchors):
        '''
        Applies a NMS refinement algorithm on the proposals output by the RPN
        '''
        probabilities = results['scores'][0] # bg prob, fg prob
        deltas = results['refinements'][0]
        mode = 'grasping_points'
        all_boxes = utils.apply_box_deltas(anchors, deltas, mode,
                                           len(config.RPN_GRASP_ANGLES))
        # Filter out boxes with center coordinates out of the image
        radius = ((0.5 * all_boxes[:, 2]) ** 2 + (0.5 * all_boxes[:, 3]) ** 2) ** 0.5

        invalid_x = np.where(all_boxes[:, 0] + radius > config.IMAGE_SHAPE[1])[0]
        all_boxes = np.delete(all_boxes, invalid_x, axis = 0)
        probabilities = np.delete(probabilities, invalid_x, axis = 0)
        radius = np.delete(radius, invalid_x)

        invalid_x = np.where(all_boxes[:, 0] - radius < 0)[0]
        all_boxes = np.delete(all_boxes, invalid_x, axis=0)
        probabilities = np.delete(probabilities, invalid_x, axis=0)
        radius = np.delete(radius, invalid_x)

        invalid_y = np.where(all_boxes[:, 1] + radius > config.IMAGE_SHAPE[0])[0]
        all_boxes = np.delete(all_boxes, invalid_y, axis = 0)
        probabilities = np.delete(probabilities, invalid_y, axis = 0)
        radius = np.delete(radius, invalid_y)

        invalid_y = np.where(all_boxes[:, 1] - radius < 0)[0]
        all_boxes = np.delete(all_boxes, invalid_y, axis=0)
        probabilities = np.delete(probabilities, invalid_y, axis=0)

        sorting_ix = np.argsort(probabilities[:, 1])[::-1][:20]
        # top_boxes = all_boxes[probabilities[:,1] > 0.9]
        # top_box_probabilities = probabilities[probabilities[:,1] > 0.9]
        top_boxes = all_boxes[sorting_ix]
        top_box_probabilities = probabilities[sorting_ix]
        top_boxes, top_box_probabilities, pre_nms_boxes, pre_nms_scores = self.orient_box_nms(top_boxes, top_box_probabilities)

        return top_boxes, pre_nms_boxes

    def get_channel_means(self):
        depth_sum = 0
        red_sum = 0
        green_sum = 0
        blue_sum = 0
        for id in self.image_ids[:1000]:
            image = self.load_image(id)
            mean_red = np.mean(image[:, :, 0])
            red_sum = red_sum + mean_red
            mean_green = np.mean(image[:, :, 1])
            green_sum = green_sum + mean_green
            mean_blue = np.mean(image[:, :, 2])
            blue_sum = blue_sum + mean_blue
            mean_depth = np.mean(image[:, :, 3])
            depth_sum = depth_sum + mean_depth

        red_channel_mean = red_sum / len(self.image_ids[:1000])
        blue_channel_mean = green_sum / len(self.image_ids[:1000])
        green_channel_mean = blue_sum / len(self.image_ids[:1000])
        depth_channel_mean = depth_sum / len(self.image_ids[:1000])
        print(red_channel_mean, green_channel_mean, blue_channel_mean, depth_channel_mean)
        return red_channel_mean, green_channel_mean, blue_channel_mean, depth_channel_mean

# SETUP ##
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Directory to save logs and trained model
MODEL_DIR = "models"
MASKRCNN_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")
config = GraspingPointsConfig()
inference_config = InferenceConfig()
DEVICE = "/gpu:0"
TEST_MODE = "inference"
mode = "grasping_points"

training_dataset = GraspingPointsDataset()
# training_dataset.construct_jacquard_dataset()
# training_dataset.load_dataset()
training_dataset.load_dataset(dataset_dir='../../../Datasets/jacquard_dataset', augmentation=True)
training_dataset.prepare()
# channel_means = np.array(training_dataset.get_channel_means())
# config.MEAN_PIXEL = np.around(channel_means, decimals = 1)

validating_dataset = GraspingPointsDataset()
validating_dataset.load_dataset(dataset_dir='../../../Datasets/jacquard_dataset', type='val_set', augmentation=True)
# validating_dataset.load_dataset(type='val_set')
validating_dataset.prepare()
#
# # # Create model in training mode
# with tf.device(DEVICE):
#     model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR,
#                               config=config, task="grasping_points")
# tf.keras.utils.plot_model(
#         model.keras_model, to_file='model.png', show_shapes=True, show_layer_names=True
#     )

# # Load weights

# # weights_path = MASKRCNN_MODEL_PATH
# weights_path = os.path.join(MODEL_DIR, "mask_rcnn_grasping_points_0060.h5")
# print("Loading weights ", weights_path)
# model.load_weights(weights_path, by_name=True)
# # model.load_weights(weights_path, by_name=True,
# #                        exclude=["conv1", "rpn_model", "rpn_class_logits",
# #                                 "rpn_class ", "rpn_bbox "])
# # model.train(training_dataset, validating_dataset,
# #                learning_rate=config.LEARNING_RATE,
# #                epochs=50,
# #                layers=r"(conv1)|(grasp_rpn\_.*)|(fpn\_.*)",
# #                task=mode)
#
# model.train(training_dataset, validating_dataset,
#                learning_rate=config.LEARNING_RATE,
#                epochs=30,
#                layers="all",
#                task=mode)
#
# model.train(training_dataset, validating_dataset,
#                learning_rate=config.LEARNING_RATE/5,
#                epochs=60,
#                layers="all",
#                task=mode)
#
# # model.train(training_dataset, validating_dataset,
# #                learning_rate=config.LEARNING_RATE/10,
# #                epochs=120,
# #                layers="all",
# #                task=mode)
# #
# # model.train(training_dataset, validating_dataset,
# #                learning_rate=config.LEARNING_RATE/5,
# #                epochs=200,
# #                layers="all",
# #                task=mode)
#
# model_path = os.path.join(MODEL_DIR, "train_id#9.h5")
# model.keras_model.save_weights(model_path)

# ######################################################################################################
# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config, task="grasping_points")

# Load weights
weights_path = os.path.join(MODEL_DIR, "train_id#9.h5")
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)
image_ids = random.choices(validating_dataset.image_ids, k=15)
for image_id in image_ids:
    # validating_dataset.load_image(image_id, validating_dataset.image_info[image_id]['augmentation'])
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(validating_dataset, config, image_id, use_mini_mask=False, mode='grasping_points')
    results = model.detect([image], verbose=1, task=mode)
    r = results[0]
    post_nms_predictions, pre_nms_predictions = validating_dataset.refine_results(r, model.anchors)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image)
    ax2.imshow(image)
    for i, rect in enumerate(pre_nms_predictions):
        rect = validating_dataset.bbox_convert_to_four_vertices([rect])
        p = patches.Polygon(rect[0], linewidth=1,edgecolor='r',facecolor='none')
        ax1.add_patch(p)
    for i, rect2 in enumerate(post_nms_predictions):
        rect2 = validating_dataset.bbox_convert_to_four_vertices([rect2])
        p2 = patches.Polygon(rect2[0], linewidth=1,edgecolor='g',facecolor='none')
        ax2.add_patch(p2)
    plt.title(validating_dataset.image_info[image_id]['path'])
    plt.show(block=False)

import code;
code.interact(local=dict(globals(), **locals()))
    # plt.savefig(os.path.join('Grasping_anchors','P'+str(level+2)+ 'center_anchors.png'))
    #
    # training_dataset.visualize_bbox(image_id, bounding_box[0], gt_class_id[i], gt_bbox[i], rgbd_image=image)

# ######################################################################################################
# normalized_anchors = model.get_anchors(config.IMAGE_SHAPE, mode='grasping_points', angles=config.RPN_GRASP_ANGLES)

# # Generate Anchors
# mode= 'grasping_points'
# backbone_shapes = modellib.compute_backbone_shapes(config, config.IMAGE_SHAPE)
# anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
#                                           config.RPN_ANCHOR_RATIOS,
#                                           backbone_shapes,
#                                           config.BACKBONE_STRIDES,
#                                           config.RPN_ANCHOR_STRIDE,
#                                           config.IMAGE_SHAPE,
#                                           mode,
#                                           config.RPN_GRASP_ANGLES)
#
# image_ids = random.choices(validating_dataset.image_ids, k=10)
# for image_id in image_ids:
#     image, image_meta, gt_class_id, gt_bbox, gt_mask =\
#         modellib.load_image_gt(validating_dataset, config, image_id, use_mini_mask=False, mode='grasping_points')
#
#     target_rpn_match, target_rpn_bbox = modellib.build_rpn_targets(
#         image.shape, model.anchors, gt_class_id, gt_bbox, model.config, mode='grasping_points')
# #     positive_anchor_ix = np.where(target_rpn_match[:] == 1)[0]
#     negative_anchor_ix = np.where(target_rpn_match[:] == -1)[0]
#     neutral_anchor_ix = np.where(target_rpn_match[:] == 0)[0]
#     positive_anchors = model.anchors[positive_anchor_ix]
#     negative_anchors = model.anchors[negative_anchor_ix]
#     neutral_anchors = model.anchors[neutral_anchor_ix]
#     # import code;
#     #
#     # code.interact(local=dict(globals(), **locals()))
#     image_final_anchors = np.where(np.logical_not(target_rpn_match[:] == 0))[0]
#     positive_anchors_mask = np.in1d(image_final_anchors, positive_anchor_ix)
#     deltas = target_rpn_bbox[positive_anchors_mask] * model.config.RPN_BBOX_STD_DEV
#     refined_anchors = utils.apply_box_deltas(positive_anchors, deltas, mode, len(config.RPN_GRASP_ANGLES))
#
#     fig, ax = plt.subplots(1, figsize=(10, 10))
#     ax.imshow(image)
#
#     for i, rect in enumerate(gt_bbox):
#         rect = validating_dataset.bbox_convert_to_four_vertices([rect])
#         p = patches.Polygon(rect[0], linewidth=1,edgecolor='g',facecolor='none')
#         ax.add_patch(p)
#     ax.set_title(validating_dataset.image_info[image_id]['path'])
#     plt.show(block=False)
#
#     fig, ax = plt.subplots(1, figsize=(10, 10))
#     ax.imshow(image)
#     print (len(positive_anchor_ix), len(negative_anchor_ix))
#
#
#     for i, rect2 in enumerate(negative_anchors):
#         rect2 = validating_dataset.bbox_convert_to_four_vertices([rect2])
#         p = patches.Polygon(rect2[0], linewidth=1,edgecolor='r',facecolor='none')
#         ax.add_patch(p)
#
#     for i, rect3 in enumerate(model.anchors[positive_anchor_ix]):
#         rect3= validating_dataset.bbox_convert_to_four_vertices([rect3])
#         q = patches.Polygon(rect3[0], linewidth=1, edgecolor='g', facecolor='none')
#         ax.add_patch(q)
#
#     for i, rect4 in enumerate(refined_anchors):
#         rect4 = validating_dataset.bbox_convert_to_four_vertices([rect4])
#         r = patches.Polygon(rect4[0], linewidth=1, edgecolor='b', facecolor='none')
#         ax.add_patch(r)
#     ax.set_title(validating_dataset.image_info[image_id]['path'])
#     plt.show(block=False)
# import code;
#
# code.interact(local=dict(globals(), **locals()))
#
#
#
# #
# log("target_rpn_match", target_rpn_match)
# log("target_rpn_bbox", target_rpn_bbox)
# positive_anchor_ix = np.where(target_rpn_match[:] == 1)[0]
# negative_anchor_ix = np.where(target_rpn_match[:] == -1)[0]
# neutral_anchor_ix = np.where(target_rpn_match[:] == 0)[0]
# positive_anchors = model.anchors[positive_anchor_ix]
# negative_anchors = model.anchors[negative_anchor_ix]
# neutral_anchors = model.anchors[neutral_anchor_ix]
# log("positive_anchors", positive_anchors)
# log("negative_anchors", negative_anchors)
# log("neutral anchors", neutral_anchors)
#
# # Apply refinement deltas to positive anchors
# # positive_anchor_indices = anchor_data[anchor_data[:,1] == 1][:,0]
# # positive_anchors = anchors[positive_anchor_indices]
# gt_boxes = gt_bbox[gt_class_id == 1]
#
# image_final_anchors = np.where(np.logical_not(target_rpn_match[:] == 0))[0]
# positive_anchors_mask = np.in1d(image_final_anchors, positive_anchor_ix)
# deltas = target_rpn_bbox[positive_anchors_mask]* model.config.RPN_BBOX_STD_DEV
# if not (deltas.shape == positive_anchors.shape):
#     print ('$$$$$$$$$$$$$$$$$$$$$$$  ERRRORRRR $$$$$$$$$$$$$$$$$$$$$')
#     import code;
#     code.interact(local=dict(globals(), **locals()))
#
# refined_anchors = utils.apply_box_deltas(positive_anchors, deltas, mode, len(config.RPN_GRASP_ANGLES))
#
# # Display positive anchors before refinement (dotted) and
# # after refinement (solid).
# visualize.draw_boxes(image, boxes=positive_anchors, refined_boxes=refined_anchors, mode=mode)

# import code; code.interact(local=dict(globals(), **locals()))

####################################### VISUALIZING ANCHORS ############################################################
#
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
#         rect = training_dataset.bbox_convert_to_four_vertices([rect])
#         p = patches.Polygon(rect[0], linewidth=1,edgecolor='r',facecolor='none')
#         ax.add_patch(p)
#     plt.savefig(os.path.join('Grasping_anchors','P'+str(level+2)+ 'center_anchors.png'))
# plt.show(block=False)
