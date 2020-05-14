import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
import open3d as o3d
import skimage.io
import math

def convert_to_depth_image(pcd, pcd_path = ''):
    xyz_load = np.asarray(pcd.points)

    depth_values = xyz_load[:,2]
    depth_values_reshaped = np.interp(depth_values, (depth_values.min(), depth_values.max()), (0, 1))

    with open(pcd_path) as f:
        points = f.readlines()

    img = np.zeros([480, 640], dtype=np.uint8)
    i = 0
    for point in points[10:]:
        index = int(point.split(' ')[-1].replace('\n', ''))
        row = int(np.floor(index / 640))
        col = int(index%640 + 1)
        img[row, col] = depth_values_reshaped[i]*255
        i += 1
    return img

def get_five_dimensional_box(coordinates):
    x_coordinates = coordinates[:, 0]
    y_coordinates = coordinates[:, 1]
    gripper_orientation = coordinates[:2, :]

    x = int((np.max(x_coordinates) - np.min(x_coordinates)) / 2) + np.min(x_coordinates)
    y = int((np.max(y_coordinates) - np.min(y_coordinates)) / 2) + np.min(y_coordinates)

    # Gripper opening distance
    w = np.linalg.norm(gripper_orientation[0] - gripper_orientation[1])

    # Gripper width - Cross product betwee
    # h = np.cross(p2-p1, p1-p3)/norm(p2-p1)
    p1 = gripper_orientation[0]
    p2 = gripper_orientation[1]
    p3 = rectange_class[i][2, :]
    h = np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1)

    deltaY = gripper_orientation[0][1] - gripper_orientation[1][1]
    deltaX = gripper_orientation[0][0] - gripper_orientation[1][0]
    theta = -1 * np.arctan2(deltaY, deltaX) * 180 / math.pi
    return [x,y,w,h,theta]


cornell_dataset_path = '../../../Datasets/Cornell-Dataset/Raw Dataset/06'
image_paths = glob.glob(cornell_dataset_path + '/**/*.png', recursive=True)

for i in list(range(len(image_paths))):
    image_paths[i] = image_paths[i].split('\\')[1].replace('r.png', '')

for image_name in image_paths:

    positive_rectangles = os.path.join(cornell_dataset_path, (image_name + 'cpos.txt'))
    negative_rectangles = os.path.join(cornell_dataset_path, (image_name + 'cneg.txt'))
    image = os.path.join(cornell_dataset_path, (image_name + 'r.png'))
    pcd_path = os.path.join(cornell_dataset_path, (image_name + '.txt'))

    pcd = o3d.io.read_point_cloud(pcd_path, format='pcd')

    depth_img = convert_to_depth_image(pcd, pcd_path = pcd_path)

    ground_truth = []
    with open(positive_rectangles) as f:
        positive_rect = f.readlines()

    with open(negative_rectangles) as f:
        negative_rect = f.readlines()

    ground_truth_data = [positive_rect, negative_rect]

    final_vertices = []
    vertices = []
    rectangles = []

    for rectange_class in ground_truth_data:
        i = 0
        rectangles = []
        for line in rectange_class:
            x = int(float(line.split(' ')[0]))
            y = int(float(line.split(' ')[1]))
            vertices.append([x,y])
            i += 1
            if i%4 == 0:
                rectangles.append(vertices)
                vertices = []
        final_vertices.append(rectangles)

    final_vertices = np.array(final_vertices)

    fig2, ax2 = plt.subplots(1)
    color_image = skimage.io.imread(image)
    rgbd_image = np.zeros([color_image.shape[0], color_image.shape[1], 4])
    rgbd_image[:, :, 0:3] = color_image
    rgbd_image[:, :, 3] = depth_img
    rgbd_image = np.array(rgbd_image).astype('uint8')

    ax2.imshow(rgbd_image)
    ax2.set_title('Visualizing rectangles')

    colors = ['g', 'r']
    j = 0

    for rectange_class in final_vertices:
        rectange_class = np.array(rectange_class)

        for i in list(range(rectange_class.shape[0])):
            rect = patches.Polygon(np.array(rectange_class[i]), linewidth=1, edgecolor=colors[j], facecolor='none')
            gripper_orientation = rectange_class[i][:2, :]

            x, y, w, h, theta = get_five_dimensional_box(rectange_class[i])

            title =  ('x = %d, y = %d, w = %d, h = %d, theta = %d' %(x,y,w,h,theta))

            ax2.plot(gripper_orientation[:, 0], gripper_orientation[:, 1], linewidth=1, color='c')
            ax2.scatter([x],[y], linewidth=2, color='k')
            ax2.add_patch(rect)
            ax2.set_title(title)

        j += 1
    plt.show()