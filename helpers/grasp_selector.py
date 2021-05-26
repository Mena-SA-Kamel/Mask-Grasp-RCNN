import numpy as np
import cv2
from matplotlib.path import Path
import open3d as o3d
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

def generate_mask_from_polygon(image_shape, polygon_vertices):
    # This function returns a mask defined by some vertices
    ny, nx = image_shape[:2]
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T
    path = Path(polygon_vertices)
    grid = path.contains_points(points)
    grid = grid.reshape((ny, nx))
    return grid

def get_images_from_frames(color_frame, aligned_depth_frame):
    color_image = np.asanyarray(color_frame[0].get_data())
    depth_image = np.asanyarray(aligned_depth_frame[0].get_data())
    return [color_image, depth_image]


def compute_approach_vector(color_frame, aligned_depth_frame, image_height, image_width, o3d_intrinsics, grasping_box):
    # This function computes the approach vector for a given grasping box in 5-dimensional coordinates
    # [x, y, w, h, theta].
    # Returns the camera frame coordinates of the
    x, y, w, h, theta = grasping_box
    # Cropping the point cloud at the central 1/3 of the grasping box
    extraction_mask_vertices = cv2.boxPoints(((x, y), (w, h / 3), theta))
    grasp_box_mask = generate_mask_from_polygon([image_height, image_width, 3], extraction_mask_vertices)
    # Masking the color and depth frames with the grasp_box_mask
    color_image, depth_image = get_images_from_frames(color_frame, aligned_depth_frame)
    masked_color = color_image * np.repeat(grasp_box_mask[..., None], 3, axis=2)
    masked_depth = depth_image * grasp_box_mask
    # Generating a point cloud by open3D for the grasping region as defined by the newly cropped color
    # and depth channels
    pcd = generate_pointcloud_from_rgbd(masked_color, masked_depth, o3d_intrinsics)
    # Estimating surface normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=50))
    # Pointing the surface normala towards the camera
    pcd.orient_normals_towards_camera_location()
    normals = np.array(pcd.normals)
    points = np.array(pcd.points)
    # Finding the point with the minumum depth - point with lowest distance in the Z direction
    min_depth_ix = np.argmin(points[:, -1])
    # approach_vector = normals[min_depth_ix]
    # Defining approach vector as the average normal vector of the central 1/3 of the grasping box
    approach_vector = np.mean(normals, axis=0)
    min_depth_point = points[min_depth_ix]
    # Approach vector needs to be pointing in the opposite direction - ie into the object because that is
    # how we plan on approaching it
    approach_vector = -1 * approach_vector
    return approach_vector


def generate_pointcloud_from_rgbd(color_image, depth_image, o3d_intrinsics):
    # Creates an Open3D point cloud
    color_stream = o3d.geometry.Image(color_image)
    depth_stream = o3d.geometry.Image(depth_image)
    o3d_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_stream, depth_stream, depth_trunc=2.0)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(o3d_rgbd_image, o3d_intrinsics)
    return pcd

def de_project_point(intrinsics, depth_frame, point):
    # This function deprojects point from the image plane to the camera frame of reference using camera intrinsic
    # parameters
    x, y = point
    distance_to_point = depth_frame.as_depth_frame().get_distance(x, y)
    de_projected_point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], distance_to_point)
    return de_projected_point

def compute_distance_3D(P1, P2):
    # This function computes the euclidean distance between two 3D points
    return np.sqrt((P1[0] - P2[0]) ** 2 + (P1[1] - P2[1]) ** 2 + (P1[2] - P2[2]) ** 2)

def compute_distance_2D(p1, p2):
    # This function computes the euclidean distance between two 2D points
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def get_camera_frame_box_coords(intrinsics, depth_frame, box_vertices):
    # This function deprojects the vertex points that make up the grasp box from the image plane to the camera frame
    de_projected_points = []
    for point in box_vertices:
        de_projected_point = de_project_point(intrinsics, depth_frame, point)
        de_projected_points.append(de_projected_point)
    return np.array(de_projected_points)

def compute_real_box_size(intrinsics, aligned_depth_frame, rect):
    rect_camera_frame = get_camera_frame_box_coords(intrinsics, aligned_depth_frame, rect)
    # Need to find the width and height of the bounding box in the world/ camera frames
    # real_width is the average of the distance between P1-P2 and P0-P3
    # real_height is the average of the distance between P0-P1 and P2-P3
    #           w
    # P2 ---------------------- P1
    # ||                        ||
    # ||            .           || h
    # ||          (0,0,0)       ||
    # P3 ---------------------- P0
    P0 = rect_camera_frame[0]
    P1 = rect_camera_frame[1]
    P2 = rect_camera_frame[2]
    P3 = rect_camera_frame[3]

    real_width = np.mean([compute_distance_3D(P1, P2), compute_distance_3D(P0, P3)])
    real_height = np.mean([compute_distance_3D(P0, P1), compute_distance_3D(P2, P3)])
    # print('Grasp Width: ', real_width, 'Grasp Height: ', real_height)
    return [real_width, real_height]


def generate_points_in_world_frame(real_width, real_height):
    # This function generates the points in the world frame of reference
    return np.array([[real_width/2, -real_height/2, 0],
                    [real_width/2, real_height/2, 0],
                    [-real_width/2, real_height/2, 0],
                    [-real_width/2, -real_height/2, 0]])



def compute_wrist_orientation(approach_vector, theta):
    # Computes the wrist orientation in the camera frame
    # Need to get the projection of vx onto the approach vector (normal), to get the projection of vx on
    # the plane perperndicular to the surface normal
    vz = approach_vector
    vx = np.array([1, 0, 0])
    proj_n_vx = (np.dot(vx, approach_vector) / (np.linalg.norm(approach_vector) ** 2)) * approach_vector
    vx = vx - proj_n_vx
    vx = vx / np.linalg.norm(vx)
    vy = np.cross(vz, vx)
    vy = vy / np.linalg.norm(vy)
    vx = vx.reshape([3, 1])
    vy = vy.reshape([3, 1])
    vz = vz.reshape([3, 1])
    V = np.concatenate([vx, vy, vz], axis=-1)
    return V

def derive_motor_angles_v0(orientation_matrix):
    orientation_matrix = np.around(orientation_matrix, decimals=2)

    b = orientation_matrix[0, 1]
    e = orientation_matrix[1, 1]
    i = orientation_matrix[2, 2]
    g = orientation_matrix[2, 0]
    h = orientation_matrix[2, 1]

    s2 = h
    c2 = np.sqrt(1 - s2 ** 2) # c2 is either a positive or negative angle. We need to try those angle combinations,
                                # and choose the angle combination with the lowest sum of angles
    angle_signs = [1, -1]
    join_sum = []
    joint_combinations = np.zeros((2,3))
    for j, angle_sign in enumerate(angle_signs):
        c2_i = c2 * angle_sign
        if c2_i == 0:
            print("Potentially Gimbal lock")
        sign = np.sign(c2_i)
        if sign == 0:
            sign = 1
        c2_i = sign*np.maximum(0.0001, np.abs(c2_i)) # Gimbal lock case - Need to look into this
        s1 = -b / c2_i
        c1 = e / c2_i
        s3 = -i / c2_i
        c3 = g / c2_i
        theta_1 = np.arctan2(s1, c1) / (np.pi / 180)
        theta_2 = np.arctan2(s2, c2_i) / (np.pi / 180)
        theta_3 = np.arctan2(s3, c3) / (np.pi / 180)
        joint_combinations[j,:] = np.array([theta_1,theta_2, theta_3])
        join_sum.append(np.sum(np.abs(np.array([theta_1,theta_2, theta_3]))))
    # theta_1, theta_2, theta_3 = joint_combinations[np.argmin(join_sum)] # Choosing the angle combo with the least
                                                                        # deviation required
    theta_1, theta_2, theta_3 = joint_combinations[0]  # Choosing the angle combo with the least
    return [theta_1, theta_2, theta_3]

def select_grasp_box(realsense_orientation, top_grasp_boxes, image_width, image_height, center_crop_size, color_frame,
                     aligned_depth_frame, o3d_intrinsics, dataset_object, intrinsics):
    cam_pitch, _, cam_roll = realsense_orientation[0]
    # Need to select the box that requires the least deviation from current arm orientation
    # Storing the high probability grasp orientations in Direction Cosine Matrix Format (DCM)
    potential_grasps = np.zeros((top_grasp_boxes.shape[0], 3, 3))
    box_sizes = np.zeros((top_grasp_boxes.shape[0], 2))
    for k, five_dim_box in enumerate(top_grasp_boxes):
        # Need to crop out the point cloud at the oriented rectangle bounds
        # Create a mask to specify the region to extract the surface normals from. Based on Lenz et al.,
        # the approach vector is estimated as the surface normal calculated at the point of minumum depth in
        # the central one third (horizontally) of the rectangle

        x, y, w, h, theta = five_dim_box
        # Undoing the center cropping of the image
        x = int(x + (image_width - center_crop_size) // 2)
        y = int(y + (image_height - center_crop_size) // 2)
        # Getting grasping box vertices in the image plane
        rect_vertices = cv2.boxPoints(((x, y), (w, h), theta))
        # Calculating the approach vector for the grasp box
        approach_vector = compute_approach_vector(color_frame, aligned_depth_frame, image_height,
                                                  image_width, o3d_intrinsics, [x, y, w, h, theta])
        # Getting the XYZ coordinates of the grasping box center <x, y>
        box_center = de_project_point(intrinsics[0], aligned_depth_frame[0], [x, y])
        # Wrapping angles so they are in [-90, 90] range
        theta = dataset_object.wrap_angle_around_90(np.array([theta]))[0]
        theta = theta * (np.pi / 180)  # network outputs positive angles in bottom right quadrant
        # Computing the grasp box size in camera coordinates, size in meters
        real_width, real_height = compute_real_box_size(intrinsics[0], aligned_depth_frame[0], rect_vertices)
        box_vert_obj_frame = generate_points_in_world_frame(real_width, real_height)
        # Computing the desired wrist orientation in the camera frame
        V = compute_wrist_orientation(approach_vector, theta)
        rotation_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                               [np.sin(theta), np.cos(theta), 0],
                               [0, 0, 1]])
        approach_vector_orientation = np.dot(V, rotation_z)

        # Defining rotation matrix to go from the shoulder to camera frame

        R_shoulder_camera = R.from_euler('xyz', [[cam_pitch, 0, cam_roll]],
                                         degrees=True).as_matrix().squeeze()
        grasp_orientation_shoulder = np.dot(R_shoulder_camera, approach_vector_orientation)
        potential_grasps[k] = grasp_orientation_shoulder
        box_sizes[k] = np.array([real_width, real_height])

    joint_deviations = np.zeros(potential_grasps.shape[0])
    joint_angles = np.zeros((potential_grasps.shape[0], 3))
    for i, grasp_orientation_shoulder in enumerate(potential_grasps):
        grasp_orientation_shoulder_2 = grasp_orientation_shoulder.copy()
        grasp_orientation_shoulder_2[:, :2] = grasp_orientation_shoulder_2[:, :2] * -1  # Equivalent graspbox
        equivalent_grasps = [grasp_orientation_shoulder, grasp_orientation_shoulder_2]
        equivalent_joint_deviations = np.zeros(2)
        equivalent_joint_angles = np.zeros((2, 3))
        for j, equivalent_grasp in enumerate(equivalent_grasps):
            theta1_t, theta2_t, theta3_t = derive_motor_angles_v0(equivalent_grasp)
            equivalent_joint_deviations[j] = np.sum(np.abs(np.array([theta1_t, theta2_t, theta3_t])))
            equivalent_joint_angles[j] = np.array([theta1_t, theta2_t, theta3_t])
        theta1_t, theta2_t, theta3_t = equivalent_joint_angles[np.argmin(equivalent_joint_deviations)]
        joint_deviations[i] = np.sum(np.abs(np.array([theta1_t, theta2_t, theta3_t])))
        joint_angles[i] = np.array([theta1_t, theta2_t, theta3_t])
        potential_grasps[i] = equivalent_grasps[np.argmin(equivalent_joint_deviations)]

    grasp_box_index = np.argmin(joint_deviations)
    grasp_DCM = potential_grasps[grasp_box_index]
    real_width, real_height = box_sizes[grasp_box_index]
    return [grasp_DCM, grasp_box_index, real_width, real_height]