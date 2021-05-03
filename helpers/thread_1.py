import zmq
import msgpack
import numpy as np
import cv2

def get_gaze_points_in_realsense_frame(avg_gaze, rvec, tvec, realsense_intrinsics_matrix,
                                       image_width, center_crop_size, image_height):
    gaze_points = np.array(avg_gaze).reshape(-1, 1, 3)
    gaze_points_realsense_image, jacobian = cv2.projectPoints(gaze_points, rvec, tvec,
                                                              realsense_intrinsics_matrix, None)
    gaze_x_realsense, gaze_y_realsense = gaze_points_realsense_image.squeeze().astype('uint16')
    gaze_x_realsense = int(gaze_x_realsense - (image_width - center_crop_size) / 2)
    gaze_y_realsense = int(gaze_y_realsense - (image_height - center_crop_size) / 2)
    return [gaze_x_realsense, gaze_y_realsense]


def initialize_pupil_tracker():
    # Establishing connection to eye tracker
    ctx = zmq.Context()
    # The REQ talks to Pupil remote and receives the session unique IPC SUB PORT
    pupil_remote = ctx.socket(zmq.REQ)
    ip = 'localhost'  # If you talk to a different machine use its IP.
    port = 50020  # The port defaults to 50020. Set in Pupil Capture GUI.
    pupil_remote.connect(f'tcp://{ip}:{port}')
    # Request 'SUB_PORT' for reading data
    pupil_remote.send_string('SUB_PORT')
    sub_port = pupil_remote.recv_string()
    # Request 'PUB_PORT' for writing dataYour location
    pupil_remote.send_string('PUB_PORT')
    pub_port = pupil_remote.recv_string()
    subscriber = ctx.socket(zmq.SUB)
    subscriber.connect(f'tcp://{ip}:{sub_port}')
    subscriber.subscribe('gaze.')  # receive all gaze messages
    return subscriber

def thread1(subscriber, avg_gaze, terminate, rvec, tvec, realsense_intrinsics_matrix, image_width,
                      image_height, center_crop_size):
    while not terminate[0]:
        topic, payload = subscriber.recv_multipart()
        message = msgpack.loads(payload)
        gaze_point_3d = message[b'gaze_point_3d']
        avg_gaze[:] = get_gaze_points_in_realsense_frame(gaze_point_3d, rvec, tvec,
                                                         realsense_intrinsics_matrix,
                                                         image_width, center_crop_size,
                                                         image_height)
        # avg_gaze = [int(image_width / 2), int(image_height / 2)]