import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.io
import numpy as np
import random


def generate_random_color():
    r = random.random()  # *255
    b = random.random()  # *255
    g = random.random()  # *255
    return (r, g, b)

def load_mot17_detections(folder_path):
    mot_video_sequence_name = folder_path.split('\\')[1]
    video_sequence_path = folder_path.replace(folder_path.split('\\')[-1], mot_video_sequence_name)
    # Checks if a numpy array of detections already exists
    if os.path.exists(video_sequence_path):
        return np.loadtxt(video_sequence_path)
    with open(folder_path) as f:
        raw_detections = f.readlines()
    num_columns = int(np.char.count(raw_detections[0], ',')) + 1
    detections = np.zeros([0, num_columns]).astype('float32')
    for detection in raw_detections:
        detection = detection.strip('\n').split(',')
        detection = np.array(detection).astype('float32')
        detection = np.reshape(detection, [-1, detection.shape[0]])
        detections = np.append(detections, detection, axis=0)
    np.savetxt(video_sequence_path, detections)
    return detections


mot17_dataset_path = '../../../Datasets/MOT17/train'
video_name = 'MOT17-04-FRCNN'
video_path = os.path.join(mot17_dataset_path, video_name, 'img1')

# Defining paths for the text files containing the GT and prediction txt files
mot_gt_labels_path = os.path.join(mot17_dataset_path, video_name, 'gt', 'gt.txt')
mot_pred_labels_path = os.path.join(mot17_dataset_path, video_name, 'det', 'det.txt')

gt_detections = load_mot17_detections(mot_gt_labels_path)
gt_det_ids = np.unique(np.split(gt_detections, indices_or_sections=9, axis=-1)[1])

pred_detections = load_mot17_detections(mot_pred_labels_path)
pred_det_ids = np.unique(np.split(pred_detections, indices_or_sections=7, axis=-1)[1])

# Defining the number of frames to visualize
counter = 0
num_frames_to_show = 10
delta = 30 # delta is the sampling period, i.e., for a 30 fps, this means a sampling period of 1s

# Generating a unique color for each tracked item
detections = gt_detections
det_ids = gt_det_ids
colours = []
for i in det_ids:
    colours.append(generate_random_color())

# Storing object center locations
track_history = np.zeros((num_frames_to_show, det_ids.shape[0], 2))

for frame in os.listdir(video_path):
    frame_image = skimage.io.imread(os.path.join(video_path, frame))
    frame_number = int(frame.strip('.jpg'))
    if (frame_number%delta == 0):
        frame_detections = detections[detections[:, 0] == frame_number]
        _, det_id, x, y, w, h, *_ = np.split(frame_detections, indices_or_sections=frame_detections.shape[-1], axis=-1)

        fig, ax = plt.subplots()
        for i in range(det_id.shape[0]):
            box_id = int(det_id[i][0])
            if box_id != -1:
                track_history[counter, box_id-1] = [x[i]+w[i]/2, y[i]+h[i]/2]
                p = patches.Rectangle((x[i], y[i]), w[i], h[i], angle=0, edgecolor=colours[box_id-1],
                                    linewidth=1, facecolor='none')
            else:
                p = patches.Rectangle((x[i], y[i]), w[i], h[i], angle=0, edgecolor=generate_random_color(),
                                      linewidth=1, facecolor='none')
            ax.add_patch(p)
        counter += 1
        ax.imshow(frame_image)
        if counter == num_frames_to_show:
            break

for i in range(track_history.shape[1]):
    instance_history = track_history[:, i, :]
    instance_history = instance_history[np.where(np.sum(instance_history, axis=-1)!=0)]
    ax.plot(instance_history[:, 0], instance_history[:, 1], color=colours[i])
plt.show()
