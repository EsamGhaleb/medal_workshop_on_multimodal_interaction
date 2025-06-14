import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from IPython.display import HTML
import random


mmpose_flip_index = np.concatenate(([0,2,1,4,3,6,5],[17,18,19,20,21,22,23,24,25,26],[7,8,9,10,11,12,13,14,15,16]), axis=0) 
selected_points = np.array([0, 5, 6, 7, 8, 9, 10, 91, 95, 96, 99, 100, 103, 104, 107, 108, 111, 112, 116, 117, 120, 121, 124, 125, 128, 129, 132])

def miror_poses(data_numpy):
   C,T,V,M = data_numpy.shape
   assert C == 3 # x,y,c
   data_numpy[0, :, :, :] = 1920 - data_numpy[0, :, :, :]
   return data_numpy

def shift_poses(data_numpy):
   C,T,V,M = data_numpy.shape
   print(data_numpy.shape)
   assert V == 27
   for i in range(V): # for each joint
      data_numpy[0, :, i, :] += random.uniform(-20.0, 20.0)  # Random shift in x
      data_numpy[1, :, i, :] += random.uniform(-20.0, 20.0)  # Random shift in y
   return data_numpy
def scale_poses(data_numpy):
   C,T,V,M = data_numpy.shape
   # make the scale random based on the image size
   scales = (random.uniform(0.5, 1.5), random.uniform(0.5, 1.5))
   data_numpy[0, :, :, :] = data_numpy[0, :, :, :] * scales[0]
   data_numpy[1, :, :, :] = data_numpy[1, :, :, :] * scales[1]
   return data_numpy

def process_frame(poses, frame_index, video_capture):
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, img = video_capture.read()
    if not ret:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_with_pose = plot_pose(img, poses[frame_index])
    return img_with_pose

def plot_pose(img, result, scale=(1.0, 1.0)):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    ORANGE = (0, 165, 255)
    PINK = (203, 192, 255)

    # result = result[selected_points]
    # Define limb colors
    limb_colors = [GREEN, BLUE, BLUE, BLUE, BLUE, YELLOW, ORANGE, YELLOW, ORANGE,
                   YELLOW, ORANGE, PINK, RED, PINK, RED, PINK, RED, GREEN, GREEN, GREEN,
                   GREEN, GREEN, GREEN, GREEN, GREEN, GREEN, GREEN, GREEN, GREEN]
    l_pair = ((5, 6), (5, 7),
              (6, 8), (8, 10), (7, 9), (9, 11),
              (12, 13), (12, 14), (12, 16), (12, 18), (12, 20),
              (14, 15), (16, 17), (18, 19), (20, 21),
              (22, 23), (22, 24), (22, 26), (22, 28), (22, 30),
              (24, 25), (26, 27), (28, 29), (30, 31),
              (10, 12), (11, 22))

    part_line = {}
    kp_preds = result[:, :2]
    kp_scores = result[:, 2]
    # Draw keypoints
    for n in range(kp_scores.shape[0]):
        cor_x, cor_y = int(round(kp_preds[n, 0] * scale[0])), int(round(kp_preds[n, 1] * scale[1]))

        # if flip_horizontal:
        #     cor_x = img_width - cor_x  # Flip horizontally

        part_line[n] = (cor_x, cor_y)
        cv2.circle(img, (cor_x, cor_y), 2, (0, 0, 255), -1)

    # Draw limbs with colors
    for i, (start_p, end_p) in enumerate(l_pair):
        start_p = int(start_p) - 5
        end_p = int(end_p) - 5
        if start_p in part_line and end_p in part_line:
            start_p = part_line[start_p]
            end_p = part_line[end_p]
            cv2.line(img, start_p, end_p, limb_colors[i], 2)
    
    return img
