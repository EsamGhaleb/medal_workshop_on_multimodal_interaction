import os
from speach import elan
import pandas as pd
import numpy as np
import glob
import torch 
data_path='data/gesture_form_similarity_coding.csv'
selected = np.concatenate(([0,5,6,7,8,9,10], [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0)
max_frame = 60
num_joints = 27
num_channels = 3
max_body_true = 1
sign_27_2 =  ((5, 6), (5, 7), (6, 8), (8, 10), (7, 9), (9, 11), 
                              (12,13),(12,14),(12,16),(12,18),(12,20),
                              (14,15),(16,17),(18,19),(20,21),
                              (22,23),(22,24),(22,26),(22,28),(22,30),
                              (24,25),(26,27),(28,29),(30,31),
                              (10,12),(11,22))

def process_poses(poses):
   # add one dimension to the pose at the end
   poses = np.expand_dims(poses, axis=-1)
   # original shape is T, V, C, M
   T, V, C, M = poses.shape
   poses = np.transpose(poses, (2, 0, 1, 3))

   mirrod_poses = miror_poses(poses.copy())
   return poses, mirrod_poses

        
def load_keypoints_dict():
   # read all poses in f'data/final_poses/poses_{pair}_synced_pp{speaker}.npy'
   processed_keypoints_dict = {}
   mirrored_keypoints_dict = {}
   for file in glob.glob('data/selected_poses/*.npy'):
      # file format = data/selected_poses/poses_pair04_synced_ppA.npy 
      pair = file.split('/')[-1].split('_')[1]
      speaker = file.split('/')[-1].split('_')[-1].split('.')[0][-1]
      pair_speaker = f"{pair}_{speaker}"
      try:
         processed_keypoints_dict[pair_speaker], mirrored_keypoints_dict[pair_speaker] = process_poses(np.load(file)) #np.load(keypoints_path)
      except Exception as e:
         print(e)
         print('Error in loading the keypoints for the pair:', pair, speaker)
         continue
   return processed_keypoints_dict, mirrored_keypoints_dict


def get_bone(skel):
   N, C, T, V, M = skel.shape
   bone_skel = np.zeros((N, C, T, V, M), dtype=np.float32)
   bone_skel[:, :C, :, :, :] = skel.copy()
   for v1, v2 in sign_27_2:
      v1 -= 5
      v2 -= 5
      bone_skel[:, :, :, v2, :] = skel[:, :, :, v2, :] - skel[:, :, :, v1, :]
   return bone_skel

def miror_poses(data_numpy):
   C,T,V,M = data_numpy.shape
   assert C == 3 # x,y,c
   data_numpy[0, :, :, :] = 1920 - data_numpy[0, :, :, :]
   return data_numpy