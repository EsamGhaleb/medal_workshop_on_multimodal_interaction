from __future__ import print_function
import sys
sys.path.extend(['.'])
project_directory = ''
import argparse
import pickle
import glob
import sys
import os
from itertools import product
from collections import Counter

import numpy as np
from tqdm import tqdm
import pandas as pd
from einops import rearrange

# audio_path = '/home/eghaleb/data/{}_synced_pp{}.wav'
audio_path = "/home/atuin/b105dc/data/datasets/rasenberg_phd/3_data/processed_data/audio_video/{}/{}_synced_pp{}.wav"

markersbody = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_OUTER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
          'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 
          'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX',
          'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
          'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']

markershands = ['LEFT_WRIST', 'LEFT_THUMB_CMC', 'LEFT_THUMB_MCP', 'LEFT_THUMB_IP', 'LEFT_THUMB_TIP', 'LEFT_INDEX_FINGER_MCP',
              'LEFT_INDEX_FINGER_PIP', 'LEFT_INDEX_FINGER_DIP', 'LEFT_INDEX_FINGER_TIP', 'LEFT_MIDDLE_FINGER_MCP', 
               'LEFT_MIDDLE_FINGER_PIP', 'LEFT_MIDDLE_FINGER_DIP', 'LEFT_MIDDLE_FINGER_TIP', 'LEFT_RING_FINGER_MCP', 
               'LEFT_RING_FINGER_PIP', 'LEFT_RING_FINGER_DIP', 'LEFT_RING_FINGER_TIP', 'LEFT_PINKY_FINGER_MCP', 
               'LEFT_PINKY_FINGER_PIP', 'LEFT_PINKY_FINGER_DIP', 'LEFT_PINKY_FINGER_TIP',
              'RIGHT_WRIST', 'RIGHT_THUMB_CMC', 'RIGHT_THUMB_MCP', 'RIGHT_THUMB_IP', 'RIGHT_THUMB_TIP', 'RIGHT_INDEX_FINGER_MCP',
              'RIGHT_INDEX_FINGER_PIP', 'RIGHT_INDEX_FINGER_DIP', 'RIGHT_INDEX_FINGER_TIP', 'RIGHT_MIDDLE_FINGER_MCP', 
               'RIGHT_MIDDLE_FINGER_PIP', 'RIGHT_MIDDLE_FINGER_DIP', 'RIGHT_MIDDLE_FINGER_TIP', 'RIGHT_RING_FINGER_MCP', 
               'RIGHT_RING_FINGER_PIP', 'RIGHT_RING_FINGER_DIP', 'RIGHT_RING_FINGER_TIP', 'RIGHT_PINKY_FINGER_MCP', 
               'RIGHT_PINKY_FINGER_PIP', 'RIGHT_PINKY_FINGER_DIP', 'RIGHT_PINKY_FINGER_TIP']
# from the markersbody, get me the indices of, nose, left eye, right eye, left shoulder, right shoulder, left hip, right hip, left elbow, right elbow
selected_markers = [0, 2, 5, 11, 12, 13, 14]# write the indices of markerhands plus 33 (the number of markersbody)
selected_markers.extend([33 + i for i in range(len(markershands))])

print(selected_markers)

# number of frames in a video to be considered
max_rgb_frame = 15 # 67% OF THE ICONIC GESTURES HAVE THIS LENGTH, which is the mean duration of the iconic gestures. We can keep this as it is since we are using the sliding windows approach, to determin the boundaris of the iconic gestures
sample_rate = 2
iou = 100
max_body_true = 1
max_frame = max_rgb_frame
num_frames = max_rgb_frame
num_channels = 3
keypoints_sampling_rate = 2 # this is the sampling rate of the keypoints, chose every 4th frame
max_seq_len = 40 # this is the upper bound of the number of frames in a video, given the confidence interval of 95% and the number of frames in the videos

# Audio parameters

buffer_and_window_in_seconds = {0.0: 0.48, 0.25:0.72, 0.5: 0.96}
sys.path.extend(['../'])

selected_joints = {
    '27': np.concatenate(([0,5,6,7,8,9,10], 
                    [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0), #27
   'CABB': selected_markers
}
 
def generate_data_sw(
        data_dict, 
        pair,
        speaker,
        all_speakers_fps,
        sequences_lengths,
        all_sample_names,
        sequences_labels,
        current_ind,
        history=40, 
        time_offset=2, 
        num_frames=max_rgb_frame, 
        audio_path='/Users/esamghaleb/Documents/ResearchData/CABB Small Dataset/processed_audio_video/{}/{}_synced_pp{}.wav',
        buffer=0.0
        ):
    """
    Generate data for sliding window analysis.

    Args:
        data_dict (dict): Dictionary containing the data.
        gestures_info (DataFrame): Information about gestures.
        pair (str): Pair identifier.
        speaker (str): Speaker identifier.
        all_speakers_fps (list): List to store generated data.
        sequences_lengths (list): List to store sequence lengths.
        all_sample_names (list): List to store generated sample names.
        sequences_labels (list): List to store sequence labels.
        current_ind (int): Current index.
        history (int, optional): Length of the history. Defaults to 40.
        time_offset (int, optional): Time offset. Defaults to 2.
        num_frames (int, optional): Number of frames. Defaults to 18.

    Returns:
        tuple: A tuple containing the following elements:
            - all_speakers_fps (list): Updated list of generated data.
            - sequences_lengths (list): Updated list of sequence lengths.
            - all_sample_names (list): Updated list of generated sample names.
            - sequences_labels (list): Updated list of sequence labels.
            - [None]*len(sequences_lengths) (list): List of None values.
            - current_ind (int): Updated current index.
    """
    sample_format = "{:6}_{:1}_{:06d}_{:06d}"
    data = data_dict[(pair, speaker)]
    start_ind = 0
    end_ind = data.shape[0] - (history + num_frames) * time_offset
    FPS = 50
    # Calculate samples per video frame
    # Calculate the step size for an offset of 2 frames
    buffer_in_frames = buffer * FPS
    num_frames_in_seconds = (num_frames+buffer_in_frames) / FPS
    for t in tqdm(range(start_ind, end_ind, history), leave=False):
        arr = np.stack([
            data[t + i*time_offset : t + (history+i)*time_offset : time_offset, ...]
            for i in range(num_frames)
        ])
        arr = rearrange(arr, 'f t k d -> t d f k')
        all_speakers_fps[current_ind] = arr    
        sample_names = [
            sample_format.format(pair, speaker, i, i+num_frames)
            for i in range(t, t + history*time_offset, time_offset)
        ]
        all_sample_names[current_ind] = sample_names
        current_ind += 1
    return all_speakers_fps, sequences_lengths, all_sample_names, sequences_labels, [None]*len(sequences_lengths), current_ind
    

def load_data(keypoints_path, config):
    data_dict = dict()
    for i, path in tqdm(enumerate(glob.glob(keypoints_path)), leave=True):
        print(path)
        pair = path.split('/')[-1].split('.')[0].split('_')[0]
        speaker = path.split('/')[-1].split('.')[0].split('_')[1]
        if speaker == 'a':
            speaker = 'A'
        elif speaker == 'b':
            speaker = 'B'
        data_dict[(pair, speaker)] = np.load(path)
    return data_dict

def get_data_size(data_dict, pair_speaker, history, time_offset, num_frames):
    total_num_samples = 0
    for pair in pair_speaker:
        x = data_dict[pair].shape[0]
        case_num_samples = (x - (history + num_frames) * time_offset) // (history) + 1
        total_num_samples += case_num_samples
    return total_num_samples

def get_part_pair_speaker(data_dict):
    print('Number of pairs and speakers: {}'.format(data_dict.keys()))
    # extract the unique pair and speaker from data dict
    unique_pair_speaker = {elem for elem in data_dict.keys()}
    unique_pair_speaker = sorted(unique_pair_speaker)
    print('Number of unique pair and speaker: {}'.format(len(unique_pair_speaker)))
    return unique_pair_speaker


def gendata(
        label_path, 
        out_path, 
        keypoints_path,
        history=40, 
        part='train', 
        config='27', 
        save_video=False,
        audio_path=None,
        buffer=0.0
        ):
    time_offset = 2
    
    data_dict = load_data(keypoints_path, config)
    
    pair_speaker = get_part_pair_speaker(data_dict)
    total_num_samples = get_data_size(data_dict, pair_speaker, history, time_offset, num_frames)
    all_speakers_fps = np.zeros((total_num_samples, history, 3, num_frames, int(config)), dtype=np.float32)

    sequences_lengths = history * np.ones(total_num_samples)
    all_sample_names = np.empty((total_num_samples, history), dtype='U22')
    sequences_labels = np.empty((total_num_samples, history), dtype='U47')
    
    all_pair_speaker_referent = []
    current_ind = 0
    for pair, speaker in tqdm(pair_speaker, leave=True):
        ret_val = generate_data_sw(
            data_dict=data_dict,
            pair=pair,
            speaker=speaker,
            all_speakers_fps=all_speakers_fps,
            sequences_lengths=sequences_lengths,
            all_sample_names=all_sample_names,
            sequences_labels=sequences_labels,
            current_ind=current_ind,
            history=history,
            audio_path=audio_path,
            buffer=buffer
        )
        all_speakers_fps, sequences_lengths, all_sample_names, sequences_labels, _, current_ind = ret_val
     
    if current_ind < total_num_samples:
        all_speakers_fps = all_speakers_fps[:current_ind]
        sequences_lengths = sequences_lengths[:current_ind]
        all_sample_names = all_sample_names[:current_ind]
        sequences_labels =  sequences_labels[:current_ind]

    with open(label_path, 'wb') as f:
        pickle.dump((all_sample_names, all_pair_speaker_referent, sequences_labels, sequences_lengths), f)
    all_speakers_fps = np.array(all_speakers_fps) 
    print(all_speakers_fps.shape)
    np.save(out_path, all_speakers_fps)
    print('saved to {}'.format(out_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process audio and video data')
    parser.add_argument('--keypoints-path', type=str, help='The path to the numpy file', default='./data/sho_final_vitposes/*.npy')
    args = parser.parse_args()
    keypoints_path = args.keypoints_path
    # '/home/hpc/b105dc/b105dc10/co-speech-v2/co-speech-gesture-detection/data/videos/npy3/{}_synced_pp{}.npy'
    out_folder = project_directory + '.'
    history = 40
    # gestures_info contains the following columns: ['pair', 'speaker', 'start_frame', 'end_frame', 'referent', 'label', 'is_gesture', 'keypoints']
    out_folder = project_directory+'data/sho_data/'  
    # check if the out_folder exists
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for buffer in [0.0]:
        data_out_path = '{}data_27_joint_buffer_{}.npy'.format(out_folder, buffer)
        audio_out_path = '{}audio_buffer_{}.npy'.format(out_folder, buffer)
        label_out_path = '{}27_labels_buffer_{}.pkl'.format(out_folder, buffer)
        referents_speakers_path = '{}27_referent_speakers_buffer_{}.pkl'.format(out_folder, buffer)
        gendata(
            label_out_path, 
            data_out_path, 
            keypoints_path,
            history=history, 
            audio_path=audio_path,
            buffer=buffer
        )