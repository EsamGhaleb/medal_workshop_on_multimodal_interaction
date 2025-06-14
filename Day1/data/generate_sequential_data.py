from __future__ import print_function
import sys
sys.path.extend(['.'])
project_directory = ''
import argparse
import pickle
import sys
import os
from itertools import product
from collections import Counter

import numpy as np
import librosa
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
    '59': np.concatenate((np.arange(0,17), np.arange(91,133)), axis=0), #59
    '31': np.concatenate((np.arange(0,11), [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0), #31
    '27': np.concatenate(([0,5,6,7,8,9,10], 
                    [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0), #27
   'CABB': selected_markers
}



def calculate_overlap(start1, end1, start2, end2):
    # Find the earliest and latest start and end times
    earliest_start = min(start1, start2)
    latest_end = max(end1, end2)
    latest_start = max(start1, start2)
    earliest_end = min(end1, end2)

    # Calculate the overlap, if any
    overlap = 0
    if latest_start < earliest_end:
        overlap = (earliest_end - latest_start)

    # Calculate the total duration
    total_duration = (end1-start1)

    # Calculate the percentage overlap
    if total_duration > 0:
        percentage_overlap = (overlap / total_duration) # * 100
    else:
        percentage_overlap = 0
    return percentage_overlap

      
def overlap_percentage(window, annotation):
    # Calculate the overlap as the intersection of the two ranges
    overlap = max(0, min(window[1], annotation[1]) - max(window[0], annotation[0]) + 1)

    # Calculate the length of the window and annotation
    window_length = window[1] - window[0] + 1
    annotation_length = annotation[1] - annotation[0] + 1

    # Check if the window is entirely within the annotation or vice versa
    if window[0] >= annotation[0] and window[1] <= annotation[1]:  # window within annotation
        percentage = overlap / window_length
        status = 'full'
        started_ended = 'inside'
    elif annotation[0] >= window[0] and annotation[1] <= window[1]:  # annotation within window
        percentage = overlap / annotation_length
        status = 'full'
        started_ended = 'inside'
    elif window[1] < annotation[1]:  # Window is on the left side of the annotation
        percentage = overlap / window_length
        if percentage < 0.05:
            status = 'outside'
        elif percentage < 0.25:
            status = 'starting'
        elif percentage < 0.5:
            status = 'early'
        elif percentage < 0.75:
            status = 'middle'
        else:
            status = 'full'
        started_ended = 'started'
    else:  # Window is on the right side of the annotation
        percentage = overlap / window_length
        if percentage < 0.05:
            status = 'outside'
        elif percentage < 0.25:
            status = 'ending'
        elif percentage < 0.5:
            status = 'late'
        elif percentage < 0.75:
            status = 'middle'
        else:
            status = 'full'
        started_ended = 'ended'

    # select the first 2 decimal points of the percentage
    percentage = int(percentage * 10000) / 10000
    return percentage, status, started_ended

def find_status(
        start_frame, 
        end_frame,
        data_start_frame,
        data_end_frame,
        gesture_type,
        gestures_info
        ):
    window = (start_frame, end_frame)
    annotation = (data_start_frame, data_end_frame)
    if gesture_type != 'iconic':
        # first calculate the overlap percentages the current window has with all the gestures 
        # if there is an overlap more than 10% with any of the gestures, then this window is not good
        overlap_percentages = gestures_info.apply(lambda x: calculate_overlap(start_frame, end_frame, x['start_frame'], x['end_frame']), axis=1)
        if overlap_percentages.max() > 0.1:
            return -1, "outsie", ""
        percentage = overlap_percentages.max()
        status = 'outside'
        started_ended = ""
    else:
        percentage, status, started_ended = overlap_percentage(window, annotation)
    return percentage, status, started_ended
 
 
def generate_data_sw(
        data_dict, 
        gestures_info,
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

    label_format = "overlap_percentage_{:7.5f}_status_{:7}"
    sample_format = "{:6}_{:1}_{:06d}_{:06d}"
    data = data_dict[(pair, speaker)]
    start_ind = 0
    end_ind = data.shape[0] - (history + num_frames) * time_offset
    FPS = 29.97002997002997
    # Calculate samples per video frame
    samples_per_frame = int(sample_rate / FPS)  # 533 samples
    # Calculate the step size for an offset of 2 frames
    buffer_in_frames = buffer * FPS
    num_frames_in_seconds = (num_frames+buffer_in_frames) / FPS
    corresponding_audio_window_size = int(num_frames_in_seconds * sample_rate)
    for t in tqdm(range(start_ind, end_ind, time_offset*history), leave=False):
        arr = np.stack([
            data[t + i*time_offset : t + (history+i)*time_offset : time_offset, ...]
            for i in range(num_frames)
        ])
        arr = rearrange(arr, 'f t k d -> t d f k')
        all_speakers_fps[current_ind] = arr
        percentages = list()
        statuses = list()
        for i in range(t, t + history*time_offset, time_offset):
            start_frame = i
            end_frame = i + num_frames
            window = (start_frame, end_frame)

            # start frame for the audio 
            audio_start_frame = start_frame * samples_per_frame
            # transform the audio window 
            # Extract features using VGGish
            # Extract features using Wav2Vec2
            # For demonstration, just print the shape of last_hidden_state
            # apply mean pooling on the extracted features
            # convert the tensor to numpy array
            all_intersections = gestures_info.apply(
                lambda x: 
                overlap_percentage(window, (x["start_frame"], x["end_frame"])), axis=1
                )
            all_intersections = list(all_intersections)
            if len(all_intersections) == 0:
                status = 'outside'
                percentage = 0.0
            else:
                percentages_i = [elem[0] for elem in all_intersections]
                percentage_i_argmax = percentages_i.index(max(percentages_i))
                percentage = percentages_i[percentage_i_argmax]
                status = [elem[1] for elem in all_intersections][percentage_i_argmax]
                
            percentages.append(percentage)
            statuses.append(status)
        label = [label_format.format(p,s) for p, s in zip(percentages, statuses)]
        sample_names = [
            sample_format.format(pair, speaker, i, i+num_frames)
            for i in range(t, t + history*time_offset, time_offset)
        ]
        all_sample_names[current_ind] = sample_names
        sequences_labels[current_ind] = label
        current_ind += 1
    return all_speakers_fps, sequences_lengths, all_sample_names, sequences_labels, [None]*len(sequences_lengths), current_ind
    

def load_data(keypoints_path, pairs, speakers, config):
    data_dict = dict.fromkeys(product(pairs, speakers))
    for pair, speaker in product(pairs, speakers):
        audio_path = keypoints_path.format(pair, speaker)
        keypoints = np.load(audio_path)
        selected = selected_joints[config]
        keypoints = keypoints[:, selected, :]
        data_dict[(pair, speaker)] = keypoints
    return data_dict

def get_data_size(data_dict, pair_speaker, history, time_offset, num_frames):
    total_num_samples = 0
    for pair in pair_speaker:
        x = data_dict[pair].shape[0]
        case_num_samples = (x - (history + num_frames) * time_offset) // (time_offset * history) + 1
        total_num_samples += case_num_samples
    return total_num_samples

def get_part_pair_speaker(data_df):
    unique_pair_speaker = {tuple(elem) for elem in data_df[["pair", "speaker"]].values}
    unique_pair_speaker = sorted(unique_pair_speaker)
    return unique_pair_speaker


def gendata(
        all_data, 
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
    pairs = np.unique(all_data['pair'].to_numpy())
    speakers = np.unique(all_data['speaker'].to_numpy())
    data_dict = load_data(keypoints_path, pairs, speakers, config)
    pair_speaker = get_part_pair_speaker(all_data)
    total_num_samples = get_data_size(data_dict, pair_speaker, history, time_offset, num_frames)
    all_speakers_fps = np.zeros((total_num_samples, history, 3, num_frames, int(config)))

    sequences_lengths = history * np.ones(total_num_samples)
    all_sample_names = np.empty((total_num_samples, history), dtype='U22')
    sequences_labels = np.empty((total_num_samples, history), dtype='U47')
    
    all_pair_speaker_referent = []
    current_ind = 0
    for pair, speaker in tqdm(pair_speaker, leave=True):
        data = all_data[(all_data['pair'] == pair) & (all_data['speaker'] == speaker)]
        data = data.reset_index(drop=True)
        # get the number of iconic gestures
        data = data[['start_frame', 'end_frame', 'label', 'speaker', 'pair', 'is_gesture']]
        # remove duplicate rows
        data = data.drop_duplicates()
        iconic_gestures = data[data['is_gesture'] == 'gesture']
        ret_val = generate_data_sw(
            data_dict=data_dict,
            gestures_info=iconic_gestures,
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
     
    c = Counter()
    for label in sequences_labels:
        c.update([elem.split('_')[-1] for elem in label])

    overhead = c[''] // history
    print('overhead: {}'.format(overhead))
    if overhead > 0:
        all_speakers_fps = all_speakers_fps[:-overhead]
        sequences_lengths = sequences_lengths[:-overhead]
        all_sample_names = all_sample_names[:-overhead]
        sequences_labels = sequences_labels[:-overhead]

    with open(label_path, 'wb') as f:
        pickle.dump((all_sample_names, all_pair_speaker_referent, sequences_labels, sequences_lengths), f)
    all_speakers_fps = np.array(all_speakers_fps) 
    print(all_speakers_fps.shape)
    np.save(out_path, all_speakers_fps)
    print('saved to {}'.format(out_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process audio and video data')
    parser.add_argument('--keypoints-path', type=str, help='The path to the numpy file', default='../gestural_alignment/data/final_poses/poses_{}_synced_pp{}.npy')
    args = parser.parse_args()
    keypoints_path = args.keypoints_path
    # '/home/hpc/b105dc/b105dc10/co-speech-v2/co-speech-gesture-detection/data/videos/npy3/{}_synced_pp{}.npy'
    out_folder = project_directory + '.'
    history = 40
    new_gestures_info_path = project_directory+'data/gestures_info.pkl'
    if os.path.exists(new_gestures_info_path):
        gestures_info = pd.read_pickle(new_gestures_info_path)
    else:
        raise ValueError("Provide correct path to the mmpose pickle file!")
    # gestures_info contains the following columns: ['pair', 'speaker', 'start_frame', 'end_frame', 'referent', 'label', 'is_gesture', 'keypoints']
    out_folder = project_directory+'data/mm_data/'
    # drop the keypoint column
    labels = np.array(gestures_info['label'].to_list())
    unique_labels = np.unique(labels)
    gesture_ids = {}
    for i, label in enumerate(unique_labels):
        gesture_ids[label] = i
    discrete_labels = []
    for label in labels:
        discrete_labels.append(gesture_ids[label])
    gestures_info['pair_speaker'] = gestures_info.apply(lambda x: str(x['pair']) + '_' + str(x['speaker']), axis=1)    
    # check if the out_folder exists
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for buffer in [0.0]:
        data_out_path = '{}data_27_joint_buffer_{}.npy'.format(out_folder, buffer)
        audio_out_path = '{}audio_buffer_{}.npy'.format(out_folder, buffer)
        label_out_path = '{}27_labels_buffer_{}.pkl'.format(out_folder, buffer)
        referents_speakers_path = '{}27_referent_speakers_buffer_{}.pkl'.format(out_folder, buffer)
        gendata(
            gestures_info, 
            label_out_path, 
            data_out_path, 
            keypoints_path,
            history=history, 
            audio_path=audio_path,
            buffer=buffer
        )