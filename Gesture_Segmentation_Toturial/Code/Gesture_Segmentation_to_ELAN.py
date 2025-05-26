
import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import label_binarize
import pickle
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time
from moviepy.editor import VideoFileClip
from IPython.display import Video
from itertools import combinations
from collections import defaultdict
from elan_data import ELAN_Data
from scipy.signal import find_peaks


from tqdm import tqdm

# upload labels
buffer = 0.5
fps = 29.97
labels_path = "data/mm_data_vggish/27_labels_buffer_{}.pkl".format(buffer)

videos_path = "/home/eghaleb/data/{}_synced_pp{}.mp4"
SAMPLE_RATE = 16000



pairs_mappings = {}
for i in range(1, 150, 2):
   j = i + 1
   key = int(f"{i}{j}")
   value = f"{i:03}{j:03}"
   key = int(value)
   pairs_mappings[key] = value
   if i == 149 and j == 150:
      break
sample_pairs_mappings = {k: pairs_mappings[k] for k in list(pairs_mappings)}
sample_pairs_mappings



def upload_models_result(results_path, pairs_mapping=pairs_mappings):
   with open(results_path, 'rb') as f:
      results = pickle.load(f)
   all_results_details = pd.DataFrame() 
   results_dict = {}
   speakers_mapping = {0: 'A', 1: 'B'}
   for fold in tqdm(range(5)):
      if fold == 2:
         continue
      n_gpus, n_samples, num_seq = results[fold]['labels'].shape
      labels = results[fold]['labels'].reshape(n_gpus * n_samples* num_seq)
      preds = results[fold]['preds'].reshape(n_gpus * n_samples * num_seq, results[fold]['preds'].shape[-1])
      speaker_ID = results[fold]['speaker_ID'].reshape(n_gpus * n_samples * num_seq)
      pair_ID = results[fold]['pair_ID'].reshape(n_gpus * n_samples * num_seq)
      pair_speaker = [f"{pairs_mapping[int(pair)]}_{speakers_mapping[int(speaker)]}" for pair, speaker in zip(pair_ID, speaker_ID)]
      # repeat pair_speaker for each frame, 40 times
      start_frames = results[fold]['start_frames'].reshape(-1)
      end_frames = results[fold]['end_frames'].reshape(-1)
      folds = np.array([fold] * len(labels))
      results_dict['labels'] = labels
      if fold == 0:
         results_dict['preds'] = preds 
      else:
         results_dict['preds'] = results_dict['preds'] +  preds
      results_dict['pair_speaker'] = pair_speaker
      results_dict['start_frames'] = start_frames
      results_dict['end_frames'] = end_frames
      results_dict['folds'] = folds 
      
   # convert to dataframe
   results_dict['preds'] = results_dict['preds']/5
   all_results_details = pd.DataFrame(results_dict, columns=['labels', 'start_frames', 'end_frames', 'folds'])
   # add preds
   all_results_details['preds'] = results_dict['preds'].tolist()
   all_results_details['pair_speaker'] = results_dict['pair_speaker']
   return all_results_details 

# find local maxima
def divide_segment(gesture_predictions, start_frame, end_frame):
    # find the local maxima
    average_gesture_predictions = gesture_predictions[start_frame:end_frame]
    min_peaks, min_peaks_dict = find_peaks(-average_gesture_predictions)
    peaks, peaks_dict = find_peaks(average_gesture_predictions)
    # print('Peaks:', peaks)
    sub_segments = defaultdict(list)
    # assert len(peaks) == len(min_peaks)+1 or len(peaks) < 5
    if len(peaks) == 0 or len(min_peaks) == 0 or len(peaks) == 1:
        sub_segments['start_frame'].append(start_frame)
        sub_segments['end_frame'].append(end_frame)
    else:
        for peak_id, peak in enumerate(peaks[:-1]):
            # print('Peak:', peak)
            # print('Peak ID:', peak_id)
            sub_segments['start_frame'].append(start_frame)
            if peak_id == len(peaks)-2:
                sub_segments['end_frame'].append(end_frame)
            else:
                sub_segments['end_frame'].append(min_peaks[peak_id])
            start_frame = min_peaks[peak_id]+1
    return sub_segments
            
           



def save_elan_files(pair_speaker, binar_gesture_predictions, gesture_predictions, min_gesture_predictions, fps, model='skeleton'):
   new_eaf = ELAN_Data.create_eaf("final_eaf_files/{}.gesture_{}_final.eaf".format(pair_speaker, model), audio="{}_synced_pp{}.mp4".format(pair, speaker),
                                    tiers=["{} model".format(model)],
                                    remove_default=True)

   i = 0
   while i < len(binar_gesture_predictions):
      label = binar_gesture_predictions[i]
      if label:
         # print('start_frame=', i)
         start_frame = i
         start_ts = start_frame / fps
         for end_frame in range(i, len(binar_gesture_predictions)):
               # print('Gesture prediction {:.2f}'.format(gesture_predictions[end_frame]))
               # print('Minimum prediction {:.2f}'.format(min_gesture_predictions[end_frame]))
               # print('Gesture prediction {:.2f}'.format(gesture_predictions[end_frame]))
               if not binar_gesture_predictions[end_frame]:
                  # print('end_frame=', end_frame)
                  # print('Duration:', end_frame - start_frame)
                  i = end_frame+1
                  average_gesture_predictions = gesture_predictions[start_frame:end_frame]
                  phases = divide_segment(average_gesture_predictions, 0, len(average_gesture_predictions))
                  phases['start_frame'] = np.array(phases['start_frame'])+start_frame
                  phases['end_frame'] = np.array(phases['end_frame'])+start_frame
                  num_phases = len(phases['start_frame'])   
                  for phase_id in range(num_phases):
                     #  print('Phase start frame:', phases['start_frame'][phase_id])
                      phase_start_frame = phases['start_frame'][phase_id]
                      phase_end_frame = phases['end_frame'][phase_id]
                      start_ts = phase_start_frame / fps * 1000
                      end_ts = phase_end_frame / fps * 1000
                      mean_preds = gesture_predictions[phase_start_frame:phase_end_frame].mean()
                      new_eaf.add_segment("{} model".format(model), start=start_ts, stop=end_ts, annotation="prob-{:.2f}".format(mean_preds))
                  break 
               else:
                  i += 1
      else:
         i += 1
   # ed.write()
   new_eaf.save_ELAN(raise_error_if_unmodified=False)


def get_predictions (results_details):
   predictions = results_details["preds"]
   predictions_n = np.array([float(pred[0]) for pred in predictions])
   predictions_g = np.array([float(pred[1]) for pred in predictions])
   predictions = np.column_stack((predictions_n, predictions_g))
   # apply softmax
   predictions = softmax(predictions, axis=1)
   return predictions
def get_class_predictions(results_details):
   predictions = results_details["preds"]
   predictions_n = np.array([float(pred[0]) for pred in predictions])
   predictions_g = np.array([float(pred[1]) for pred in predictions])
   predictions = np.column_stack((predictions_n, predictions_g))
   # apply softmax
   predictions = softmax(predictions, axis=1)
   results_details["gesture_preds"] = predictions[:, 1]
   results_details["gesture_preds"] = results_details["gesture_preds"].astype(float)
   results_details["neutral_preds"] = predictions[:, 0]
   results_details["neutral_preds"] = results_details["neutral_preds"].astype(float)
   return results_details, predictions
# best speech results

all_results_path = {
   # 'skeleton': 'tb_logs/Appr_Skeleton_fold_4_lr_5e-05_subject_joint_False_gesture_unit_True_ft_speech_False_vggish_False_speech_buffer_0.0_offset_2_schedular_plateau_audio_encoder_False_skeleton_encoder_True_crf_False_bs_40_sc_True_final/test_results.pkl',
   'skeleton&speech': 'tb_logs/Appr_EarlyFusion_fold_4_lr_0.0001_subject_joint_False_gesture_unit_True_ft_speech_False_vggish_False_speech_buffer_0.0_offset_2_schedular_plateau_audio_encoder_False_skeleton_encoder_False_crf_False_bs_24_sc_False_/test_results.pkl'}


for key in all_results_path:
   results_path = all_results_path[key]
   model = key
   all_results = upload_models_result(results_path)
   all_results, all_preds = get_class_predictions(all_results)
   videos_path = "/home/eghaleb/data/{}_synced_pp{}.mp4"
   SAMPLE_RATE = 16000


   all_results['speaker'] = all_results['pair_speaker'].apply(lambda x: x.split("_")[1])

   pair_speaker = "119120_B"
   all_pairs_speakers = all_results['pair_speaker'].unique()
   for pair_speaker in tqdm(all_pairs_speakers, total=len(all_pairs_speakers)):
   #   pair_speaker = '035036_A'
      pair = pair_speaker.split("_")[0]
      speaker = pair_speaker.split("_")[1]
      poses = np.load("data/selected_poses/poses_{}_synced_pp{}.npy".format(pair, speaker), allow_pickle=True)
      pair_speaker_results = all_results[all_results['pair_speaker'] == pair_speaker]
      gesture_predictions = np.zeros(len(poses))
      min_gesture_predictions = np.zeros(len(poses))
      for frame, pose in tqdm(enumerate(poses), total=len(poses)):
            # get the rows where the frame is between start and end frames
            row = pair_speaker_results[(pair_speaker_results['start_frames'] <= frame) & (pair_speaker_results['end_frames'] >= frame)]
            # take the average of the gesture predictions
            gesture_predictions[frame] = row['gesture_preds'].mean()
            min_gesture_predictions[frame] = row['gesture_preds'].min()
      binar_gesture_predictions = gesture_predictions > 0.5
      save_elan_files(pair_speaker, binar_gesture_predictions, gesture_predictions,min_gesture_predictions, fps, model=model)
   #   break



