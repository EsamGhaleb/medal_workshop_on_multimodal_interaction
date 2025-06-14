
import numpy as np
from scipy.special import softmax
import pandas as pd
from collections import defaultdict
from elan_data import ELAN_Data
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

from tqdm import tqdm

# upload labels
fps = 25
SAMPLE_RATE = 16000



pairs_mappings = {}
for i in range(1, 100):
   key = int(f"{i}")
   value = f"{i:03}"
   key = int(value)
   pairs_mappings[key] = value
sample_pairs_mappings = {k: pairs_mappings[k] for k in list(pairs_mappings)}
sample_pairs_mappings



def upload_models_result(results, pairs_mapping=pairs_mappings):
   # with open(results_path, 'rb') as f:
   #    results = pickle.load(f)
   all_results_details = pd.DataFrame() 
   results_dict = {}
   speakers_mapping = {0: 'B', 1: 'C'}
   for fold in tqdm(range(5)):
      n_samples, num_seq = results[fold]['labels'].shape
      print(f"Fold {fold} - Number of samples: {n_samples}, Number of sequences: {num_seq}")
      labels = results[fold]['labels'].reshape(n_samples* num_seq)
      preds = results[fold]['preds'].reshape(n_samples * num_seq, results[fold]['preds'].shape[-1])
      
      results[fold]['speaker_ID'] = np.array([num_seq * [pair_speaker_ID] for pair_speaker_ID in results[fold]['speaker_ID']])
      results[fold]['pair_ID'] = np.array([num_seq * [pair_ID] for pair_ID in results[fold]['pair_ID']])
      results[fold]['start_frames'] = np.array([num_seq * [start_frame] for start_frame in results[fold]['start_frames']])
      results[fold]['end_frames'] = np.array([num_seq * [end_frame] for end_frame in results[fold]['end_frames']])
      speaker_ID = results[fold]['speaker_ID'].reshape(n_samples * num_seq)
      pair_ID = results[fold]['pair_ID'].reshape(n_samples * num_seq)
      pair_speaker = [f"{int(pair)}_{speakers_mapping[int(speaker)]}" for pair, speaker in zip(pair_ID, speaker_ID)]
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
            
def save_elan_files(elan_file_path, video_output_path, binar_gesture_predictions, gesture_predictions, fps, model='skeleton'):
 
   # existing_elan_file = ELAN_Data.from_file('elan_files/{}_SignAcc_{}_CP.eaf'.format(pair, day))
   # check if the folder exists, if not create it
   import os
   if os.path.exists('eaf_files') is False:
      os.makedirs('eaf_files')
   new_eaf = ELAN_Data.create_eaf(elan_file_path, audio=video_output_path, tiers=["{} model".format(model)], remove_default=True)

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
                  i = end_frame
                  # average_gesture_predictions = gesture_predictions[start_frame:end_frame]
                  # phases = divide_segment(average_gesture_predictions, 0, len(average_gesture_predictions))
                  # phases['start_frame'] = np.array(phases['start_frame'])+start_frame
                  # phases['end_frame'] = np.array(phases['end_frame'])+start_frame
                  # num_phases = len(phases['start_frame'])   
                  # for phase_id in range(num_phases):
                  #    #  print('Phase start frame:', phases['start_frame'][phase_id])
                  #     phase_start_frame = phases['start_frame'][phase_id]
                  #     phase_end_frame = phases['end_frame'][phase_id]
                  start_ts = start_frame / fps * 1000
                  end_ts = end_frame / fps * 1000
                  if end_frame - start_frame <= 2:
                     break
                     # print('Duration:', end_frame - start_frame)
                  mean_preds = gesture_predictions[start_frame:end_frame].mean()
                  new_eaf.add_segment("{} model".format(model), start=start_ts, stop=end_ts, annotation="prob-{:.2f}".format(mean_preds))
                  # existing_elan_file.add_segment("Sign".format(model), start=start_ts, stop=end_ts, annotation="prob-{:.2f}".format(mean_preds))
                  break 
               else:
                  i += 1
      else:
         i += 1
   # ed.write()
   new_eaf.save_ELAN(raise_error_if_unmodified=False)
   # existing_elan_file.save_ELAN(raise_error_if_unmodified=False, rename=existing_elan_file_path)


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

def get_elan_files(results, fps=25, model='skeleton', threshold=0.55, file_path='test_videos/tedtalk.npy', video_output_path=None):
   all_results = upload_models_result(results)
   all_results, all_preds = get_class_predictions(all_results)
   elan_file_path = file_path.replace('.npy', '_segmentation_results.eaf')
   all_results['speaker'] = all_results['pair_speaker'].apply(lambda x: x.split("_")[1])

   all_pairs_speakers = all_results['pair_speaker'].unique()
   for pair_speaker in tqdm(all_pairs_speakers, total=len(all_pairs_speakers)):
   #   pair_speaker = '035036_A'
      pair = pair_speaker.split("_")[0]
      speaker = pair_speaker.split("_")[1]
      print('Pair:', pair)
      print('Speaker:', speaker)
      pair_speaker_results = all_results[all_results['pair_speaker'] == pair_speaker]
      gesture_predictions = pair_speaker_results['preds'].to_numpy()
      gesture_predictions = get_predictions(pair_speaker_results)[:, 1]
      
      binar_gesture_predictions = gesture_predictions >= threshold #NOTE: 0.40 is the threshold which we can change and see what is the best threshold
      smoothing_factor = int(0.2 * fps)
      smoothed = median_filter(binar_gesture_predictions, size=smoothing_factor)
      binar_gesture_predictions = smoothed
      # assign -1 to the frames where the gesture predictions are less than 0.5
      gesture_predictions[binar_gesture_predictions == 0] = -1
      save_elan_files(elan_file_path, video_output_path, binar_gesture_predictions, gesture_predictions, fps, model=model)



