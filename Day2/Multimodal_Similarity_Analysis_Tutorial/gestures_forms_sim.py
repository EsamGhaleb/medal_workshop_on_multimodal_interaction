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
from tqdm import tqdm
# import model from ../model
import sys
from model.skeleton_speech_models import GSSModel
# computer the correlation between similarity_score and skeleton_feats_sim
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
from data.read_process_poses import load_keypoints_dict, get_bone


def plot_boxenplot(gestures_form,   x = 'similarity_score', y = 'skeleton_feats_similarity'):
   from statannotations.Annotator import Annotator
   sns.set_theme(style="whitegrid")
   ax = sns.barplot(x=x, y=y, data=gestures_form, hue='similarity_score', palette='Paired')
   means = gestures_form.groupby('similarity_score')[y].mean()
   # remove legend
   ax.get_legend().remove()

   order = [0, 1, 2, 3, 4, 5]
   pairs=[(5, 0), (5, 1), (5, 2), (5, 3), (5, 4)]
   annotator = Annotator(ax, pairs, data=gestures_form, x=x, y=y, order=order)
   annotator.configure(test='t-test_ind', text_format='star', loc='inside')
   annotator.apply_and_annotate()
   y_max = ax.get_ylim()[1]-0.0
   for i, mean in enumerate(means):
      ax.text(i, y_max, round(mean, 2), fontsize=15, color='blue')

   # rename the x label 
   ax.set_xlabel('Number of Similar Features')
   ax.set_ylabel('Cosine Similarity')
   return ax

def get_speech_waveform(pair_speaker, from_ts, to_ts, audio_dict, device):
   speech_buffer = 0.5
   speech_start_time = from_ts-speech_buffer
   speech_end_time = to_ts+speech_buffer
   sample_rate = 16000
   # convert the speech_start_frame and speech_end_frame to speech frame
   start_frame = int(speech_start_time * sample_rate)
   end_frame = int(speech_end_time * sample_rate)
   if start_frame < 0:
      padding = np.zeros(abs(start_frame))
      start_frame = 0
      audio_data = np.concatenate(
         (padding, audio_dict[pair_speaker]['audio'][start_frame:end_frame])
      )
   elif end_frame > len(audio_dict[pair_speaker]['audio']):
      padding = np.zeros(end_frame - len(audio_dict[pair_speaker]['audio']))
      end_frame = len(audio_dict[pair_speaker]['audio'])
      audio_data = np.concatenate(
         (audio_dict[pair_speaker]['audio'][start_frame:end_frame], padding)
      )
   else:
      audio_data = audio_dict[pair_speaker]['audio'][start_frame:end_frame]
   
   # convert audio data to torch tensor
   audio_data = torch.tensor(audio_data).float().to(device).unsqueeze(0).unsqueeze(0)
   return audio_data

   
def get_all_poses(poses):
   # reset index of the data data/segmented_gestures/small_gesture_results.pkl
   data = pd.read_pickle('data/segmented_gestures/small_gesture_results.pkl')
   data = data[data['pair_speaker'].isin(poses.keys())]
   data = data.reset_index(drop=True)
   all_poses = np.zeros((len(data), 3, 30, 27, 1))
   for index, row in tqdm(data.iterrows(), total=data.shape[0], desc='Loading poses...'):
      start_frame = int(row['start_frames']) #- round(gesture_buffer * fps)
      end_frame = int(row['end_frames']) #+ round(gesture_buffer * fps)
      middle_frame = (start_frame + end_frame) // 2
      # take 15 frames before and after the middle frame
      start_frame = middle_frame - 15
      end_frame = start_frame + 30
      pair_speaker = row['pair_speaker']
      C, T, V, _= poses[pair_speaker].shape
      all_poses[index] = poses[pair_speaker][:, start_frame:end_frame, :, :]
   
   N, C, T, V, M = all_poses.shape
   mean_map = all_poses.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
   std_map = all_poses.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))
   return mean_map, std_map

def process_skeleton(skel, mirorred_skel):
   skel[0,:,:,:] = skel[0,:,:,:] - skel[0,:,0,0].mean(axis=0)
   skel[1,:,:,:] = skel[1,:,:,:] - skel[1,:,0,0].mean(axis=0)
   mirorred_skel[0,:,:,:] = mirorred_skel[0,:,:,:] - mirorred_skel[0,:,0,0].mean(axis=0)
   mirorred_skel[1,:,:,:] = mirorred_skel[1,:,:,:] - mirorred_skel[1,:,0,0].mean(axis=0)
   mirorred_skel = np.expand_dims(mirorred_skel, axis=0)
   skel = np.expand_dims(skel, axis=0)
   # concatenate the two skeletons
   skel = np.concatenate([skel, mirorred_skel], axis=0)
   
   return skel
 
def get_distances(two_speakers_feats, cos=torch.nn.CosineSimilarity(dim=1, eps=1e-6)):
   cosine_distances = []
   batch1 = torch.tensor(two_speakers_feats[0])
   batch2 = torch.tensor(two_speakers_feats[1])
   for vec1 in batch1:
      for vec2 in batch2:
         # Compute cosine similarity and then convert it to cosine distance
         cos_sim = cos(vec1.unsqueeze(0), vec2.unsqueeze(0))
         cosine_distances.append(cos_sim.item())
   # return the average 
   average_sim = np.mean(cosine_distances)
   return average_sim


def time_to_seconds(time_str):
    # Split the time string by colon and period to separate hours, minutes, seconds, and milliseconds
    hours, minutes, seconds = time_str.split(':')
    seconds, milliseconds = seconds.split('.')

    # Convert hours, minutes, seconds, and milliseconds to integers
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    milliseconds = int(milliseconds)

    # Calculate total seconds
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000

    return total_seconds


def measure_sim_gestures(sup_model, processed_keypoints_dict, mirrored_keypoints_dict, multimodal=True, bones=False, both=False, before_head_feats=False, skeleton_backbone='jointsformer', modalities=['skeleton', 'speech'], audio_dict=None):
   cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
   gestures_form = pd.read_csv('data/gesture_form_similarity_coding.csv', sep=';')
   aligned_tagged_speech_per_word_small = pd.read_csv('dialog_utils/data/aligned_tagged_speech_per_word_small.csv')
   # gesture similarity encoded as H1P0S0M0R0, build a dictionary for each letter
   # gestures_form['gesture_similarity_coding'].apply(lambda x: x.split('0'))
   similarity_dict = {'H': [], 'P': [], 'S': [], 'M': [], 'R': []}
   similarity_scores = []
   gestures_form['handedness'] = [int(x.split('H')[1][0]) for x in gestures_form['gesture_similarity_coding']]
   gestures_form['position'] = [int(x.split('P')[1][0]) for x in gestures_form['gesture_similarity_coding']]
   gestures_form['shape'] = [int(x.split('S')[1][0]) for x in gestures_form['gesture_similarity_coding']]
   gestures_form['movement'] = [int(x.split('M')[1][0]) for x in gestures_form['gesture_similarity_coding']]
   gestures_form['rotation'] = [int(x.split('R')[1][0]) for x in gestures_form['gesture_similarity_coding']]
   for index, row in tqdm(gestures_form.iterrows(), total=gestures_form.shape[0]):
      for letter in similarity_dict.keys():
         similarity_dict[letter].append(int(row['gesture_similarity_coding'].split(letter)[1][0]))
      similarit_score = 0
      for letter in similarity_dict.keys():
         if similarity_dict[letter][-1] == 9:
            similarit_score += 0
         else:
            similarit_score += similarity_dict[letter][-1]
      similarity_scores.append(similarit_score)
   # count the number of 5 in the similarity scores
   similarity_scores = np.array(similarity_scores)
   similarity_scores[similarity_scores == 5].shape[0]/similarity_scores.shape[0]
   gestures_form['similarity_score'] = similarity_scores
   content_words_pos = ['ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB', 'NUM']  # removed 'INTJ'
   sup_model.eval()
   # for device, get the sup_model to the device
   device = sup_model.parameters().__next__().device
   gestures_form['skeleton_feats_similarity'] = 0
   concatenated_features = []
   for i, row in tqdm(gestures_form.iterrows(), total=gestures_form.shape[0], desc='Calculating the similarity between the two speakers'):
      pair = row['pairnr']
      if pair < 10:
         pair = 'pair0'+str(pair)
      else:
         pair = 'pair'+str(pair)
      two_speakers_feats = []
      try:
         for speaker in ['A', 'B']:
            from_ts = row['{}_begin_hmsm'.format(speaker)]
            to_ts = row['{}_end_hmsm'.format(speaker)]            
            start_frame = int(time_to_seconds(from_ts) * 29.97)
            end_frame = int(time_to_seconds(to_ts) * 29.97)
            if end_frame - start_frame < 2:
               end_frame = end_frame + 2
               start_frame = start_frame - 2
            if end_frame - start_frame > 60: ## if the gesture is too long, we choose the middle 60 frames
               # choose the middle 60 frames
               start_frame = start_frame + (end_frame - start_frame - 60) // 2
               end_frame = start_frame + 60
            
            
            pair_speaker = pair+'_'+speaker
            from_ts = start_frame/29.97
            to_ts = end_frame/29.97
            
            if 'speech' in modalities:
               speech_waveform = get_speech_waveform(pair_speaker, from_ts, to_ts, audio_dict, device)
            else:
               speech_waveform = None
               
            # the frame here is in seconds
            speech_start_time = from_ts-1
            speech_end_time = to_ts + 1
            speech_info = aligned_tagged_speech_per_word_small[(aligned_tagged_speech_per_word_small['pair_name'] == pair) & (aligned_tagged_speech_per_word_small['speaker'] == speaker) & (aligned_tagged_speech_per_word_small['from_ts'] >= speech_start_time) & (aligned_tagged_speech_per_word_small['to_ts'] <= speech_end_time)]
            if not speech_info.empty:
               sentences = ' '.join(speech_info['word']).split()
               pos_seq = ' '.join(speech_info['pos']).split()
               assert len(sentences) == len(pos_seq)
               content_words = ' '.join([sentences[i] for i in range(len(sentences)) if pos_seq[i] in content_words_pos])
               sentences = ' '.join(sentences)
            else:
               sentences = 'no speech'
         
            skel = processed_keypoints_dict[pair_speaker][:, start_frame:end_frame, :, :]
            mirored_skel = mirrored_keypoints_dict[pair_speaker][:, start_frame:end_frame, :, :]
            if speaker == 'B':
               skel, mirored_skel = mirored_skel, skel
            skel = process_skeleton(skel, mirored_skel)
            if bones:
               skel = get_bone(skel)
            # construct a random speech waveform of 2 seconds, sr=16000
            # convert the numpy array to tensor
            skel = torch.tensor(skel).to(device)
            # convert skel to float
            skel = skel.float()
            skel = skel.squeeze().float()
            if skeleton_backbone != 'stgcn':
               skel = skel.permute(0, 2, 3, 1)
            else:
               skel = skel.unsqueeze(-1)
            with torch.no_grad():
               # skeleton_feats = gcn_model(skel)
               mm_outputs = sup_model(skel, utterances=[sentences], features=modalities, speech_waveform=speech_waveform)
               if mm_outputs['semantic_features'] is not None:
                  semantic_feats = mm_outputs['semantic_features']
               skeleton_feats = mm_outputs['skeleton_features']
               # normalize the skeleton features
               skeleton_feats = torch.nn.functional.normalize(skeleton_feats, dim=1)
               concatenated_skeleton_feats = np.concatenate(skeleton_feats.cpu().numpy(), axis=0)
            two_speakers_feats.append(skeleton_feats.cpu().numpy())
         concatenated_features.append(concatenated_skeleton_feats)
         similarity = get_distances(two_speakers_feats, cos=cos)
         gestures_form.at[i, 'skeleton_feats_similarity'] = similarity.item()
      except Exception as e:
         print(e)
         print('Error in calculating the similarity for the pair:', pair, i)
         continue
   # calculate the correlation between the similarity scores and the skeleton_feats_similarity
   pearson_correlation, _ = pearsonr(gestures_form['similarity_score'], gestures_form['skeleton_feats_similarity'])
   spearman_correlation, _ = spearmanr(gestures_form['similarity_score'], gestures_form['skeleton_feats_similarity'])
   # print the correlation
   print('Pearson correlation:', pearson_correlation)
   print('Spearman correlation:', spearman_correlation)
   concatenated_features = np.concatenate(concatenated_features, axis=0)
   # gestures_form['concatenated_features'] = concatenated_features.tolist()

   difference = gestures_form[gestures_form['similarity_score']==5]['skeleton_feats_similarity'].mean() - gestures_form[gestures_form['similarity_score']==1]['skeleton_feats_similarity'].mean()
   
   return spearman_correlation, difference, gestures_form

if __name__ == '__main__':
   processed_keypoints_dict, mirrored_keypoints_dict = load_keypoints_dict() 
   sup_model = GSSModel()
   measure_sim_gestures(sup_model=sup_model, processed_keypoints_dict=processed_keypoints_dict, mirrored_keypoints_dict=mirrored_keypoints_dict)