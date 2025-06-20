import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from functools import partial
from scipy.spatial.distance import cdist
from tqdm import tqdm

# Imports from your project modules (adjust paths as needed)
from dialog_utils.prepare_gesture_referents_dataset import get_detailed_gestures
from gestures_forms_sim import process_skeleton, get_bone, get_speech_waveform
from data.read_process_poses import load_keypoints_dict

def compute_topk_accuracies(retrieval_df, gestures_df, k_list=[1,2,3,4,5], skip_missing=True):
    # Dictionary to collect binary correctness for each k.
    acc_dict = {f"top{k}": [] for k in k_list}
    
    for row in retrieval_df.itertuples():
        query_idx = row.gesture_index
        query_obj = gestures_df.at[query_idx, 'object']
        candidates = row.top_k_similar_gestures  # This is a list of candidate indices.
        
        # If skipping missing correct labels, check if any candidate exists.
        if skip_missing and not candidates:
            continue  
        
        # For each k, check if at least one candidate in the top-k has the correct object.
        for k in k_list:
            cand_subset = candidates[:k]
            found = any(gestures_df.at[c, 'object'] == query_obj for c in cand_subset)
            acc_dict[f"top{k}"].append(1 if found else 0)
    
    # Compute average only over the queries that were evaluated.
    return {f"top{k}": np.mean(acc_dict[f"top{k}"]) if len(acc_dict[f"top{k}"]) > 0 else np.nan
            for k in k_list}


def group_compute_topk_accuracies(retrieval_df, gestures_df, k_list=[1,2,3,4,5]):
    # Merge on the query gesture index (the gestures_df index)
    merged = retrieval_df.merge(gestures_df[['pair', 'speaker', 'object']],
                                 left_on='gesture_index', right_index=True)
    group_results = []
    for (pair, speaker), group in merged.groupby(['pair', 'speaker']):
        accs = compute_topk_accuracies(group, gestures_df, k_list=k_list)
        accs['pair'] = pair
        accs['speaker'] = speaker
        group_results.append(accs)
    return pd.DataFrame(group_results)
def filter_candidate_gestures(df, query_speaker, query_pair, mode, exclude_same_speaker):
    if mode == 'only_speaker':
        # retrieval from same dialogue and same speaker
        return df[(df['pair'] == query_pair) & (df['speaker'] == query_speaker)]
    elif mode == 'pair':
        # retrieval from same dialogue
        if exclude_same_speaker:
            return df[(df['pair'] == query_pair) & (df['speaker'] != query_speaker)]
        else:
            return df[df['pair'] == query_pair]
    elif mode == 'all':
        # retrieval from all gestures across dialogues
        # When excluding, only remove the exact match (same speaker and same pair)
        if exclude_same_speaker:
            return df[~((df['speaker'] == query_speaker) & (df['pair'] == query_pair))]
        else:
            # Exclude only the gesture itself (the calling row will be removed later)
            return df
    else:
        raise ValueError("Invalid mode provided.")


def find_top_k_gestures(distances, gestures_df, top_k,
                        mode='all', exclude_same_speaker=True):
    results = []
    for row in tqdm(gestures_df.itertuples(), total=gestures_df.shape[0]):
        i = row.Index
        query_speaker = row.speaker
        query_pair = row.pair
        query_object = row.object

        # Filter candidate gestures based on desired mode
        candidates = filter_candidate_gestures(gestures_df, query_speaker, query_pair, mode, exclude_same_speaker)
        # Always exclude the query gesture itself
        candidates = candidates.drop(i, errors='ignore')

        # Check whether any candidate shares the same object
        if candidates.empty or (query_object not in candidates['object'].values):
            results.append((i, []))
            continue

        indices = candidates.index.tolist()
        dists = distances[i, indices]
        top_k_inds = np.array(indices)[dists.argsort()[:top_k]].tolist()
        results.append((i, top_k_inds))
        
    return pd.DataFrame(results, columns=['gesture_index', 'top_k_similar_gestures'])


def check_same_object(top_k_df, gestures_df):
    """
    For each query gesture, check if the top-1 similar gesture refers to the same object.
    """
    top_k_df['same_object'] = False
    top_k_df['object'] = ""
    top_k_df['similar_object'] = ""

    for row in tqdm(top_k_df.itertuples(), total=top_k_df.shape[0]):
        i = row.gesture_index
        similar_inds = row.top_k_similar_gestures
        if not similar_inds:
            continue
        query_obj = gestures_df.at[i, 'object']
        candidate_obj = gestures_df.at[similar_inds[0], 'object']
        top_k_df.at[i, 'object'] = query_obj
        top_k_df.at[i, 'similar_object'] = candidate_obj
        if query_obj == candidate_obj:
            top_k_df.at[i, 'same_object'] = True

    return top_k_df


def calculate_accuracy(df):
    return df['same_object'].value_counts(normalize=True).get(True, 0)


def manual_implementation_object_retrieval(jointsformer,
                                           processed_keypoints_dict,
                                           mirrored_keypoints_dict,
                                           sekeleton_backbone,
                                           multimodal=True,
                                           bones=False,
                                           before_head_feats=False,
                                           gestures_and_speech_info=None,
                                           modalities=['skeleton', 'semantic'],
                                           audio_dict=None, model_type=None):
   """
   Manually implement the object retrieval functionality.
   """
   device = next(jointsformer.parameters()).device

   if gestures_and_speech_info is None:
      gestures_and_speech_info = get_detailed_gestures()

   # Reset indices and filter iconic gestures
   gestures_and_speech_info = gestures_and_speech_info.reset_index(drop=True)
   gestures_and_speech_info['pair_speaker'] = gestures_and_speech_info['pair'] + '_' + gestures_and_speech_info['speaker']
   gestures_and_speech_info = gestures_and_speech_info[gestures_and_speech_info['type'] == 'iconic'].reset_index(drop=True)

   jointsformer.eval()

   num_gestures = len(gestures_and_speech_info)
   if bones:
      feature_name = 'bone_features'
      feat_shape = 20
   elif not multimodal and not bones:
      feature_name = 'skeleton_features'
      feat_shape = 20 
   elif sekeleton_backbone == 'stgcn' and multimodal:
      feature_name = 'stgcn_features'
      feat_shape = 20  
   else:
      if model_type is not None:
         feature_name = model_type
      else:
         feature_name = 'transformer_features'
      feat_shape = 20  

   gestures_and_speech_info[feature_name] = [np.zeros(feat_shape) for _ in range(num_gestures)]
   gestures_and_speech_info['keypoints_features'] = [np.zeros(81) for _ in range(num_gestures)]
   gestures_and_speech_info['semantic_features'] = [np.zeros(16) for _ in range(num_gestures)]
   gestures_and_speech_info['speech_features'] = gestures_and_speech_info.apply(lambda x: np.zeros(16), axis=1)

   # Process each gesture
   for row in tqdm(gestures_and_speech_info.itertuples(), total=num_gestures):
      i = row.Index
      pair = row.pair
      speaker = row.speaker
      from_ts = row.from_ts
      to_ts = row.to_ts
      start_frame = int(from_ts * 29.97)
      end_frame = int(to_ts * 29.97)
      if end_frame - start_frame < 2:
         start_frame -= 2
         end_frame += 2
      if end_frame - start_frame > 60:
         # Choose the middle 60 frames; this is because the model expects 60 frames
         shift = (end_frame - start_frame - 60) // 2
         start_frame += shift
         end_frame = start_frame + 60

      pair_speaker = f"{pair}_{speaker}"
      skel = processed_keypoints_dict[pair_speaker][:, start_frame:end_frame, :, :]
      mirrored_skel = mirrored_keypoints_dict[pair_speaker][:, start_frame:end_frame, :, :]
      if speaker == 'B':
         skel, mirrored_skel = mirrored_skel, skel

      speech_waveform = None
      if 'speech' in modalities:
         speech_waveform = get_speech_waveform(pair_speaker, from_ts, to_ts, audio_dict, device)

      skel = process_skeleton(skel, mirrored_skel)
      bone_skel = get_bone(skel)
      skel_tensor = torch.tensor(skel, device=device, dtype=torch.float32).squeeze()
      bone_tensor = torch.tensor(bone_skel, device=device, dtype=torch.float32)

      if sekeleton_backbone != 'stgcn':
         skel_tensor = skel_tensor.permute(0, 2, 3, 1)
      else:
         skel_tensor = skel_tensor.unsqueeze(-1)
      if bones:
         skel_tensor = bone_tensor

      utterance = row.processed_utterance
      with torch.no_grad():
         mm_outputs = jointsformer(
               skel_tensor, eval=True, utterances=[utterance],
               features=modalities, before_head_feats=before_head_feats,
               speech_waveform=speech_waveform
         )
         if mm_outputs.get('one_branch_cross_modal', False):
               skeleton_feats = mm_outputs['skeleton_features'][0]
         else:
               skeleton_feats = torch.cat([mm_outputs['skeleton_features'][0], mm_outputs['skeleton_features'][1]], dim=0)
         skeleton_feats = torch.nn.functional.normalize(skeleton_feats, dim=0)

         if mm_outputs.get('semantic_features') is not None:
               semantic_feats = mm_outputs['semantic_features'].squeeze()
               semantic_feats = torch.nn.functional.normalize(semantic_feats, dim=0)
               gestures_and_speech_info.at[i, 'semantic_features'] = semantic_feats.cpu().numpy()

         if mm_outputs.get('speech_features') is not None:
               speech_feats = mm_outputs['speech_features'].squeeze()
               speech_feats = torch.nn.functional.normalize(speech_feats, dim=0)
               gestures_and_speech_info.at[i, 'speech_features'] = speech_feats.cpu().numpy()

      skeleton_feats = skeleton_feats.reshape(1, -1)
      gestures_and_speech_info.at[i, feature_name] = skeleton_feats[0].cpu().numpy()

      if sekeleton_backbone == 'stgcn':
         skel_tensor = skel_tensor.squeeze(-1).permute(0, 2, 3, 1)
      skel_mean = skel_tensor.mean(dim=1).view(skel_tensor.size(0), -1)
      skel_norm = torch.nn.functional.normalize(skel_mean, dim=1)
      gestures_and_speech_info.at[i, 'keypoints_features'] = skel_norm[0].cpu().numpy()


   return gestures_and_speech_info


if __name__ == '__main__':
   ''' this is the main function that loads the pretrained models and computes the similarity between gestures'''
   # all the baseline models are loaded here
   processed_keypoints_dict, mirrored_keypoints_dict = load_keypoints_dict()
   speakers_pairs = processed_keypoints_dict.keys()
   # remove pairs without 'pair'
   speakers_pairs = [speaker_pair for speaker_pair in speakers_pairs if 'pair' in speaker_pair]
   # audio_dict = load_audio_dict(speakers_pairs, audio_path='data/audio_files/{}_synced_pp{}.wav', audio_sample_rate=16000)

   gestures_info = get_detailed_gestures()
   gestures_info['round'] = gestures_info['round'].astype(int)
   gestures_info['object'] = gestures_info['object'].astype(int)

   from lib.data.datareader_cabb import load_keypoints_dict
   model_types = {
        'multimodal-x-skeleton-semantic': 
            {'modalities': ['skeleton', 'semantic'], 'hidden_dim': 256, 'cross_modal':True,'attentive_pooling':False, 'model_weights':  'wpretrained_models/multimodal-x_skeleton_text_correlation=0.30.ckpt'},
        'multimodal-skeleton-semantic': 
             {'modalities': ['skeleton', 'semantic'], 'hidden_dim': 256, 'cross_modal':False,'attentive_pooling':False, 'model_weights':  'wpretrained_models/multimodal_skeleton_text_correlation=0.29.ckpt'},
        
    }
   for model_type in model_types.keys():
      print(f'Loaded model {model_type} with modalities {model_types[model_type]["modalities"]} and weights from {model_types[model_type]["model_weights"]}')
      pretrained_model = load_pretrained_model(model_params=model_types[model_type]) # you can take this function from save_extracted_embeddings.py
      pretrained_model.eval()
      retrieval_results, gestures_info = manual_implementation_object_retrieval(pretrained_model , processed_keypoints_dict, mirrored_keypoints_dict, sekeleton_backbone='jointsformer', gestures_and_speech_info=gestures_info, modalities=model_types[model_type]['modalities'], audio_dict={}, model_type=model_type)
         # rename the column 'transformer_features' to the model type
   
