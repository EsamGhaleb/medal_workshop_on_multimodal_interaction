import os
import pickle
import numpy as np
from speach import elan
import glob
from collections import defaultdict
import pandas as pd
def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

# read the elan files and build a dictionary of tiers
def build_dialogue_dict(eaf, pair, verbose=False):
    complete_dialogue_tiers = defaultdict(list)
    gesture_lists = ['B_RH_gesture', 'B_LH_gesture', 'A_RH_gesture', 'A_LH_gesture', 'B_RH_gesture_type', 'B_LH_gesture_type', 'A_RH_gesture_type', 'A_LH_gesture_type', 'B_RH_gesture_referent', 'B_LH_gesture_referent', 'A_RH_gesture_referent', 'A_LH_gesture_referent']  
    for tier in eaf:
        if verbose:
            print(f"{tier.ID} | Participant: {tier.participant} | Type: {tier.type_ref}")
        for ann in tier:
            _from_ts = ann.from_ts.sec
            _to_ts = ann.to_ts.sec
            _duration = ann.duration
            if tier.ID in ['B_po', 'A_po']:
                tier_name = 'speakers'
            else:
                tier_name = tier.ID
            if tier_name in gesture_lists:
                speaker = tier_name.split('_')[0]
                information_type = tier_name.split('_')[-1]
                this_tier = {'from_ts': _from_ts, 'to_ts': _to_ts, 'duration': _duration, 'is_gesture': ann.value, 'speaker': speaker, 'pair': pair, 'info_type': information_type}
                complete_dialogue_tiers['gestures_info'].append(this_tier)
            else:
                this_tier = {'from_ts': _from_ts, 'to_ts': _to_ts, 'duration': _duration, 'value': ann.value}
                complete_dialogue_tiers[tier_name].append(this_tier)
            
            if verbose:
                print(f"{ann.ID.rjust(4, ' ')}. [{ann.from_ts} :: {ann.to_ts}] {ann.text}")
                print(tier.ID+ ': value '+ann.value)
                print(tier.ID+ ': text '+ann.text)
    return complete_dialogue_tiers

def get_gesture_info(): 
   all_gestures_info = []
   elan_files_path = os.path.expanduser('../data/ElanFiles/New_ELAN_files/')

   for root, dirs, files in os.walk(elan_files_path):
      for file in files:
         if file.endswith('.eaf'):
               file_path = os.path.join(root, file)
               eaf = elan.read_eaf(file_path)
               pair = file_path.split('/')[-1].split('.')[0]
               complete_dialogue_tiers = build_dialogue_dict(eaf, pair, verbose=False)
               all_gestures_info.extend(complete_dialogue_tiers['gestures_info'])

   # covert the list of dictionaries to a dataframe
   all_gestures_info = pd.DataFrame(all_gestures_info)
   # remove duplicates
   all_gestures_info.drop_duplicates(subset=['from_ts', 'to_ts', 'is_gesture', 'speaker', 'pair'], inplace=True, ignore_index=True)
   # check the number of duplicates based on the from_ts, to_ts, speaker, pair
   all_gestures_info.groupby(['from_ts', 'to_ts', 'speaker', 'pair']).count().sort_values(by='is_gesture', ascending=False)
   # merge the is_gesture column based on the from_ts, to_ts, speaker, pair
   all_gestures = all_gestures_info.groupby(['from_ts', 'to_ts', 'speaker', 'pair']).agg({'is_gesture': lambda x: '_'.join(x)}).reset_index()
   all_gestures['type'] = all_gestures['is_gesture'].apply(lambda x: x.split('_')[-1])
   all_gestures['duration'] = all_gestures['to_ts'] - all_gestures['from_ts']
   # remove non-gesture 
   # gestures_info = gestures_info[gestures_info['is_gesture'] != 'non-gesture']
   return all_gestures
if __name__ == "__main__":
   get_gesture_info()