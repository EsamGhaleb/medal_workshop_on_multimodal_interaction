import sys
sys.path.append('../dialog_utils')
import os
import numpy as np
from speach import elan
import glob
from collections import defaultdict
import pandas as pd
import torch
import whisper
from tqdm import tqdm
import librosa
import spacy


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
                if pair == 'pair19':
                    print('pair19')
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

def read_manually_segmented_gestures():
    all_gestures_info = []
    elan_files_path = os.path.expanduser('data/ElanFiles//New_ELAN_files/')

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
    all_gestures_info.drop_duplicates(subset=['from_ts', 'to_ts', 'is_gesture', 'speaker', 'pair', 'info_type'], inplace=True, ignore_index=True)
    gestures_info = all_gestures_info[all_gestures_info['info_type'] == 'gesture']
    types = all_gestures_info[all_gestures_info['info_type'] == 'type']
    referents = all_gestures_info[all_gestures_info['info_type'] == 'referent']
    # merge the three dataframes
    gestures_info = gestures_info.merge(types, on=['from_ts', 'to_ts', 'speaker', 'pair'], how='left')
    gestures_info = gestures_info.merge(referents, on=['from_ts', 'to_ts', 'speaker', 'pair'], how='left')
    gestures_info.rename(columns={'is_gesture_x': 'is_gesture', 'is_gesture_y': 'type', 'is_gesture': 'referent'}, inplace=True)
    # remove the info_type_x and info_type_x and info_type
    gestures_info.drop(columns=['info_type_x', 'info_type_y', 'info_type', 'duration_y', 'duration_x'], inplace=True)
    # fill the nan values in the type and referent columns with empty strings
    gestures_info['type'].fillna('', inplace=True)
    gestures_info['referent'].fillna('', inplace=True)
    gestures_info['duration'] = gestures_info['to_ts'] - gestures_info['from_ts']
    # remove is_gesture which are not gesture
    gestures_info = gestures_info[gestures_info['is_gesture'] == 'gesture']
    # reindex the dataframe
    gestures_info.reset_index(drop=True, inplace=True)
    gestures_info['pair_speaker'] = gestures_info['pair'] + '_' + gestures_info['speaker']
    gestures_info['segmentation method'] = 'manual'
    gestures_info['probability'] = 1
    
    # now read automatically segmented gestures
    segmented_gestures_path = 'dialog_utils/data/final_eaf_files/*.eaf'
    eaf_files = glob.glob(segmented_gestures_path)
    all_gestures_info = []
    for file_path in eaf_files:
        eaf = elan.read_eaf(file_path)
        pair = file_path.split('/')[-1].split('.')[0]
        for tier in eaf:
            for ann in tier:
                _from_ts = ann.from_ts.sec
                _to_ts = ann.to_ts.sec
                _duration = ann.duration
                tier_name = tier.ID
                pair_speaker = file_path.split('/')[-1].split('.')[0]
                pair = pair_speaker.split('_')[0]
                speaker = pair_speaker.split('_')[1]
                # from_ts	to_ts	is_gesture	speaker	pair	type	duration	referent	pair_speaker	turn
                this_tier = {'from_ts': _from_ts, 'to_ts': _to_ts, 'duration': _duration, 'probability': ann.value, 'speaker': speaker, 'pair': pair, 'segmentation method': tier_name, 'is_gesture': 'gesture', 'type': '', 'referent': '', 'pair_speaker': pair_speaker}
                all_gestures_info.append(this_tier)
    # convert the list of dictionaries to a dataframe
    segmented_gestures_info = pd.DataFrame(all_gestures_info)
    segmented_gestures_info['pair_speaker'] = segmented_gestures_info['pair'] + '_' + segmented_gestures_info['speaker']
    segmented_gestures_info['duration'] = segmented_gestures_info['to_ts'] - segmented_gestures_info['from_ts']

    # merge the two dataframes
    gestures_info = pd.concat([gestures_info, segmented_gestures_info])
    return gestures_info
if __name__ == "__main__":
    gestures_info = read_manually_segmented_gestures()
    aligned_tagged_speech_per_word_small = pd.read_csv('dialog_utils/dialign/dialogues/lg/aligned_tagged_speech_per_word_small.csv')
    aligned_tagged_speech_per_word_large = pd.read_csv('dialog_utils/dialign/dialogues/lg/aligned_tagged_speech_per_word_large.csv')
    unique_speakers = gestures_info['pair_speaker'].unique()
    
    use_model = True
    audio_dict = {}
    # get unique speakers
    for a_speaker in tqdm(unique_speakers):
        pair, speaker = a_speaker.split('_')
        pair_speaker = f"{pair}_{speaker}"
        audio_path = os.path.join("data", f"{pair}_synced_pp{speaker}.wav")
        input_audio = librosa.load(audio_path, sr=16000)[0]
        audio_dict[pair_speaker] = input_audio
        
    model = whisper.load_model("large-v3")
    nlp = spacy.load("nl_core_news_lg")
    model.forced_decoder_ids = None
    model.eval()
    # model to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # make a list euqal to the number of samples
    buffer = 0.5
    contain_text = np.bool_(np.zeros(len(gestures_info)))
    words = np.zeros(len(gestures_info), dtype=object)
    lemmas = np.zeros(len(gestures_info), dtype=object)
    pos = np.zeros(len(gestures_info), dtype=object)
    lemmas_pos = np.zeros(len(gestures_info), dtype=object)
    transcriptions = np.zeros(len(gestures_info), dtype=object)
    # loop over labels and generate text
    special_cases = ['-', '`', '\'', '!', '.', ',', ';', ':', '?', '=', '(', ')', '[', ']', '{', '}', '/', '\\', ']?', '*', '***', '..', '...']
    options = whisper.DecodingOptions()
    for i, row in tqdm(gestures_info.iterrows(), total=len(gestures_info)):
        pair_speaker = row['pair_speaker']
        pair, speaker = pair_speaker.split('_')
        orig_str_pair = pair
        pair = str(pair)
        speaker = str(speaker)
        start_time = row['from_ts']
        end_time = row['to_ts']
        speech_end_frame = end_time + buffer
        speech_start_frame = start_time - buffer

        words[i] = ''
        contain_text[i] = False
        lemmas[i] = ''
        pos[i] = ''
        lemmas_pos[i] = ''
        transcriptions[i] = ''
        
        if row['segmentation method'] == 'manual':
            # check which rows in aligned_tagged_speech_per_word_small are within the speech_start_frame and speech_end_frame
            speech_info = aligned_tagged_speech_per_word_small[(aligned_tagged_speech_per_word_small['pair_name'] == pair) & (aligned_tagged_speech_per_word_small['speaker'] == speaker) & (aligned_tagged_speech_per_word_small['from_ts'] >= speech_start_frame) & (aligned_tagged_speech_per_word_small['to_ts'] <= speech_end_frame) & (aligned_tagged_speech_per_word_small['speaker'] == speaker)]
        else:
            pair = int(pair)
            speech_info = aligned_tagged_speech_per_word_large[(aligned_tagged_speech_per_word_large['pair_name'] == pair) & (aligned_tagged_speech_per_word_large['speaker'] == speaker) & (aligned_tagged_speech_per_word_large['from_ts'] >= speech_start_frame) & (aligned_tagged_speech_per_word_large['to_ts'] <= speech_end_frame) & (aligned_tagged_speech_per_word_large['speaker'] == speaker)]
            
        if not speech_info.empty:
            print("Manual transcription")
            # get word, lemma, and pos
            words[i] = ' '.join(speech_info['word'])
            lemmas[i] = ' '.join(speech_info['lemma'])
            pos[i] = ' '.join(speech_info['pos'])
            lemmas_pos[i] = ' '.join(speech_info['lemma_pos'])
            transcriptions[i] = 'manual'
            contain_text[i] = True
        else:
            print("Automatic transcription")
            try:
                sr = 16000
                audio_segment = audio_dict[f'{orig_str_pair}_{speaker}'][int(speech_start_frame*sr):int(speech_end_frame*sr)]
                input_features = whisper.pad_or_trim(audio_segment)
                mel = whisper.log_mel_spectrogram(input_features, n_mels=128).to(device)
                # decode the audio
                result = whisper.decode(model, mel, options)
                text = result.text
                contain_text[i] = bool(text)
                if bool(text):
                    print(f"{pair_speaker}: {text} for {speech_start_frame} to {speech_end_frame}")
                for special_case in special_cases:
                    text = text.replace(special_case, '')
                if not text or result.language != 'nl':
                    raise Exception("No text found")
                text = text.strip()
                pos_utterance = nlp(str(text))
                lemmas_with_pos = ''
                pos_sequence = ''
                lemmas_sequence = ''
                text_lemma_pos = ''
                utterance_idx = 0
                final_utterance = []
                for idx, token in enumerate(pos_utterance):
                    lemmas_sequence += token.lemma_+' '
                    pos_sequence += token.pos_+' '
                    lemmas_with_pos += token.lemma_+'#'+token.pos_+' '
                    text_lemma_pos += token.text+'_'+token.lemma_+'#'+token.pos_+' '
                
                words[i] = text        
                lemmas[i] = lemmas_sequence 
                pos[i] = pos_sequence
                lemmas_pos[i] = lemmas_with_pos
                transcriptions[i] = 'automatic'   
            
            except Exception as e:
                print(f"Error processing sample {i} with error {e}")
                words[i] = ''
                contain_text[i] = False
                lemmas[i] = ''
                pos[i] = ''
                lemmas_pos[i] = ''
                transcriptions[i] = ''
                
    
    gestures_info['words'] = words   
    gestures_info['contain_text'] = contain_text
    gestures_info['lemmas'] = lemmas
    gestures_info['pos'] = pos
    gestures_info['lemmas_pos'] = lemmas_pos
    gestures_info['transcriptions'] = transcriptions

    gestures_info.to_csv('dialog_utils/data/gestures_info_with_text.csv', index=False)