import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from dialog_utils.read_labeled_and_pos_target_data import prepare_dialogue_shared_expressions_and_turns_info
from dialog_utils.data_classes import Turn, Gesture, Utterance
from dialog_utils.prepare_annotate_dialogues import get_gestures_info, get_turn_info


def get_gestures_and_speech_info(turns_info):
   print("Extracting gestures and speech information...")
   print("NOTE: This function requires the 'whisper' and 'spacy' libraries to be installed.")
   import spacy
   import whisper
   pairs_name = ['pair10', 'pair17', 'pair15', 'pair21', 'pair18', 'pair24', 'pair16', 'pair13', 'pair07', 'pair22', 'pair20',  'pair08', 'pair05', 'pair11', 'pair23', 'pair04', 'pair09', 'pair12', 'pair14' ]
   gestures_and_speech_info = []
   for pair in pairs_name:
      pair_turns_info = turns_info[pair]
      for turn in pair_turns_info:
         if turn.gesture:
            for gesture in turn.gesture:
               gesture_info = {'pair': pair, 'turn': turn.ID, 'round': turn.round, 'type': gesture.g_type, 'is_gesture': gesture.is_gesture, 'from_ts': gesture.g_from_ts, 'to_ts': gesture.g_to_ts, 'hand': gesture.g_hand, 'referent': gesture.g_referent, 'comment': gesture.g_comment, 'object': turn.target, 'utterance': turn.utterance, 'speaker': turn.speaker, 'trial': turn.trial, 'director': turn.director, 'correct_answer': turn.correct_answer, 'given_answer': turn.given_answer, 'accuracy': turn.accuracy, 'processed_utterance': turn.processed_utterance, 'pos_sequence': turn.pos_sequence, 'lemmas_sequence': turn.lemmas_sequence, 'text_lemma_pos': turn.text_lemma_pos, 'turn_from_ts': turn.from_ts, 'turn_to_ts': turn.to_ts}
               gestures_and_speech_info.append(gesture_info)
   gestures_and_speech_info = pd.DataFrame(gestures_and_speech_info)
   gestures_and_speech_info['duration'] = gestures_and_speech_info['to_ts'] - gestures_and_speech_info['from_ts']
   gestures_and_speech_info = gestures_and_speech_info.fillna('')
   gesture_data = gestures_and_speech_info.groupby('type').count().reset_index().rename(columns={'pair': 'count'})[['type', 'count']]
   gesture_data['normalized_count'] = gesture_data['count'] / gesture_data['count'].sum()
   gesture_data['normalized_count'] = gesture_data['normalized_count'].apply(lambda x: round(x, 2))
   # fill nan values in gestures_info with empty strings
   gestures_and_speech_info = gestures_and_speech_info.fillna('')
   # group gestures by pair, turn, type, is_gesture, from_ts, to_ts, referent, comment, and duration and other attributes
   gestures_and_speech_info = gestures_and_speech_info.groupby(['pair', 'turn', 'round', 'type', 'is_gesture', 'from_ts', 'to_ts', 'referent', 'comment', 'duration', 'object', 'utterance', 'speaker', 'trial', 'director', 'correct_answer', 'given_answer', 'accuracy', 'processed_utterance', 'pos_sequence', 'lemmas_sequence', 'text_lemma_pos', 'turn_from_ts', 'turn_to_ts']).agg({'hand': lambda x: ', '.join(x)}).reset_index()
   gestures_and_speech_info = gestures_and_speech_info[gestures_and_speech_info['is_gesture'] == 'gesture']
   
   aligned_tagged_speech_per_word_small = pd.read_csv('dialog_utils/dialign/dialogues/lg/aligned_tagged_speech_per_word_small.csv')
   buffer = 0.5
   contain_text = np.bool_(np.zeros(len(gestures_and_speech_info)))
   words = np.zeros(len(gestures_and_speech_info), dtype=object)
   lemmas = np.zeros(len(gestures_and_speech_info), dtype=object)
   pos = np.zeros(len(gestures_and_speech_info), dtype=object)
   lemmas_pos = np.zeros(len(gestures_and_speech_info), dtype=object)
   transcriptions = np.zeros(len(gestures_and_speech_info), dtype=object)
   # reset gestures_and_speech_info
   gestures_and_speech_info.reset_index(drop=True, inplace=True)

   gestures_and_speech_info['pair_speaker'] = gestures_and_speech_info['pair'] + '_' + gestures_and_speech_info['speaker']
   unique_speakers = gestures_and_speech_info['pair_speaker'].unique()
    
   use_model = True
   audio_dict = {}
   # get unique speakers
   for a_speaker in tqdm(unique_speakers):
      pair, speaker = a_speaker.split('_')
      pair_speaker = f"{pair}_{speaker}"
      audio_path = os.path.join("data", f"{pair}_synced_pp{speaker}.wav")
      input_audio = whisper.load_audio(audio_path)
      audio_dict[pair_speaker] = input_audio
   model = whisper.load_model("large-v3")
   nlp = spacy.load("nl_core_news_lg")
   model.forced_decoder_ids = None
   model.eval()
   # model to gpu
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model.to(device)
   options = whisper.DecodingOptions()
        
   for row_index, row in tqdm(gestures_and_speech_info.iterrows(), total=len(gestures_and_speech_info)):
      pair = row['pair']
      speaker = row['speaker']
      start_time = row['from_ts']
      end_time = row['to_ts']
      speech_end_frame = end_time + buffer
      speech_start_frame = start_time - buffer
      # check which rows in aligned_tagged_speech_per_word_small are within the speech_start_frame and speech_end_frame
      speech_info = aligned_tagged_speech_per_word_small[(aligned_tagged_speech_per_word_small['pair_name'] == pair) & (aligned_tagged_speech_per_word_small['speaker'] == speaker) & (aligned_tagged_speech_per_word_small['from_ts'] >= speech_start_frame) & (aligned_tagged_speech_per_word_small['to_ts'] <= speech_end_frame)]
      if not speech_info.empty:
         words[row_index] = ' '.join(speech_info['word'])
         lemmas[row_index] = ' '.join(speech_info['lemma'])
         pos[row_index] = ' '.join(speech_info['pos'])
         lemmas_pos[row_index] = ' '.join(speech_info['lemma_pos'])
         transcriptions[row_index] = 'manual'
         contain_text[row_index] = True
      else:
         print("Automatic transcription")
         try:
               sr = 16000
               audio_segment = audio_dict[f'{pair}_{speaker}'][int(speech_start_frame*sr):int(speech_end_frame*sr)]
               input_features = whisper.pad_or_trim(audio_segment)
               mel = whisper.log_mel_spectrogram(input_features, n_mels=128).to(device)
               # decode the audio
               result = whisper.decode(model, mel, options)
               text = result.text
               contain_text[row_index] = bool(text)
               if bool(text):
                  print(f"{pair_speaker}: {text} for {speech_start_frame} to {speech_end_frame}")
                  
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
               
               words[row_index] = text        
               lemmas[row_index] = lemmas_sequence 
               pos[row_index] = pos_sequence
               lemmas_pos[row_index] = lemmas_with_pos
               transcriptions[row_index] = 'automatic'   
         
         except Exception as e:
               print(f"Error processing sample {row_index} with error {e}")
               words[row_index] = ''
               contain_text[row_index] = False
   gestures_and_speech_info['words'] = words
   gestures_and_speech_info['lemmas'] = lemmas
   gestures_and_speech_info['pos'] = pos
   gestures_and_speech_info['lemmas_pos'] = lemmas_pos
   gestures_and_speech_info['transcriptions'] = transcriptions
   gestures_and_speech_info['contain_text'] = contain_text 
   return gestures_and_speech_info, gesture_data

def get_detailed_gestures(gestures_path='./dialog_utils/data/gestures_referents_info.csv'):
   # first check if the file exists
   try:
      gestures_and_speech_info = pd.read_csv(gestures_path)
      return gestures_and_speech_info
   except:
      print('File not found. Generating the detailed gestures info...')
   # code to set up the paths
   dialign_output = './dialog_utils/dialign/dialogues/lg_output/'
   turn_info_path = './dialog_utils/dialign/dialogues/lg/'
   fribbles_path = "./dialog_utils/data/Fribbles/{}.jpg"
   videos_path = '~/data/{}_synced_overview.mp4'

   pos_func_words = ['DET', 'PRON', 'ADP', 'CCONJ', 'SCONJ', 'AUX', 'PART', 'PUNCT', 'SYM', 'X', 'INTJ', 'NUM', 'SPACE']
   content_word_pos = ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV']

   shared_constructions_info, turns_info = prepare_dialogue_shared_expressions_and_turns_info(dialign_output, turn_info_path)
   speech_and_objects = []
   for pair in turns_info.keys():
      for this_turn in turns_info[pair]:
         speech_and_objects.append({'utterance': this_turn.utterance, 'object': this_turn.target, 'pair': pair, 'turn_ID': this_turn.ID, 'speaker': this_turn.speaker, 'utterance': this_turn.utterance, 'target': this_turn.target, 'trial': this_turn.trial, 'director': this_turn.director, 'correct_answer': this_turn.correct_answer, 'given_answer': this_turn.given_answer, 'accuracy': this_turn.accuracy, 'pos_sequence': this_turn.pos_sequence, 'lemmas_sequence': this_turn.lemmas_sequence, 'text_lemma_pos': this_turn.text_lemma_pos, 'round': this_turn.round, 'from_ts': this_turn.from_ts, 'to_ts': this_turn.to_ts})
   speech_and_objects = pd.DataFrame(speech_and_objects)
   
   gestures_and_speech_info, gesture_data = get_gestures_and_speech_info(turns_info)
   # check duplicates in gestures_and_speech_info (using the 'pair', 'turn', 'round', 'type', 'is_gesture', 'from_ts', 'to_ts')
   duplicates = gestures_and_speech_info[gestures_and_speech_info.duplicated(subset=['pair', 'turn', 'round', 'type', 'is_gesture', 'from_ts', 'to_ts'], keep=False)]
   # remove duplicates from gestures_and_speech_info
   gestures_and_speech_info = gestures_and_speech_info.drop_duplicates(subset=['pair', 'turn', 'round', 'type', 'is_gesture', 'from_ts', 'to_ts'], keep='first')
   duplicates = gestures_and_speech_info[gestures_and_speech_info.duplicated(subset=['pair', 'from_ts', 'to_ts'], keep=False)]
   gestures_and_speech_info = gestures_and_speech_info.drop_duplicates(subset=['pair', 'from_ts', 'to_ts'], keep='first')
   # reset index
   gestures_and_speech_info = gestures_and_speech_info.reset_index(drop=True)
   gestures_and_speech_info['segmentation method'] = 'manual'
   # rename utterance column to speech
   gestures_and_speech_info = gestures_and_speech_info.rename(columns={'utterance': 'speech'})
    
   # save gestures_and_speech_info to a csv file
   gestures_and_speech_info.to_csv(gestures_path, index=False)
   return gestures_and_speech_info

if __name__ == '__main__':
   gestures_and_speech_info = get_detailed_gestures()
   print(gestures_and_speech_info.head())

   