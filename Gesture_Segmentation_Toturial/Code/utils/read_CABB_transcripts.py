import re
# accessing nested tiers
from collections import defaultdict
import glob
import pandas as pd
from speach import elan

import whisperx
import numpy as np
from whisperx.audio import SAMPLE_RATE, load_audio
from whisperx.alignment import get_trellis, backtrack, merge_repeats, merge_words
# from whisperx.decoding import DecodingOptions, DecodingResult
# from whisperx.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from whisperx.utils import exact_div, format_timestamp, optional_int, optional_float, str2bool #, write_ass
import torch
import numpy as np
import os
from typing import List, Optional, Tuple, Union, Iterator, TYPE_CHECKING
import re
import numpy as np
import torch
from tqdm import tqdm
from itertools import product
import librosa
LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]

DEFAULT_ALIGN_MODELS_TORCH = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
}

DEFAULT_ALIGN_MODELS_HF = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
}

def get_audio(audio: Union[str, np.ndarray, torch.Tensor, dict], speaker: str):
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        if isinstance(audio, dict):
            return audio[speaker]
        audio = torch.from_numpy(audio)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)

    return audio

def align(
    transcript: Union[Iterator[dict], List[dict]],
    model: torch.nn.Module,
    align_model_metadata: dict,
    orig_audio: Union[str, np.ndarray, torch.Tensor, dict],
    device: str,
    extend_duration: float = 0.0,
    start_from_previous: bool = True,
    drop_non_aligned_words: bool = False,
):
   #  print("Performing alignment...")


    model_dictionary = align_model_metadata['dictionary']
    model_lang = align_model_metadata['language']
    model_type = align_model_metadata['type']

    prev_t2 = 0
    word_segments_list = []
    for idx, segment in enumerate(tqdm(transcript)):
        audio = get_audio(orig_audio, segment['speaker'])
        MAX_DURATION = audio.shape[0] / SAMPLE_RATE

        t1 = max(segment['start'] - extend_duration, 0)
        t2 = min(segment['end'] + extend_duration, MAX_DURATION)
        # if start_from_previous and t1 < prev_t2:
        #     t1 = prev_t2

        f1 = int(t1 * SAMPLE_RATE)
        f2 = int(t2 * SAMPLE_RATE)
      #   print(idx)
        waveform_segment = audio[f1:f2]
        # convert waveform to torch tensor
        waveform_segment = torch.from_numpy(waveform_segment).unsqueeze(0)
        with torch.inference_mode():
            if model_type == "torchaudio":
                emissions, _ = model(waveform_segment.to(device))
            elif model_type == "huggingface":
                emissions = model(waveform_segment.to(device)).logits
            else:
                raise NotImplementedError(f"Align model of type {model_type} not supported.")
            emissions = torch.log_softmax(emissions, dim=-1)
        emission = emissions[0].cpu().detach()
        transcription = segment['text'].strip()
        if model_lang not in LANGUAGES_WITHOUT_SPACES:
            t_words = transcription.split(' ')
        else:
            t_words = [c for c in transcription]

        t_words_clean = [''.join([w for w in word if w.lower() in model_dictionary.keys()]) for word in t_words]
        t_words_nonempty = [x for x in t_words_clean if x != ""]
        t_words_nonempty_idx = [x for x in range(len(t_words_clean)) if t_words_clean[x] != ""]
        segment['word-level'] = []

        if len(t_words_nonempty) > 0:
            transcription_cleaned = "|".join(t_words_nonempty).lower()
            tokens = [model_dictionary[c] for c in transcription_cleaned]
            trellis = get_trellis(emission, tokens)
            try:
                path = backtrack(trellis, emission, tokens)
            except:
                segment['word-level'].append({"text": segment['text'], "start": segment['start'], "end":segment['end']})
                word_segments_list.append({"text": segment['text'], "start": segment['start'], "end":segment['end']})
                print('*************')
                continue
            try:
               segments = merge_repeats(path, transcription_cleaned)
            except Exception as e:
               print(e)
               print('*************')
               return None 
            word_segments = merge_words(segments)
            ratio = waveform_segment.size(0) / (trellis.size(0) - 1)

            duration = t2 - t1
            local = []
            t_local = [None] * len(t_words)
            for wdx, word in enumerate(word_segments):
                t1_ = ratio * word.start
                t2_ = ratio * word.end
                local.append((t1_, t2_))
                t_local[t_words_nonempty_idx[wdx]] = (t1_ * duration + t1, t2_ * duration + t1)
            t1_actual = t1 + local[0][0] * duration
            t2_actual = t1 + local[-1][1] * duration

            segment['start'] = t1_actual
            segment['end'] = t2_actual
            prev_t2 = segment['end']

            # for the .ass output
            for x in range(len(t_local)):
                curr_word = t_words[x]
                curr_timestamp = t_local[x]
                if curr_timestamp is not None:
                    segment['word-level'].append({"text": curr_word, "start": curr_timestamp[0], "end": curr_timestamp[1]})
                else:
                    segment['word-level'].append({"text": curr_word, "start": None, "end": None})

            # for per-word .srt ouput
            # merge missing words to previous, or merge with next word ahead if idx == 0
            for x in range(len(t_local)):
                curr_word = t_words[x]
                curr_timestamp = t_local[x]
                if curr_timestamp is not None:
                    word_segments_list.append({"text": curr_word, "start": curr_timestamp[0], "end": curr_timestamp[1]})
                elif not drop_non_aligned_words:
                    # then we merge
                    if x == 0:
                        t_words[x+1] = " ".join([curr_word, t_words[x+1]])
                    else:
                        word_segments_list[-1]['text'] += ' ' + curr_word
        else:
            # then we resort back to original whisper timestamps
            # segment['start] and segment['end'] are unchanged
            prev_t2 = 0
            segment['word-level'].append({"text": segment['text'], "start": segment['start'], "end":segment['end']})
            word_segments_list.append({"text": segment['text'], "start": segment['start'], "end":segment['end']})

        print(f"[{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}] {segment['text']}")
        print(f"  {segment['word-level']}")
    return {"segments": transcript, "word_segments": word_segments_list}

class Tier:
    ''' This class holds information for each tier'''
    def __init__(self, tier_name, from_ts, to_ts, value, duration):
        self.tier_name = tier_name
        self.from_ts = from_ts
        self.to_ts = to_ts
        self.value = value
        self.duration = duration
    def __str__(self) -> str:
        return 'Tier name is {} with value \"{}\"'.format(self.tier_name, self.value)
    
def temp_build_dialogue_dict(eaf, verbose=False):
    complete_dialogue_tiers = defaultdict(list)
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
            # rows.append((tier.ID, tier.participant, _from_ts, _to_ts, _duration, ann.value))            
            this_tier = Tier(tier_name=tier.ID, from_ts=_from_ts, to_ts=_to_ts, duration=_duration, value=ann.value)
            complete_dialogue_tiers[tier_name].append(this_tier)
            
            if verbose:
                print(f"{ann.ID.rjust(4, ' ')}. [{ann.from_ts} :: {ann.to_ts}] {ann.text}")
                print(tier.ID+ ': value '+ann.value)
                print(tier.ID+ ': text '+ann.text)
    return complete_dialogue_tiers

def get_non_vocal_sounds_and_targets_mentions(dataset_name, dataset):
    non_vocal_sounds_and_targets_mentions = []
    
    for filepath in glob.iglob(dataset[dataset_name]["dataset_path"]):
        eaf = elan.read_eaf(filepath)
        complete_dialogue_tiers = temp_build_dialogue_dict(eaf)
        for tier in complete_dialogue_tiers['speakers']:
            utterance = tier.value
            if dataset_name == 'small':
                matching = re.findall(r"\#.*?\#", " ".join(utterance.strip().split()))
                # target_matching = re.findall(r"\*.*?\*", " ".join(utterance.strip().split()))
            else:
                matching = re.findall(r"\*.*?\*", " ".join(utterance.strip().split()))
                # target_matching = re.findall(r"\#.*?\#", " ".join(utterance.strip().split()))
            if len(matching) > 0:
                for match in matching:
                    non_vocal_sounds_and_targets_mentions.append(match)
            elif dataset[dataset_name]['Non-verbal_sounds_marker'] in utterance:
                print(utterance)
      # add the following: #lip smack#
    non_vocal_sounds_and_targets_mentions.extend(["#lip smack#", "#sounds of the microphonesmack one's lips#", "#smack one's lipsbreath laugh#", "#smackinhale#", "#smack one's lipsbreath laugh#", "#coughsmack one's lips#", "#smack one's lipsbreath laugh#", "#smackinhale#", "#inhalesmack#", "#swallowsmack#", "#airthudairthudairthudairthudairthud#", "#stuttersmack one's lips#", "stutterstutter", "#stuttersmack one's lips#", "#smack one's lipsthinking sound#", "#airthudairthudairthudairthudairthud#", "#click of the computer", "#tongueclick# #tongueclick# #tongueclick# #tongueclick# #tongueclick#", "#click of the computer#", "#claps hands#"])
    return set(non_vocal_sounds_and_targets_mentions)
 
 
def normalize_utterance(utterance, dataset_name, dataset, verbose=False):
    print('Before: ', utterance)
    # utterance = utterance.replace(dataset[dataset_name]['repairs_marker'], "")
    for key, value in dataset[dataset_name]['dutch_numbers'].items():
        utterance = utterance.replace(key, value) 
    for key, value in dataset[dataset_name]['dutch_capital_letters'].items():
        utterance = utterance.replace(key, value)
    if dataset[dataset_name]['Non-verbal_sounds_marker'] in utterance:
        if verbose:
            print('Before: ', utterance)
        for expression in dataset[dataset_name]['Non-verbal_sounds_and_target_mentions']: # this is special for both datasets
            utterance = utterance.replace(expression, '')
        if verbose:
            print('After: ', utterance)
    if '(' in utterance or ')' in utterance:
        if verbose:
            print('Before: ', utterance)
        for inaduable_speech_marker in dataset[dataset_name]['inaudiable_speech_markers']:
            utterance = utterance.replace(inaduable_speech_marker, '')
        if verbose:
            print('After: ', utterance)
    if dataset[dataset_name]['targets_marker'] in utterance:
        if verbose:
            print('Before: ', utterance)
        utterance = utterance.replace(dataset[dataset_name]['targets_marker'], '')
        if verbose:
            print('After: ', utterance)
    if '\\' in utterance:
        if verbose:
            print('Before: ', utterance)
        utterance = utterance.replace("\\", '')
        if verbose:
            print('After: ', utterance)
    if '-' in utterance and dataset_name == 'large':
        if verbose:
            print('Before: ', utterance)
        utterance = utterance.replace("-", '')
        if verbose:
            print('After: ', utterance)
    if '/' in utterance:
        if verbose:
            print('Before: ', utterance)
        utterance = utterance.replace("/", ' ')
        if verbose:
            print('After: ', utterance)
    
    utterance = " ".join([word.replace('=-', 'IS') if len(word.split('?')[-1])==0 else ' '.join(word.split('?')).replace('=-', 'IS') for word in utterance.split()])
    utterance = utterance.strip()
    if utterance != '':
        if utterance[-1] in ['`', '\'', '!', '.', ',', ';', ':', '?']:
            utterance = utterance[:-1]
    # utterance = " ".join(utterance.split())
    print('After: ', utterance)

    return utterance

if __name__ == "__main__":
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   alignment_model, alignment_metadata = whisperx.load_align_model("nl", device=device)
   
   dataset ={}
   dataset_name = 'small'
   dataset[dataset_name] = {}
   inaudiable_speech_markers = ['(?)', '()', '( )', '(', ')', '(?])']
   dataset[dataset_name]["dataset_path"] = "/home/eghaleb/data/*.eaf"   
   dataset[dataset_name]['name'] = dataset_name
   dataset[dataset_name]['Non-verbal_sounds_matcher'] = r"\#.*?\#"
   dataset[dataset_name]['Non-verbal_sounds_marker'] = '#'
   dataset[dataset_name]['targets_marker'] = '*'
   dataset[dataset_name]['target_matcher'] = r"\*.*?\*"
   dataset[dataset_name]['repairs_marker'] = '-'
   dataset[dataset_name]['inaudiable_speech_markers'] = inaudiable_speech_markers
   dataset[dataset_name]['Non-verbal_sounds_and_target_mentions'] = get_non_vocal_sounds_and_targets_mentions(dataset_name, dataset)

   # A dictionary of dutch numbers (between two * character) with their written form, from 1 to 16 
   dataset[dataset_name]['dutch_numbers'] = {'*1*': 'een', '*2*': 'twee', '*3*': 'drie', '*4*': 'vier', '*5*': 'vijf', '*6*': 'zes', '*7*': 'zeven', '*8*': 'acht', '*9*': 'negen', '*10*': 'tien', '*11*': 'elf', '*12*': 'twaalf', '*13*': 'dertien', '*14*': 'veertien', '*15*': 'vijfteen', '*16*': 'zestien'}

   # A dictionary with the first 16 dutch capital letters between two * characters, and the letters without the * characters
   dataset[dataset_name]['dutch_capital_letters'] = {'*A*': 'A', '*B*': 'B', '*C*': 'C', '*D*': 'D', '*E*': 'E', '*F*': 'F', '*G*': 'G', '*H*': 'H', '*I*': 'I', '*J*': 'J', '*K*': 'K', '*L*': 'L', '*M*': 'M', '*N*': 'N', '*O*': 'O', '*P*': 'P'}
   # A list of the first 16 dutch capital letters between two * characters
   dataset[dataset_name]['dutch_capital_letters_list'] = ['*A*', '*B*', '*C*', '*D*', '*E*', '*F*', '*G*', '*H*', '*I*', '*J*', '*K*', '*L*', '*M*', '*N*', '*O*', '*P*']
      # read utterances from eaf files
   speech_transcripts = []
   for filepath in tqdm(glob.iglob("./data/Latest_ELAN_files/*.eaf")):
      eaf = elan.read_eaf(filepath)
      pair = os.path.basename(filepath).split('_')[0].split('.')[0]
      for tier in eaf:
         if tier.ID in ['B_po', 'A_po']:
            for ann in tier:
               _from_ts = ann.from_ts.sec
               _to_ts = ann.to_ts.sec
               _duration = ann.duration
               speaker = tier.ID.split('_')[0]
               text = ann.value
               speech_transcripts.append([pair, speaker, _from_ts, _to_ts, _duration, text])
   speech_transcripts = pd.DataFrame(speech_transcripts, columns=['pair', 'speaker', 'from_ts', 'to_ts', 'duration', 'text'])
   
       
   unique_pair_speaker = {tuple(elem) for elem in speech_transcripts[["pair", "speaker"]].values}
   unique_pair_speaker = sorted(unique_pair_speaker)
   audio_dict = {}
   # get unique speakers
   for pair, speaker in tqdm(unique_pair_speaker):
      pair_speaker = f"{pair}_{speaker}"
      audio_path = os.path.join("../gesture_representation_and_resolution/data/audio_files/{}_synced_pp{}.wav".format(pair, speaker))
      # audio_dict[pair_speaker]  = load_audio(audio_path)
      if not os.path.exists(audio_path):
         print(f"Audio file {audio_path} does not exist")
         # remove the rows of pair_speaker from the speech_transcripts
         speech_transcripts = speech_transcripts[speech_transcripts['pair'] != pair]
         continue
      audio_dict[pair_speaker]  = librosa.load(audio_path, sr=16000)[0]
   
   print(speech_transcripts.shape)
   AB_result_aligned = {'segments': [], 'word_segments': []}
   all_transcripts = []
   for i, row in tqdm(speech_transcripts.iterrows()):
      pair_speaker = f"{row['pair']}_{row['speaker']}"
      start = row['from_ts']
      end = row['to_ts']
      text = row['text']
      original_text = text
      normalized_utterance = normalize_utterance(text, dataset_name, dataset)
      if normalize_utterance != '':
         if 'oh zo hij is leuk' in normalized_utterance: # wrong transcription, it is longer than the audio segment
            normalized_utterance = 'oh zo'
         # replace * and # with empty string
         normalized_utterance = normalized_utterance.replace('*', '')
         normalized_utterance = normalized_utterance.replace('#', '')
         turns_transcript = {'text': normalized_utterance, 'start': row['from_ts'], 'end': row['to_ts'], 'speaker': pair_speaker}
         
         turn_result_aligned = align([turns_transcript], alignment_model, alignment_metadata, audio_dict, device=device)
         if turn_result_aligned is None:
            continue 
         word_level = turn_result_aligned['segments'][0]['word-level']
         AB_result_aligned['segments'].append(turn_result_aligned['segments'][0])
         AB_result_aligned['word_segments'].append(turn_result_aligned['word_segments'][0])
         utterance_speech = []
         for text in word_level:
               # utterance_speech.append(Utterance(text['text'], text['start'], text['end']))
            print(text['text'], text['start'], text['end'])
            all_transcripts.append([pair_speaker, text['text'], text['start'], text['end'], start, end, original_text, row['speaker'], row['pair']])
   
   all_transcripts = pd.DataFrame(all_transcripts, columns=['pair_speaker', 'text', 'from_ts', 'to_ts', 'utterance_from_ts', 'utterance_to_ts', 'original_utterance', 'speaker', 'pair'])

   # save to pickle
   all_transcripts.to_pickle('aligned_transcripts.pkl')
   
   

