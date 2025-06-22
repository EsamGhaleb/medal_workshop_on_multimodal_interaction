import sys
sys.path.append('./dialog_utils')
import re
from collections import defaultdict
import glob
from speach import elan
import csv
import pickle

import numpy as np
from whisperx.audio import load_audio
import whisperx
from align_speech_time_wise import align
from data_classes import Tier, Turn, Utterance, Gesture 


import torch
import os
import argparse


def get_gestures_info(Turns, dialogue, participant_ID, g_hand, verbose=False):
    gesture_n = participant_ID+'_'+g_hand+'H_gesture'
    turns_to_ts = np.array([turn.to_ts for turn in Turns])
    turns_from_ts = np.array([turn.from_ts for turn in Turns])
    speakers = [turn.speaker for turn in Turns]
    for gesture_idx, gesture in enumerate(dialogue[gesture_n]):
        gesture_from_ts = gesture.from_ts
        gesture_to_ts = gesture.to_ts 
        overlaps = np.minimum(turns_to_ts, gesture_to_ts) - np.maximum(turns_from_ts, gesture_from_ts)
        # Exclude overlaps where speaker is not the same as the participant_ID by assigning -1000 to them
        #NOTE this is done to avoid the case where the gesture is for another speaker
        overlaps = np.where(np.array(speakers) == participant_ID, overlaps, -1000)
        turn_index = np.argmax(overlaps)
        try:
            assert participant_ID == Turns[turn_index].speaker
        except:
            print('Gesture is not for the same speaker')
            #NOTE: just keep counter for the number of gestures that are not for the same speaker
        if verbose:
            print('Utterance from {} to {} :'.format(Turns[turn_index].from_ts, Turns[turn_index].to_ts))
            
        g_to_ts = gesture.to_ts
        g_from_ts = gesture.from_ts
        is_gesture = gesture.value
        if is_gesture == 'non-gesture':
            print('Non-gesture found!')
        
        # gesture should usually come with type, referent, and comment
        g_comment = None
        g_type = None
        g_referent = None
        #get the comment/referent/type from their tiers
        g_type_index = [idx for idx, gesture_type in enumerate(dialogue[gesture_n+'_type']) if gesture_type.from_ts == g_from_ts]
        if g_type_index:
            assert dialogue[gesture_n+'_type'][g_type_index[0]].to_ts == g_to_ts
            g_type = dialogue[gesture_n+'_type'][g_type_index[0]].value

        g_comment_index = [idx for idx, gesture_comment in enumerate(dialogue[gesture_n+'_comment']) if gesture_comment.from_ts == g_from_ts]
        if g_comment_index:
            assert dialogue[gesture_n+'_comment'][g_comment_index[0]].to_ts == g_to_ts
            g_comment = dialogue[gesture_n+'_comment'][g_comment_index[0]].value
        g_referent_index = [idx for idx, gesture_referent in enumerate(dialogue[gesture_n+'_referent']) if gesture_referent.from_ts == g_from_ts]

        if g_referent_index:
            assert dialogue[gesture_n+'_referent'][g_referent_index[0]].to_ts == g_to_ts
            g_referent = dialogue[gesture_n+'_referent'][g_referent_index[0]] .value
        if verbose:
            print('Gesture is for the {}'.format(participant_ID+g_hand))
            print('Gesture is from {} to {}'.format(g_from_ts, g_to_ts))
            print('Is it gesture?: {}'.format(is_gesture))
            print('Gesture type is: {}'.format(g_type))
            print('Gesture referent is: {}'.format(g_referent))
            print('Gesture comment is: {}'.format(g_comment))
        this_gesture = Gesture(is_gesture, g_from_ts, g_to_ts, g_type, g_referent, g_comment, g_hand, participant_ID)
        
        Turns[turn_index].set_gesture(this_gesture)
        
    return Turns

def get_turn_info(dialogue, pair_name, tier, trial_index, turn_ID, dataset = 'small', verbose=False):
    if verbose:
        message = 'new turn'
        x = message.center(20, "-")
        print(x)

    utterance = tier.value
    duration = tier.duration
    from_ts = tier.from_ts
    to_ts = tier.to_ts
    speaker = tier.tier_name.split('_')[0]
  
    trial = dialogue['trial'][trial_index].value
    # if dataset == 'large' and trial.split('_')[-1] == 'loc': # if the trial is a location trial, we do not use it as we are now focusing on the referential trials
    #     if verbose:
    #         print(utterance)
    #         print ('Location trial, skipping: {}'.format(trial))
    #     return None
    target = dialogue['target'][trial_index].value
    accuracy = dialogue['accuracy'][trial_index].value
    correct_answer = dialogue['correct_answer'][trial_index].value
    given_answer = dialogue['given_answer'][trial_index].value
    director = dialogue['director'][trial_index].value

    if verbose:
        print()
        print('Utterance from {} to {} :'.format(from_ts, to_ts))
        print('Utterance {}'.format(utterance))
        print('Trial from {} to {} :'.format(dialogue['trial'][trial_index].from_ts, dialogue['trial'][trial_index].to_ts))
        print('Speaker {}'.format(speaker))
        print('Director {}'.format(director))
        print('Trial {}'.format(trial))
        print('Target {}'.format(target))
        print('Accuracy {}'.format(accuracy))
        print('Given answer {}'.format(given_answer))
        print('Correct answer {}'.format(correct_answer))
    round = trial.split('.')[0]
    if dataset == 'large':
        trial = trial.split('.')[1].split('_')[0]
    elif dataset == 'small':
        trial = trial.split('.')[1]
    this_turn = Turn(pair_name, speaker, turn_ID, turn_ID, utterance, duration,
    from_ts, to_ts, round, trial, target, director, correct_answer, given_answer, accuracy, dataset)
    return this_turn



def build_dialogue_dict(eaf, verbose=False):
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
def clean_and_pos_tag_a_turn(pair_name, turn, dataset_name, dataset, nlp, verbose=False):
    utterance = turn.utterance_speech
    include_word = np.ones(len(utterance), dtype=bool)
    # remove these special cases from the utterance --> cleaning the data
    for idx, word_level in enumerate(utterance):
        word = word_level.word
        if word == '' or word == ' ':
            include_word[idx] = False
            continue
        if word[-1] == dataset[dataset_name]['repairs_marker']: # Self-repairs: e.g., “aan de zijkant twee ku- kubussen” --> this especial for the large dataset
            if verbose:
                print(' Self-repairs: e.g., “aan de zijkant twee ku- kubussen”')
                print(utterance)
            include_word[idx] = False
            continue
        if word[-1] == '?' and len(word) > 1:
            if word[-2] == dataset[dataset_name]['repairs_marker']: # Self-repairs: e.g., “aan de zijkant twee ku- kubussen” --> this especial for the large dataset
                if verbose:
                    print(' Self-repairs: e.g., “aan de zijkant twee ku- kubussen”')
                    print(utterance)
                include_word[idx] = False
                continue
        if word in dataset[dataset_name]['interaction_words']:
            include_word[idx] = False
            continue # remove interactional markers
        if word[0] == dataset[dataset_name]['Non-verbal_sounds_marker']:
            if verbose:
                print(utterance)
            include_word[idx] = False
            continue # remove encoded laughs
        if (len(word) > 1) & (word[-1] == dataset[dataset_name]['Non-verbal_sounds_marker']):
            if verbose:
                print(utterance)
            include_word[idx] = False
            continue
    utterance = [word_level for idx, word_level in enumerate(utterance) if include_word[idx]]
    if len(utterance) == 0:
        return None, None
    processed_utterance = ' '.join([word_level.word for word_level in utterance])
    pos_utterance = nlp(str(processed_utterance))
    lemmas_with_pos = ''
    pos_sequence = ''
    lemmas_sequence = ''
    text_lemma_pos = ''
    utterance_idx = 0
    final_utterance = []
    
    special_cases = ['-', '`', '\'', '!', '.', ',', ';', ':', '?', '=', '(', ')', '[', ']', '{', '}', '/', '\\', ']?']
    for idx, token in enumerate(pos_utterance):
        lemmas_sequence += token.lemma_+' '
        pos_sequence += token.pos_+' '
        lemmas_with_pos += token.lemma_+'#'+token.pos_+' '
        text_lemma_pos += token.text+'_'+token.lemma_+'#'+token.pos_+' '
        print(pair_name)
        print(token.text, '---',utterance[utterance_idx].word)
        final_utterance.append(Utterance(token.text, utterance[utterance_idx].from_ts, utterance[utterance_idx].to_ts, token.lemma_, token.pos_, token.lemma_+'#'+token.pos_))
        # unfortunately, spacy separate special characters from the words, so we need to skip them and not cound them when taking the time of the words
        if (utterance[utterance_idx].word[-1] in special_cases) and (not token.text in special_cases):
            assert token.text == utterance[utterance_idx].word[:-1] # this mean that the word was not complete in the transcript and spacy still separated the special character from the word
            continue 
        else:
            if token.text in special_cases:
                assert token.text == utterance[utterance_idx].word[-1]
            else:
                assert token.text == utterance[utterance_idx].word
            utterance_idx += 1
    turn.set_pos_sequence(pos_sequence)
    turn.set_lemmas_with_pos(lemmas_with_pos)
    turn.set_lemmas_sequence(lemmas_sequence)
    turn.set_text_lemma_pos(text_lemma_pos)
    turn.set_utterance_speech(final_utterance)
    turn.set_processed_utterance(processed_utterance)
    return turn, processed_utterance

    
def get_non_vocal_sounds_and_targets_mentions(dataset_name, dataset):
    non_vocal_sounds_and_targets_mentions = []
    for filepath in glob.iglob(dataset[dataset_name]["dataset_path"]):
        eaf = elan.read_eaf(filepath)
        complete_dialogue_tiers = build_dialogue_dict(eaf, verbose=False)
        complete_dialogue_tiers['speakers'].sort(key=lambda x: x.from_ts, reverse=False)
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

    if dataset_name == 'large':
        return set(non_vocal_sounds_and_targets_mentions+ ["*sounds of the microphonesmack one's lips*", "*smack one's lipsbreath laugh*", "*smackinhale*", "*smack one's lipsbreath laugh*", "*coughsmack one's lips*", "*smack one's lipsbreath laugh*", "*smackinhale*", "*inhalesmack*", "*swallowsmack*", "*airthudairthudairthudairthudairthud*", "*stuttersmack one's lips*", "stutterstutter", "*stuttersmack one's lips*", "*smack one's lipsthinking sound*", "*airthudairthudairthudairthudairthud*", "*click of the computer", "*tongueclick* *tongueclick* *tongueclick* *tongueclick* *tongueclick*", "*click of the computer*", '*sounds of the microphone*'])
    return set(non_vocal_sounds_and_targets_mentions)

def get_interactional_words():
    interaction_words = []
    interaction_words_path = './data/CABB_data/words_lists/interaction_words.txt'
    with open(interaction_words_path) as f:
        lines = f.readlines()
    for word in lines:
        word = word.split("\n")[0]
        interaction_words.append(word)
    return interaction_words


def find_max_overlap_segment_vectorized(trail_from_ts, trail_to_ts, from_ts, to_ts):
    # Step 1: Check for overlaps (boolean mask)
    overlap_mask = (trail_from_ts <= to_ts) & (trail_to_ts >= from_ts)

    # Step 2: Calculate overlap start and end for the overlapping segments
    overlap_start = np.maximum(trail_from_ts, from_ts)
    overlap_end = np.minimum(trail_to_ts, to_ts)

    # Step 3: Calculate the overlap duration (apply mask)
    overlap_duration = np.maximum(0, overlap_end - overlap_start)

    # Apply the mask to ensure we only consider overlapping segments
    overlap_duration = overlap_duration * overlap_mask

    # Step 4: Find the maximum overlap duration and its index
    max_overlap_index = np.argmax(overlap_duration)

    # If no overlap is found, return None
    if overlap_duration[max_overlap_index] == 0:
        return None
    else:
        return max_overlap_index

def prepare_and_write_target_trials_per_pair(dataset_name, dataset, nlp, alignment_model, alignment_metadata, device, verbose=False, align_speech=True, save_files=True):
    non_used_pairs = []
    speech_and_objects = []
    speech_existance = False
    aligned_tagged_speech_per_word = []
    for filepath in glob.iglob(dataset[dataset_name]["dataset_path"]):
        targets_turns = defaultdict(list)
        pair_name  = re.findall(dataset[dataset_name]['pair_name_pattern'], filepath)[0]

        # check if AB_audio_path exists
        object_file =  dataset[dataset_name]["output_path"]+pair_name.split('.')[0]+'.pickle'
        eaf = elan.read_eaf(filepath)
        complete_dialogue_tiers = build_dialogue_dict(eaf, verbose=verbose)
        Turns = []
        AB_result_aligned = {'segments': [], 'word_segments': []}
        trial_index = 0
        #sort the dialogue tiers based on the start time
        trail_to_ts = np.array([tier.to_ts for tier in complete_dialogue_tiers['trial']])
        trail_from_ts = np.array([tier.from_ts for tier in complete_dialogue_tiers['trial']])

        complete_dialogue_tiers['speakers'].sort(key=lambda x: x.from_ts, reverse=verbose)
        
        try:
            if dataset_name =='small':
                A_audio_path = dataset[dataset_name]['audio_path'].format(pair_name, 'A')
                B_audio_path = dataset[dataset_name]['audio_path'].format(pair_name, 'B')
                AB_audio_path = dataset[dataset_name]['audio_path'].format(pair_name, 'AB')
            else:
                A_audio_path = dataset[dataset_name]['audio_path'].format(pair_name, 'A')
                B_audio_path = dataset[dataset_name]['audio_path'].format(pair_name, 'B')
                AB_audio_path = dataset[dataset_name]['audio_path'].format(pair_name, 'AB')
            AB_audios = {'A': load_audio(A_audio_path), 'B': load_audio(B_audio_path)}
            speech_existance = True

        except Exception as e:
            print('Audio file not found for the pair: ', A_audio_path)
            print('Audio file not found for the pair: ', B_audio_path)
            print('Error: ', e)
            non_used_pairs.append(pair_name)
            speech_existance = False
        transcript = []
        # with open(label_file, 'w', encoding='utf8', newline='') as tsv_file:
            # tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        prev_trial_index = 0
        for turn_ID, tier in enumerate(complete_dialogue_tiers['speakers']):
            from_ts = tier.from_ts
            to_ts = tier.to_ts
            if trial_index >= dataset[dataset_name]['max_trials']:
                # print('Turn is out of the trials range')# the number of trials depends on the dataset
                break
            try: # I noticed that there are some turns that are out of the trials range, e.g., these turns are between two trails. 
                overlaps = np.minimum(trail_to_ts, to_ts) - np.maximum(trail_from_ts, from_ts)
                # overlaps[overlaps < 0] = 0
                trial_index = np.argmax(overlaps)
                overl_trial_index = find_max_overlap_segment_vectorized(trail_from_ts, trail_to_ts, from_ts, to_ts)
                assert trial_index == overl_trial_index # if this fails, it means that the turn is not overlapping with a given trial (e.g., it is at the end)
                prev_trial_index = trial_index
            except:
                if prev_trial_index < dataset[dataset_name]['max_trials']-1:
                    print('Turn is overlapping two trials') # invalid turns since they do not belong to any trial
                else:
                    print('Turn is out of the trials range and should be skipped')
                continue # best is to skip this turn
            this_turn = get_turn_info(complete_dialogue_tiers, get_turn_info, tier, trial_index, turn_ID, dataset_name, verbose=verbose)
            if this_turn is not None:
                normalized_utterance = normalize_utterance(this_turn.utterance, dataset_name, dataset, verbose=False)
                if 'oh zo hij is leuk' in normalized_utterance: # wrong transcription, it is longer than the audio segment
                    normalized_utterance = 'oh zo'
                if normalized_utterance != "":
                    if align_speech and speech_existance:
                        turns_transcript = {'text': normalized_utterance, 'start': this_turn.from_ts, 'end': this_turn.to_ts, 'speaker': this_turn.speaker}
                        transcript.append(turns_transcript)
                        turn_result_aligned = align([turns_transcript], alignment_model, alignment_metadata, AB_audios, device)
                        word_level = turn_result_aligned['segments'][0]['word-level']
                        AB_result_aligned['segments'].append(turn_result_aligned['segments'][0])
                        AB_result_aligned['word_segments'].append(turn_result_aligned['word_segments'][0])
                        utterance_speech = []
                        for text in word_level:
                            utterance_speech.append(Utterance(text['text'], text['start'], text['end']))
                    else:
                        utterance_speech = []
                        for word in normalized_utterance.split():
                            utterance_speech.append(Utterance(word, this_turn.from_ts, this_turn.to_ts))
                    this_turn.set_utterance_speech(utterance_speech)
                    this_turn, processed_utterance = clean_and_pos_tag_a_turn(pair_name, this_turn, dataset_name, dataset, nlp, verbose=verbose)                    
                    if this_turn is not None:
                        #NOTE: this database can be used to get the word-level information for each gesture
                        for word in this_turn.utterance_speech:
                            aligned_tagged_speech_per_word.append({'pair_name': pair_name, 'turn_ID': this_turn.ID, 'speaker': this_turn.speaker, 'word': word.word, 'lemma': word.lemma, 'pos': word.pos, 'lemma_pos': word.lemma_pos, 'from_ts': word.from_ts, 'to_ts': word.to_ts})
                        this_turn_ID = len(Turns)
                        this_turn.set_ID(this_turn_ID)
                        target_turn = len(targets_turns[this_turn.target])
                        this_turn.set_target_turn(target_turn)
                        Turns.append(this_turn)
                        targets_turns[this_turn.target].append(this_turn)
                        speech_and_objects.append({'pair_name': pair_name, 'turn_ID': this_turn.ID, 'speaker': this_turn.speaker, 'utterance': this_turn.utterance, 'target': this_turn.target, 'trial': this_turn.trial, 'director': this_turn.director, 'correct_answer': this_turn.correct_answer, 'given_answer': this_turn.given_answer, 'accuracy': this_turn.accuracy, 'processed_utterance': this_turn.processed_utterance, 'pos_sequence': this_turn.pos_sequence, 'lemmas_sequence': this_turn.lemmas_sequence, 'text_lemma_pos': this_turn.text_lemma_pos, 'round': this_turn.round})
        # NOTE: Now add gestures to the closest turn, regardless of the speaker, since a gesture of a speaker might not overlap with a turn or a turn by the same speaker
        if dataset_name == 'small':
            Turns = get_gestures_info(Turns, complete_dialogue_tiers, 'A', 'R', verbose=verbose)
            Turns = get_gestures_info(Turns, complete_dialogue_tiers, 'A', 'L', verbose=verbose)
            Turns = get_gestures_info(Turns, complete_dialogue_tiers, 'B', 'R', verbose=verbose)
            Turns = get_gestures_info(Turns, complete_dialogue_tiers, 'B', 'L', verbose=verbose)
         
        if save_files:
            # with open(object_file, 'wb') as handle:
            with open(object_file, 'wb') as handle:
                pickle.dump(Turns, handle, protocol=pickle.HIGHEST_PROTOCOL)
            fribbles = targets_turns.keys()
            for fribble in fribbles:
                label_file = dataset[dataset_name]["output_path"]+pair_name.split('.')[0]+"_"+fribble+'.tsv'
                object_file =  dataset[dataset_name]["output_path"]+pair_name.split('.')[0]+"_"+fribble+'.pickle'
                with open(label_file, 'w', encoding='utf8', newline='') as tsv_file:
                    tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
                    for turn in targets_turns[fribble]:
                        tsv_writer.writerow([turn.speaker+': ', turn.lemmas_sequence])
                with open(object_file, 'wb') as handle:
                    pickle.dump(targets_turns[fribble], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
       
        print("Well done!")
    import pandas as pd
    speech_and_objects = pd.DataFrame(speech_and_objects)
    aligned_tagged_speech_per_word = pd.DataFrame(aligned_tagged_speech_per_word)
     # save the speech and objects to a csv file
    speech_and_objects.to_csv(dataset[dataset_name]["output_path"]+'speech_and_objects_{}.csv'.format(dataset_name), index=False)
    aligned_tagged_speech_per_word.to_csv(dataset[dataset_name]["output_path"]+'aligned_tagged_speech_per_word_{}.csv'.format(dataset_name), index=False)
    return non_used_pairs
  
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process dataset paths and configurations.")
    parser.add_argument('--spacy_model', type=str, default='nl_core_news_lg', choices=['nl_core_news_lg', 'nl_core_news_md', 'nl_core_news_sm'], help='Spacy model to use')
    parser.add_argument('--audio_path', type=str, default='{}_synced_pp{}.wav', help='Audio path template')
    parser.add_argument('--elan_path_small', type=str, default='ElanFiles/New_ELAN_files/pair*.eaf', help='ELAN files path for small dataset')
    parser.add_argument('--elan_path_large', type=str, default='ElanFiles/New_ELAN_files/*complete.eaf', help='ELAN files path for large dataset')
    parser.add_argument('--output_path', type=str, default='dialog_utils/dialign/dialogues/{}/', help='Output path template')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    print("NOTE: This script is designed to work with the CABB dataset, which is in Dutch. It uses Spacy for NLP tasks and WhisperX for speech alignment.")
    import spacy
    print('Using Spacy model:', args.spacy_model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    spacy_model = args.spacy_model
    
    alignment_model, alignment_metadata = whisperx.load_align_model("nl", device=device)
    
    inaudiable_speech_markers = ['(?)', '()', '( )', '(', ')', '(?])', '?]', '[?', ']?']
    nlp = spacy.load(spacy_model)
    dataset = {}

    for dataset_name, elan_path, max_trials, pair_name_pattern, non_verbal_sounds_matcher, non_verbal_sounds_marker, targets_marker, target_matcher, repairs_marker in [
        ('small', args.elan_path_small, 96, r"/([\w\d]+)\.", r"\#.*?\#", '#', '*', r"\*.*?\*", '-'),
        ('large', args.elan_path_large, 192, r"/(\d+)_", r"\*.*?\*", '*', '#', r"\#.*?\#", ']')
    ]:
        dataset[dataset_name] = {
            'audio_path': args.audio_path,
            'dataset_path': elan_path,
            'output_path': args.output_path.format(spacy_model.split('_')[-1]),
            'name': dataset_name,
            'max_trials': max_trials,
            'pair_name_pattern': pair_name_pattern,
            'Non-verbal_sounds_matcher': non_verbal_sounds_matcher,
            'Non-verbal_sounds_marker': non_verbal_sounds_marker,
            'targets_marker': targets_marker,
            'target_matcher': target_matcher,
            'repairs_marker': repairs_marker,
            'inaudiable_speech_markers': inaudiable_speech_markers,
            'interaction_words': get_interactional_words(),
        }
        Non_verbal_sounds_and_target_mentions = get_non_vocal_sounds_and_targets_mentions(dataset_name, dataset)
        dataset[dataset_name]['Non-verbal_sounds_and_target_mentions'] = Non_verbal_sounds_and_target_mentions
        
        if not os.path.exists(dataset[dataset_name]['output_path']):
            os.makedirs(dataset[dataset_name]['output_path'])

        dataset[dataset_name]['dutch_numbers'] = {f'{targets_marker}{i}{targets_marker}': name for i, name in enumerate(['een', 'twee', 'drie', 'vier', 'vijf', 'zes', 'zeven', 'acht', 'negen', 'tien', 'elf', 'twaalf', 'dertien', 'veertien', 'vijfteen', 'zestien'], 1)}
        dataset[dataset_name]['dutch_capital_letters'] = {f'{targets_marker}{chr(65+i)}{targets_marker}': chr(65+i) for i in range(16)}
        dataset[dataset_name]['dutch_capital_letters_list'] = [f'{targets_marker}{chr(65+i)}{targets_marker}' for i in range(16)]
        non_used_pairs = prepare_and_write_target_trials_per_pair(dataset_name, dataset, nlp, alignment_model, alignment_metadata, device, verbose=False)