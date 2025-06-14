import sys
# add ui_utils to the path
# sys.path.append('./dialog_utils/')

import numpy as np
import pandas as pd
import os
from scipy.stats import entropy
import math 
import dill as pickle
import re 
from collections import defaultdict, Counter
from dialog_utils.data_classes import Turn, Utterance, Gesture
from dialog_utils.prepare_annotate_dialogues import get_gestures_info, get_turn_info

def load_pickle(file_name):
   with open(file_name, 'rb') as reader:
        b = pickle.load(reader)
        return b

def calculate_entropy(targets):
    bins=np.arange(0, 17)
    max_entropy = math.log(len(bins)-1,2)
    exp_entropy = entropy(np.histogram(np.array(targets)-1, bins=bins)[0], base=2)/max_entropy
    return exp_entropy

def read_annotated_pos_data(dialign_output, turn_info_path):
   shared_constructions_info = defaultdict(list)
   targets_turns_info = defaultdict(list)
   turns_info = defaultdict()
   for f in os.listdir(dialign_output):
      if f.endswith('_tsv-lexicon.tsv') and not f.startswith('.'):
         filepath = os.path.join(dialign_output, f)
         files_parts = filepath.split('_')
         fribble_ID = files_parts[-2]
         pair_name = files_parts[-3].split('/')[1]
         all_turns_info_path = turn_info_path+pair_name+'.pickle'
         targets_turns_info_path = turn_info_path+pair_name+'_'+fribble_ID+'.pickle'
         pair_target_shared_expressions = pd.read_csv(filepath, sep='\t', header=0)
         with open(targets_turns_info_path, 'rb') as reader:
               targets_pair_turns_info = pickle.load(reader)
        
         for i, row in pair_target_shared_expressions.iterrows():
            turns =  [str(targets_pair_turns_info[int(turn)].ID) for turn in row['Turns'].split(',')]
            pair_target_shared_expressions.loc[i, 'Turns'] = ", ".join(turns)
            pair_target_shared_expressions.loc[i, 'Establishment turn'] = targets_pair_turns_info[int(row['Establishment turn'])].ID
            pair_target_shared_expressions.loc[i, 'Spanning'] = targets_pair_turns_info[int(row['Establishment turn'])].ID - int(turns[0])
         shared_constructions_info[pair_name].append(pair_target_shared_expressions)
         targets_turns_info[pair_name].extend(targets_pair_turns_info)
         with open(all_turns_info_path, 'rb') as reader:
               turns_info[pair_name] = pickle.load(reader)
   for pair_name in shared_constructions_info.keys(): 
      shared_constructions_info[pair_name]= pd.concat(shared_constructions_info[pair_name], axis=0)
      shared_constructions_info[pair_name].reset_index(drop=True, inplace=True)
   return shared_constructions_info, turns_info


def read_annotated_pos_data_self_repeatitions(dialign_output, turn_info_path):
   shared_constructions_info = defaultdict(list)
   targets_turns_info = defaultdict(list)
   turns_info = defaultdict()
   for f in os.listdir(dialign_output):
      if (f.endswith('_tsv-lexicon-self-rep-A.tsv') or f.endswith('_tsv-lexicon-self-rep-B.tsv'))  and not f.startswith('.'):
         filepath = os.path.join(dialign_output, f)
         files_parts = filepath.split('_')
         fribble_ID = files_parts[-2]
         pair_name = files_parts[-3].split('/')[1]
         all_turns_info_path = turn_info_path+pair_name+'.pickle'
         targets_turns_info_path = turn_info_path+pair_name+'_'+fribble_ID+'.pickle'
         pair_target_shared_expressions = pd.read_csv(filepath, sep='\t', header=0)
         with open(targets_turns_info_path, 'rb') as reader:
               targets_pair_turns_info = pickle.load(reader)
        
         for i, row in pair_target_shared_expressions.iterrows():
            turns =  [str(targets_pair_turns_info[int(turn)].ID) for turn in row['Turns'].split(',')]
            pair_target_shared_expressions.loc[i, 'Turns'] = ", ".join(turns)
            pair_target_shared_expressions.loc[i, 'Establishment turn'] = targets_pair_turns_info[int(row['Establishment turn'])].ID
            pair_target_shared_expressions.loc[i, 'Spanning'] = targets_pair_turns_info[int(row['Establishment turn'])].ID - int(turns[0])
         shared_constructions_info[pair_name].append(pair_target_shared_expressions)
         targets_turns_info[pair_name].extend(targets_pair_turns_info)
         with open(all_turns_info_path, 'rb') as reader:
               turns_info[pair_name] = pickle.load(reader)
   for pair_name in shared_constructions_info.keys(): 
      shared_constructions_info[pair_name]= pd.concat(shared_constructions_info[pair_name], axis=0)
      shared_constructions_info[pair_name].reset_index(drop=True, inplace=True)
   return shared_constructions_info, turns_info


def create_exp_info_row(expression, length, targets, target_fribbles, num_targets, exp_entropy, speakers, freq, free_freq, priming, spanning_rounds, spanning_time, duration_to_emerge, turns_to_emerge, rounds_to_emerge, turns, rounds, establishment_round, establishment_turn,  first_round, last_round, initiators, shared_exp_pos_seq, shared_exp_pos, pair, dataset, from_ts, to_ts, establishment_ts):
    return { 'exp': expression, 'length': length, 'fribbles': targets, 'target_fribbles': target_fribbles,  '#fribbles': num_targets, 'fribbles entropy': exp_entropy, 'speakers': speakers, 'freq': freq, 'free. freq': free_freq, 'priming': priming, 'spanning_rounds': spanning_rounds, 'spanning_time': spanning_time, 'duration_to_emerge': duration_to_emerge, 'turns_to_emerge':turns_to_emerge, 'rounds_to_emerge': rounds_to_emerge, 'turns': turns, 'rounds': rounds, 'estab_round': establishment_round, 'estab_turn': establishment_turn, 'first_round': first_round, 'last_round': last_round, 'initiator': initiators, 'pos_seq': shared_exp_pos_seq, 'exp with pos': shared_exp_pos, 'pair': pair, 'dataset': dataset, 'from_ts': from_ts, 'to_ts': to_ts, 'establishment_ts': establishment_ts}

def extract_all_shared_exp_info(shared_constructions_info, turns_info):
    first_row = True
    exp_info = pd.DataFrame()
    for pair in shared_constructions_info:
        shared_expressions = shared_constructions_info[pair]['Surface Form'].to_list()
        all_turns = shared_constructions_info[pair]['Turns'].to_list()
        priming = shared_constructions_info[pair]['Priming'].to_list()
        free_freq = shared_constructions_info[pair]['Free Freq.'].to_list()
        priming = shared_constructions_info[pair]['Priming'].to_list()
        spanning = shared_constructions_info[pair]['Spanning'].to_list()
        freqs = shared_constructions_info[pair]['Freq.'].to_list()
        lengths = shared_constructions_info[pair]['Size'].to_list()
        initiators = shared_constructions_info[pair]['First Speaker'].to_list()
        establishment_turns = shared_constructions_info[pair]['Establishment turn'].to_list()
        shared_exp_pos = shared_constructions_info[pair]['Exp with POS'].to_list()
        shared_exp_pos_seq = shared_constructions_info[pair]['POS'].to_list()
        # create a list of pair the same size as the number of shared expressions
    
        for idx, turns in enumerate(all_turns):
            all_turns[idx] = np.array(turns.split(','), dtype=int)
        for idx, turns in enumerate(all_turns):
            targets = [int(turns_info[pair][turn].target) for turn in turns]
            dataset = turns_info[pair][turns[0]].dataset
            speakers = [turns_info[pair][turn].speaker for turn in turns]
            rounds = [int(turns_info[pair][turn].round) for turn in turns]
            num_targets = len(set(targets))
            establishment_turn = int(establishment_turns[idx])
            establishment_round = int(turns_info[pair][establishment_turn].round)         
            
            spanning_time = turns_info[pair][turns[-1]].to_ts - turns_info[pair][establishment_turn].from_ts 
            spanning_rounds = int(turns_info[pair][turns[-1]].round) - int(turns_info[pair][turns[0]].round)

            last_round = turns_info[pair][turns[-1]].round
            first_round = turns_info[pair][turns[0]].round
            rounds_to_emerge = int(turns_info[pair][establishment_turn].round) - int(turns_info[pair][turns[0]].round)
            turns_to_emerge = establishment_turn - turns[0]
            
            duration_to_emerge = turns_info[pair][turns[-1]].to_ts - turns_info[pair][establishment_turn].from_ts
            exp_entropy = calculate_entropy(targets)
            target_fribbles = np.unique(targets)
            from_ts = [turns_info[pair][a_turn].from_ts for a_turn in turns]
            to_ts = [turns_info[pair][a_turn].to_ts for a_turn in turns]
            establishment_ts = turns_info[pair][establishment_turn].from_ts
        

            this_exp_info = create_exp_info_row(shared_expressions[idx], lengths[idx], targets, target_fribbles, num_targets, exp_entropy, speakers, freqs[idx], free_freq[idx], priming[idx], spanning_rounds, spanning_time, duration_to_emerge, turns_to_emerge, rounds_to_emerge, turns, rounds, establishment_round, establishment_turn, first_round, last_round, initiators[idx], shared_exp_pos_seq[idx], shared_exp_pos[idx], pair, dataset, from_ts, to_ts, establishment_ts)
            if first_row:
                exp_info = pd.Series(this_exp_info).to_frame().T
                first_row = False
            else:
                exp_info = pd.concat([exp_info, pd.Series(this_exp_info).to_frame().T])
                # print(exp_info)
          
                
    exp_info = exp_info.reset_index(drop=True)
    return exp_info

def prepare_dialogue_shared_expressions_and_turns_info(dialign_output, turn_info_path, self_repition=False):
    if self_repition:
        shared_constructions_info, turns_info = read_annotated_pos_data_self_repeatitions(dialign_output, turn_info_path)
    else:
        shared_constructions_info, turns_info = read_annotated_pos_data(dialign_output, turn_info_path)
    for pair in shared_constructions_info:
        shared_exp_pos = shared_constructions_info[pair]['Surface Form'].copy()
        shared_exp_turns = shared_constructions_info[pair]['Turns'].copy()
        for idx, turns in enumerate(shared_exp_turns):
            shared_exp_turns[idx] = np.array(turns.split(','), dtype=int)
        shared_exp = []
        pos_seq_shared_exp = []
        expressions_with_pos = []
        for exp_idx, exp in enumerate(shared_exp_pos):
            pos_seq = []
            turns = shared_exp_turns[exp_idx]        
            exp_patterns = r"(" + r"#[\w]+ ".join(exp.split(' '))+ r"#[\w]+)"
            exp_patterns = exp_patterns.replace('##', '#')
            exp_patterns = exp_patterns.replace('?', '\?')
            # exp = exp_patterns.replace(', #PUNCT', ',#PUNCT')
            exp_patterns = exp_patterns.replace(')#', '\)#')
            exp_patterns = exp_patterns.replace('((', '(\(')
            exp_patterns = exp_patterns.replace('*', '\*')
            exp_with_pos_all_turns = []
            for turn in turns:
                turn = int(turn)
                exp_with_pos = re.findall(exp_patterns, turns_info[pair][turn].lemmas_with_pos)
                # print(turns_info[pair][turn].pos_sequence)
                exp_with_pos_all_turns.append(exp_with_pos[0])
            exp_with_pos_all_turns = Counter(exp_with_pos_all_turns)
            exp_with_pos = max(exp_with_pos_all_turns, key=exp_with_pos_all_turns.get)
            # exp_with_pos = exp_with_pos.replace('_', '#')
            expressions_with_pos.append(exp_with_pos)
            for word in exp_with_pos.split(' '):
                pos_seq.append(word.split('#')[1])
            pos_seq_shared_exp.append(" ".join(pos_seq))
        # shared_constructions_info[pair]['Surface Form'] = shared_exp
        assert len(pos_seq_shared_exp) == len(shared_exp_pos)
        shared_constructions_info[pair]['POS'] = pos_seq_shared_exp
        shared_constructions_info[pair]['Exp with POS'] = expressions_with_pos 
    return shared_constructions_info, turns_info

def all_func_words(pos_seq, pos_func_words):
    tokenized_pos_seq = pos_seq.split(' ')
    return all(word in pos_func_words for word in tokenized_pos_seq)

def read_stop_function_words():
   """Reads stop function words from file and returns them as a list."""
   most_common_words = pd.read_excel('../data/SUBTLEX-NL.cd-3SDsmean.xlsx')  
   frequencies = most_common_words['FREQlemma'].to_numpy()
   # select the most frequent words based on threshold used in Marlou's study
   highest_frequencies = frequencies >= 85918
   stop_words = most_common_words['Word'].to_numpy()[highest_frequencies]

   function_words = [] 
   with open('/Users/esamghaleb/Documents/ResearchData/CABB Large Dataset/Results/Word_counts/1_preprocessing/input/function_words.txt', encoding="ISO-8859-1") as f:
      lines = f.readlines()
   for word in lines:
      word = word.split("\n")[0]
      function_words.append(word)
   return stop_words, function_words

def get_linked_expressions_for_single_fribble(dialign_output, turn_info_path):
    shared_constructions_info, turns_info = prepare_dialogue_shared_expressions_and_turns_info(dialign_output, turn_info_path)
    exp_info = extract_all_shared_exp_info(shared_constructions_info, turns_info)
    pos_func_words = ['DET', 'PRON', 'ADP', 'CCONJ', 'SCONJ', 'AUX', 'PART', 'PUNCT', 'SYM', 'X', 'INTJ'] # https://universaldependencies.org/u/pos/
    stop_words, function_words = read_stop_function_words()
    exp_with_only_func_words = np.logical_not([all_func_words(pos_seq, pos_func_words) for idx, pos_seq in enumerate(exp_info['pos_seq'].to_list())])
    exp_info = exp_info[exp_with_only_func_words]

    exp_with_only_func_words = np.logical_not([all_func_words(exp, function_words) for idx, exp in enumerate(exp_info['exp'].to_list())])
    exp_info = exp_info[exp_with_only_func_words]

    exp_with_only_stop_words = np.logical_not([all_func_words(exp, stop_words) for idx, exp in enumerate(exp_info['exp'].to_list())])
    exp_info = exp_info[exp_with_only_stop_words]
    exp_info['shared expressions'] = exp_info['exp'].apply(lambda x: [x])
    # check if there are nan values in exp_info
    assert not exp_info.isnull().values.any()
    exp_info = exp_info[exp_info['#fribbles'] == 1]
    # gestures_info, gesture_data = get_gestures_info(turns_info)
    return exp_info, turns_info,shared_constructions_info, function_words, pos_func_words 
def assert_match_betwee_shared_exp_and_actual_utterances(turns_info, shared_constructions_info):
    all_pairs = shared_constructions_info.keys()
    for pair in all_pairs:
        for row_idx, row in shared_constructions_info[pair].iterrows():
            exp = row['Exp with POS']
            exp_patterns = [r"[\w'?]+_"+exp_pos for exp_pos in exp.split(' ')]
            exp_patterns_to_extract_tokens = r"(" + " ".join(exp_patterns)+ r")"
            exp_patterns_to_extract_tokens = exp_patterns_to_extract_tokens.replace(r'?', r'\?')
            turns = row['Turns'].split(',')
            exp_lemmas = " ".join([exp_pos.split('#')[0] for exp_pos in exp.split(' ')])
            for turn in turns:
                assert exp_lemmas in turns_info[pair][int(turn)].lemmas_sequence

if __name__ == "__main__":
    dialign_output = 'code/dialign/output_targets_riws_lemma/'
    turn_info_path = 'code/dialign/targets_riws_lemma/'
    fribbles_path = "/Users/esamghaleb/Documents/ResearchData/CABB Small Dataset/Fribbles/{}.jpg"
    videos_path = '/Users/esamghaleb/Documents/ResearchData/CABB Small Dataset/processed_audio_video/{}/{}_synced_overview.mp4'
    temp_videos = 'temp_lex_align_videos/{}/{}_{}.mp4'
    temp_videos_path = '/'.join(temp_videos.split('/')[0:-1])
    shared_constructions_info, turns_info = prepare_dialogue_shared_expressions_and_turns_info(dialign_output, turn_info_path)
    exp_info = extract_all_shared_exp_info(shared_constructions_info, turns_info)
    print('wtfffff')
