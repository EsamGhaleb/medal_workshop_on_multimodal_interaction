import numpy as np
from collections import defaultdict
import pandas as pd
import pickle
import os

def load_pickle(file_name):
   with open(file_name, 'rb') as handle:
        b = pickle.load(handle)
        return b
def read_annotated_data(dialign_output, turn_info_path):
    shared_constructions_info = defaultdict()
    turns_info = defaultdict()
    for f in os.listdir(dialign_output):
        if f.endswith('_tsv-lexicon.tsv') and not f.startswith('.'):
            filepath = os.path.join(dialign_output, f)
            files_parts = filepath.split('_')
            pair_name = files_parts[1].split('/')[1]
            Turn_info_path = turn_info_path+pair_name+'_'+files_parts[2]+"_objects.pickle"
            shared_constructions_info[pair_name] = pd.read_csv(filepath, sep='\t', header=0)
            turns_info[pair_name] = load_pickle(Turn_info_path)
    return shared_constructions_info, turns_info

# Extract information related to turns of shared expressions and establishing turns
def extract_shared_expressions_info(shared_constructions_info, turns_info, plot_per_pair=False):
# use a pair to demonstrate the idea of turns information
    all_expressions_turn_sizes = defaultdict(lambda: defaultdict(int))
    all_expressions_turns = defaultdict(int)
    all_establishment_turn_per_round_sizes = defaultdict(lambda: defaultdict(int))

    for target_pair in shared_constructions_info:
        all_turns = shared_constructions_info[target_pair]['Turns'].to_list()
        free_freq = shared_constructions_info[target_pair]['Free Freq.'].to_list()
        freq = shared_constructions_info[target_pair]['Freq.'].to_list()
        size = shared_constructions_info[target_pair]['Size'].to_list()
        establishment_turns = shared_constructions_info[target_pair]['Establishment turn'].to_list()
        for idx, turns in enumerate(all_turns):
            all_turns[idx] = np.array(turns.split(','), dtype=int)
        expressions_turn_sizes = defaultdict(lambda: defaultdict(int))
        expressions_turns = defaultdict(int)
        establishment_turn_per_round = defaultdict(int)
        establishment_turn_per_round_sizes = defaultdict(lambda: defaultdict(int))

        for idx, turns in enumerate(all_turns):
            # print(' Target construction is: '+shared_constructions[idx]) 
            establishment_turn = establishment_turns[idx]
            round = turns_info[target_pair][establishment_turn].round
            establishment_turn_per_round[round] += 1
            establishment_turn_per_round_sizes[round][size[idx]] += 1
            all_establishment_turn_per_round_sizes[round][size[idx]] += 1

            for turn_idx, turn in enumerate(all_turns[idx]):
                # turn related info: director, target (fribble), round, trial, accuracy, gestures
                turn_speaker = turns_info[target_pair][turn].speaker
                turn_director = turns_info[target_pair][turn].director
                turn_target = turns_info[target_pair][turn].target
                turn_round = turns_info[target_pair][turn].round
                expressions_turns[turn_round] += 1
                expressions_turn_sizes[turn_round][size[idx]] += 1
                all_expressions_turns[turn_round] += 1
                all_expressions_turn_sizes[turn_round][size[idx]] += 1

        pd_expressions_turns = pd.DataFrame(expressions_turns, index=[0]).T.sort_index()
        if plot_per_pair:
            pd_expressions_turns.plot(kind="bar", figsize=(10, 5), title="Distribution of expressions over round in {}".format(target_pair), xlabel="round", ylabel="occurances")
        expressions_turn_sizes = pd.DataFrame(expressions_turn_sizes).sort_index().T.sort_index()
        if plot_per_pair:
            expressions_turn_sizes.plot(kind="bar", stacked=True, figsize=(10, 5), title="Expressions length over round in {}".format(target_pair), xlabel="round", ylabel="occurances and their length")
        establishment_turn_per_round = pd.DataFrame(establishment_turn_per_round, index=[0]).T.sort_index()
        if plot_per_pair:
            establishment_turn_per_round.plot(kind="bar", stacked=True, figsize=(10, 5), title="Establishment turn in {}".format(target_pair), xlabel="round", ylabel="occurances")
        establishment_turn_per_round_sizes = pd.DataFrame(establishment_turn_per_round_sizes).sort_index().T.sort_index()
        if plot_per_pair:
            establishment_turn_per_round_sizes.plot(kind="bar", stacked=True, figsize=(10, 5), title="Establishment turns' length in {}".format(target_pair), xlabel="round", ylabel="sizes")
    pd_all_expressions_turns = pd.DataFrame(all_expressions_turns, index=[0]).T.sort_index()
    pd_all_expressions_turn_sizes = pd.DataFrame(all_expressions_turn_sizes).sort_index().T.sort_index()
    pd_all_establishment_turn_per_round_sizes = pd.DataFrame(all_establishment_turn_per_round_sizes).sort_index().T.sort_index()

    return pd_all_expressions_turns, pd_all_expressions_turn_sizes, pd_all_establishment_turn_per_round_sizes