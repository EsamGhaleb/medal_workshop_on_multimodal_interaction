import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from model.skeleton_speech_models import GSSModel
from data.read_process_poses import load_keypoints_dict
from dialog_utils.prepare_gesture_referents_dataset import get_detailed_gestures
from object_retrieval_pair import manual_implementation_object_retrieval
from gestures_forms_sim import measure_sim_gestures

pd.set_option("display.precision", 2)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask

def get_embeddings(sentences, model, tokenizer):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        model_output = model(**encoded_input)
    return mean_pooling(model_output, encoded_input['attention_mask'])

def load_models():

    # BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")
    bert_model = AutoModel.from_pretrained("GroNLP/bert-base-dutch-cased", output_hidden_states=True).to(DEVICE)
    bert_model.eval()

    return bert_model, tokenizer

def load_pretrained_model(model_params=None):
    model_path = model_params['model_weights'] 
    # remove the model weights from the model params
    del model_params['model_weights']
    pretrained_model = GSSModel(**model_params)
    weights_dict = torch.load(model_path, map_location=torch.device('cpu'))
    for key in pretrained_model.state_dict().keys():
        pretrained_model.state_dict()[key].copy_(weights_dict['state_dict']['model.'+key])
    return pretrained_model.to(DEVICE)


def main():
    processed_keypoints_dict, mirrored_keypoints_dict = load_keypoints_dict()
    speakers_pairs = processed_keypoints_dict.keys()
    # remove pairs without 'pair' --> this is to focus on CABB-S pairs
    speakers_pairs = [speaker_pair for speaker_pair in speakers_pairs if 'pair' in speaker_pair]

    audio_dict = {} #load_audio_dict(speakers_pairs, audio_path='data/audio_files/{}_synced_pp{}.wav', audio_sample_rate=16000)
    gestures_info = get_detailed_gestures()
    gestures_info['round'] = gestures_info['round'].astype(int)
    gestures_info['object'] = gestures_info['object'].astype(int)
    bert_model, tokenizer = load_models()
    gestures_info = gestures_info[gestures_info['type'] == 'iconic'].reset_index(drop=True)

    # get semantic embeddings
    gestures_info['semantic_embeddings'] = gestures_info['processed_utterance'].apply(
        lambda x: get_embeddings([x], bert_model, tokenizer).squeeze(0).cpu().numpy()
    )
    
    gestures_info['referent'].unique()
    # remove -REP, -rep, -WF, -wf from referent
    to_remove = ['-REP', '-rep', '-WF', '-wf', 'ref', 'REF']
    gestures_info['referent_clean'] = gestures_info['referent'].apply(lambda x: x if not x.endswith(tuple(to_remove)) else x.split('-')[0])
    # explode the referent_clean column with '+' as separator
    gestures_info_exploded = gestures_info.assign(referent_clean=gestures_info['referent_clean'].str.split('+')).explode('referent_clean')
    gestures_info_exploded['referent_clean'] = gestures_info_exploded['referent_clean'].str.strip()
    # replace - with 0 in referent_clean
    gestures_info_exploded['referent_clean'] = gestures_info_exploded['referent_clean'].str.replace('-', '0')
    gestures_info_exploded['referent_clean'] = gestures_info_exploded['referent_clean'].str.replace('\'', '')
    # remove undecided from referent_clean
    gestures_info_exploded = gestures_info_exploded[gestures_info_exploded['referent_clean'] != 'undecided']
    gestures_info_exploded['referent_clean'] = gestures_info_exploded.apply(lambda x: f'{17}' if 'general' in x['referent_clean'] else x['referent_clean'], axis=1)
    gestures_info_exploded['referent_clean'] = gestures_info_exploded.apply(lambda x: f'{0}' if 'main' in x['referent_clean'] else x['referent_clean'], axis=1)
        
    #TODO: load the models from our work properly. For now, we are using hardcoded paths
    model_types = {
        'multimodal-x-skeleton-semantic': 
            {'modalities': ['skeleton', 'semantic'], 'hidden_dim': 256, 'cross_modal':True,'attentive_pooling':False, 'model_weights':  'pretrained_models/multimodal-x_skeleton_text_correlation=0.30.ckpt'},
        'multimodal-skeleton-semantic': 
             {'modalities': ['skeleton', 'semantic'], 'hidden_dim': 256, 'cross_modal':False,'attentive_pooling':False, 'model_weights':  'pretrained_models/multimodal_skeleton_text_correlation=0.29.ckpt'},
        'unimodal_skeleton': 
            {'modalities': ['skeleton'], 'hidden_dim': 256, 'cross_modal':False, 'attentive_pooling':False, 'model_weights':  'pretrained_models/unimodal_correlation=0.22.ckpt'},
    }
    all_model_results = [] 
    
    for model_type in model_types.keys():
        print(f'Loaded model {model_type} with modalities {model_types[model_type]["modalities"]} and weights from {model_types[model_type]["model_weights"]}')
        pretrained_model = load_pretrained_model(model_params=model_types[model_type])
        pretrained_model.eval()
        pretrained_model.to(DEVICE)
        spearman_correlation, difference, combined_gestures_form = measure_sim_gestures(pretrained_model , multimodal=True,processed_keypoints_dict=processed_keypoints_dict, mirrored_keypoints_dict=mirrored_keypoints_dict, modalities=model_types[model_type]['modalities'],audio_dict=audio_dict)

        gestures_info_exploded = manual_implementation_object_retrieval(pretrained_model , processed_keypoints_dict, mirrored_keypoints_dict, sekeleton_backbone='jointsformer', gestures_and_speech_info=gestures_info_exploded, modalities=model_types[model_type]['modalities'], audio_dict=audio_dict,model_type=model_type)
        # rename the column 'transformer_features' to the model type
        gestures_info_exploded.rename(columns={'transformer_features': model_type}, inplace=True)        

           
    # random skeleton features (baseline)
    gestures_info_exploded['random_skeleton_features'] = gestures_info_exploded['multimodal-skeleton-semantic'].apply(
        lambda x: np.random.rand(x.shape[0], 1)
    )
    
    gestures_info_exploded['semantic+multimodal-x'] = gestures_info_exploded.apply( lambda x: np.concatenate([x['multimodal-x-skeleton-semantic'], x['semantic_embeddings']]), axis=1)
    gestures_info_exploded['semantic+multimodal'] = gestures_info_exploded.apply( lambda x: np.concatenate([x['multimodal-skeleton-semantic'], x['semantic_embeddings']]), axis=1)
    gestures_info_exploded['semantic+unimodal'] = gestures_info_exploded.apply( lambda x: np.concatenate([x['unimodal_skeleton'], x['semantic_embeddings']]), axis=1)
    

    gestures_info_exploded.to_pickle('data/gestures_info_exploded_subparts.pkl')
    

if __name__ == "__main__":
    main()
