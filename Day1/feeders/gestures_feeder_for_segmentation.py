import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import sys
import random
sys.path.extend(['../'])
from torchaudio import functional as F
import glob
import os
from utils.mediapipe_augmentations import Compose, CenterNormalize3D, Jitter3D, RandomRotate3D, RandomScale3D, RandomTranslate3D, RandomShear3D, RandomFlip3D


# from utils.utils_data import crop_scale

from tqdm import tqdm
mediapipe_flip_index = np.concatenate(([0,2,1,4,3,6,5], [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48], [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26, 27], ), axis=0) 

mmpose_flip_index = np.concatenate(([0,2,1,4,3,6,5],[17,18,19,20,21,22,23,24,25,26],[7,8,9,10,11,12,13,14,15,16]), axis=0) 


def miror_poses(data_numpy):
   C,T,V,M = data_numpy.shape
   assert C == 3 # x,y,c
   data_numpy[0, :, :, :] = 1920 - data_numpy[0, :, :, :]
   return data_numpy

def process_poses(poses):
   # add one dimension to the pose at the end
   poses = np.expand_dims(poses, axis=-1)
   # original shape is T, V, C, M
   T, V, C, M = poses.shape
   poses = np.transpose(poses, (2, 0, 1, 3))

   mirrod_poses = miror_poses(poses.copy())
   return poses, mirrod_poses
       
def load_keypoints_dict(all_pair_speakers):
   # read all poses in f'data/final_poses/poses_{pair}_synced_pp{speaker}.npy'
   processed_keypoints_dict = {}
   for file in tqdm(glob.glob('./data/mediapipe_outputs/*.npy'), desc='Loading keypoints...', total=len(glob.glob('./data/mediapipe_outputs/*.npy'))):
      # file format = data/mediapipe_outputs/poses_pair04_synced_ppA.npy 
      pair = file.split('/')[-1].split('_')[0]
      speaker = file.split('/')[-1].split('_')[-1].split('.')[0][-1]
      pair_speaker = f"{pair}_{speaker}"
      if pair_speaker not in all_pair_speakers:
            continue
      try:
         processed_keypoints_dict[pair_speaker] = np.load(file)
      except Exception as e:
         print(e)
         print('Error in loading the keypoints for the pair:', pair, speaker)
         continue
   return processed_keypoints_dict


class CABBFeeder(Dataset):
    def __init__(
            self,
            random_choose=True,
            random_shift=True,
            random_move=True,
            window_size=30,
            normalization=True,
            debug=False,
            use_mmap=True,
            random_mirror=True,
            random_mirror_p=0.5,
            is_vector=False,
            fold=0,
            sample_rate1=16000,
            debug_audio=True,
            data_path='./dialog_utils/data/gestures_info_with_text.csv',
            poses_path='./data/',
            modalities=["skeleton", "speech"],
            audio_path='./data/audio_files/{}_synced_pp{}.wav',
            n_views=2,
            apply_skeleton_augmentations=True,
            skeleton_augmentations=None,
            random_scale=True,
            fps=30000/1001,
            speech_buffer=0.0,
            gesture_buffer=0.5,
            apply_speech_augmentations=False,
            global_normalization=False,
            phase='train',
            task='segmentation',
            use_only_small_dataset=True,
            filter_text=True,
            crop_scale=False,
            skeleton_backbone='jointsformer',
            eval_on_small_dataset=True,
            train_data=True,
            **kwargs
    ):
        """ 
        :param poses_path: path to poses 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param data_path: path to data
        :param modalities: list of modalities
        :param n_views: number of views (1 or 2)
        :param apply_augmentations: whether to apply augmentations to view1, view2 is augmented by default
        """
        self.data_path = data_path
        self.poses_path = poses_path
        self.audio_path = audio_path
        self.modalities = modalities
        self.phase = phase
        self.apply_speech_augmentations = apply_speech_augmentations
        self.global_normalization = global_normalization
        self.apply_skeleton_augmentations=apply_skeleton_augmentations
        self.skeleton_augmentations=skeleton_augmentations
        self.random_scale = random_scale
        self.task = task
        self.use_only_small_dataset = use_only_small_dataset
        self.skeleton_backbone = skeleton_backbone
        self.eval_on_small_dataset = eval_on_small_dataset
        self.train_data = train_data
        self.fps = fps
        self.kwargs = kwargs

        if 'semantic' in self.modalities:
            self.transcriptions = pd.read_pickle('./dialog_utils/data/aligned_transcripts.pkl')


        if apply_skeleton_augmentations:
            assert self.skeleton_augmentations is not None, "Augmentations are not provided"
        self.n_views=n_views
        

        self.debug_audio = debug_audio
        self.debug = debug
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.random_mirror = random_mirror
        self.random_mirror_p = random_mirror_p
        self.audio_sample_rate = sample_rate1
        self.fps = fps
        self.speech_buffer = speech_buffer
        self.gesture_buffer = gesture_buffer

        self.filter_text = filter_text
        self.crop_scale = crop_scale

        self.data = None
        self.mirrored_poses = None
        self.pairs_speakers = None
        self.poses = None
        self.is_vector = is_vector
        self.load_data()

        if 'speech' not in self.modalities:
            self.audio_dict = None
       
        if self.global_normalization:
            self.get_mean_map()
        self.audio_sample_per_frame = self.audio_sample_rate/self.fps
        
    def prepared_unlabeled_segmentation_sequences(self):
        max_sequnce_length = self.window_size
        frame_offset = self.window_size
        # iterate over the poses
        gesture_sequences_data = []
        # convert start_frames and end_frames to int
        
        for pair_speaker in tqdm(self.pairs_speakers, desc='Preparing segmentation sequences...', total=len(self.pairs_speakers)):
            # select the rows of the pair_speaker
           
            total_frames = self.poses[pair_speaker].shape[0]
            frame_gesture_list = [0] * total_frames
          
            start_frame = 0
            end_frame = self.poses[pair_speaker].shape[0] - max_sequnce_length
            
            for f in range(start_frame, end_frame, frame_offset):
                labels = frame_gesture_list[f:f + max_sequnce_length]
                if sum(labels) == 0:
                    is_gesture = 'non-gesture'
                else:
                    is_gesture = 'gesture'
                gesture_instance = {
                    'pair_speaker': pair_speaker,
                    'start_frames': f,
                    'end_frames': f + max_sequnce_length,
                    'labels': frame_gesture_list[f:f + max_sequnce_length],
                    'speaker': pair_speaker.split('_')[1],
                    'pair': pair_speaker.split('_')[0],
                    'is_gesture': is_gesture,
                    'contain_text': False,
                    'words': 'geen speech',
                    'segmentation method': 'manual',
                    'from_ts': f / self.fps,
                    'to_ts': (f + max_sequnce_length) / self.fps,
                    'probability': 1.0,
                    'referent': 'None',
                    'window_size': max_sequnce_length,
                }  
                    
                gesture_sequences_data.append(gesture_instance)
        self.data = pd.DataFrame(gesture_sequences_data) 
        # save the gesture sequences to a csv file
        self.data.to_csv('./data/dilay_all_gesture_sequences.csv', index=False)
        
    def prepared_segmentation_sequences(self):
        max_sequnce_length = self.window_size
        frame_offset = self.window_size
        gesture_buffer = int(self.gesture_buffer * self.fps)
        # iterate over the poses
        gesture_sequences_data = []
        # convert start_frames and end_frames to int
        self.data['start_frames'] = self.data['start_frames'].astype(int)
        self.data['end_frames'] = self.data['end_frames'].astype(int)
        for pair_speaker in tqdm(self.pairs_speakers, desc='Preparing segmentation sequences...', total=len(self.pairs_speakers)):
            # select the rows of the pair_speaker
            data = self.data[self.data['pair_speaker'] == pair_speaker]
            # reset the index of the data
            data.reset_index(drop=True, inplace=True)
            total_frames = self.poses[pair_speaker].shape[0]
            frame_gesture_list = [0] * total_frames
            # Set the frames with gestures to 1
            start_frames = data['start_frames'].values
            end_frames = data['end_frames'].values
            # Iterate over the start and end frames to set the frames with gestures to 1
            for start_frame, end_frame in zip(start_frames, end_frames):
                for i in range(start_frame-gesture_buffer, end_frame + gesture_buffer): # allow for some buffer to cover the gesture unit
                    frame_gesture_list[i] = 1

            # TODO check how to incorporate the frame offset
            start_frame = 0
            end_frame = self.poses[pair_speaker].shape[0] - max_sequnce_length
            if 'semantic' in self.modalities:
                self.transcriptions['pair_speaker'] = self.transcriptions['pair'] + '_' + self.transcriptions['speaker']
            for f in range(start_frame, end_frame, frame_offset):
                labels = frame_gesture_list[f:f + max_sequnce_length]
                if sum(labels) == 0:
                    is_gesture = 'non-gesture'
                else:
                    is_gesture = 'gesture'
                gesture_instance = {
                    'pair_speaker': pair_speaker,
                    'start_frames': f,
                    'end_frames': f + max_sequnce_length,
                    'labels': frame_gesture_list[f:f + max_sequnce_length],
                    'speaker': pair_speaker.split('_')[1],
                    'pair': pair_speaker.split('_')[0],
                    'is_gesture': is_gesture,
                    'contain_text': False,
                    'words': 'geen speech',
                    'segmentation method': 'manual',
                    'from_ts': f / self.fps,
                    'to_ts': (f + max_sequnce_length) / self.fps,
                    'probability': 1.0,
                    'referent': 'None',
                    'window_size': max_sequnce_length,
                }  
                    
                gesture_sequences_data.append(gesture_instance)
        self.data = pd.DataFrame(gesture_sequences_data) 
        # save the gesture sequences to a csv file
        self.data.to_csv('./data/all_gesture_sequences.csv', index=False)
             

    def load_skeletal_data(self):
        if self.phase == 'test':
            poses_path = '/Users/esagha/Projects/medal_workshop_on_multimodal_interaction/Gesture_Segmentation_Toturial/Code/data/test_keypoints/'
            self.poses = {}
            
            self.pairs_speakers = []
            for file in tqdm(glob.glob(os.path.join(poses_path, '*.npy')), desc='Loading keypoints...', total=len(glob.glob(os.path.join(poses_path, '*.npy')))):
                participant_ID = file.split('/')[-1].replace('_all_kpts_17', '').split('.')[0]
                type = file.split('/')[-2]
                data = np.load(file)
                if data.shape[0] == 0:
                    print('Empty file: {}'.format(file))
                    continue
                data = np.load(file)[:, :, :-1]  # remove the last dimension
                participant_ID = participant_ID.replace('S', '')
                day = participant_ID[-1]
                participant_ID = participant_ID[:-1]
                self.poses[participant_ID+'_' + day] = data
                self.pairs_speakers.append(participant_ID+'_' + day)

            self.prepared_unlabeled_segmentation_sequences()
        else:
            # data: N C V T M
            if self.data_path.endswith('.csv'):
                self.data = pd.read_csv(self.data_path)
                self.data['start_frames'] = self.data['from_ts'] * self.fps
                self.data['end_frames'] = self.data['to_ts'] * self.fps
            elif self.data_path.endswith('.pkl'):
                self.data = pd.read_pickle(self.data_path)
            self.data = self.data.reset_index(drop=True)
            if self.eval_on_small_dataset and not self.use_only_small_dataset and self.task != 'segmentation':
                if self.train_data: 
                    if self.data_path.endswith('.csv'):
                        self.data = self.data[self.data['segmentation method'] != 'manual']
                        # self.data = self.data[self.data['segmentation method'] == 'skeleton&speech model']
                    else:
                        self.data = self.data[self.data['segmentation method'] != 'manual']
                else: 
                    self.data = self.data[self.data['segmentation method'] == 'manual']
            elif self.task == 'segmentation' or self.use_only_small_dataset:
                self.data = self.data[self.data['segmentation method'] == 'manual']
            if self.filter_text:
                self.data = self.data[self.data["words"] != "0"]
                # Select rows where the number of words in 'words' and 'pos' columns are equal
                self.data['word_count'] = self.data['words'].apply(lambda x: len(str(x).split()))
                self.data['pos_count'] = self.data['pos'].apply(lambda x: len(str(x).split()))
                self.data = self.data[self.data['word_count'] == self.data['pos_count']]
                # Filter rows where at least one POS tag is in content_words_pos
                self.data = self.data[self.data['has_content']]

                # Keep only content words in the 'words' column
                def filter_content_words(words, pos):
                    return ' '.join([word for word, p in zip(words.split(), pos.split()) if p in content_words_pos])

                self.data['words'] = self.data.apply(lambda row: filter_content_words(row['words'], row['pos']), axis=1)

                # Remove rows where all words have been filtered out
                self.data = self.data[self.data['words'] != '']

                # Drop the temporary column used for filtering
                self.data = self.data.drop(columns=['has_content', 'word_count', 'pos_count'])
                
                # reindex the data
                self.data = self.data.reset_index(drop=True)
                print(self.data.shape)
            # convert from_ts and to_ts to frames
            self.data = self.data.reset_index(drop=True)
            if self.debug:
                # select rows of pair04
                pairs = self.data['pair_speaker'].unique()
                # select 4 pairs
                pairs = pairs[:4]
                self.data = self.data[self.data['pair_speaker'].isin(pairs)]
            self.pairs_speakers = self.data['pair_speaker'].unique()
            print('Number of pairs:', len(self.pairs_speakers))
            self.poses = {}
            self.poses = load_keypoints_dict(self.pairs_speakers)
            # select only poses of the pairs in the data
            self.poses = {pair_speaker: self.poses[pair_speaker] for pair_speaker in self.pairs_speakers}
            if self.task == 'segmentation':
                # # check if the there is data saved
                # if './data/all_gesture_sequences.csv' in glob.glob('./data/all_gesture_sequences.csv'):
                #     self.data = pd.read_csv('./data/all_gesture_sequences.csv')
                # else:
                self.prepared_segmentation_sequences()
      

    def load_data(self):
        self.load_skeletal_data()

    # TODO: define augmentations from HAR project. Should the strength of augmentations be the same?
    def augment_skeleton(self, data_numpy):
        possible_augmentations = [
            Jitter3D(sigma=0.02),
            RandomRotate3D(max_angle=20),
            RandomScale3D(0.9, 1.1),
            RandomTranslate3D(0.05),
            RandomShear3D(0.05),
            RandomFlip3D(),
        ]
        choosen_augmentations = random.sample(possible_augmentations, k=random.randint(1, len(possible_augmentations)))
        augmentations = Compose(choosen_augmentations)
        
        return augmentations(data_numpy)

    

    def get_mean_map(self):
        data = self.get_all_poses()
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def get_all_poses(self):
        # reset index of the data
        all_poses = np.zeros((len(self.data), 3, 30, 27, 1))
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0], desc='Loading poses...'):
            start_frame = int(row['start_frames'])  # - round(self.gesture_buffer * self.fps)
            end_frame = int(row['end_frames'])  # + round(self.gesture_buffer * self.fps)
            middle_frame = (start_frame + end_frame) // 2
            # take 15 frames before and after the middle frame
            start_frame = middle_frame - 15
            end_frame = start_frame + 30
            pair_speaker = row['pair_speaker']
            all_poses[index] = self.poses[pair_speaker][:, start_frame:end_frame, :, :]
        return all_poses

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        img_width = 1920
        img_height = 1080
        if 'labels' in self.data.columns and self.task == 'segmentation':
            item = {
                "label": np.array(self.data.iloc[index]['labels'])
            }
        else:
            item = {
                "label": 0
            }
        row = self.data.iloc[index]
        
       
        start_frame = int(row['start_frames'])
        end_frame = int(row['end_frames'])
        if self.random_choose and self.phase == 'train':
            # random choose a portion
            if end_frame - start_frame > self.window_size:
                start_frame = np.random.randint(start_frame, end_frame - self.window_size + 1)
            else:
                middle_frame = (start_frame + end_frame) // 2
                # take 15 frames before and after the middle frame 
                start_frame = middle_frame - round(self.window_size / 2)
            end_frame = start_frame + self.window_size
        
        pair_speaker = row['pair_speaker']
      
        # add speaker and pair to the item
        if self.phase == 'test':
            speaker_dict = {'B': 0, 'C': 1}
        else:
            speaker_dict = {'A': 0, 'B': 1}
        item['speaker_ID'] = speaker_dict[pair_speaker.split('_')[1]]
        item['pair_ID'] = int(pair_speaker.split('_')[0].split('pair')[-1])
        # add frame_ID to the item
        frame_IDs = np.arange(start_frame, end_frame)
        item['frame_IDs'] = frame_IDs
        item['start_frames'] = start_frame
        item['end_frames'] = end_frame
        if "skeleton" in self.modalities:
           
            skeleton_data = self.poses[pair_speaker][start_frame:end_frame, :, :]          
            item["skeleton"] = {}
            # skeleton_data = self.data[index]
            skeleton_data_numpy = np.array(skeleton_data)
            # divide x and y by the image width and height
            # F, J, C = x.shape
            item["skeleton"]["orig"] = torch.tensor(skeleton_data_numpy, dtype=torch.float32)
            # skeleton data augmentation and view2 generation
            if self.apply_skeleton_augmentations:
                skeleton_data_numpy_1 = self.augment_skeleton(
                    skeleton_data_numpy)
                # convert to torch tensor
                skeleton_data_numpy_1 = torch.tensor(skeleton_data_numpy_1, dtype=torch.float32)
                # 
                item["skeleton"]["view1"] = skeleton_data_numpy_1
                
        for key in item['skeleton']:
            if item['skeleton'][key].shape[2] == 3:
                # divide x and y by the image width and height
                item['skeleton'][key][:, :, 0] = item['skeleton'][key][:, :, 0] #/ img_width
                item['skeleton'][key][:, :, 1] = item['skeleton'][key][:, :, 1] #/ img_height
        return item


if __name__ == '__main__':
    # test this class
    feeder = CABBFeeder(
        apply_skeleton_augmentations=False,
        task='segmentation',
        random_choose=False,
        window_size=120,
        modalities=["skeleton"],
        phase='test'
    )
    # get the first item
    item = feeder[0]
    print(item.keys())