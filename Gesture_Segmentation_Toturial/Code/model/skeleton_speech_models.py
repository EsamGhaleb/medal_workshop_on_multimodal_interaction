import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.extend(['../'])
from model.wav2vec2_wrapper import Wav2Vec2CNN
dependencies = ['torch', 'numpy', 'resampy', 'soundfile']

from model.decouple_gcn_attn_sequential import Model as STGCN
import torch
weights_path = '27_2_finetuned/joint_finetuned.pt'

class SupConWav2vec2GCN(nn.Module):
    """backbone + projection head"""
    def __init__(self, feat_dim=128, w2v2_type='multilingual', modalities=['skeleton', 'speech'], fusion='late', pre_trained_gcn=True):
        super(SupConWav2vec2GCN, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pre_trained_gcn = pre_trained_gcn
        self.modality = modalities
        self.fusion = fusion
        if 'speech' in modalities:
            self.speech_model = Wav2Vec2CNN(w2v2_type=w2v2_type)
        if 'skeleton' in modalities:
            self.gcn_model = STGCN(device=device)
            if pre_trained_gcn:
                self.gcn_model.load_state_dict(torch.load(weights_path))
            sekeleton_feat_dim = 256
            if 'text' in modalities:
                feat_dim = 768
                middle_dim = 256
            else:
                feat_dim = 128
                middle_dim = 128
        if fusion == 'late' and 'speech' in modalities and 'skeleton' in modalities:
            self.skeleton_head = nn.Sequential(
                    nn.Linear(sekeleton_feat_dim, middle_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(middle_dim, feat_dim)
                )
            self.speech_head = nn.Sequential(
                    nn.Linear(128, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, feat_dim)
                )
        elif fusion == 'concat' and 'speech' in modalities and 'skeleton' in modalities:
            self.mm_head = nn.Sequential(
                    nn.Linear(sekeleton_feat_dim+128, middle_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(middle_dim, feat_dim)
                )
        elif fusion == 'none' and 'speech' in modalities:
            self.speech_head = nn.Sequential(
                    nn.Linear(128, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, feat_dim)
                )
        elif fusion == 'none' and 'skeleton' in modalities:
            self.skeleton_head = nn.Sequential(
                    nn.Linear(sekeleton_feat_dim, middle_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(middle_dim, feat_dim)
                )
        else:
            raise NotImplementedError(
                'fusion not supported: {}'.format(fusion))
    
    def forward(self, skeleton=None, speech_waveform=None, speech_lengths=None, eval=False, features=['skeleton', 'speech'], before_head_feats = False):
        # Skeleton forward pass
        if 'skeleton' in self.modality:
            skeleton_features = self.gcn_model(skeleton)
            if eval and 'speech' not in features:
                if before_head_feats:
                    return F.normalize(skeleton_features, dim=1)
                else:
                    return F.normalize(self.skeleton_head(skeleton_features), dim=1)
        # Speech forward pass
        if 'speech' in self.modality:
            speech_features = self.speech_model(speech_waveform, lengths=speech_lengths)
            if eval and 'skeleton' not in features:
                return F.normalize(self.speech_head(speech_features), dim=1)
        # Late fusion: Apply skeleton and speech projections separately and normalize features per modality
        if self.fusion == 'late' and 'speech' in self.modality and 'skeleton' in self.modality:
            skeleton_feat = F.normalize(self.skeleton_head(skeleton_features), dim=1)
            speech_feat = F.normalize(self.speech_head(speech_features), dim=1)
            return skeleton_feat, speech_feat
        # Concat fusion: Concatenate features, apply mm projection head and normalize the multimodal projected feature vector
        elif self.fusion == 'concat' and 'speech' in self.modality and 'skeleton' in self.modality:
            mm_feat = F.normalize(self.mm_head(torch.cat([skeleton_features, speech_features], dim=1)), dim=1)
            return mm_feat
        # Unimodal speech: normalizaed projections for speech features only
        elif self.fusion == 'none' and 'speech' in self.modality:
            return F.normalize(self.speech_head(speech_features), dim=1)
        # Unimodal skeleton: normalizaed projections for skeleton features only
        elif self.fusion == 'none' and 'skeleton' in self.modality:
            return F.normalize(self.skeleton_head(skeleton_features), dim=1)
