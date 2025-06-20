import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.extend(['../'])

from model.wav2vec2_wrapper import Wav2Vec2CNN
from model.decouple_gcn_attn_sequential import Model as STGCN
from model.DSTformer import DSTformer
from model.CrossDSTformer import CrossDSTformer
from model.model_object import ObjectNet
from functools import partial
from model.semantic_pool import BertjePoolingModule

weights_path = '27_2_finetuned/joint_finetuned.pt'

class GSSModel(nn.Module):
    """
    GSSModel is a multimodal model designed to process data from multiple sources:
        Gestures (skeletons), Speech, and Semantics (text).

    Args:
        feat_dim (int): Dimension of the feature representation. Default is 128.
        w2v2_type (str): Type of Wav2Vec2 model to use. Default is 'multilingual'.
        modalities (list): Modalities to be used. Options include ['skeleton', 'speech', 'semantic'].
        fusion (str): Fusion strategy to be used. Default is 'late'.
        pre_trained_gcn (bool): Use a pre-trained GCN model if True. Default is True.
        skeleton_backbone (str): Backbone model for skeleton data. Default is 'stgcn'.
        hidden_dim (int): Size of the hidden dimension. Default is 256.
        bertje_dim (int): Dimension size for the Bertje model. Default is 768.
        freeze_bertje (bool): Freeze the Bertje model during training if True. Default is True.
        attentive_pooling (bool): Use attentive pooling if True. Default is True.
        attentive_pooling_skeleton (bool): Use attentive pooling for skeleton if True. Default is True.
        use_robbert (bool): Use the RobBERT model if True. Default is False.
        cross_modal (bool): Use cross-modal attention if True. Default is False.
        dont_pool (bool): Do not pool the hidden states if True. Default is False.

    Attributes:
        pre_trained_gcn (bool): Indicates if a pre-trained GCN model is used.
        modality (list): Modalities being used.
        fusion (str): Fusion strategy being used.
        skeleton_backbone (str): Backbone model for skeleton data.
        speech_model (Wav2Vec2CNN): Model for processing speech data.
        skeleton_model (nn.Module): Model for processing skeleton data.
        semantic (BertjePoolingModule): Module for processing semantic data.
        skeleton_head (nn.Sequential): Projection head for skeleton data.
        speech_head (nn.Sequential): Projection head for speech data.
    """
    def __init__(
            self,
            feat_dim=128,
            w2v2_type='multilingual',
            modalities=['skeleton', 'speech', 'semantic'],
            fusion='late',
            pre_trained_gcn=True,
            skeleton_backbone='jointformer',
            hidden_dim=128,
            bertje_dim=768,
            freeze_bertje=True,
            attentive_pooling=True,
            attentive_pooling_skeleton=False,
            use_robbert=False,
            weights_path=None,
            wasserstein_distance=False,
            cross_modal=False,
            dont_pool=False,
            maxlen=60,
            multimodal_embeddings_dim=768,
            **kwargs
    ):
        super(GSSModel, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pre_trained_gcn = pre_trained_gcn
        self.modality = modalities
        self.fusion = fusion
        self.skeleton_backbone = skeleton_backbone
        self.wasserstein_distance = 'wasserstein' in kwargs.get('loss_types', [])
        self.cross_modal = cross_modal
        self.dont_pool = dont_pool
        self.attentive_pooling = attentive_pooling
        # check if 'one_branch_cross_modal' is in kwargs
        self.one_branch_cross_modal = kwargs.get('one_branch_cross_modal', False)
        self.loss_types = kwargs.get('loss_types', [])
        if 'matching' in self.loss_types:
            self.classifier = nn.Sequential(
                nn.Linear(kwargs.get('feat_dim', 128), 1),
            )
        # Encoders
        if 'semantic' in modalities:
            if cross_modal:
                multimodal_embeddings_dim = 768
            self.semantic = BertjePoolingModule(
                freeze_bertje=freeze_bertje,
                use_attentive_pooling=attentive_pooling,
                use_robbert=use_robbert,
                dont_pool=dont_pool
            )
        if 'speech' in modalities:
            if cross_modal:
                apply_cnns = False
                multimodal_embeddings_dim = 1024
            else:
                apply_cnns = True
            self.speech_model = Wav2Vec2CNN(w2v2_type=w2v2_type, apply_cnns=apply_cnns)
        if 'skeleton' in modalities:
            if skeleton_backbone == 'stgcn' and not cross_modal:
                self.skeleton_model = STGCN(device=device)
                if pre_trained_gcn and weights_path is not None:
                    self.skeleton_model.load_state_dict(torch.load(weights_path))
            else:
                if cross_modal:
                    assert attentive_pooling == False, 'Cross-modal requires attentive pooling to be False'
                    jointsformer = CrossDSTformer(
                        dim_in=3,
                        dim_out=3,
                        dim_feat=hidden_dim,
                        dim_rep=hidden_dim,
                        depth=4,
                        num_heads=8,
                        mlp_ratio=2,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        maxlen=60,
                        num_joints=27,
                        multimodal_embeddings_dim=multimodal_embeddings_dim
                    )
                else:
                    jointsformer = DSTformer(
                        dim_in=3,
                        dim_out=3,
                        dim_feat=hidden_dim,
                        dim_rep=hidden_dim,
                        depth=4,
                        num_heads=8,
                        mlp_ratio=2,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        maxlen=60,
                        num_joints=27
                    )
                self.skeleton_model = ObjectNet(
                    backbone=jointsformer,
                    dim_rep=hidden_dim,
                    version='embed',
                    hidden_dim=hidden_dim,
                    attentive_pooling=attentive_pooling_skeleton,
                    local_features=self.wasserstein_distance
                )
            middle_dim = int(hidden_dim // 2)
       

        # Projection heads
        if 'skeleton' in modalities:
            self.skeleton_head = nn.Sequential(
                    nn.Linear(hidden_dim, middle_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(middle_dim, feat_dim)
                )
        if 'speech' in modalities:
            self.speech_head = nn.Sequential(
                    nn.Linear(128, middle_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(middle_dim, feat_dim)
                )
        if 'semantic' in modalities:
            self.semantic_projection = torch.nn.Sequential(
                torch.nn.Linear(bertje_dim, middle_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(middle_dim, feat_dim)
            )
            if self.wasserstein_distance:
                self.semantic_local_projection = torch.nn.Sequential(
                    torch.nn.Linear(bertje_dim, middle_dim),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(middle_dim, feat_dim)
                )

    def forward(
            self,
            skeleton=None,
            speech_waveform=None,
            utterances=None,
            speech_lengths=None,
            eval=False,
            features=['skeleton', 'speech', 'semantic'],
            before_head_feats=False,
            reconstruction=True,
            representations=True,
            vicreg=False,
    ):
        # Skeleton forward pass
        skeleton_features = None
        semantic_features = None
        semantic_outputs = {}
        speech_outputs = {}
        crossmodal_inputs = {}

        if self.cross_modal:
            assert 'semantic' in self.modality or 'speech' in self.modality
            assert not ('semantic' in self.modality and 'speech' in self.modality)
        # Speech forward pass
        if 'speech' in self.modality and speech_waveform is not None:
            speech_outputs = self.speech_model(speech_waveform, lengths=speech_lengths)

        # Semantic forward pass
        if 'semantic' in self.modality and utterances is not None:
            semantic_outputs = self.semantic(utterances)
            semantic_features = semantic_outputs["global"]
            if self.wasserstein_distance:
                semantic_local_features = semantic_outputs["local"]
                semantic_local_features = self.semantic_local_projection(semantic_local_features)
        if 'speech' in self.modality and self.cross_modal:
            crossmodal_inputs = {
                "local": speech_outputs["local"],
                "attention_mask": speech_outputs["attention_mask"]
            }
        if 'semantic' in self.modality and self.cross_modal:
            crossmodal_inputs = {
                "local": semantic_outputs["local"],
                "attention_mask": semantic_outputs["attention_mask"]
            }
        skeleton_outputs = self.skeleton_model(
            skeleton,
            crossmodal_inputs=crossmodal_inputs,
            get_rep=representations,
            get_pred=reconstruction,
            one_branch_cross_modal=self.one_branch_cross_modal
        )
        skeleton_features = skeleton_outputs["global"]
        if self.wasserstein_distance:
            skeleton_local_features = skeleton_outputs["local"]

        if semantic_features is not None:
            semantic_features = self.semantic_projection(semantic_features)
            if not vicreg:
                semantic_features = F.normalize(semantic_features, dim=1)
        if skeleton_features is not None:
            skeleton_features = self.skeleton_head(skeleton_features)
            if not vicreg:
                skeleton_features = F.normalize(skeleton_features, dim=1)
        if 'speech' in self.modality and speech_waveform is not None:
            speech_outputs['global'] = self.speech_head(speech_outputs['global'])
            if not vicreg:
                speech_outputs['global'] = F.normalize(speech_outputs['global'], dim=1)
        if 'matching' in self.loss_types:   
            matching_predictions = self.classifier(skeleton_features)
        return {
            "skeleton_features": skeleton_features,
            "skeleton_representations": skeleton_outputs["representations"] if representations else None,
            "speech_features": speech_outputs["global"] if 'speech' in self.modality and speech_waveform is not None else None,
            "semantic_features": semantic_features,
            "semantic_local_features": semantic_local_features if self.wasserstein_distance else None,
            "skeleton_local_features": skeleton_local_features if self.wasserstein_distance else None,
            "skeleton_predictions": skeleton_outputs["skeleton_predictions"] if reconstruction else None,
            "one_branch_cross_modal": self.one_branch_cross_modal,
            "matching_predictions": matching_predictions if 'matching' in self.loss_types else None
        }
        
    
    
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