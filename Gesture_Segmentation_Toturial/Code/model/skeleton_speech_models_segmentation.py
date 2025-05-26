import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.extend(['../'])

from model.wav2vec2_wrapper import Wav2Vec2CNN
from model.decouple_gcn_attn_sequential import Model as STGCN
from model.DSTformer import DSTformer
from model.CrossDSTformer_Segmentation import CrossDSTformer
from functools import partial
from model.semantic_pool import BertjePoolingModule
from model.my_transformer import TransformerEncoder
    
weights_path = '27_2_finetuned/joint_finetuned.pt'

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

# ---------------------
# Helper segmentation head
# ---------------------
class SegmentHead(nn.Module):
    def __init__(self, in_size, hidden_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )
    
    def forward(self, x, **kwargs):
        """
        Input: (N, T, J, C)
        Takes the mean over joints then applies a projection head.
        """
        # apply the pooling only if skeleton modality is present
        if 'skeleton' in kwargs.get('modalities', []):
            x = torch.mean(x, dim=2)  # (N, T, C)
        return self.head(x)

# ---------------------
# ObjectNet: handles the backbone, crossmodal, and fusion logic
# ---------------------
class ObjectNet(nn.Module):
    def __init__(
            self,
            backbone=None,
            crossmodal_encoder=None,
            dim_rep=512,
            hidden_dim=768,
            num_joints=27,
            fusion='late',
            **kwargs
    ):
        """
        fusion: one of {'early', 'late', 'crossmodal', 'unimodal'}.
        """
        super(ObjectNet, self).__init__()
        self.backbone = backbone
        # self.crossmodal_encoder = crossmodal_encoder
        self.feat_J = num_joints
        self.fusion = fusion  # This value determines the fusion strategy
        
        # The main head for skeleton representation.
        if 'skeleton' in kwargs.get('modalities', []):
            self.head = SegmentHead(in_size=dim_rep, hidden_dim=hidden_dim)
        
        # For late fusion, a separate head for the crossmodal branch.
        if self.fusion == 'late' or (self.fusion == 'unimodal' and any(m in ['speech', 'semantic'] for m in kwargs.get('modalities', []))):
            multimodal_embeddings_dim = kwargs.get('multimodal_embeddings_dim', 768)
            self.crossmodal_head = SegmentHead(in_size=multimodal_embeddings_dim, hidden_dim=hidden_dim)

    def forward(self, x, **kwargs):
        """
        x: tensor of shape (N, T, 27, C) corresponding to skeleton data.
        kwargs should include:
          - modalities: list of strings (e.g., ['skeleton', 'speech', 'semantic'])
          - crossmodal_inputs: dict with key 'local' (required if a crossmodal branch is used)
        """
        modalities = kwargs.get('modalities', [])
        fusion_type = self.fusion

        skeleton_output = None
        crossmodal_output = None

        # Process skeleton modality if available.
        if 'skeleton' in modalities:
            # For early fusion, concatenate the crossmodal inputs into the skeleton data.
            if fusion_type == 'early' and kwargs.get('crossmodal_inputs') is not None:
                x = self._early_fuse(x, kwargs['crossmodal_inputs'])
            # Process the (possibly fused) skeleton data through the backbone.
            outputs = self.backbone(x, return_rep=True, return_pred=True, **kwargs)
            skeleton_output = self.head(outputs['representations'], **kwargs)
        
        # Process crossmodal branch if the fusion is late or if unimodal with crossmodal input.
        if fusion_type in ['late', 'unimodal'] and any(m in modalities for m in ['speech', 'semantic']):
            crossmodal_output = self._process_crossmodal(kwargs)
            maxlen = kwargs.get('maxlen', 120)
            N, T, C = crossmodal_output.shape
            crossmodal_output= crossmodal_output.permute(0, 2, 1).contiguous()
            crossmodal_output = F.interpolate(
                    crossmodal_output, size=maxlen, mode='nearest')
            # reshape back to (N, T, C)
            crossmodal_output = crossmodal_output.permute(0, 2, 1).contiguous()
        
        # Return based on fusion strategy.
        if fusion_type in ['early', 'crossmodal']:
            return skeleton_output
        elif fusion_type == 'late':
            # Combine the two branches (assuming same shape)
            return skeleton_output + crossmodal_output
        elif fusion_type == 'unimodal':
            # If skeleton branch exists, use it; otherwise use the crossmodal branch.
            return skeleton_output if skeleton_output is not None else crossmodal_output
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def _early_fuse(self, skeleton, crossmodal_inputs):
        """
        Early fusion: concatenate crossmodal features to the skeleton input features.
        Assumes crossmodal_inputs['local'] is compatible in shape.
        """
        return torch.cat([skeleton, crossmodal_inputs['local']], dim=-1)

    def _process_crossmodal(self, kwargs):
        """
        Late fusion (or unimodal crossmodal branch): process crossmodal inputs using the crossmodal encoder and head.
        It reshapes the crossmodal input to (N, T, -1) and then applies the submodules.
        """
        crossmodal_inputs = kwargs['crossmodal_inputs']['local']
        # # Transform encoder expects input of shape (T, N, C).
        # crossmodal_inputs = crossmodal_inputs.permute(1, 0, 2).contiguous()  # (T, N, C)
        # # Reshape to (N, T, C) for the crossmodal encoder.
        # crossmodal_features = self.crossmodal_encoder(crossmodal_inputs).permute(1, 0, 2).contiguous()  # (N, T, C)
        return self.crossmodal_head(crossmodal_inputs, **kwargs)

# ---------------------
# GSSModel: handles multiple modalities and selects the appropriate segmentation network
# ---------------------
class GSSModel(nn.Module):
    """
    GSSModel is a multimodal model designed to process data from multiple sources:
    Gestures (skeletons), Speech, and Semantics (text). The fusion strategy is controlled by the
    parameter 'fusion' (which can be 'early', 'late', 'crossmodal', or 'unimodal').
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
            cross_modal=False,
            dont_pool=False,
            maxlen=120,
            multimodal_embeddings_dim=768,
            apply_cnns=False,
            **kwargs
    ):
        super(GSSModel, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.modality = modalities
        self.fusion = fusion
        self.skeleton_backbone = skeleton_backbone
        self.cross_modal = cross_modal
        self.dont_pool = dont_pool
        self.attentive_pooling = attentive_pooling
        self.one_branch_cross_modal = kwargs.get('one_branch_cross_modal', False)
        self.loss_types = kwargs.get('loss_types', [])
        self.maxlen = maxlen

        # Instantiate semantic and speech modules if needed.
        if 'semantic' in modalities:
            # Adjust multimodal_embeddings_dim if cross_modal is True.
            self.semantic = BertjePoolingModule(
                freeze_bertje=freeze_bertje,
                use_attentive_pooling=attentive_pooling,
                use_robbert=use_robbert,
                dont_pool=dont_pool
            )
        if 'speech' in modalities:
            self.speech_model = Wav2Vec2CNN(
                w2v2_type=w2v2_type,
                apply_cnns=apply_cnns,
            )

        # Set up the skeleton branch backbone.
        if 'skeleton' in modalities:
            if skeleton_backbone == 'stgcn' and not cross_modal:
                self.segmentation_model = ObjectNet(
                    backbone=STGCN(device=device),
                    dim_rep=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_joints=27,
                    fusion=fusion,
                    **kwargs
                )
                if pre_trained_gcn and weights_path is not None:
                    self.segmentation_model.backbone.load_state_dict(torch.load(weights_path))
            else:
                # Use jointsformer-based backbones.
                if cross_modal:
                    jointsformer = CrossDSTformer(
                        dim_in=3,
                        dim_out=3,
                        dim_feat=hidden_dim,
                        dim_rep=hidden_dim,
                        depth=4,
                        num_heads=8,
                        mlp_ratio=1,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        maxlen=maxlen,
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
                        mlp_ratio=1,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        maxlen=maxlen,
                        num_joints=27
                    )
                
        # If no skeleton modality is available but speech or semantic is,
        # instantiate ObjectNet using only the crossmodal encoder.
        if any(mod in modalities for mod in ['speech', 'semantic']) and not 'skeleton' in modalities:
            assert fusion == 'unimodal', "Fusion type must be 'unimodal' when skeleton is not present."
            self.segmentation_model = ObjectNet(
                crossmodal_encoder=TransformerEncoder(
                    input_dim=multimodal_embeddings_dim,
                    model_dim=hidden_dim,
                    num_layers=6,
                    num_heads=8
                ),
                dim_rep=hidden_dim,
                hidden_dim=hidden_dim,
                num_joints=27,
                fusion=fusion,
                modalities=modalities,
                multimodal_embeddings_dim=multimodal_embeddings_dim,
                **kwargs
            )
        elif 'skeleton' in modalities and any(mod in modalities for mod in ['speech', 'semantic']):
            assert fusion in ['late', 'crossmodal', 'unimodal'], "Fusion type must be 'late', 'crossmodal', or 'unimodal' when skeleton is present."
            self.segmentation_model = ObjectNet(
                    backbone=jointsformer,
                    crossmodal_encoder=TransformerEncoder(
                        input_dim=multimodal_embeddings_dim,
                        model_dim=hidden_dim,
                        num_layers=6,
                        num_heads=8
                    ),
                    dim_rep=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_joints=27,
                    fusion=fusion,
                    modalities=modalities,
                    multimodal_embeddings_dim=multimodal_embeddings_dim,
                    **kwargs
                )
        elif 'skeleton' in modalities and not any(mod in modalities for mod in ['speech', 'semantic']):
            self.segmentation_model = ObjectNet(
                backbone=jointsformer,
                dim_rep=hidden_dim,
                hidden_dim=hidden_dim,
                num_joints=27,
                fusion=fusion,
                modalities=modalities,
                multimodal_embeddings_dim=multimodal_embeddings_dim,
                **kwargs
            )
        else:
            raise NotImplementedError(
                'Fusion not supported: {}'.format(fusion)
            )

    def forward(
            self,
            skeleton=None,
            speech_waveform=None,
            utterances=None,
            speech_lengths=None,
    ):
        # Containers for outputs and crossmodal inputs.
        semantic_outputs = {}
        speech_outputs = {}
        crossmodal_inputs = {}

        # Ensure that cross_modal is only set for one of speech or semantic.
        if self.cross_modal:
            assert 'semantic' in self.modality or 'speech' in self.modality
            assert not ('semantic' in self.modality and 'speech' in self.modality)

        # Process speech if available.
        if 'speech' in self.modality and speech_waveform is not None:
            speech_outputs = self.speech_model(speech_waveform, lengths=speech_lengths)
            
            crossmodal_inputs = {
                "local": speech_outputs["local"],
                "attention_mask": speech_outputs.get("attention_mask", None)
            }

        # Process semantic data if available.
        if 'semantic' in self.modality and utterances is not None:
            crossmodal_inputs = self.semantic(utterances)
            crossmodal_inputs["local"] = crossmodal_inputs["local"].transpose(1, 2)
            crossmodal_inputs["local"] = F.interpolate(
                crossmodal_inputs["local"], size=self.maxlen, mode='linear', align_corners=True
            )
            crossmodal_inputs["local"] = crossmodal_inputs["local"].transpose(1, 2)
            crossmodal_inputs['attention_mask'] = crossmodal_inputs.get("attention_mask", None)
        

        # Forward pass through the segmentation model.
        segmentation_outputs = self.segmentation_model(
            skeleton,
            crossmodal_inputs=crossmodal_inputs,
            modalities=self.modality,
            one_branch_cross_modal=self.one_branch_cross_modal,
            maxlen=self.maxlen
        )
        return segmentation_outputs

    
    
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