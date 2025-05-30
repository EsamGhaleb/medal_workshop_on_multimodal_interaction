import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.extend(['../'])

from model.decouple_gcn_attn_sequential import Model as STGCN
from model.DSTformer import DSTformer
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
        x = torch.mean(x, dim=2)  # (N, T, C)
        return self.head(x)

# ---------------------
# ObjectNet: handles the backbone, crossmodal, and fusion logic
# ---------------------
class ObjectNet(nn.Module):
    def __init__(
            self,
            backbone=None,
            dim_rep=512,
            hidden_dim=768,
            num_joints=46
    ):
        """
        fusion: one of {'early', 'late', 'crossmodal', 'unimodal'}.
        """
        super(ObjectNet, self).__init__()
        self.backbone = backbone
        # self.crossmodal_encoder = crossmodal_encoder
        self.feat_J = num_joints
        
        # The main head for skeleton representation.
        self.head = SegmentHead(in_size=dim_rep, hidden_dim=hidden_dim)
        
    def forward(self, x, **kwargs):
        """
        x: tensor of shape (N, T, 46, C) corresponding to skeleton data.
        kwargs should include:
          - modalities: list of strings (e.g., ['skeleton', 'speech', 'semantic'])
          - crossmodal_inputs: dict with key 'local' (required if a crossmodal branch is used)
        """
        skeleton_output = None

      
        # Process the (possibly fused) skeleton data through the backbone.
        outputs = self.backbone(x, return_rep=True, return_pred=True, **kwargs)
        skeleton_output = self.head(outputs['representations'], **kwargs)
    
        return skeleton_output
    

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
            modalities=['skeleton', 'speech', 'semantic'],
            fusion='late',
            skeleton_backbone='jointformer',
            hidden_dim=128,
            cross_modal=False,
            dont_pool=False,
            maxlen=120,
            depth=2,
            num_heads=4,
            mlp_ratio=1,
            **kwargs
    ):
        super(GSSModel, self).__init__()
        self.modality = modalities
        self.fusion = fusion
        self.skeleton_backbone = skeleton_backbone
        self.cross_modal = cross_modal
        self.dont_pool = dont_pool
        self.one_branch_cross_modal = kwargs.get('one_branch_cross_modal', False)
        self.loss_types = kwargs.get('loss_types', [])
        self.maxlen = maxlen

        jointsformer = DSTformer(
            dim_in=3,
            dim_out=3,
            dim_feat=hidden_dim,
            dim_rep=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            maxlen=maxlen,
            num_joints=46
        )
  
        self.segmentation_model = ObjectNet(
            backbone=jointsformer,
            dim_rep=hidden_dim,
            hidden_dim=hidden_dim,
            num_joints=46,
        )
      

    def forward(
            self,
            skeleton=None,
            speech_waveform=None,
            utterances=None,
            speech_lengths=None,
    ):
        # Containers for outputs and crossmodal inputs.

        # Forward pass through the segmentation model.
        segmentation_outputs = self.segmentation_model(
            skeleton,
        )
        return segmentation_outputs

    