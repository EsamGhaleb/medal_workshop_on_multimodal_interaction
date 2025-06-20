import torch.nn as nn
import torch.nn.functional as F

from model.attentive_pooler import AttentivePooler


class ObjectHeadClassification(nn.Module):
    def __init__(self, dropout_ratio=0., dim_rep=256, num_classes=16, num_joints=27, hidden_dim=1024):
        super(ObjectHeadClassification, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim_rep*num_joints, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, feat):
        '''
            Input: (N, T, J, C)
        '''
        N, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 2, 3, 1)      # (N, T, J, C) -> (N, J, C, T)
        feat = feat.mean(dim=-1)
        feat = feat.reshape(N, -1)           # (N, J*C)
        feat = self.fc1(feat)
        feat = self.bn(feat)
        feat = self.relu(feat)
        feat = self.fc2(feat)
        return feat


class ObjectHeadEmbed(nn.Module):
    def __init__(
            self,
            dropout_ratio=0.,
            dim_rep=256,
            num_joints=27,
            hidden_dim=1024,
            attentive_pooling=False,
            local_features=False,
    ):
        super(ObjectHeadEmbed, self).__init__()
        self.attentive_pooling = attentive_pooling
        self.local_features = local_features
        if attentive_pooling:
            self.attentive_pooler = AttentivePooler(
                num_queries=1,
                embed_dim=dim_rep,
                num_heads=8,
                mlp_ratio=4.0,
                depth=1,
            )
        if local_features:
            self.local_network = nn.Sequential(
                nn.Conv1d(dim_rep, dim_rep, kernel_size=2),
                nn.ReLU(inplace=True),
            )
        else:
            self.dropout = nn.Dropout(p=dropout_ratio)
            self.fc1 = nn.Linear(dim_rep*num_joints, hidden_dim)

    def forward(self, feat, gestures_similarity=False):
        outputs = {}
        if self.attentive_pooling:
            outputs["global"] = self._attentive_pooling(feat)
        else:
            outputs["global"] = self._mean_pooling(feat, gestures_similarity)
        if self.local_features:
            outputs["local"] = self._local_features(feat)
        return outputs

    def _mean_pooling(self, feat, gestures_similarity=False):
        '''
            Input: (N, T, J, C)
        '''
        N, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 2, 3, 1)      # (N, T, J, C) -> (N, J, C, T)
        feat = feat.mean(dim=-1)
        if gestures_similarity:
            return feat.mean(dim=1)
        feat = feat.reshape(N, -1)           # (N, J*C)
        feat = self.fc1(feat)
        # feat = feat.mean(dim=1)
        feat = F.normalize(feat, dim=-1)
        return feat

    def _attentive_pooling(self, feat):
        feat = feat.mean(dim=2)  # N, T, J, C -> N, T, C
        feat = self.attentive_pooler(feat).squeeze()
        feat = F.normalize(feat, dim=-1)
        return feat

    def _local_features(self, feat):
        '''
            Input: (N, T, J, C)
        '''
        # print("local features shape before")
        # print(feat.shape)
        feat = feat.mean(dim=2).squeeze()
        feat = feat.permute(0, 2, 1)  # N, T, C -> N, C, T
        feat = self.local_network(feat).permute(0, 2, 1)  # N, C, T -> N, T, C
        # print("local features shape after")
        # print(feat.shape)
        return feat


class ObjectNet(nn.Module):
    def __init__(
            self,
            backbone,
            dim_rep=512,
            num_classes=16,
            dropout_ratio=0.,
            version='class',
            hidden_dim=768,
            num_joints=27,
            attentive_pooling=False,
            local_features=False,
    ):
        super(ObjectNet, self).__init__()
        self.backbone = backbone
        self.feat_J = num_joints
        if version == 'class':
            self.head = ObjectHeadClassification(
                dropout_ratio=dropout_ratio,
                dim_rep=dim_rep,
                num_classes=num_classes,
                num_joints=num_joints
            )
        elif version == 'embed':
            self.head = ObjectHeadEmbed(
                dropout_ratio=dropout_ratio,
                dim_rep=dim_rep,
                hidden_dim=hidden_dim,
                num_joints=num_joints,
                attentive_pooling=attentive_pooling,
                local_features=local_features,
            )
        else:
            raise ValueError('Unknown ObjectNet version.')

    def forward(self, x, get_rep=True, get_pred=True, gestures_similarity=False,  **kwargs):
        '''
            Input: (N x T x 27 x 3)
        '''
        N, T, J, C = x.shape
        assert get_rep or get_pred, "At least one of get_rep and get_pred should be True"
        # special case when get_rep=False and get_pred=True
        if not get_rep and get_pred: #TODO: think about a better way to do this
            gestures_similarity = True
            get_rep = True # we need the representation to compute the similarity
        outputs =  self.backbone(x, return_rep=get_rep, return_pred=get_pred, **kwargs)
        # pass the representation to the head
        outputs['representations'] = outputs['representations'].reshape([N, T, self.feat_J, -1])      # (N, T, J, C)
        outs = self.head(outputs['representations'], gestures_similarity=gestures_similarity)
        outputs.update(outs)
        return outputs
