import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn
from torch.autograd import Variable


class NTXentMM(LightningModule):
    """
    Multimodal adaptation of NTXent, according to the original CMC paper.
    Adapted from: https://github.com/razzu/cmc-cmkm/blob/main/models/cmc.py 
    """
    def __init__(self, batch_size, temperature=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features_1, features_2):
        self.batch_size = features_1.shape[0]
        logits, labels, pos, neg = self.get_infoNCE_logits_labels(features_1, features_2, self.batch_size, self.temperature)
        return self.criterion(logits, labels), pos, neg
    
    @staticmethod
    def get_cosine_sim_matrix(features_1, features_2, normalize=False):
        if normalize:
            features_1 = F.normalize(features_1, dim=1)
            features_2 = F.normalize(features_2, dim=1)
        similarity_matrix = torch.matmul(features_1, features_2.T)
        return similarity_matrix

    def get_infoNCE_logits_labels(self, features_1, features_2, batch_size, modalities=2, temperature=0.1):
        # Let M1 and M2 be abbreviations for the first and the second modality, respectively.

        # Computes similarity matrix by multiplication, shape: (batch_size, batch_size).
        # This computes the similarity between each sample in M1 with each sample in M2.
        similarity_matrix = NTXentMM.get_cosine_sim_matrix(features_1, features_2)

        # We need to formulate (2 * batch_size) instance discrimination problems:
        # -> each instance from M1 with each instance from M2
        # -> each instance from M2 with each instance from M1

        # Similarities on the main diagonal are from positive pairs, and are the same in both directions.
        mask = torch.eye(batch_size, dtype=torch.bool)
        positives_m1_m2 = similarity_matrix[mask].view(batch_size, -1)
        positives_m2_m1 = similarity_matrix[mask].view(batch_size, -1)
        positives = torch.cat([positives_m1_m2, positives_m2_m1], dim=0)

        # The rest of the similarities are from negative pairs. Row-wise for the loss from M1 to M2, and column-wise for the loss from M2 to M1.
        negatives_m1_m2 = similarity_matrix[~mask].view(batch_size, -1)
        negatives_m2_m1 = similarity_matrix.T[~mask].view(batch_size, -1)
        negatives = torch.cat([negatives_m1_m2, negatives_m2_m1])
        
        # Reshuffle the values in each row so that positive similarities are in the first column.
        logits = torch.cat([positives, negatives], dim=1)

        # Labels are a zero vector because all positive logits are in the 0th column.
        labels = torch.zeros(2 * batch_size)

        logits = logits / temperature

        return logits, labels.long().to(logits.device), positives.mean(), negatives.mean()


class NTXent(LightningModule):
    def __init__(self, batch_size, n_views=2, temperature=0.1):
        super().__init__()
        self.n_views = n_views
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        batch_size = int(x.shape[0] / 2)
        logits, labels, pos, neg = self.get_infoNCE_logits_labels(x, batch_size, self.n_views, self.temperature)
        return self.criterion(logits, labels), pos, neg
    
    def get_infoNCE_logits_labels(self, features, batch_size, n_views=2, temperature=0.1):
        """
            Implementation from https://github.com/sthalles/SimCLR/blob/master/simclr.py
        """
        positives, negatives = get_pos_neg_similarities(features, batch_size, n_views)

        # reshuffles values in each row so that positive similarity value for each row is in the first column
        logits = torch.cat([positives, negatives], dim=1)
        # labels is a zero vector because all positive logits are in the 0th column
        labels = torch.zeros(logits.shape[0])

        logits = logits / temperature

        return logits, labels.long().to(logits.device), positives.mean(), negatives.mean()

def get_pos_neg_similarities(features, batch_size, n_views):
    # creates a vector with labels [0, 1, 2, 0, 1, 2] 
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    # creates matrix where 1 is on the main diagonal and where indexes of the same intances match (e.g. [0, 4][1, 5] for batch_size=3 and n_views=2) 
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    # computes similarity matrix by multiplication, shape: (batch_size * n_views, batch_size * n_views)
    similarity_matrix = get_cosine_sim_matrix(features)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool)#.to(self.args.device)
    # mask out the main diagonal - output has one column less 
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix_wo_diag = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix_wo_diag[labels.bool()].view(labels.shape[0], -1)
    # select only the negatives 
    negatives = similarity_matrix_wo_diag[~labels.bool()].view(similarity_matrix_wo_diag.shape[0], -1)
    return positives, negatives

def get_cosine_sim_matrix(features):
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    return similarity_matrix


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        bsz = int(features.shape[0] / 2)
        positives, negatives = get_pos_neg_similarities(features, bsz, n_views=2)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        device = (torch.device('cuda')
                if features.is_cuda
                else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss, positives.mean(), negatives.mean()


class WeightedFocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1).squeeze()
        input = input.view(-1, input.shape[-1])

        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        target = target.view(-1, 1).long()
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()