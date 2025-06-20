import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn
from torch.autograd import Variable
import numpy as np


# Numpy-based errors
def get_bone_information(data):
    # (N, T, 27, 3) --> size in our implementation (N, T, V, C)
    # N, C, T, V = predicted.shape --> original size for bone information
    bone_id = [(5, 6), (5, 7),
                              (6, 8), (8, 10), (7, 9), (9, 11), 
                              (12,13),(12,14),(12,16),(12,18),(12,20),
                              (14,15),(16,17),(18,19),(20,21),
                              (22,23),(22,24),(22,26),(22,28),(22,30),
                              (24,25),(26,27),(28,29),(30,31),
                              (10,12),(11,22)]
    fp_sp = torch.zeros_like(data)
    for v1, v2 in bone_id:
        v1 -= 5
        v2 -= 5
        # fp_sp[:, :, :, v2] = predicted[:, :, :, v2] - predicted[:, :, :, v1] --> original implementation
        fp_sp[:, :, v2, :] = data[:, :, v2, :] - data[:, :, v1, :]
    return fp_sp
def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return np.mean(np.linalg.norm(predicted - target, axis=len(target.shape)-1), axis=1)

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1), axis=1)

# PyTorch-based errors (for losses)

def loss_mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))
    
def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))

def loss_2d_weighted(predicted, target, conf):
    assert predicted.shape == target.shape
    predicted_2d = predicted[:,:,:,:2]
    target_2d = target[:,:,:,:2]
    diff = (predicted_2d - target_2d) * conf
    return torch.mean(torch.norm(diff, dim=-1))
    
def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return loss_mpjpe(scale * predicted, target)

def weighted_bonelen_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.001 * torch.pow(predict_3d_length - gt_3d_length, 2).mean()
    return loss_length

def weighted_boneratio_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.1 * torch.pow((predict_3d_length - gt_3d_length)/gt_3d_length, 2).mean()
    return loss_length

def get_limb_lens(x):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
    '''
    limbs_id = [[0,1], [1,2], [2,3],
         [0,4], [4,5], [5,6],
         [0,7], [7,8], [8,9], [9,10],
         [8,11], [11,12], [12,13],
         [8,14], [14,15], [15,16]
        ]
    limbs = x[:,:,limbs_id,:]
    limbs = limbs[:,:,:,0,:]-limbs[:,:,:,1,:]
    limb_lens = torch.norm(limbs, dim=-1)
    return limb_lens

def loss_limb_var(x):
    '''
        Input: (N, T, 17, 3)
    '''
    if x.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(x.device)
    limb_lens = get_limb_lens(x)
    limb_lens_var = torch.var(limb_lens, dim=1)
    limb_loss_var = torch.mean(limb_lens_var)
    return limb_loss_var

def loss_limb_gt(x, gt):
    '''
        Input: (N, T, 17, 3), (N, T, 17, 3)
    '''
    limb_lens_x = get_limb_lens(x)
    limb_lens_gt = get_limb_lens(gt) # (N, T, 16)
    return nn.L1Loss()(limb_lens_x, limb_lens_gt)

def loss_velocity(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    if predicted.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(predicted.device)
    velocity_predicted = predicted[:,1:] - predicted[:,:-1]
    velocity_target = target[:,1:] - target[:,:-1]
    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=-1))

def loss_joint(predicted, target):
    assert predicted.shape == target.shape
    return nn.L1Loss()(predicted, target)


        
def loss_bone(predicted, target):
    assert predicted.shape == target.shape
    predicted_bone = get_bone_information(predicted)
    target_bone = get_bone_information(target)
    return nn.L1Loss()(predicted_bone, target_bone)



def get_angles(x):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
    '''
    limbs_id = [[0,1], [1,2], [2,3],
         [0,4], [4,5], [5,6],
         [0,7], [7,8], [8,9], [9,10],
         [8,11], [11,12], [12,13],
         [8,14], [14,15], [15,16]
        ]
    angle_id = [[ 0,  3],
                [ 0,  6],
                [ 3,  6],
                [ 0,  1],
                [ 1,  2],
                [ 3,  4],
                [ 4,  5],
                [ 6,  7],
                [ 7, 10],
                [ 7, 13],
                [ 8, 13],
                [10, 13],
                [ 7,  8],
                [ 8,  9],
                [10, 11],
                [11, 12],
                [13, 14],
                [14, 15] ]
    eps = 1e-7
    limbs = x[:,:,limbs_id,:]
    limbs = limbs[:,:,:,0,:]-limbs[:,:,:,1,:]
    angles = limbs[:,:,angle_id,:]
    angle_cos = F.cosine_similarity(angles[:,:,:,0,:], angles[:,:,:,1,:], dim=-1)
    return torch.acos(angle_cos.clamp(-1+eps, 1-eps)) 

def loss_angle(x, gt):
    '''
        Input: (N, T, 17, 3), (N, T, 17, 3)
    '''
    limb_angles_x = get_angles(x)
    limb_angles_gt = get_angles(gt)
    return nn.L1Loss()(limb_angles_x, limb_angles_gt)

def loss_angle_velocity(x, gt):
    """
    Mean per-angle velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert x.shape == gt.shape
    if x.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(x.device)
    x_a = get_angles(x)
    gt_a = get_angles(gt)
    x_av = x_a[:,1:] - x_a[:,:-1]
    gt_av = gt_a[:,1:] - gt_a[:,:-1]
    return nn.L1Loss()(x_av, gt_av)


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
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class VicReg(nn.Module):
    """
    Implementation of VicReg adapted from https://github.com/facebookresearch/vicreg/
    """
    def __init__(
        self,
        embedding_size,
        ssl_batch_size,
        sim_coeff=25,
        std_coeff=25,
        cov_coeff=1,
        **kwargs
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.ssl_batch_size = ssl_batch_size

        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def _compute_vicreg_loss(self, x, y):
        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.ssl_batch_size - 1)
        cov_y = (y.T @ y) / (self.ssl_batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.embedding_size
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.embedding_size)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )

        return loss, repr_loss, std_loss, cov_loss

    def forward(self, x, y):
        return self._compute_vicreg_loss(x, y)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class WassersteinDistanceLoss(nn.Module):
    """
    Wasserstein distance loss between local features of different modalities.
    Adapted from: https://github.com/ShramanPramanick/VoLTA/blob/master/Pre-training/main.py#L213 
    """
    def __init__(self, iteration=30):
        super().__init__()
        self.iteration = iteration

    def forward(self, local_features_1, local_features_2):
        cos_distance = cost_matrix_batch_torch(local_features_1.transpose(2, 1), local_features_2.transpose(2, 1))
        cos_distance = cos_distance.transpose(1, 2)
        beta = 0.1
        min_score = cos_distance.min()
        max_score = cos_distance.max()
        threshold = min_score + beta * (max_score - min_score)
        cos_dist = torch.nn.functional.relu(cos_distance - threshold)

        wd = IPOT_distance_torch_batch_uniform(
            cos_dist,
            local_features_1.size(0),
            local_features_1.size(1),
            local_features_2.size(1),
            iteration=self.iteration
        )
        return wd.mean()
