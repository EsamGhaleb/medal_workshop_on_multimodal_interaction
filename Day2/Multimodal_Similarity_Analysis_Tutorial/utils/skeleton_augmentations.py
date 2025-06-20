import random
import numpy as np
import torch
from scipy import signal


class Jittering():
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        # add one dimension to the input tensor
        if len(x.shape) == 4:
            x = x.squeeze()
        noise = np.random.normal(loc=0, scale=self.sigma, size=x.shape)
        x = x + torch.tensor(noise).float()
        if len(x.shape) == 3:
            x = x.unsqueeze(-1)
        return x


class RecenterJoints(object):
    """Changes the center of the coordinate system of the entire sample to the position of the
    anchor joint in the first frame.

    Args:
        anchor_joint_index (int): Index of the joint which will become the new coordinate center. 
            Usually a hip or spine joint.
    """
    
    def __init__(self, anchor_joint_index):
        self.anchor_joint_index = anchor_joint_index

    def __call__(self, sample):
        if len(sample.shape) == 4:
            sample = sample.squeeze()
        anchor_joint = sample[self.anchor_joint_index, :, 0]
        sample = (sample - np.expand_dims(anchor_joint, axis=1))
        if len(sample.shape) == 3:
            sample = sample.unsqueeze(-1)
        return sample


class NormalizeDistances(object):
    """Normalizes the sample based on the distance between the specified pair of anchor joints.

    Args:
        anchor_joint_1_index (int): First anchor joint.
        anchor_joint_2_index (int): Second anchor joint.
    """

    def __init__(self, anchor_joint_1_index, anchor_joint_2_index):
        self.anchor_joint_1_index = anchor_joint_1_index
        self.anchor_joint_2_index = anchor_joint_2_index

    def __call__(self, sample):
        if len(sample.shape) == 4:
            sample = sample.squeeze()
        anchor_joint_1 = sample[self.anchor_joint_1_index, :, 0]
        anchor_joint_2 = sample[self.anchor_joint_2_index, :, 0]
        norm_distance = np.linalg.norm(anchor_joint_1 - anchor_joint_2)
        sample = sample / norm_distance
        if len(sample.shape) == 3:
            sample = sample.unsqueeze(-1)
        return sample


class SkeletonSampler:
    """
    Resamples a skeleton frame sequence from any size of timesteps to the given size.
    """

    def __init__(self, size=125):
        """
        Initiates transform with a target number for resize
        :param size: int
        """
        self.size = size

    def __call__(self, x):
        """
        Uses scipy signal resample function to downsample/upsample the signals (joint positions) to the given size
        :param x: ndarray
        :return: ndarray
        """
        if len(x.shape) == 4:
            x = x.squeeze()
        x = signal.resample(x, self.size, axis=2)
        if len(x.shape) == 3:
            x = x.unsqueeze(-1)
        return x

class GeometricTransform:
    """
    Generic class for geometric transformations in 2D and 3D, offering some helper functions.
    """
    def __init__(self, dimensions):
        self.dimensions = dimensions

    def to_homogeneous_coordinates(self, x):
        """
        Expects input as tensors of shape CxFxJ (channels x frames x joints).
        Returns tensors of shape FxJx(C+1) (proper shape for batch matrix multiplications).
        """
        x = x.permute(1, 2, 0)
        x = torch.cat((x, torch.ones((*x.shape[:2], 1))), dim=2)
        return x

    def to_cartesian_coordinates(self, x):
        """
        Expects input as tensors of shape FxJx(C+1).
        Returns tensors of shape CxFxJ (channels x frames x joints).
        """
        x = x[:, :, :self.dimensions].permute(2, 0, 1)
        return x

    def build_translation_matrix(self, n_frames, offsets):
        T = torch.eye(self.dimensions + 1)
        T = T.repeat([n_frames, 1, 1])
        T[:, :self.dimensions, -1] = offsets
        T = T.float()
        return T

    def build_rotation_matrix(self, n_frames, angles):
        """
        2D: rotate around origin.
        3D: rotate around each axis.
        """
        if self.dimensions == 2:
            theta = np.radians(angles[0])
            c = np.cos(theta)
            s = np.sin(theta)
            R = torch.eye(self.dimensions + 1)
            R[0][0] = c
            R[0][1] = -s
            R[1][0] = s
            R[1][1] = c
        else:
            thetax, thetay, thetaz = np.radians(angles[0]), np.radians(angles[1]), np.radians(angles[2])
            cx, cy, cz = np.cos(thetax), np.cos(thetay), np.cos(thetaz)
            sx, sy, sz = np.sin(thetax), np.sin(thetay), np.sin(thetaz)
            
            # Rotate around X axis.
            Rx = torch.eye(self.dimensions + 1)
            Rx[1, 1] = cx
            Rx[1, 2] = -sx
            Rx[2, 1] = sx
            Rx[2, 2] = cx

            # Rotate around Y axis.
            Ry = torch.eye(self.dimensions + 1)
            Ry[0, 0] = cy
            Ry[0, 2] = sy
            Ry[2, 0] = -sy
            Ry[2, 2] = cy

            # Rotate around Z axis.
            Rz = torch.eye(self.dimensions + 1)
            Rz[0][0] = cz
            Rz[0][1] = -sz
            Rz[1][0] = sz
            Rz[1][1] = cz

            R = Rx @ Ry @ Rz

        R = R.repeat([n_frames, 1, 1]).float()
        return R

    def build_scaling_matrix(self, n_frames, factors):
        S = torch.diagflat(torch.tensor([*factors, 1]))
        S = S.repeat([n_frames, 1, 1]).float()
        return S

    def build_shear_matrix(self, n_frames, factors):
        if (self.dimensions == 2):
            shx = factors[0]
            shy = factors[1]
            S = torch.tensor([
                [1,   shx, 0],
                [shy, 1,   0],
                [0,   0,   1]
            ])
        else:
            shx = factors[0]
            shy = factors[1]
            shz = factors[2]
            S = torch.tensor([
                [1,   shx, shx, 0],
                [shy, 1,   shy, 0],
                [shz, shz, 1,   0],
                [0,   0,   0,   1]
            ])
        S = S.repeat([n_frames, 1, 1]).float()
        return S


class RandomRotation(GeometricTransform):
    """
    Data augmentation transform, applies a random rotation to each joint of
    the skeleton around a given anchor joint.
    Expects input as ndarrays of shape (CxFxJ, where J=joints, F=frames).
    
    2D: rotates around the given anchor joint.
    3D: rotates around each axis with a different angle.
    """
    def __init__(self, dimensions, start=-5, end=5, step=5, anchor_joint=0):
        super(RandomRotation, self).__init__(dimensions)
        self.choices_list = list(range(start, end + 1, step))
        self.anchor_joint = anchor_joint
    
    def __call__(self, x):
        if len(x.shape) == 4:
            x = x.squeeze()
        x = x.float()
        x = self.to_homogeneous_coordinates(x) # FxJxC

        T1 = self.build_translation_matrix(x.shape[0], x[:, 0, :self.dimensions])
        T2 = self.build_translation_matrix(x.shape[0], -x[:, 0, :self.dimensions])
        if self.dimensions == 2:
            angles = [np.random.choice(self.choices_list)]
        else:
            angles = np.random.choice(self.choices_list, size=3)
        R = self.build_rotation_matrix(x.shape[0], angles)

        x = T1 @ R @ T2 @ x.permute(0, 2, 1)
        x = x.permute((0, 2, 1))
        x = self.to_cartesian_coordinates(x)
        if len(x.shape) == 3:
            x = x.unsqueeze(-1)
        return x


class RandomScale(GeometricTransform):
    """
    Scales by a random factor across each dimension, centered in a given anchor joint.
    Expects input as ndarrays of shape (CxFxJ, where J=joints, F=frames).
    """

    def __init__(self, dimensions, min_p=0.9, max_p=1.1, anchor_joint=1):
        super(RandomScale, self).__init__(dimensions)
        self.min_p = min_p
        self.max_p = max_p
        self.anchor_joint = anchor_joint

    def __call__(self, x):
        if len(x.shape) == 4:
            x = x.squeeze()
        x = self.to_homogeneous_coordinates(x) # FxJxC

        T1 = self.build_translation_matrix(x.shape[0], x[:, 0, :self.dimensions])
        T2 = self.build_translation_matrix(x.shape[0], -x[:, 0, :self.dimensions])
        S = self.build_scaling_matrix(x.shape[0], np.random.uniform(self.min_p, self.max_p, size=self.dimensions))

        x = T1 @ S @ T2 @ x.permute(0, 2, 1)
        x = x.permute((0, 2, 1))
        x = self.to_cartesian_coordinates(x)
        if len(x.shape) == 3:
            x = x.unsqueeze(-1)
        return x


class RandomShear(GeometricTransform):
    """
    Applies a shear transformation in all directions by a random factor (chosen from a normal distribution
    with a mean of 0).
    Expects as input tensors of shape CxFxJ (C = channels, F = frames, J = joints).
    """
    def __init__(self, dimensions, beta=0.1):
        super(RandomShear, self).__init__(dimensions)
        self.beta = beta

    def __call__(self, x):
        if len(x.shape) == 4:
            x = x.squeeze()
        x = self.to_homogeneous_coordinates(x) # FxJxC

        S = self.build_shear_matrix(x.shape[0], np.random.normal(loc=0., scale=self.beta, size=self.dimensions))

        x = S @ x.permute(0, 2, 1)
        x = x.permute((0, 2, 1))
        x = self.to_cartesian_coordinates(x)
        if len(x.shape) == 3:
            x = x.unsqueeze(-1)
        return x


class VariableLengthRandomCrop:
    """
    Takes a crop of a variable length from a random position within the sequence. The resulting samples
    will be contiguous (i.e. will not loop around the end of the sequence).
    The length of the crop is in [crop_p... 1] * sample_size. 
    Expects input as ndarrays of shape (JxCxF, where J=joints, C=channels, F=frames).
    """

    def __init__(self, crop_p=0.7):
        self.crop_p = crop_p

    def __call__(self, x):
        if len(x.shape) == 4:
            x = x.squeeze()
        sample_length = x.shape[2]
        crop_length = int(random.uniform(self.crop_p, 1) * sample_length)
        max_crop_start = sample_length - crop_length
        crop_start = random.randint(0, max_crop_start)
        x = x[:, :, crop_start:crop_start+crop_length]
        if len(x.shape) == 3:
            x = x.unsqueeze(-1)
        return x


class CropAndResize:
    """
    Takes a random crop of variable length from a random position within the sequence. The crop
    is then resized to the given fixed size.
    Expects as input tensors of shape CxFxJ (C = channels, F = frames, J = joints).
    TODO: wrap around the edges, for more variation?
    """
    def __init__(self, min_p=0.7, max_p=0.9, size=50):
        self.min_p = min_p
        self.max_p = max_p
        self.size = size

    def __call__(self, x):
        if len(x.shape) == 4:
            x = x.squeeze()
        sample_length = x.shape[1]
        crop_length = int(random.uniform(self.min_p, self.max_p) * sample_length)
        max_crop_start = sample_length - crop_length
        crop_start = random.randint(0, max_crop_start)
        cropped = x[:, crop_start:crop_start+crop_length, :]
        result = torch.tensor(signal.resample(cropped, self.size, axis=1)).float()
        if len(result.shape) == 3:
            result = result.unsqueeze(-1)
        return result

class MirrorPoses:
    """
    Mirrors the poses horizontally.
    Expects as input tensors of shape CxFxVxM (C = channels, F = frames, V = joints, M = instances).
    """
    def __init__(self, frame_width=1920, num_channels=3):
        self.frame_width = frame_width
        self.num_channels = num_channels
    def __call__(self, x):
        C, T, V, M = x.shape
        assert C == self.num_channels  # x, y, c
        x[0, :, :, :] = self.frame_width - x[0, :, :, :]
        return x

class ShiftPoses:
    """
    Applies a random shift to each joint in the poses.
    Expects as input tensors of shape CxFxVxM (C = channels, F = frames, V = joints, M = instances).
    """
    def __init__(self, max_shift=30.0, min_shift=-30.0, num_joints=27):
        self.max_shift = max_shift
        self.min_shift = min_shift
        self.num_joints = num_joints
    def __call__(self, x):
        C, T, V, M = x.shape
        assert V == self.num_joints
        for i in range(V):  # For each joint
            shift_x = random.uniform(-self.min_shift, self.max_shift)  # Random shift in x
            shift_y = random.uniform(-self.min_shift, self.max_shift)  # Random shift in y
            x[0, :, i, :] += shift_x
            x[1, :, i, :] += shift_y
        return x

class ScalePoses:
    """
    Scales the poses by a random factor based on the image size.
    Expects as input tensors of shape CxFxVxM (C = channels, F = frames, V = joints, M = instances).
    """
    def __init__(self, min_scale=0.5, max_scale=1.5):
        self.min_scale = min_scale
        self.max_scale = max_scale
    def __call__(self, x):
        C, T, V, M = x.shape
        scales = (random.uniform(self.min_scale, self.max_scale), random.uniform(self.min_scale, self.max_scale))
        x[0, :, :, :] *= scales[0]
        x[1, :, :, :] *= scales[1]
        return x

class AutoPadding:
    """
    Pads the sequence to a specified size. If the sequence is shorter than the size,
    it's padded with zeros either randomly or at the beginning.
    Expects input as tensors of shape CxFxVxM (C = channels, F = frames, V = joints, M = instances).
    """
    
    def __init__(self, size, random_pad=False):
        self.size = size
        self.random_pad = random_pad

    def __call__(self, x):
        C, T, V, M = x.shape
        if T < self.size:
            begin = random.randint(0, self.size - T) if self.random_pad else 0
            data_numpy_padded = np.zeros((C, self.size, V, M))
            data_numpy_padded[:, begin:begin + T, :, :] = x
            return data_numpy_padded
        else:
            return x
  
    

class RandomChoose:
    """
    Randomly selects a segment from the sequence. If the sequence is shorter than the specified size,
    it can be automatically padded.
    Expects input as tensors of shape CxFxVxM (C = channels, F = frames, V = joints, M = instances).
    """

    def __init__(self, size, auto_pad=True):
        self.size = size
        self.auto_pad = auto_pad

    def __call__(self, x):
        C, T, V, M = x.shape
        if T == self.size:
            return x
        elif T < self.size:
            if self.auto_pad:
                return AutoPadding(self.size, random_pad=True)(x)
            else:
                return x
        else:
            begin = random.randint(0, T - self.size)
            return x[:, begin:begin + self.size, :, :]
  
class RandomMove:
    """
    Applies random rotations, scalings, and translations to the sequence.
    Expects input as tensors of shape CxFxVxM (C = channels, F = frames, V = joints, M = instances).
    """

    def __init__(self, angle_candidate=None, scale_candidate=None,
                 transform_candidate=None, move_time_candidate=None):
        if angle_candidate is None:
            angle_candidate = [-10., -5., 0., 5., 10.]
        if scale_candidate is None:
            scale_candidate = [0.9, 1.0, 1.1]
        if transform_candidate is None:
            transform_candidate = [-0.2, -0.1, 0.0, 0.1, 0.2]
        if move_time_candidate is None:
            move_time_candidate = [1]

        self.angle_candidate = angle_candidate
        self.scale_candidate = scale_candidate
        self.transform_candidate = transform_candidate
        self.move_time_candidate = move_time_candidate

    def __call__(self, x):
        C, T, V, M = x.shape
        move_time = random.choice(self.move_time_candidate)
        if move_time == 0 or move_time <=0.1:
            # Handle the zero case, e.g., skip or assign a default move_time
            move_time = 1  # or another appropriate fallback
        node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
        node = np.append(node, T)
        num_node = len(node)

        A = np.random.choice(self.angle_candidate, num_node)
        S = np.random.choice(self.scale_candidate, num_node)
        T_x = np.random.choice(self.transform_candidate, num_node)
        T_y = np.random.choice(self.transform_candidate, num_node)

        a = np.zeros(T)
        s = np.zeros(T)
        t_x = np.zeros(T)
        t_y = np.zeros(T)

        # Linspace interpolation
        for i in range(num_node - 1):
            a[node[i]:node[i + 1]] = np.linspace(
                A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
            s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                                 node[i + 1] - node[i])
            t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                                   node[i + 1] - node[i])
            t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                                   node[i + 1] - node[i])

        theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                          [np.sin(a) * s, np.cos(a) * s]])  # Rotation matrix

        # Perform transformation
        for i_frame in range(T):
            xy = x[0:2, i_frame, :, :]
            new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
            new_xy[0] += t_x[i_frame]
            new_xy[1] += t_y[i_frame]  # Translation
            x[0:2, i_frame, :, :] = torch.from_numpy(new_xy.reshape(2, V, M)).type_as(x)  # Convert back to tensor and ensure type consistency
        return x
