import numpy as np

class Jitter3D:
    """Add Gaussian noise to (x,y,z) channels of skeleton data."""
    def __init__(self, sigma=0.02):
        self.sigma = sigma  # standard deviation relative to normalized coords

    def __call__(self, data):
        # data: ndarray shape (J, F, 4)
        noise = np.random.normal(0, self.sigma, size=data[:, :, :3].shape)
        data[:, :, :3] += noise
        return data

class RandomRotate3D:
    """Rotate 3D skeleton around a random axis by an angle in degrees."""
    def __init__(self, max_angle=30):
        self.max_angle = max_angle  # degrees

    def __call__(self, data):
        # data: ndarray (J, F, 4)
        theta = np.deg2rad(np.random.uniform(-self.max_angle, self.max_angle))
        axis = np.random.choice(['x','y','z'])
        c, s = np.cos(theta), np.sin(theta)

        # Rotation matrices
        if axis == 'x':
            R = np.array([[1,0,0],[0,c,-s],[0,s,c]])
        elif axis == 'y':
            R = np.array([[c,0,s],[0,1,0],[-s,0,c]])
        else:  # z
            R = np.array([[c,-s,0],[s,c,0],[0,0,1]])

        coords = data[:, :, :3].reshape(-1,3).T  # (3, J*F)
        rotated = (R @ coords).T.reshape(data[:, :, :3].shape)
        data[:, :, :3] = rotated
        return data


class RandomScale3D:
    """Scale 3D skeleton by a random factor per axis."""
    def __init__(self, min_scale=0.9, max_scale=1.1):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, data):
        scales = np.random.uniform(self.min_scale, self.max_scale, size=3)
        data[:, :, :3] *= scales.reshape(1,1,3)
        return data

class RandomTranslate3D:
    """Translate 3D skeleton by a random offset."""
    def __init__(self, max_translate=0.1):
        self.max_translate = max_translate

    def __call__(self, data):
        offsets = np.random.uniform(-self.max_translate, self.max_translate, size=3)
        data[:, :, :3] += offsets.reshape(1,1,3)
        return data

class CenterNormalize3D:
    """
    Center skeleton so that a reference joint (e.g., root or hip) is at the origin.
    data: ndarray of shape (J, F, 4) where last dim = (x, y, z, visibility)
    """
    def __init__(self, root_index=0):
        self.root_index = root_index

    def __call__(self, data):
        # data[root_index] has shape (F, 4)
        root_coords = data[self.root_index, :, :3]               # (F, 3)
        data[:, :, :3] = data[:, :, :3] - root_coords[None, :, :]
        return data
class RandomFlip3D:
      """Randomly flip 3D skeleton along a random axis."""
      def __init__(self):
         pass
      def __call__(self, data):
         axis = np.random.choice(['x', 'y'])
         if axis == 'x':
               data[:, :, 0] *= -1
         elif axis == 'y':
               data[:, :, 1] *= -1
        #  else:  # z
        #        data[:, :, 2] *= -1
         return data
      
class RandomShear3D:
    """Apply random shear in 3D space."""
    def __init__(self, max_shear=0.1):
        self.max_shear = max_shear

    def __call__(self, data):
        sh = np.random.uniform(-self.max_shear, self.max_shear, size=3)
        S = np.array([
            [1, sh[0], sh[1]],
            [sh[0], 1, sh[2]],
            [sh[1], sh[2], 1]
        ])
        coords = data[:, :, :3].reshape(-1,3).T
        sheared = (S @ coords).T.reshape(data[:, :, :3].shape)
        data[:, :, :3] = sheared
        return data

class RandomCropTemporal:
    """Crop a random subsequence of specified length."""
    def __init__(self, target_length):
        self.target_length = target_length

    def __call__(self, data):
        J, F, C = data.shape
        if F <= self.target_length:
            return data
        start = np.random.randint(0, F - self.target_length)
        return data[:, start:start+self.target_length, :]

from scipy.signal import resample

class TemporalResample:
    """Resample frames to a fixed number via interpolation."""
    def __init__(self, target_frames):
        self.target = target_frames

    def __call__(self, data):
        J, F, C = data.shape
        data_resampled = resample(data, self.target, axis=1)
        return data_resampled

class TimeReverse:
    """Reverse the order of frames."""
    def __call__(self, data):
        return data[:, ::-1, :]


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Example pipeline
    mediapipe_flip_index = np.concatenate(( [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                                            [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42], [43, 44, 45]))
    pipeline = Compose([
        CenterNormalize3D(root_index=0),  # Assuming root joint is at index 0
        Jitter3D(sigma=0.02),
        RandomRotate3D(max_angle=20),
        RandomScale3D(0.9,1.1),
        RandomTranslate3D(0.05),
        RandomShear3D(0.05),
        RandomFlip3D(),
        # RandomCropTemporal(target_length=100),
        # TemporalResample(target_frames=120),
        # TimeReverse()
        ])
    skeleton_data = np.load('/Users/esagha/Projects/medal_workshop_on_multimodal_interaction/Gesture_Segmentation_Toturial/Code/data/mediapipe_outputs/pair04_synced_ppA.npy')
    augmented = pipeline(skeleton_data)  # skeleton_data shape (J, F, 4)


    # Apply the pipeline to the skeleton data
    augmented_data = pipeline(skeleton_data)
    print("Original shape:", skeleton_data.shape)
    print("Augmented shape:", augmented_data.shape)
