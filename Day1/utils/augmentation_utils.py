from torchvision import transforms

from utils.skeleton_augmentations import *

def compose_random_augmentations(modality, config_dict):
    skeleton_augmentations = {
        'jittering': Jittering,
        'crop_and_resize': CropAndResize,
        'scaling': RandomScale,
        'rotation': RandomRotation,
        'shear': RandomShear,
        'mirror_poses': MirrorPoses,
        'shift_poses': ShiftPoses,
        'scale_poses': ScalePoses,
        # 'random_choose': RandomChoose,
        'random_move': RandomMove
    }

    all_augmentations = {
        "skeleton": skeleton_augmentations
    }
    transforms_list = []
    augmentations_for_modality = all_augmentations[modality]
    for key in config_dict:
        if config_dict[key]['apply']:
            if 'parameters' not in config_dict[key]:
                config_dict[key]['parameters'] = {}
            augmentation = augmentations_for_modality[key](**config_dict[key]['parameters'])
            probability = config_dict[key]['probability']
            transforms_list.append(transforms.RandomApply([augmentation], p=probability))
    return transforms_list 
