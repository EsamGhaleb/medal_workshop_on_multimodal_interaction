import datetime
import os
import numpy as np
import argparse
import errno
import random
import pytorch_lightning as L
import wandb
import json
import pickle

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold 
from utils.tools import get_config
from collections import defaultdict
from pytorch_lightning import Trainer
from pytorch_lightning import loggers

from callbacks.init_callbacks import (
    setup_early_stopping_callback,
    setup_model_checkpoint_callback
)
from torch.utils.data import random_split
from feeders.gestures_feeder_for_segmentation import CABBFeeder
from utils.utils import load_yaml_to_dict

from model.segment_net import SegmentNet
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from model.skeleton_speech_models_segmentation import GSSModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/segmentation/CABB_segment_basic.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='CABB_Segmentation', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default=None, type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-ld', '--limited_data_segmentation', default=None, type=float, help='limited data segmentation')
    parser.add_argument('--devices', nargs="+", default=[-1], help="device ids to use")
    parser.add_argument('--phase', default='eval', type=str, help='eval or test')
    parser.add_argument('--apply_skeleton_augmentations', default=True, type=bool, help='apply skeleton augmentations')
    # add skeleton_augmentations_path
    parser.add_argument(
        '--skeleton_augmentations_path',
        default='config/augmentations/skeleton_simple_aug.yaml',
        help='the path to the skeleton augmentations',
        type=str
    )
    parser.add_argument("--debug", action="store_true", default=False)
    opts = parser.parse_args()
    return opts

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    L.seed_everything(seed)

def save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss):
    print('Saving checkpoint to', chk_path)
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model_pos': model_pos.state_dict(),
        'min_loss' : min_loss
    }, chk_path)


def load_objectnet_backbone_from_checkpoint(backbone, checkpoint, use_gpu=True):
    """
    Load ObjectNet backbone weights from a checkpoint.

    Args:
        backbone: The backbone model to load weights into.
        checkpoint (str): Path to the checkpoint file.
        use_gpu (bool): Whether to use GPU for loading. Defaults to True.

    Returns:
        None
    """
    device = torch.device("cuda" if use_gpu else "cpu")
    state_dict = torch.load(checkpoint)["state_dict"]
    state_dict_backbone = {key.partition("backbone.")[2]: state_dict[key] for key in state_dict if "backbone" in key}
    return backbone.load_state_dict(state_dict_backbone, strict=False)

def train_with_config(args, opts):
    print(args)
    try:
        os.makedirs(opts.checkpoint, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)

    print('Loading dataset...')
    trainloader_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 12,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True
    }

    testloader_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 12,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True
    }

    skeleton_augmentations = load_yaml_to_dict(opts.skeleton_augmentations_path)
    cabb_feeder = CABBFeeder(
        phase=args.phase,
        task='segmentation',
        skeleton_augmentations=skeleton_augmentations,
        window_size=args.model_args['maxlen'],
        modalities=args.model_args['modalities'],
        filter_text=False,
        w2v2_type=args.model_args['w2v2_type'],
        apply_skeleton_augmentations=args.apply_skeleton_augmentations,
    )
    
    speaker_ID = cabb_feeder.data['pair_speaker']
    gkf = GroupKFold(n_splits=5)
    splitting_method = gkf.split(cabb_feeder, groups=speaker_ID)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    splitting_method = kf.split(cabb_feeder)

    # splitting_method = gkf.split(cabb_feeder, groups=speaker_ID)

    experiment_group = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"    
   
    results = defaultdict(dict)
    for fold, (train_index, test_index) in enumerate(splitting_method):
        print(train_index, test_index)
        print(f"Starting Fold {fold + 1}")
        args['fold'] = fold+1

        # Create fold-specific directory
        fold_dir = os.path.join(opts.checkpoint, f"fold_{fold + 1}")
        os.makedirs(fold_dir, exist_ok=True)

        if opts.limited_data_segmentation is not None:
            train_index = np.random.choice(
                train_index,
                int(len(train_index) * opts.limited_data_segmentation),
                replace=False
            )

        train_dataset = torch.utils.data.Subset(cabb_feeder, train_index)
        test_dataset = torch.utils.data.Subset(cabb_feeder, test_index)

        if args.phase != 'test':
            cabb_loader_2d = DataLoader(train_dataset, **trainloader_params)
            cabb_loader_2d_val = DataLoader(test_dataset, **testloader_params)
        else:
            cabb_loader_2d = DataLoader(cabb_feeder, **trainloader_params)
            cabb_loader_2d_val = DataLoader(cabb_feeder, **testloader_params)

        # Load model backbone
        model_backbone = GSSModel(**args.model_args)
        model_params = sum(p.numel() for p in model_backbone.parameters())
        print('INFO: Trainable parameter count:', model_params)

        model_weights = args.pretrained if hasattr(args, 'pretrained') else None
        args['limited_data_segmentation'] = opts.limited_data_segmentation

        

        # Define the model
        segmentation_model = SegmentNet(
            backbone=model_backbone,
            optimizer="adamw",
            args=args,
            lr=args.learning_rate,
        )
        # if model_weights is not None:
        #     state_dict = torch.load(model_weights, map_location=torch.device('cpu'))["state_dict"]
        #     # load the state dict into the model
        #     segmentation_model.load_state_dict(state_dict, strict=True)


        # Experiment ID and logging
        experiment_id = f"{experiment_group}_fold_{fold + 1}"
        experiment_info = vars(opts)
        experiment_info.update(args)

        wandb_logger = loggers.WandbLogger(
            config=experiment_info,
            entity="sensor_har",
            project="CABB_segmentation",
            name=experiment_id,
            id=experiment_id,
            save_dir=fold_dir,
            group=experiment_group
        )
        tb_logger = TensorBoardLogger(fold_dir, name=experiment_id, version=experiment_id)

        loggers_list = [tb_logger, wandb_logger]

        callbacks = [
            # setup_early_stopping_callback("val/segmentation_loss"),
            setup_model_checkpoint_callback(
                os.path.join(fold_dir, "checkpoints"),
                "val/segmentation_loss",
                "CABB",
                "segmentation",
                experiment_id
            ),
        ]

        if torch.cuda.is_available():
            devices = [int(d) for d in opts.devices] if len(opts.devices) > 1 else int(opts.devices[0])
            # Trainer Configuration
            trainer = Trainer(
                logger=loggers_list,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=devices,
                num_nodes=1,
                deterministic=True,
                max_epochs=args.epochs,
                default_root_dir=fold_dir,
                callbacks=callbacks,
                strategy="ddp_find_unused_parameters_true",
                accumulate_grad_batches=1,
                limit_train_batches=None if not opts.debug else 10,
                limit_val_batches=None if not opts.debug else 10,
            )
        else:
            devices = None
            trainer = Trainer(
                logger=loggers_list,
                accelerator="cpu",
                max_epochs=args.epochs,
                default_root_dir=fold_dir,
                callbacks=callbacks,
                accumulate_grad_batches=1,
                limit_train_batches=None if not opts.debug else 10,
                limit_val_batches=None if not opts.debug else 10,
            )
        if args.phase == 'train':
        # Train the model
            trainer.fit(segmentation_model, cabb_loader_2d, cabb_loader_2d_val)
        else:
            # Evaluate the model
            checkpoint_path = "CABB_Segmentation/fold_{}/checkpoints/latest/last.ckpt".format(fold + 1)
            trainer.test(segmentation_model, dataloaders=cabb_loader_2d_val, ckpt_path=checkpoint_path)
            # Save results for the fold
       
        wandb.finish()

        results[fold]['labels'] = segmentation_model.models_results['test']['labels']
        results[fold]['preds'] = segmentation_model.models_results['test']['preds']
        results[fold]['loss'] = segmentation_model.models_results['test']['loss']
        results[fold]['start_frames'] = segmentation_model.models_results['test']['start_frames']
        results[fold]['end_frames'] = segmentation_model.models_results['test']['end_frames']
        results[fold]['speaker_ID'] = segmentation_model.models_results['test']['speaker_ID']
        results[fold]['pair_ID'] = segmentation_model.models_results['test']['pair_ID']
    # save the results
    if args.phase == 'train':
        results_path = fold_dir+"/results.pkl"  
    elif args.phase == 'test':
        results_path = fold_dir+"/dilay_test_results.pkl"
    print(f"Saving results to {results_path}")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
        

if __name__ == "__main__":
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    train_with_config(args, opts)
