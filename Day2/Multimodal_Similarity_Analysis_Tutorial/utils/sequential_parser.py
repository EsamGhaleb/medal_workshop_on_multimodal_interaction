import argparse
import numpy as np

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='speech and skeleton models for co-speech gesture analysis')
    parser.add_argument(
        '--work-dir',
        default='work-dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument('-Experiment_name', default='')
    parser.add_argument(
        '--config',
        default='configs/SSL/train_ssl_clip_on_gestures_augmented_combined.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=4,
        help='the number of worker for data loader')
    parser.add_argument(
        '--feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    parser.add_argument(
        '--audio-model',
        default='wav2vec2_wrapper',
        help='the model for audio')
    parser.add_argument(
        '--audio-model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='Adam', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=True, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--keep_rate',
        type=float,
        default=0.9,
        help='keep probability for drop')
    parser.add_argument(
        '--groups',
        type=int,
        default=8,
        help='decouple groups')
    parser.add_argument(
        '--pretrained-model',
        type=str,
        default='',
        help='pretrained model name'
    )
    parser.add_argument(
        '--loss-function',
        type=str,
        default='NTXent',
        help='loss function'
    )

   
    # New Arguments
    parser.add_argument('--accumulate_grad_batches', type=int, default=4, help='accumulate gradient batches')
    parser.add_argument(
        '--all_audio_path', 
        type=str, 
        default='~/data/{}_synced_pp{}.wav',
        help='Path to the processed audio data'
    ) 
    parser.add_argument(
        '--fusion',
        default=None,
        help='the arguments of fusion model')
    parser.add_argument(
        '--fusion-args',
        default=dict(),
        help='the arguments of fusion model')
    parser.add_argument(
        "--scheduler",
        type=str
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience of the scheduler"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=50,
        help="Patience of the early stopping, this can be different from the scheduler & longer for the cross modal fusion as it takes longer to train"
    )

    
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_rate_decay', type=float, default=0.1,
                        help='decay rate for learning rate')

    # model dataset
    parser.add_argument('--model', type=str, default='bimodal')
    parser.add_argument('--dataset', type=str, default='CABB',
                        choices=['cifar10', 'cifar100', 'path', 'CABB'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    
    # add skeleton_augmentations_path
    parser.add_argument(
        '--skeleton_augmentations_path',
        default='config/augmentations/skeleton_simple_aug.yaml',
        help='the path to the skeleton augmentations',
        type=str
    )

    parser.add_argument(
        '--wandb_entity',
        default='sensor_har',
        choices=["sensor_har", "none"],
        help='Name of wandb logger. Set to none not to use wandb and use tensorboard only.',
        type=str
    )
    return parser