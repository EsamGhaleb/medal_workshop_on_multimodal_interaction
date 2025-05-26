import argparse

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
        description='Decoupling Graph Convolution Network with DropGraph Module')
    parser.add_argument(
        '--work-dir',
        default='/home/eghaleb/Projects/SL-GCN/work-dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument('-Experiment_name', default='')
    parser.add_argument(
        '--config',
        default='./config/detection/test_joints_sho.yaml',
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
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--viterabi-interval',
        type=int,
        default=20,
        help='the interval for applying viterabi algorithm (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')
    parser.add_argument(
        '--pretrained',
        type=bool,
        default=True,
        help='whether to use pretrained model')

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

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
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
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
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
        default='WeightedFocalLoss',
        help='loss function'
    )
    parser.add_argument('--only_train_part', default=True)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)

       # labeler
    parser.add_argument('--labeler', default=None, help='the labeler will be used')
    parser.add_argument(
        '--labeler-args',
        type=dict,
        default=dict(),
        help='the arguments of the labeler')
   
    # New Arguments
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--gesture_unit', type=bool, default=True, help='gesture unit flag')
    parser.add_argument('--subject_joint', type=bool, default=True, help='subject joint flag')
    parser.add_argument('--cross_attn', type=bool, default=False, help='cross attention flag')
    parser.add_argument('--fine_tuned_audio_model', type=bool, default=False, help='fine-tuned audio model flag')
    parser.add_argument('--vggish', type=bool, default=False, help='VGGish flag')
    parser.add_argument('--accumulate_grad_batches', type=int, default=4, help='accumulate gradient batches')
    parser.add_argument(
        '--all_audio_path', 
        type=str, 
        default='/Users/esamghaleb/Documents/ResearchData/CABB Small Dataset/processed_audio_video/{}/{}_synced_pp{}.wav',
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
        '--use-crf',
        type=bool,
        default=False,
        help='use crf or not') 
    parser.add_argument(
        "--scheduler",
        type=str
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Patience of the scheduler"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=80,
        help="Patience of the early stopping, this can be different from the scheduler & longer for the cross modal fusion as it takes longer to train"
    )
    parser.add_argument(
        '--wandb_entity',
        default='sensor_har',
        choices=["sensor_har", "none"],
        help='Name of wandb logger. Set to none not to use wandb and use tensorboard only.',
        type=str
    )
    return parser