
from collections import defaultdict
import warnings

import numpy as np
import yaml
import pickle
import torch
import lightning as L
import time
import torch
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
# from torchcrf import CRF
from torchmetrics import MetricCollection
from sklearn.preprocessing import label_binarize
from scipy.special import softmax

from torchmetrics.classification import (
    BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
)
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score


from detection_parser import get_parser
from feeders.audio_video_feeder import SequentialAudioSkeletonFeeder, WrapperDataset
from feeders.audio_video_test_feeder_sho import SequentialAudioSkeletonTestFeeder as TestFeeder
from model.losses import WeightedFocalLoss

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

torch.set_float32_matmul_precision("medium")

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
 
class SpeechSkeletonModel(L.LightningModule):
    def __init__(self, arg):
        super().__init__()
        model = import_class(arg.fusion)
        self.model = model(**arg.fusion_args)
        metrics = MetricCollection([
            BinaryAccuracy(),
            BinaryPrecision(),
            BinaryRecall(),
            BinaryF1Score()
        ])
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.classes = ['non-gesture', 'gesture']
        self.lr = arg.lr
        self.num_epoch = arg.num_epoch
        self.scheduler = arg.scheduler
        self.loss = WeightedFocalLoss(gamma=2, alpha=0.25)
        self.metrics = ['Accuracy', 'Precision', 'Recall', 'F1Score']
        self.modalities = ['audio', 'skeleton', 'fused']
        if arg.use_crf:
            # raise NotImplementedError
            raise NotImplementedError
            # self.model_crf = CRF(num_tags=len(self.classes), batch_first=True)
        self.use_crf = arg.use_crf
        self.patience = arg.patience
        self.save_hyperparameters()
        self.models_results = defaultdict(dict)
        phases = ['train', 'val', 'test']
        keys = ['labels', 'preds', 'loss', 'start_frames', 'end_frames', 'speaker_ID', 'pair_ID']
        for phase in phases:
            for key in keys:
                self.models_results[phase][key] = torch.Tensor()
        
    def compute_and_log_metrics(self, orig_preds, orig_labels, stage, modality, compute_loss=True):
        if compute_loss:
            if self.use_crf:
                loss = -self.model_crf(orig_preds, orig_labels, reduction='token_mean')
            else:
                loss = self.loss(orig_preds, orig_labels)
            # self.log(f"{stage}_{modality}_loss", loss, sync_dist=True)
        N, T, C = orig_preds.shape
        if self.use_crf: #TODO: you can not use crf after the modifications
            preds = np.array(self.model_crf.decode(orig_preds))
            preds = preds.reshape(N * T)
            # convert the preds to torch tensor
            preds = torch.from_numpy(preds).to(self.device)
            # raise NotImplementedError
            raise NotImplementedError
        else:
            preds = orig_preds.view(N * T, C)
            preds = torch.argmax(preds, dim=-1)
        labels = orig_labels.view(N * T)

        labels = labels.squeeze().detach()
        preds = torch.round(preds).detach().squeeze()
        if modality == 'fused': 
            if stage == "train":
                self.train_metrics(preds, labels)
            else:
                self.val_metrics(preds, labels)
        return loss
    def report_metrics(self, stage, modality, log=True):
        # preds = torch.cat(self.models_results[stage]['preds']).detach().cpu().numpy()   
        # labels = torch.cat(self.models_results[stage]['labels']).detach().cpu().numpy()
        preds = self.models_results[stage]['preds'].detach().cpu().numpy()
        labels = self.models_results[stage]['labels'].detach().cpu().numpy()
        labels = labels.reshape(-1)  # This will flatten the array
        preds = preds.reshape(-1, preds.shape[-1])
        precision, recall, f1, _ = precision_recall_fscore_support(labels, np.argmax(preds, axis=1), average=None)
        for i in range(len(precision)):
            self.log("Precision/{}_{}_{}".format(stage, modality, self.classes[i]), precision[i], prog_bar=True, sync_dist=True)
            self.log("Recall/{}_{}_{}".format(stage, modality, self.classes[i]), recall[i], prog_bar=True, sync_dist=True)
            self.log("F1/{}_{}_{}".format(stage, modality, self.classes[i]), f1[i], prog_bar=True, sync_dist=True)
        precision, recall, f1, _ = precision_recall_fscore_support(
                labels, np.argmax(preds, axis=1), average='macro')
        self.log("Recall/{}_{}_macro".format(stage, modality), recall, prog_bar=True, sync_dist=True)
        self.log("F1/{}_{}_macro".format(stage, modality), f1, prog_bar=True, sync_dist=True)
        self.log("Precision/{}_{}_macro".format(stage, modality), precision, prog_bar=True, sync_dist=True)

        preds = softmax(preds, axis=1)
        # replace labels == 0 with 'neutral' and labels == 1 with 'gesture'
        labels = np.where(labels == 0, self.classes[0], self.classes[1])
        target_names = np.array(self.classes)
        y_onehot_test = label_binarize(labels, classes=target_names)
        if y_onehot_test.shape[1] == 1:
            y_onehot_test = np.concatenate([1-y_onehot_test, y_onehot_test], axis=1)

        micro_roc_auc_ovr = roc_auc_score(
            y_onehot_test,
            preds,
            multi_class="ovr",
            average="micro")
        macro_roc_auc_ovr = roc_auc_score(
            y_onehot_test,
            preds,
            multi_class="ovr",
            average="macro")
        self.log("AUC/{}_{}_micro".format(stage, modality), micro_roc_auc_ovr, prog_bar=True, sync_dist=True)
        self.log("AUC/{}_{}_macro".format(stage, modality), macro_roc_auc_ovr, prog_bar=True, sync_dist=True)
        
        # Compute mean average precision score
        macro_average_precision = average_precision_score(y_onehot_test, preds, average='macro')
        micro_average_precision = average_precision_score(y_onehot_test, preds, average='micro')
        gesture_average_precision = average_precision_score(y_onehot_test[:, 1], preds[:, 1])
        self.log("MAP/{}_{}_micro".format(stage, modality), micro_average_precision, prog_bar=True, sync_dist=True)
        self.log("MAP/{}_{}_macro".format(stage, modality), macro_average_precision, prog_bar=True, sync_dist=True)
        self.log("MAP/{}_{}_gesture".format(stage, modality), gesture_average_precision, prog_bar=True, sync_dist=True)
        
        all_losses = self.models_results[stage]['loss'].detach().cpu().numpy()
        self.log(f"{stage}_{modality}_loss", np.mean(all_losses), sync_dist=True)

            
    def _process_step(self, batch, phase):
        """
        Process a training, validation, or test step.

        Args:
        batch (tuple): The input batch data.
        phase (str): The phase of the model ('train', 'val', or 'test').

        Returns:
        torch.Tensor: The computed loss for the batch.
        """
        audio, skeleton, labels, _, speaker_frames = batch
        _, _, fused_preds = self.model(audio, skeleton)
        loss = self.compute_and_log_metrics(fused_preds, labels, phase, 'fused')
        self.models_results[phase]['labels'] = torch.cat([self.models_results[phase]['labels'].cpu(), labels.detach().cpu()])
        self.models_results[phase]['preds'] = torch.cat([self.models_results[phase]['preds'].cpu(), fused_preds.detach().float().cpu()])
        self.models_results[phase]['loss'] = torch.cat([self.models_results[phase]['loss'].cpu(), loss.detach().cpu().unsqueeze(0)])
        self.models_results[phase]['start_frames'] = torch.cat([self.models_results[phase]['start_frames'].cpu(), speaker_frames['start_frames'].detach().cpu()])
        self.models_results[phase]['end_frames'] = torch.cat([self.models_results[phase]['end_frames'].cpu(), speaker_frames['end_frames'].detach().cpu()])
        self.models_results[phase]['speaker_ID'] = torch.cat([self.models_results[phase]['speaker_ID'].cpu(), speaker_frames['speaker_ID'].detach().cpu()])
        self.models_results[phase]['pair_ID'] = torch.cat([self.models_results[phase]['pair_ID'].cpu(), speaker_frames['pair_ID'].detach().cpu()])

        return loss
    
    def training_step(self, batch, batch_idx):
        return self._process_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._process_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._process_step(batch, 'test')

    def _handle_epoch_end(self, phase, verbose=False, test_phase=False) -> None:
        """
        Handles the end of an epoch for training, validation, or testing.
        Args:
        phase (str): The phase of the model ('train', 'val', or 'test').
        verbose (bool): Whether to provide verbose output.
        """
        # Gather results from all processes
        for key in ['labels', 'preds', 'loss', 'start_frames', 'end_frames', 'speaker_ID', 'pair_ID']:
            self.models_results[phase][key] = self.all_gather(self.models_results[phase][key])

        # Report metrics and reset the results for the next epoch
        self.report_metrics(phase, 'fused')
        for key in ['labels', 'preds', 'loss', 'start_frames', 'end_frames', 'speaker_ID', 'pair_ID']:
            if test_phase:
                # convert the list to numpy array
                self.models_results[phase][key] = self.models_results[phase][key].detach().cpu().numpy()
            else:
                self.models_results[phase][key] = torch.Tensor()
            
    def on_train_epoch_end(self, outputs=None) -> None:
        train_metrics_values = self.train_metrics.compute()
        self.log_dict(train_metrics_values)
        self.train_metrics.reset()
        self._handle_epoch_end('train')
        
    def on_validation_epoch_end(self, outputs=None) -> None:
        val_metrics_values = self.val_metrics.compute()
        self.log_dict(val_metrics_values)
        self.val_metrics.reset()
        self._handle_epoch_end('val')
        
    def on_test_epoch_end(self, outputs=None) -> None:
        # gather the predictions and labels
        self._handle_epoch_end('test', test_phase=True)
   
    def configure_optimizers(self):
        # if self.model.vggish:
        #     optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # else:
            # check first if the model has audio_model
        if hasattr(self.model, 'audio_model'):
            audio_model_params = list(self.model.audio_model.parameters())
            audio_model_param_ids = set(id(p) for p in audio_model_params)
            base_params = [p for p in self.parameters() if id(p) not in audio_model_param_ids]
            optimizer = torch.optim.Adam([
                {'params': base_params},
                {'params': audio_model_params, 'lr': self.lr * 0.1}
            ], lr=self.lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        if self.scheduler == 'MultiStepLR':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[int(self.num_epoch * 0.8), int(self.num_epoch * 0.9)],
                gamma=0.1
            )
        if self.scheduler == "constant":
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0 ** epoch)
        elif self.scheduler == "plateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.2,
                    patience=self.patience,
                    min_lr=1e-7,
                    verbose=True
                )
        elif self.scheduler == "linear":
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=self.num_epoch
            )
        elif self.scheduler == "exponential":
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=0.95
            )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch", "monitor": "val_fused_loss"}]

def main(lr=1e-4):
    L.seed_everything(42)
    parser = get_parser()
    # load arg form config file
    fold = 0
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
                if k not in key:
                    print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()
    arg.feeder_args['fold'] = fold
    arg.feeder_args['vggish'] = arg.fusion_args['vggish']
    arg.feeder_args['speech_buffer'] = arg.fusion_args['speech_buffer']
    arg.fusion_args['sanity_check'] = arg.feeder_args['sanity_check']

    # Assuming `sequential_audio_dataset` is a list-like or NumPy array-like object with your data
    if arg.phase == 'test':
        large_cabb_test_dataset = TestFeeder(**arg.feeder_args)
        large_cabb_test_dataset_loader = torch.utils.data.DataLoader(
            large_cabb_test_dataset, 
            batch_size=arg.batch_size, 
            shuffle=False, 
            num_workers=4,
        )
    results = defaultdict(dict)
    models_directory = "tb_logs/"
    for fold in range(5):
        # Here, you would set up your configuration as before
        arg.feeder_args['fold']=fold
        arg.feeder_args['vggish'] = arg.fusion_args['vggish']
        arg.feeder_args['speech_buffer'] = arg.fusion_args['speech_buffer']
        arg.Experiment_name = default_arg['Experiment_name'].format(
        arg.fusion.split('.')[-1],
            fold,
            arg.lr,
            arg.feeder_args['subject_joint'],
            arg.feeder_args['gesture_unit'],
            arg.fine_tuned_audio_model,
            arg.fusion_args['vggish'],
            arg.fusion_args['speech_buffer'],
            arg.fusion_args['offset'],
            arg.scheduler,
            arg.fusion_args['encoder_for_audio'],
            arg.fusion_args['encoder_for_skeleton'],
            arg.use_crf,
            arg.batch_size,
            arg.feeder_args['sanity_check']
        )
        print(arg.Experiment_name)
        model = SpeechSkeletonModel(arg=arg)

        # Now, instead of using random_split, use the indices provided by KFold to create train and test datasets
        # Model 
        logger_name = arg.Experiment_name+'_inference_for_sho_data'
        tb_logger = TensorBoardLogger(models_directory, name=logger_name)

        data_type = arg.fusion_args['weights_path'].split('/')[1].split('_')[0]
  
        loggers = [tb_logger]
        wandb_name = 'fold_{}_{}_bsz_{}_lr_{}_time_{}'.format(fold, data_type, arg.batch_size, arg.lr, str(time.time()))
        # for Mounika, she does not have wandb access to our project, so it will be none 
        if arg.wandb_entity != "none" and False:
            experiment_info = vars(arg)
            project="GestureDetection_" 

            wandb_logger = WandbLogger(
                config=experiment_info,
                entity=arg.wandb_entity,
                project=project,
                name=wandb_name, # temporary solution for unique experiment names in wandb
                id=wandb_name
            )
            loggers.append(wandb_logger)
        
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        early_stopping = EarlyStopping(
            monitor="F1/val_fused_gesture",
            mode="max", # Since you are monitoring a f1
            patience=arg.early_stopping_patience,
            verbose=True
        )
        save_top_k = ModelCheckpoint(
            filename="{epoch}-{F1/val_fused_gesture:.2f}-{val_fused_loss:.2f}",
            monitor="F1/val_fused_gesture",
            save_top_k=1,
            every_n_epochs=1,  # Check at every epoch
            mode="max"         # Since you are monitoring a f1
        )
        if torch.cuda.is_available():
            trainer = L.Trainer(
                # gradient_clip_val=0.25, 
                max_epochs=arg.num_epoch, 
                logger=loggers, 
                accelerator="gpu", 
                devices=-1, 
                num_nodes=1,
                accumulate_grad_batches=arg.accumulate_grad_batches, 
                callbacks=[
                    lr_monitor, 
                    early_stopping, 
                    save_top_k
                    ], 
                strategy="ddp_find_unused_parameters_true",
                enable_progress_bar=False,
                precision="bf16",
                num_sanity_val_steps=2,
                default_root_dir=models_directory+arg.Experiment_name
                )
        else:
            trainer = L.Trainer(
                accelerator="cpu",
                max_epochs=arg.num_epoch, 
                logger=loggers, 
                accumulate_grad_batches=arg.accumulate_grad_batches, 
                callbacks=[
                    lr_monitor, 
                    early_stopping, 
                    save_top_k
                    ], 
                enable_progress_bar=False,
                num_sanity_val_steps=2,
                default_root_dir=models_directory+arg.Experiment_name
                )
        # Note that the training steps are removed from this script
        if arg.phase == 'test':
            # TODO: load the best model model
            trainer.test(model, large_cabb_test_dataset_loader, ckpt_path=models_directory+arg.Experiment_name+"/last_epoch.ckpt")
        results[fold]['labels'] = model.models_results['test']['labels']
        results[fold]['preds'] = model.models_results['test']['preds']
        results[fold]['loss'] = model.models_results['test']['loss']
        results[fold]['start_frames'] = model.models_results['test']['start_frames']
        results[fold]['end_frames'] = model.models_results['test']['end_frames']
        results[fold]['speaker_ID'] = model.models_results['test']['speaker_ID']
        results[fold]['pair_ID'] = model.models_results['test']['pair_ID']
    # save the results
    if arg.phase == 'train':
        results_path = models_directory+arg.Experiment_name+"/results.pkl"  
    elif arg.phase == 'test':
        results_path = models_directory+arg.Experiment_name+"/sho_test_results.pkl"
    print(f"Saving results to {results_path}")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
main()