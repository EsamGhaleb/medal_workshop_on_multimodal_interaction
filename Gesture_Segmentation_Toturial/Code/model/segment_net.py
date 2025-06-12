import copy
from collections import defaultdict

import torch
import numpy as np
import torch.optim as optim
from torch import nn
from pytorch_lightning import LightningModule
from torchmetrics import MetricCollection
from sklearn.preprocessing import label_binarize
from scipy.special import softmax
from model.losses import WeightedFocalLoss
from torchmetrics.classification import (
    BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
)
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.metrics import roc_auc_score, average_precision_score

class SegmentNet(LightningModule):
    def __init__(
            self,
            backbone,
            optimizer,
            args,
            lr,
            backbone_ckpt = None,
            
    ) -> None:
        super().__init__()
                
        self.backbone = backbone
        if backbone_ckpt is not None:
            weights = torch.load(backbone_ckpt)
            # weights structure from in masked recognition object_net.backbone.[backbone weight name]
            backbone_weights = [".".join(key.split(".")[2:]) for key in weights["state_dict"].keys() if "backbone" in key]
            self.backbone.load_state_dict(backbone_weights)
            print(f"Weights loaded from {backbone_ckpt}")

        self.optimizer = optimizer
        self.args = args
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        
        metrics = MetricCollection([
            BinaryAccuracy(),
            BinaryPrecision(),
            BinaryRecall(),
            BinaryF1Score()
        ])
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.classes = ['non-gesture', 'gesture']
        self.criterion = WeightedFocalLoss(gamma=2, alpha=0.25)
        self.metrics = ['Accuracy', 'Precision', 'Recall', 'F1Score']
        self.modalities = ['audio', 'skeleton', 'fused']
        self.models_results = defaultdict(dict)
        phases = ['train', 'val', 'test']
        keys = ['labels', 'preds', 'loss', 'frame_IDs', 'speaker_ID', 'pair_ID', 'end_frames', 'start_frames']
        for phase in phases:
            for key in keys:
                self.models_results[phase][key] = torch.Tensor()

    def forward(self, processed_batch):
        x = self.backbone(skeleton = processed_batch["skeleton"]["orig"], speech_waveform = processed_batch["speech"]["orig"],utterances = processed_batch["utterance"])
        return x

    def _shared_step(self, processed_batch, phase):
        if 'utterance' not in processed_batch:
            processed_batch['utterance'] = None
        if 'skeleton' not in processed_batch:
            processed_batch['skeleton'] = {}
            processed_batch['skeleton']['orig'] = None
        if 'speech' not in processed_batch: 
            processed_batch['speech'] = {}
            processed_batch['speech']['orig'] = None
        else:
            processed_batch['speech']['orig'] = processed_batch['speech']['orig'].float()
        if phase == 'train' and 'skeleton' in processed_batch and self.args.apply_skeleton_augmentations:
            if 'view1' in processed_batch['skeleton']:
                processed_batch['skeleton']['orig'] = processed_batch['skeleton']['view1'].float()
        elif phase in ['val', 'test'] and 'skeleton' in processed_batch:
            processed_batch['skeleton']['orig'] = processed_batch['skeleton']['orig'].float()
            
        predictions = self(processed_batch)
        loss = self.criterion(predictions, processed_batch["label"])
        labels = processed_batch["label"]
        self.log(f'{phase}/segmentation_loss', loss)
        self.models_results[phase]['labels'] = torch.cat([self.models_results[phase]['labels'].cpu(), labels.detach().cpu()])
        self.models_results[phase]['preds'] = torch.cat([self.models_results[phase]['preds'].cpu(), predictions.detach().float().cpu()])
        self.models_results[phase]['loss'] = torch.cat([self.models_results[phase]['loss'].cpu(), loss.detach().cpu().unsqueeze(0)])
        self.models_results[phase]['frame_IDs'] = torch.cat([self.models_results[phase]['frame_IDs'].cpu(), processed_batch['frame_IDs'].detach().cpu()])
        self.models_results[phase]['speaker_ID'] = torch.cat([self.models_results[phase]['speaker_ID'].cpu(), processed_batch['speaker_ID'].cpu()])
        self.models_results[phase]['pair_ID'] = torch.cat([self.models_results[phase]['pair_ID'].cpu(), processed_batch['pair_ID'].cpu()])
        self.models_results[phase]['start_frames'] = torch.cat([self.models_results[phase]['start_frames'].cpu(), processed_batch['start_frames'].detach().cpu()])
        self.models_results[phase]['end_frames'] = torch.cat([self.models_results[phase]['end_frames'].cpu(), processed_batch['end_frames'].detach().cpu()])
        
        return loss, predictions, labels
    
    def _handle_epoch_end(self, phase, verbose=False, test_phase=False) -> None:
        """
        Handles the end of an epoch for training, validation, or testing.
        Args:
        phase (str): The phase of the model ('train', 'val', or 'test').
        verbose (bool): Whether to provide verbose output.
        """
        # Gather results from all processes
        for key in ['labels', 'preds', 'loss', 'frame_IDs', 'speaker_ID', 'pair_ID', 'end_frames', 'start_frames']:
            self.models_results[phase][key] = self.all_gather(self.models_results[phase][key])

        # Report metrics and reset the results for the next epoch
        self.report_metrics(phase, '')
        for key in ['labels', 'preds', 'loss', 'frame_IDs', 'speaker_ID', 'pair_ID', 'end_frames', 'start_frames']:
            if test_phase:
                # convert the list to numpy array
                self.models_results[phase][key] = self.models_results[phase][key].detach().cpu().numpy()
            else:
                self.models_results[phase][key] = torch.Tensor()
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
        
    def _shared_eval_step(self, batch, phase):
        _, out, labels = self._shared_step(batch, phase)
        # Compute metrics for metrics between outputs and ground-truth labels, e.g., accuracy, f1-score, etc.

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss, out, labels = self._shared_step(batch, "test")
        return loss

    def on_train_epoch_end(self, outputs=None) -> None:
        # train_metrics_values = self.train_metrics.compute()
        # self.log_dict(train_metrics_values)
        # self.train_metrics.reset()
        self._handle_epoch_end('train')
        
    def on_validation_epoch_end(self, outputs=None) -> None:
        # val_metrics_values = self.val_metrics.compute()
        # self.log_dict(val_metrics_values)
        # self.val_metrics.reset()
        self._handle_epoch_end('val')
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=self.args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=10,
            gamma=0.5
        )
        return {
            "optimizer": optimizer,
            "scheduler": scheduler
        }


