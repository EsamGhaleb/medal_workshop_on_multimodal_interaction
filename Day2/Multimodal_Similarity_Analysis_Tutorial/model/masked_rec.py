import copy
from collections import defaultdict

import torch
import torch.optim as optim
from pytorch_lightning import LightningModule
from model.losses import NTXent, NTXentMM
from model.semantic_pool import BertjePoolingModule
from lib.model.model_object import ObjectNet
from model.decouple_gcn_attn_sequential import Model as STGCN
from transformers import AutoTokenizer, AutoModel
from model.wav2vec2_wrapper import Wav2Vec2CNN
weights_path = '27_2_finetuned/joint_finetuned.pt'

from lib.model.loss import (
    loss_2d_weighted,
    loss_angle,
    loss_angle_velocity,
    loss_limb_gt,
    loss_limb_var,
    loss_mpjpe,
    loss_velocity,
    n_mpjpe,
    loss_bone
)


class MaskedReconstructionModel(LightningModule):
    def __init__(
            self,
            backbone,
            optimizer,
            args,
            lr,
            has_3d: bool = False,
            has_gt: bool = True,
            modalities: list = ['skeleton', 'semantic'],  # 'speech', 'skeleton', 'semantic', 'image'
            loss_type: str = ['masked_reconstruction', 'contrastive'],  # 'masked_reconstruction', 'contrastive', or 'mm_contrastive'
            freeze_bertje: bool = True,
            w2v2_type: str = 'multilingual',
            skeleton_backbone: str = 'stgcn'
    ) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.args = args
        self.lr = lr
        self.has_3d = has_3d
        self.has_gt = has_gt
        self.modalities = modalities
        self.use_contrastive = "contrastive" in loss_type
        self.loss_type = loss_type
        self.use_masked_reconstruction = "masked_reconstruction" in loss_type
        self.hidden_dim = args.hidden_dim
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.skeleton_backbone = skeleton_backbone
        if skeleton_backbone == 'stgcn':
            self.object_net = STGCN(device=device)
            self.object_net.load_state_dict(torch.load(weights_path))
        else:
            self.object_net = ObjectNet(
                backbone=backbone,
                dim_rep=args.dim_rep,
                version='embed',
                hidden_dim=args.hidden_dim
            )

        if "contrastive" in loss_type or "mm_contrastive" in loss_type:
            self.skeleton_projection = torch.nn.Sequential(
                torch.nn.Linear(args.dim_rep, self.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            )

        # NOTE: unimodal contrastive is applied to skeleton only
        if "contrastive" in loss_type and "skeleton" in self.modalities:
            self.criterion = NTXent(batch_size=args.batch_size, n_views=2, temperature=args.temp)

        # NOTE: multimodal contrastive is applied to semantic and/or speech embeddings and skeleton embeddings
        if "semantic" in modalities or 'speech' in modalities:
            self.mm_criterion = NTXentMM(batch_size=args.batch_size, temperature=args.temp)

        if 'speech' in modalities:
            self.speech_model = Wav2Vec2CNN(w2v2_type=w2v2_type)
            # speech projection head
            self.speech_projection = torch.nn.Sequential(
                torch.nn.Linear(256, self.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            )

        # Load the BERTje tokenizer and model
        if 'semantic' in modalities:
            self.bertje_model = BertjePoolingModule(args)
            # semantic head
            self.semantic_projection = torch.nn.Sequential(
                torch.nn.Linear(args.bertje_dim, self.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            )

    def forward(self, x):
        x = self.object_net.forward(x, get_rep=False)
        return x

    # def get_cls_embedding(self, sentences):
    #     # get model device
    #     device = next(self.bertje_model.parameters()).device
    #     inputs = self.tokenizer(
    #         sentences,
    #         return_tensors="pt",
    #         truncation=True,
    #         padding=True,
    #         max_length=512
    #     ).to(device)
    #     outputs = self.bertje_model(**inputs)
    #     # take the average embeddings
    #     average_embeddings = torch.mean(outputs.last_hidden_state, dim=1)
    #     # cls_embedding = outputs.last_hidden_state[:, 0, :]
    #     return average_embeddings

    def _process_batch(self, batch):
        if "skeleton" in self.modalities:
            orig_skeletons = batch["skeleton"]["orig"] if "orig" in batch["skeleton"] else None
            skeletons_1 = batch["skeleton"]["view1"] if "view1" in batch["skeleton"] else None
            skeletons_2 = batch["skeleton"]["view2"] if "view2" in batch["skeleton"] else None
        else:
            skeletons_1 = None
            skeletons_2 = None
            orig_skeletons = None
        if "speech" in self.modalities:
            speech_1 = batch["speech"]["view1"] if "view1" in batch["speech"] else None
            orig_speech = batch["speech"]["orig"] if "orig" in batch["speech"] else None
            speech_1 = batch["speech"]["view1"] if "view1" in batch["speech"] else None
            speech_2 = batch["speech"]["view2"] if "view2" in batch["speech"] else None
            speech_lengths = batch["speech"]["lengths"] if "lengths" in batch["speech"] else None
        else:
            orig_speech = None
            speech_1 = None
            speech_2 = None
            speech_lengths = None
        if "semantic" in self.modalities:
            utterances = batch["utterance"]
        else:
            utterances = None

        label = batch["label"] if "label" in batch else None

        batch_input = batch['skeleton']['orig']
        batch_gt = batch['skeleton']['orig'] if self.has_gt else None
        conf = None if self.has_3d else batch_input[:, :, :, 2:]
        with torch.no_grad():
            if self.args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if not self.has_3d:
                # For 2D data, weight/confidence is at the last channel
                conf = copy.deepcopy(batch_input[:, :, :, 2:])
            if self.args.rootrel:
                batch_gt = batch_gt - batch_gt[:, :, 0:1, :]
            else:
                # Place the depth of first frame root to 0.
                batch_gt[:, :, :, 2] = batch_gt[:, :, :, 2] - batch_gt[:, 0:1, 0:1, 2]
            if self.skeleton_backbone != 'stgcn':
                if self.args.mask or self.args.noise:
                    batch_input = self.args.aug.augment2D(
                        batch_input,
                        noise=(self.args.noise and self.has_gt),
                        mask=self.args.mask
                    )
        return {
            "batch_input": batch_input,
            "batch_gt": batch_gt,
            "conf": conf,
            "orig_skeletons": orig_skeletons,
            "skeletons_1": skeletons_1,
            "skeletons_2": skeletons_2,
            "orig_speech": orig_speech,
            "speech_1": speech_1,
            "speech_2": speech_2,
            "labels": label,
            "speech_lengths": speech_lengths,
            "utterance": utterances
        }

    def _compute_loss_2d(self, predicted_pos, batch_gt, conf, phase):
        losses = defaultdict()
        loss_2d_proj = loss_2d_weighted(predicted_pos, batch_gt, conf)
        loss_velocity_2d = loss_velocity(predicted_pos, batch_gt)
        loss_bone_2d = loss_bone(predicted_pos, batch_gt)
        losses[f'{phase}/joints_loss'] = loss_2d_proj
        losses[f'{phase}/velocity_loss'] = loss_velocity_2d
        losses[f'{phase}/bone_loss'] = loss_bone_2d
        total_loss = (loss_2d_proj + loss_velocity_2d + loss_bone_2d) / 3
        losses[f'{phase}/total_loss'] = total_loss
        return total_loss, losses

    def _compute_loss_3d(self, predicted_3d_pos, batch_gt, phase):
        losses = defaultdict()
        loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
        loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
        loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
        loss_lv = loss_limb_var(predicted_3d_pos)
        loss_lg = loss_limb_gt(predicted_3d_pos, batch_gt)
        loss_a = loss_angle(predicted_3d_pos, batch_gt)
        loss_av = loss_angle_velocity(predicted_3d_pos, batch_gt)
        loss_total = loss_3d_pos + \
            self.args.lambda_scale * loss_3d_scale + \
            self.args.lambda_3d_velocity * loss_3d_velocity + \
            self.args.lambda_lv * loss_lv + \
            self.args.lambda_lg * loss_lg + \
            self.args.lambda_a * loss_a + \
            self.args.lambda_av * loss_av
        losses[f'{phase}/3d_pos'] = loss_3d_pos
        losses[f'{phase}/3d_scale'] = loss_3d_scale
        losses[f'{phase}/3d_velocity'] = loss_3d_velocity
        losses[f'{phase}/lv'] = loss_lv
        losses[f'{phase}/lg'] = loss_lg
        losses[f'{phase}/angle'] = loss_a
        losses[f'{phase}/angle_velocity'] = loss_av
        losses[f'{phase}/total_loss'] = loss_total
        return loss_total, losses

    def _compute_contrastive_loss(self, processed_batch, phase):
        loss_dict = defaultdict()
        processed_batch["orig_skeletons"] = processed_batch["orig_skeletons"].float() if processed_batch["orig_skeletons"] is not None else None
        processed_batch["skeletons_1"] = processed_batch["skeletons_1"].float() if processed_batch["skeletons_1"] is not None else None
        processed_batch["skeletons_2"] = processed_batch["skeletons_2"].float() if processed_batch["skeletons_2"] is not None else None

        if "skeleton" in self.modalities and "contrastive" in self.loss_type:
            assert processed_batch["skeletons_1"] is not None
            assert processed_batch["skeletons_2"] is not None
            skeletons = torch.cat([processed_batch["skeletons_1"], processed_batch["skeletons_2"]], dim=0)
            skeleton_features = self.skeleton_projection(self.object_net(skeletons))
            skeleton_features = torch.nn.functional.normalize(skeleton_features, dim=1)
            skeleton_loss, skeleton_pos, skeleton_neg = self.criterion(skeleton_features)
            # make dictionary for logging
            loss_dict[f'{phase}/unimodal_contrastive_loss'] = skeleton_loss
            loss_dict[f'{phase}/pos'] = skeleton_pos
            loss_dict[f'{phase}/neg'] = skeleton_neg

        if "semantic" in self.modalities or "speech" in self.modalities:
            assert processed_batch["orig_skeletons"] is not None
            skeletons = processed_batch["orig_skeletons"]
            skeleton_features = self.skeleton_projection(self.object_net(skeletons))
            skeleton_features = torch.nn.functional.normalize(skeleton_features, dim=1)

        if "semantic" in self.modalities:
            semantic_embeddings = self.bertje_model(processed_batch["utterance"])
            semantic_embeddings = self.semantic_projection(semantic_embeddings)
            semantic_embeddings = torch.nn.functional.normalize(semantic_embeddings, dim=1)
            mm_loss, mm_pos, mm_neg = self.mm_criterion(semantic_embeddings, skeleton_features)
            loss_dict[f'{phase}/mm_semantic_contrastive_loss'] = mm_loss
            loss_dict[f'{phase}/mm_semantic_pos'] = mm_pos
            loss_dict[f'{phase}/mm_semantic_neg'] = mm_neg

        if 'speech' in self.modalities:
            speech_embeddings = self.speech_model(processed_batch["orig_speech"], processed_batch["speech_lengths"])
            speech_embeddings = torch.nn.functional.normalize(speech_embeddings, dim=1)
            mm_loss, mm_pos, mm_neg = self.mm_criterion(speech_embeddings, skeleton_features)
            loss_dict[f'{phase}/mm_speech_contrastive_loss'] = mm_loss
            loss_dict[f'{phase}/mm_speech_pos'] = mm_pos
            loss_dict[f'{phase}/mm_speech_neg'] = mm_neg

        return loss_dict

    def _shared_step(self, batch, phase):
        processed_batch = self._process_batch(batch)
        contrastive_loss = 0
        masked_loss = 0
        if self.use_masked_reconstruction:
            predicted_pos = self(processed_batch["batch_input"])
            masked_loss, masked_losses_d = self._compute_loss_3d(predicted_pos, processed_batch["batch_gt"], phase) if (
                self.has_3d
            ) else self._compute_loss_2d(predicted_pos, processed_batch["batch_gt"], processed_batch["conf"], phase)
            for loss_name in masked_losses_d:
                self.log(loss_name, masked_losses_d[loss_name])
        if self.use_contrastive or "semantic" in self.modalities or "speech" in self.modalities:
            contrastive_loss_d = self._compute_contrastive_loss(processed_batch, phase)
            for loss_name in contrastive_loss_d:
                self.log(loss_name, contrastive_loss_d[loss_name])
                if "loss" in loss_name:
                    contrastive_loss += contrastive_loss_d[loss_name]

        loss = masked_loss + contrastive_loss
        self.log(f'{phase}/total_loss', loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.object_net.parameters()),
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
    
    # def on_after_backward(self):
    #     self.check_gradients()
    #     self.print_gradient_norms()

    # def check_gradients(self):
    #     for name, param in self.named_parameters():
    #         if param.requires_grad and param.grad is None:
    #             print(f"No gradient for {name}")

    # def print_gradient_norms(self):
    #     for name, param in self.named_parameters():
    #         if param.requires_grad and param.grad is not None:
    #             grad_norm = param.grad.norm().item()
    #             self.log(f'{name}_grad_norm', grad_norm)
    #             print(f"Gradient norm for {name}: {grad_norm}")
