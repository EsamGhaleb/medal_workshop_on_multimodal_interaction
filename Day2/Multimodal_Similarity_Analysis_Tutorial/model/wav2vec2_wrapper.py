from typing import List

import torch
import torchaudio
import torch.nn as nn
from pytorch_lightning import LightningModule
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from model.wav2vec2_cnn import CNN1D


class Wav2Vec2Wrapper(LightningModule):
    """ Wraps Wav2Vec 2.0 model from torch audio into the Lightning Module
    """
    def __init__(
            self,
            w2v2_type: str = 'multilingual',
            freeze: bool = True
    ):
        super().__init__()
        # TODO: add other configurations of wav2vec2.0 and integrate with Wav2VecCNN
        self.w2v2_type = w2v2_type
        if w2v2_type == 'base':
            bundle = torchaudio.pipelines.WAV2VEC2_BASE
            self.w2v2 = bundle.get_model()
        elif w2v2_type == 'multilingual':
            bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M
            self.w2v2 = bundle.get_model()
        elif w2v2_type == 'dutch_finetuned':
            model_name = 'jonatasgrosman/wav2vec2-large-xlsr-53-dutch'
            self.w2v2 = Wav2Vec2Model.from_pretrained(model_name)
        else:
            raise ValueError("wrong type of W2V2 model provided")
        
        if freeze:
            for param in self.w2v2.parameters():
                param.requires_grad = False

    def forward(self, x, lengths=None):
        if self.w2v2_type == 'base':
            internal_features, valid_lengths = self.w2v2.extract_features(x.squeeze(axis=1), lengths=lengths)
            output_local_encoder = self.w2v2.encoder.feature_projection.forward(
                self.w2v2.feature_extractor(
                    x.squeeze(axis=1), length=lengths
                )[0]
            )
            internal_features.append(output_local_encoder)
        elif self.w2v2_type == 'multilingual':
            internal_features, valid_lengths = self.w2v2.extract_features(x.squeeze(axis=1), lengths=lengths, num_layers=24)
        elif self.w2v2_type == 'dutch_finetuned':
            output= self.w2v2(x.squeeze(1), output_hidden_states=True, return_dict=True)
            # print(output.last_hidden_state.shape)
            # print(output.extract_features.shape)
            # internal_features = torch.cat(output.hidden_states, dim=1)
            internal_features = output.hidden_states
            valid_lengths = None
        return internal_features, valid_lengths


class Wav2Vec2CNN(LightningModule):
    """ CNN applied on top of the wav2vec 2.0 features and weighted average
        applied to different transformer layers from wav2vec.
        Adapted from: https://arxiv.org/pdf/2104.03502.pdf
    """
    def __init__(
            self,
            length_samples: int = 8,
            sample_rate: int = 16000,
            w2v2_type: str = 'multilingual',
            freeze: bool = True,
            out_channels: List = [128, 128],
            kernel_sizes: List = [1, 1],
            pretrained=None,
            peft_config=None,
            len_seq=99,
            apply_cnns=True
    ):
        super().__init__()
        
        self.wav2vec2 = Wav2Vec2Wrapper(w2v2_type=w2v2_type, freeze=freeze)
        if self.wav2vec2.w2v2_type == 'base':
            num_layers = 13
            feature_size = 768
        elif self.wav2vec2.w2v2_type == 'multilingual':
            num_layers = 24
            feature_size = 1024
        elif self.wav2vec2.w2v2_type == 'dutch_finetuned':
            num_layers = 25
            feature_size = 1024

        self.weighted_average = nn.Parameter(torch.ones(num_layers))  # for learnable weights
        self.cnn = CNN1D(
            in_channels=feature_size,
            len_seq="full",  # do not trim signal along temporal dimension before using cnn
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            padding=0,
            stride=1,
            pool_size=1,
            pool_padding=0
        )
        # self.out_size = self.compute_out_size(length=length_samples, sample_rate=sample_rate)

        self.save_hyperparameters()

        if pretrained is not None:
            loaded_checkpoint = torch.load(pretrained.replace('\\', '/'))
            # Pytorch lightning checkpoints store more values, and state dict needs to be accessed
            # using "state_dict" key, whereas default pytorch checkpoints store state_dict only
            if "state_dict" in loaded_checkpoint:
                loaded_checkpoint = loaded_checkpoint["state_dict"]
            self.load_state_dict(loaded_checkpoint)
            print(f'Wav2vec2CNN: succesfully loaded weights from {pretrained}')
        else:
            print("Wav2vec2CNN: NO pretrained weights loaded")

    def compute_out_size(self, length, sample_rate):
        dummy_input = torch.rand((64, 1, int(length * sample_rate))).to(self.device)
        out = self.forward(dummy_input)

        return torch.numel(out)

    def forward(self, x, lengths=None):
        speech_outputs = {
            'local': [],
            'global': [],
            'attention_mask': []
        }
        # pass data through wav2vec2
        w2v2_features, valid_lengths = self.wav2vec2(x, lengths=lengths)
        # process the features and apply weighted average
        embedding = torch.stack(w2v2_features, axis=1)
        embedding = embedding * self.weighted_average[None, :, None, None]
        embedding = torch.sum(embedding, 1) / torch.sum(self.weighted_average)
        # setting channels first
        embedding = torch.transpose(embedding, 1, 2)
        
        speech_outputs['local'] = embedding.permute(0, 2, 1)

        # zero-ing the invalid lengths
        if valid_lengths is not None:
            mask = torch.arange(embedding.shape[-1]).expand(len(valid_lengths), embedding.shape[-1]) < valid_lengths.unsqueeze(1)
            mask_before_cnn = mask.unsqueeze(1).repeat(1, embedding.shape[1], 1).float()
            mask_after_cnn = mask.unsqueeze(1).repeat(1, self.cnn.out_channels[-1], 1).float()
            embedding = embedding * mask_before_cnn

        # apply cnn layers
        outs = self.cnn(embedding, mask_after_cnn if valid_lengths is not None else None)
       
        speech_outputs['global'] = outs
        speech_outputs['attention_mask'] = valid_lengths
        return speech_outputs

if __name__ == "__main__":
    # Test the model
    model = Wav2Vec2CNN(length_samples=0.75, len_seq=49)
    print(model)
    x = torch.rand((10, 1, 8000))
    out = model(x)
    print(out.shape)
    print(out)
    print("done")   