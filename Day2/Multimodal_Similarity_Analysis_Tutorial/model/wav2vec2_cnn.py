import torch
import torch.nn as nn
from collections import OrderedDict

from pytorch_lightning import LightningModule
from typing import List, Optional


class CNN1D(LightningModule):
    def __init__(
            self,
            in_channels: int,
            len_seq: int,
            out_channels: List = [32, 64, 128],
            kernel_sizes: List = [7, 5, 3],
            stride: int = 1,
            padding: int = 1,
            pool_padding: int = 0,
            pool_size: int = None,
            pool_stride = None,
            p_drop: float = 0.2,
            pretrained: Optional[str] = None,
            average_weights: Optional[int] = None,
            **kwargs,
    ):
        """
        1D-Convolutional Network with three layers.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input data.
        len_seq : int or str
            Length of the input sequence to use with CNN. If "full", the whole incoming sequence is used.
        out_channels : list of int
            List containing the number of channels in the convolutional layers.
        kernel_sizes : list of int
            List containing the sizes of the convolutional kernels.
        stride : int
            Size of the stride.
        padding : int
            Unused, just to compute the out size.
        pool_padding : int
            Padding for maxpooling.
        pool_size : int
            Size of the maxpooling.
        p_drop : float
            Dropout value.
        pretrained : str
            Path to pretrained model.
        """
        super(CNN1D, self).__init__()
        assert len(out_channels) == len(kernel_sizes), "out_channels and kernel_size list lengths should match"

        self.len_seq = len_seq
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes

        self.name = 'cnn1d'
        self.num_layers = len(out_channels)
        self.convolutional_blocks = OrderedDict()
        self.average_weights = average_weights

        self._all_channels = [self.in_channels] + self.out_channels

        if pool_size is not None and pool_stride is not None : self.convolutional_blocks['Avgpool']=nn.AvgPool1d(kernel_size=pool_size, stride=pool_stride)


        if self.average_weights is not None: self.weighted_average = nn.Parameter(torch.ones(self.average_weights))

        for i in range(1, len(self._all_channels)):
            self.convolutional_blocks.update({f'conv_block{i}': nn.Sequential(OrderedDict([
                (
                    f'conv{i}',
                    nn.Conv1d(
                        in_channels=self._all_channels[i - 1],
                        out_channels=self._all_channels[i],
                        kernel_size=self.kernel_sizes[i - 1],
                        stride=stride,
                        padding=padding,
                    )
                ),
                (f'relu{i}', nn.ReLU(inplace=True)),
                (f'dropout{i}', nn.Dropout(p=p_drop))
            ]))})

        self.convolutional_blocks = nn.Sequential(self.convolutional_blocks)

        self.out_size = out_channels[-1]

        if pretrained is not None:
            loaded_checkpoint = torch.load(pretrained.replace('\\', '/'))
            # Pytorch lightning checkpoints store more values, and state dict needs to be accessed
            # using "state_dict" key, whereas default pytorch checkpoints store state_dict only
            if "state_dict" in loaded_checkpoint:
                loaded_checkpoint = loaded_checkpoint["state_dict"]
            self.load_state_dict(loaded_checkpoint)
            print(f'CNN1D: succesfully loaded weights from {pretrained}')
        else:
            print("CNN: NO pretrained weights loaded")

        # self.save_hyperparameters()

    def forward(self, x, valid_mask=None):
        """
        Forward pass of the CNN1D.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying CNN layers.
        """
        if self.len_seq != "full":
            x = x[..., :self.len_seq]
        if self.average_weights is not None:
            x = x*self.weighted_average[None, :, None, None]
            x = torch.sum(x, 1) / torch.sum(self.weighted_average)

        x = self.convolutional_blocks(x)

        if valid_mask is None:
            x = torch.mean(x,dim=-1)
        else:
            x = masked_mean(x, valid_mask, dim=2)

        return x


def masked_mean(tensor, mask, dim):
    """from: https://discuss.pytorch.org/t/equivalent-of-numpy-ma-array-to-mask-values-in-pytorch/53354/5
    """
    masked = torch.mul(tensor, mask)  # Apply the mask using an element-wise multiply
    return masked.sum(dim=dim) / mask.sum(dim=dim)