from typing import OrderedDict
import torch.nn as nn
import torch

class Nvidia2018EncoderBlock(nn.Module):
    def __init__(self, in_channels, downsample = False) -> None:
        super().__init__()
        if downsample:
            channels = in_channels * 2 
            conv_1_stride = 2
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=channels, kernel_size=1, stride=2),
                nn.BatchNorm3d(channels)
            )
        else:
            channels = in_channels 
            conv_1_stride = 1
            self.downsample = None
        self.block = nn.Sequential(OrderedDict([
            ('batch_norm_1', nn.BatchNorm3d(in_channels)),
            ('relu_1', nn.ReLU()),
            ('conv_1', nn.Conv3d(
                in_channels=in_channels, 
                out_channels=channels, 
                kernel_size=3, 
                stride=conv_1_stride,
                padding=1 if conv_1_stride > 1 else "same")
             ),
            ('batch_norm_2', nn.BatchNorm3d(channels)),
            ('relu_2', nn.ReLU()),
            ('conv_2', nn.Conv3d(
                in_channels=channels, 
                out_channels=channels, 
                kernel_size=3, 
                stride=1,
                padding="same")
             )
        ]))

    def forward(self, x: torch.Tensor):
        residual = x
        z = self.block(x)
        if self.downsample:
            residual = self.downsample(residual)
        return z + residual

class Nvidia2018(nn.Module):
    '''
        Adapted from the encoder of 
        "3D MRI brain tumor segmentation using autoencoder regularization" by Andriy Myronenko
        (see https://arxiv.org/abs/1810.11654)

        changes to the original:
        - using batch norm instead of group norm

        output shapes (e.g. brats (d, h, w) == (160, 192, 128)):
        - (b, c, d, h, w)
        - (b, 32, 160, 192, 128)
        - (b, 64, 80, 96, 64)
        - (b, 128, 40, 48, 32)
        - (b, 256, 20, 24, 16)
    '''
    def __init__(self) -> None:
        super().__init__()
        self.first_layer = nn.Conv3d(in_channels=4, out_channels=32, kernel_size=3, padding="same")
        self.block_1 = Nvidia2018EncoderBlock(in_channels=32, downsample=False)
        self.block_2_a = Nvidia2018EncoderBlock(in_channels=32, downsample=True)
        self.block_2_b = Nvidia2018EncoderBlock(in_channels=64, downsample=False)
        self.block_3_a = Nvidia2018EncoderBlock(in_channels=64, downsample=True)
        self.block_3_b = Nvidia2018EncoderBlock(in_channels=128, downsample=False)
        self.block_4_a = Nvidia2018EncoderBlock(in_channels=128, downsample=True)
        self.block_4_b = Nvidia2018EncoderBlock(in_channels=256, downsample=False)
        self.block_4_c = Nvidia2018EncoderBlock(in_channels=256, downsample=False)
        self.block_4_d = Nvidia2018EncoderBlock(in_channels=256, downsample=False)

    def forward(self, x: torch.Tensor):
        block_outputs = [x]

        z_1 = self.first_layer(x)
        z_1 = self.block_1(z_1)
        block_outputs.append(z_1)

        z_2 = self.block_2_a(z_1)
        z_2 = self.block_2_b(z_2)
        block_outputs.append(z_2)

        z_3 = self.block_3_a(z_2)
        z_3 = self.block_3_b(z_3)
        block_outputs.append(z_3)

        z_4 = self.block_4_a(z_3)
        z_4 = self.block_4_b(z_4)
        z_4 = self.block_4_c(z_4)
        z_4 = self.block_4_d(z_4)
        block_outputs.append(z_4)

        return block_outputs

