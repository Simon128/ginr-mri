import torch
import torch.nn as nn
from dataclasses import dataclass, field
from omegaconf import OmegaConf, MISSING

@dataclass
class ConvolutionalLTConfigSingle:
    n_layer: int = MISSING
    strides: list[int] = MISSING
    padding: list[int | str] = MISSING
    channels: list[int] = MISSING
    kernel_size: list[int] = MISSING
    backbone_idx: int = MISSING
    input_channels: int = MISSING
    batch_norm: bool = True

    @classmethod
    def create(cls, config):
        defaults = OmegaConf.structured(cls())
        config = OmegaConf.merge(defaults, config)
        return config

@dataclass
class ConvolutionalLTConfig:
    name: str = "convolutional"
    levels: list[ConvolutionalLTConfigSingle] = field(default_factory=lambda: list())

    @classmethod
    def create(cls, config):
        defaults = OmegaConf.structured(cls())
        config = OmegaConf.merge(defaults, config)
        return config

class ConvolutionalLT(nn.Module):
    def __init__(self, config: ConvolutionalLTConfig) -> None:
        super().__init__()
        self.config = config

        self.model = nn.ModuleList()
        for level in self.config.levels:
            layers = []
            in_channels = level.input_channels
            for idx in range(level.n_layer):
                if level.batch_norm:
                    layers.append(nn.BatchNorm3d(in_channels))
                layers.append(nn.ReLU())
                layers.append(nn.Conv3d(
                    in_channels, level.channels[idx], level.kernel_size[idx], 
                    stride=level.strides[idx], padding=level.padding[idx]
                ))
                in_channels = level.channels[idx]

            self.model.append(nn.Sequential(*layers))

    def forward(self, backbone_feat: list[torch.Tensor]):
        output = []

        for idx, level in enumerate(self.config.levels):
            feat = backbone_feat[level.backbone_idx]
            output.append(self.model[idx](feat))

        return output
