import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf, MISSING

@dataclass
class WeightModulatedINRConfig:
    n_layer: int = 5
    hidden_dim: list[int] = MISSING
    use_bias: bool = True
    input_dim: int = 3
    output_dim: int = 3
    ff_sigma: int = 1024
    ff_dim: int = 120
    modulated_layers: list[int] = MISSING
    backbone_feature_dimensions: list[int] = MISSING
    modulated_layers_to_backbone_features_map: dict = MISSING
    normalize_weights: bool = True

    @classmethod
    def create(cls, config):
        defaults = OmegaConf.structured(cls())
        config = OmegaConf.merge(defaults, config)
        return config

class WeightModulatedINR(nn.Module):
    def __init__(self, config: WeightModulatedINRConfig) -> None:
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList()
        self.modulated_projection = nn.ModuleList()
        curr_features = self.config.ff_dim

        for idx, c in enumerate(self.config.hidden_dim):
            if idx in self.config.modulated_layers:
                self.layers.append(nn.Identity())
                # todo
                self.modulated_projection.append(nn.Sequential(
                    nn.Conv3d(256, 128, 1, padding="same"),
                    nn.ReLU(),
                    nn.Conv3d(128, 64, 1, padding="same"),
                    nn.ReLU(),
                    nn.Conv3d(64, 32, 1, stride=2),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(960*32, c * curr_features)
                ))
            else:
                self.layers.append(nn.Sequential(
                    nn.Linear(in_features=curr_features, out_features=c),
                    nn.ReLU()
                ))
                self.modulated_projection.append(nn.Identity())
            curr_features = c

        self.layers.append(nn.Linear(in_features=curr_features, out_features=self.config.output_dim))
        self.modulated_projection.append(nn.Identity())

        # deterministic_transinr
        log_freqs = torch.linspace(0, np.log(config.ff_sigma), config.ff_dim // self.config.input_dim)
        self.ff_linear = torch.exp(log_freqs).to("cuda") # todo

    def forward(self, coord: torch.Tensor, features: list[torch.Tensor], target: torch.Tensor | None = None):
        fourier_features = torch.matmul(coord.unsqueeze(-1), self.ff_linear.unsqueeze(0))
        z = fourier_features.view(*coord.shape[:-1], -1)

        for idx, (layer, mod) in enumerate(zip(self.layers, self.modulated_projection)):
            if idx in self.config.modulated_layers:
                feat_idx = self.config.modulated_layers_to_backbone_features_map[idx]
                feat = features[feat_idx]
                proj_feat = mod(feat)
                z = torch.bmm(z.unsqueeze(1), proj_feat.view(*z.shape, -1)).squeeze(1)
            else:
                z = layer(z)

        output_dict = {
            "output": z
        }
        if target:
            output_dict["loss"] = 0

        return output_dict
