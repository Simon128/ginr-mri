import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf, MISSING
from .inr_output import INROutput

@dataclass
class WeightModulatedINRConfig:
    name: str = "weight_modulated_inr"
    n_layer: int = 5
    hidden_dim: list[int] = MISSING
    use_bias: bool = True
    input_dim: int = 3
    output_dim: int = 3
    ff_sigma: int = 1024
    ff_dim: int = 120
    modulated_layers: list[int] = MISSING
    backbone_spatial_feature_dimensions: list[int] = MISSING
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
        self.mod_biases = nn.ParameterList()
        curr_features = self.config.ff_dim * 2
        mod_idx = 0

        for idx, c in enumerate(self.config.hidden_dim):
            if idx in self.config.modulated_layers:
                self.layers.append(nn.Identity())
                self.modulated_projection.append(nn.Sequential(
                    nn.Flatten(start_dim=2),
                    nn.Linear(config.backbone_spatial_feature_dimensions[mod_idx], c)
                ))
                # todo: correct init (xavier) instead of ones
                self.mod_biases.append(nn.Parameter(torch.ones((1, c))))
                mod_idx += 1
            else:
                self.layers.append(nn.Sequential(
                    nn.Linear(in_features=curr_features, out_features=c),
                    nn.ReLU()
                ))
                self.modulated_projection.append(nn.Identity())
                self.mod_biases.append(nn.Identity())
            curr_features = c

        self.layers.append(nn.Linear(in_features=curr_features, out_features=self.config.output_dim))
        self.modulated_projection.append(nn.Identity())

        # deterministic_transinr
        log_freqs = torch.linspace(0, np.log(config.ff_sigma), config.ff_dim // self.config.input_dim)
        self.ff_linear = torch.exp(log_freqs).to("cuda") # todo
        #self.ff_linear = 2 ** torch.linspace(0, config.ff_sigma, config.ff_dim // self.config.input_dim)
        #self.ff_linear = torch.randn(self.config.input_dim, config.ff_dim).to("cuda") * config.ff_sigma  # scaler

    def compute_loss(self, preds, targets, reduction="mean"):
        assert reduction in ["mean", "sum", "none"]
        batch_size = preds.shape[0]
        sample_mses = torch.reshape((preds - targets) ** 2, (batch_size, -1)).mean(dim=-1)

        if reduction == "mean":
            total_loss = sample_mses.mean()
        elif reduction == "sum":
            total_loss = sample_mses.sum()
        else:
            total_loss = sample_mses

        return total_loss

    def forward(self, coord: torch.Tensor, features: list[torch.Tensor], target: torch.Tensor | None = None):
        fourier_features = torch.matmul(coord.unsqueeze(-1), self.ff_linear.unsqueeze(0))
        fourier_features = fourier_features.view(*coord.shape[:-1], -1)
        #fourier_features = torch.matmul(coord, self.ff_linear)
        #fourier_features = fourier_features * np.pi
        
        fourier_features = [torch.cos(fourier_features), torch.sin(fourier_features)]
        z = torch.cat(fourier_features, dim=-1)

        for idx, (layer, mod) in enumerate(zip(self.layers, self.modulated_projection)):
            if idx in self.config.modulated_layers:
                feat_idx = self.config.modulated_layers_to_backbone_features_map[idx]
                feat = features[feat_idx]
                if self.config.normalize_weights:
                    feat = F.normalize(feat, dim=1)
                proj_feat = mod(feat)
                proj_feat = proj_feat.view(z.shape[0], *z.shape[2:], -1)
                if self.config.use_bias:
                    ones = torch.ones(*z.shape[:-1], 1, device=z.device)
                    z = torch.cat([z, ones], dim=-1)
                    proj_feat = torch.cat((proj_feat, self.mod_biases[idx].unsqueeze(0).repeat((z.shape[0], 1, 1))), dim = 1)

                z = torch.bmm(z, proj_feat)
            else:
                z = layer(z)

        output = INROutput(prediction=z)

        if target is not None:
            output.loss = self.compute_loss(z, target)

        return output
