import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf, MISSING
from .inr_output import INROutput

@dataclass
class LocalityAwareINRConfig:
    name: str = "locality_aware_inr"
    n_layer: int = 5
    hidden_dim: list[int] = MISSING
    use_bias: bool = True
    coord_dim: int = MISSING
    attentive_embedding: int = MISSING
    attention_on_backbone_idx: list[int] = MISSING
    backbone_spatial_feature_dimensions: list[int] = MISSING
    attention_heads: int = MISSING
    input_dim: int = MISSING
    output_dim: int = 1
    ff_sigma: int = 1024
    ff_dim: int = 120

    @classmethod
    def create(cls, config):
        defaults = OmegaConf.structured(cls())
        config = OmegaConf.merge(defaults, config)
        return config

class FourierLinear(nn.Module):
    def __init__(self, bandwidth: int, ff_dim: int, inr_channels: int, coord_dim: int) -> None:
        super().__init__()
        self.bandwidth = bandwidth
        self.inr_channels = inr_channels
        self.coord_dim = coord_dim
        self.ff_dim = ff_dim
        n = ff_dim // (2 * coord_dim)
        self.log_freqs = torch.linspace(1, np.log(bandwidth), n).unsqueeze(0)
        self.log_freqs = torch.exp(self.log_freqs)

        self.device_set = False

        self.h_F = nn.Sequential(
            nn.Linear(ff_dim, inr_channels),
            nn.ReLU()
        )

    def forward(self, coords: torch.Tensor):
        if not self.device_set:
            self.log_freqs = self.log_freqs.to(coords.device)

        fourier_features = torch.matmul(coords.unsqueeze(-1), self.log_freqs)
        fourier_features = fourier_features.view(*coords.shape[:-1], -1)
        fourier_features = fourier_features * np.pi
        fourier_features = [torch.cos(fourier_features), torch.sin(fourier_features)]
        z = torch.cat(fourier_features, dim=-1)
        return self.h_F(z)


class LocalityAwareINR(nn.Module):
    def __init__(self, config: LocalityAwareINRConfig) -> None:
        super().__init__()
        self.config = config

        self.locality_fourier = FourierLinear(bandwidth=128, ff_dim=240, inr_channels=256, coord_dim=3)
        self.locality_attention = nn.MultiheadAttention(
            self.config.attentive_embedding,
            num_heads=config.attention_heads,
            batch_first=True
        )
        self.query_proj = nn.Linear(256, 256)
        self.key_proj = nn.Linear(256, 256)
        self.value_proj = nn.Linear(256, 256)
        self.inr_fourier_1 = FourierLinear(bandwidth=256, ff_dim=240, inr_channels=256, coord_dim=3)
        self.band_1_in = nn.Linear(256, 256)
        self.band_1_out = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.inr_fourier_2 = FourierLinear(bandwidth=128, ff_dim=240, inr_channels=256, coord_dim=3)
        self.band_2_in = nn.Linear(256, 256)
        self.band_2_out = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.final_band1 = nn.Linear(256, 1)
        self.final_band2 = nn.Linear(256, 1)

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
        z = self.locality_fourier(coord)
        _f = features[-1].flatten(start_dim=2).permute((0, 2, 1))
        z = self.locality_attention(self.query_proj(z), self.key_proj(_f), self.value_proj(_f))[0]

        int_band1 = self.band_1_in(z)
        int_band1 = int_band1 + self.inr_fourier_1(coord)
        band1 = self.band_1_out(F.relu(int_band1))

        int_band2 = self.band_2_in(z)
        int_band2 = F.relu(int_band2 + self.inr_fourier_2(coord))
        int_band2 = int_band2 + band1
        band2 = self.band_2_out(F.relu(int_band2))

        _out = self.final_band2(band2) + self.final_band1(band1)
        _out = F.sigmoid(_out)

        output = INROutput(prediction=_out)

        if target is not None:
            output.loss = self.compute_loss(_out, target)

        return output
