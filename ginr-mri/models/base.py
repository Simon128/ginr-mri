import torch.nn as nn
import torch
from dataclasses import dataclass
from omegaconf import OmegaConf, MISSING
from typing import Any

from .backbones import build_backbone
from .inr import build_inr
from ..utils import CoordSampler, CoordSamplerConfig, SubSampler, SubSamplerConfig
from .model_output import ModelOutput

@dataclass
class BaseModelConfig:
    name: str = "base"
    backbone: str = "nvidia2018"
    inr: Any = MISSING
    coord_sampler: CoordSamplerConfig = MISSING
    subsampler: SubSamplerConfig = MISSING

    @classmethod
    def create(cls, config):
        defaults = OmegaConf.structured(cls())
        config = OmegaConf.merge(defaults, config)
        return config


class BaseModel(nn.Module):
    def __init__(self, config: BaseModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = build_backbone(config.backbone)
        self.inr = build_inr(config.inr.name, config.inr)
        self.coord_sampler = CoordSampler(config.coord_sampler)
        self.subsampler = SubSampler(config.subsampler)

    def sample_coord_input(self, x, coord_range=None, upsample_ratio=1.0, device=None):
        device = device if device is not None else x.device
        coord_inputs = self.coord_sampler(x, coord_range, upsample_ratio, device)
        # add noises to the coordinates for avoid overfitting on training coordinates.
        (B, *shape, input_dim) = coord_inputs.shape
        unif_noises = torch.rand(B, *shape, input_dim, device=coord_inputs.device)
        len_coord_range = self.config.coord_sampler.coord_range[1] - self.config.coord_sampler.coord_range[0]
        coord_noises = (unif_noises - 0.5) * len_coord_range / shape[0]
        coord_inputs = coord_inputs + coord_noises
        return coord_inputs

    def forward(self, batch: tuple[torch.Tensor, torch.Tensor], coord: torch.Tensor | None = None):
        x, target = batch
        z = self.backbone(x)
        if not coord:
            coord = self.sample_coord_input(target).squeeze()
        subsample_coord_idxs = self.subsampler.subsample_coords_idx(target)
        coord = self.subsampler.subsample_coords(coord, subsample_coord_idxs)
        target = self.subsampler.subsample_xs(target, subsample_coord_idxs)
        inr_output = self.inr(coord, z, target)
        output = ModelOutput(
            loss=inr_output.loss,
            backbone_out=z,
            inr_out=inr_output,
            subsampled_coords=coord,
            subsampled_targets=target
        )
        return output
