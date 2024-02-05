import torch.nn as nn
import torch
from dataclasses import dataclass
from omegaconf import OmegaConf, MISSING
from typing import Any
import logging
import torch.distributed as dist

from .backbones import build_backbone
from .inr import build_inr
from ..utils import CoordSampler, CoordSamplerConfig, SubSampler, SubSamplerConfig
from .model_output import ModelOutput
from .inr.inr_output import INROutput
from .latent_transform import build_latent_transform

logger = logging.getLogger(__name__)

@dataclass
class BaseModelConfig:
    name: str = "base"
    backbone: Any = MISSING
    latent_transform: Any = None
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
        self.backbone = build_backbone(config.backbone.name, config.backbone)
        if config.latent_transform is not None:
            self.latent_transform = build_latent_transform(config.latent_transform.name, config.latent_transform)
        else:
            self.latent_transform = None
        self.inr = build_inr(config.inr.name, config.inr)
        self.coord_sampler = CoordSampler(config.coord_sampler)
        self.subsampler = SubSampler(config.subsampler)

    def full_prediction(self, batch: tuple[torch.Tensor, torch.Tensor, Any], verbose = False):
        x, target, _ = batch
        z = self.backbone(x)
        if self.latent_transform:
            z = self.latent_transform(z)
        coord = self.sample_coord_input(target)
        batch_size, depth, height, _, _ = coord.shape
        channels = target.shape[1]
        depthwise_outputs = []
        if verbose: 
            length = depth * height
            log_step = length // 10
            counter = 0
        for d in range(depth):
            heightwise_outputs = []
            for h in range(height):
                _in = coord[:, d, h, ...].reshape((batch_size, -1, 3))
                _out = self.inr(_in, z).prediction.moveaxis(-1, 1)
                heightwise_outputs.append(_out.view((batch_size, channels, 1, 1, -1)))
                if verbose:
                    if counter % log_step == 0:
                        logger.info(f"Full INR construction [{int(counter/length * 100)}%]")
                    counter += 1

            depthwise_outputs.append(torch.cat(heightwise_outputs, dim=3))

        if verbose:
            logger.info("Full INR construction [100%]")

        full_prediction = torch.cat(depthwise_outputs, dim=2)
        inr_loss = self.inr.compute_loss(full_prediction, target)
        output = ModelOutput(
            loss=inr_loss,
            backbone_out=z,
            inr_out=INROutput(loss=inr_loss, prediction=full_prediction),
            subsampled_coords=coord,
            subsampled_targets=target,
            loss_dict={"loss": inr_loss}
        )
        return output

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

    def forward(self, batch: tuple[torch.Tensor, torch.Tensor, Any], coord: torch.Tensor | None = None):
        x, target, _ = batch
        if dist.is_initialized():
            x = x.to(dist.get_rank())
            target = target.to(dist.get_rank())
        else:
            x = x.to("cuda")
            target = target.to("cuda")
        z = self.backbone(x)
        if self.latent_transform:
            z = self.latent_transform(z)
        if not coord:
            coord = self.sample_coord_input(target)

        subsample_coord_idxs = self.subsampler.subsample_coords_idx(target)
        coord = self.subsampler.subsample_coords(coord, subsample_coord_idxs)
        target = self.subsampler.subsample_xs(target, subsample_coord_idxs)

        inr_output = self.inr(coord, z, target)
        output = ModelOutput(
            loss=inr_output.loss,
            backbone_out=z,
            inr_out=inr_output,
            subsampled_coords=coord,
            subsampled_targets=target,
            loss_dict={"loss": inr_output.loss}
        )
        return output
