import torch
import torch.nn as nn
from dataclasses import dataclass

from omegaconf import OmegaConf, MISSING

@dataclass
class CoordSamplerConfig:
    coord_range: list[float] = MISSING

    @classmethod
    def create(cls, config):
        defaults = OmegaConf.structured(cls())
        config = OmegaConf.merge(defaults, config)
        return config

def shape2coordinate(spatial_shape, batch_size, min_value=-1.0, max_value=1.0, upsample_ratio=1, device=None):
    coords = []
    for num_s in spatial_shape:
        num_s = int(num_s * upsample_ratio)
        _coords = (0.5 + torch.arange(num_s, device=device)) / num_s
        _coords = min_value + (max_value - min_value) * _coords
        coords.append(_coords)
    coords = torch.meshgrid(*coords, indexing="ij")  
    coords = torch.stack(coords, dim=-1)
    ones_like_shape = (1,) * coords.ndim
    coords = coords.unsqueeze(0).repeat(batch_size, *ones_like_shape)
    return coords


class CoordSampler(nn.Module):
    """Generates coordinate inputs according to the given data type.
    This class can be more implemented according to the coordinates sampling strategy.
    """

    def __init__(self, config: CoordSamplerConfig):
        super().__init__()
        self.config = config
        self.coord_range = config.coord_range

    def base_sampler(self, xs, coord_range=None, upsample_ratio=1.0, device=None):
        coord_range = self.coord_range if coord_range is None else coord_range
        min_value, max_value = coord_range

        batch_size, spatial_shape = xs.shape[0], xs.shape[2:]

        return shape2coordinate(spatial_shape, batch_size, min_value, max_value, upsample_ratio, device)

    def forward(self, xs, coord_range=None, upsample_ratio=1.0, device=None):
        coords = self.base_sampler(xs, coord_range, upsample_ratio, device)
        return coords
