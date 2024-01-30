import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from omegaconf import OmegaConf, MISSING

@dataclass
class SubSamplerConfig:
    type: str | None = None
    ratio: float = 0.1

    @classmethod
    def create(cls, config):
        defaults = OmegaConf.structured(cls())
        config = OmegaConf.merge(defaults, config)
        return config

class SubSampler:
    r"""
    `SubSampler` is designed to randomly select the subset of coordinates
    to efficiently train the transformer that generates INRs of given data.
    In the training loop, `subcoord_idx` generates sub-coordinates according to the `subsampler_config`.
    Then, in the traning loop, `subsample_coords` and `subsample_xs` slices the subset of features
    according to the generated `subcoord_idx`.
    """

    def __init__(self, subsamper_config: SubSamplerConfig):
        self.config = subsamper_config
        if self.config.type is not None and self.config.ratio == 1.0:
            self.config.type = None

    def subsample_coords_idx(self, xs):
        if self.config.type is None:
            subcoord_idx = None
        elif self.config.type == "random":
            subcoord_idx = self.subsample_random_idx(xs, ratio=self.config.ratio)
        else:
            raise NotImplementedError
        return subcoord_idx

    def subsample_random_idx(self, xs, ratio=None):
        batch_size = xs.shape[0]
        spatial_dims = list(xs.shape[2:])

        subcoord_idx = []
        num_spatial_dims = np.prod(spatial_dims)
        num_subcoord = int(num_spatial_dims * ratio)
        for idx in range(batch_size):
            rand_idx = torch.randperm(num_spatial_dims, device=xs.device)
            rand_idx = rand_idx[:num_subcoord]
            subcoord_idx.append(rand_idx.unsqueeze(0))
        return torch.cat(subcoord_idx, dim=0)

    @staticmethod
    def subsample_coords(coords, subcoord_idx=None):
        if subcoord_idx is None:
            return coords

        batch_size = coords.shape[0]
        sub_coords = []
        coords = coords.view(batch_size, -1, coords.shape[-1])
        for idx in range(batch_size):
            sub_coords.append(coords[idx : idx + 1, subcoord_idx[idx]])
        sub_coords = torch.cat(sub_coords, dim=0)
        return sub_coords

    @staticmethod
    def subsample_xs(xs, subcoord_idx=None):
        if subcoord_idx is None:
            return xs

        batch_size = xs.shape[0]
        permute_idx_range = [i for i in range(2, xs.ndim)]  # note: xs is originally channel-fist data format
        xs = xs.permute(0, *permute_idx_range, 1)  # convert xs into channel last type

        xs = xs.reshape(batch_size, -1, xs.shape[-1])
        sub_xs = []
        for idx in range(batch_size):
            sub_xs.append(xs[idx : idx + 1, subcoord_idx[idx]])
        sub_xs = torch.cat(sub_xs, dim=0)
        return sub_xs
