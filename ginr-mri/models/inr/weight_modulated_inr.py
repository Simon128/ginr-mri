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
    ff_dim: int = 128
    modulated_layers: list[int] = MISSING
    backbone_feature_dimensions: list[int] = MISSING
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
        curr_features = self.config.input_dim

        for idx, c in enumerate(self.config.hidden_dim):
            if idx in self.config.modulated_layers:
                self.layers.append(nn.Identity())
                self.modulated_projection.append(nn.Linear(config.backbone_feature_dimensions[idx], curr_features * c))
                self.modulated_projection.append(nn.ReLU())
                self.modulated_projection.append(nn.Linear(curr_features * c, curr_features * c))
            else:
                self.layers.append(nn.Linear(in_features=curr_features, out_features=c))
                self.modulated_projection.append(nn.Identity())

            self.layers.append(nn.ReLU())
            curr_features = c

        self.layers.append(nn.Linear(in_features=curr_features, out_features=self.config.output_dim))

        # deterministic_transinr
        log_freqs = torch.linspace(0, np.log(config.ff_sigma), config.ff_dim // self.config.input_dim)
        self.ff_linear = torch.exp(log_freqs)

    def forward(self, coord: torch.Tensor, features: list[torch.Tensor]):
        fourier_features = torch.matmul(coord.unsqueeze(-1), self.ff_linear.unsqueeze(0))
        fourier_features = fourier_features.view(*coord.shape[:-1], -1)

        for idx, (layer, mod) in enumerate(zip(self.layers, self.modulated_projection)):
            test = 5


class HypoNet(nn.Module):
    def forward(self, coord, modulation_params_dict=None):
        """Computes the value for each coordination
        Note: `assert outputs.shape[:-1] == coord.shape[:-1]`

        Args
            coord (torch.Tensor): input coordinates to be inferenced
            modulation_params_dict (torch.nn.Parameters): the dictionary of modulation parameters.
                the keys have to be matched with the keys of self.params_dict
                If `modulation_params_dict` given, self.params_dict is modulated before inference.
                If `modulation_params_dict=None`, the inference is conducted based on base params.

        Returns
            outputs (torch.Tensor): evaluated values by INR
        """
        if modulation_params_dict is not None:
            self.check_valid_param_keys(modulation_params_dict)

        batch_size, coord_shape, input_dim = coord.shape[0], coord.shape[1:-1], coord.shape[-1]
        coord = coord.view(batch_size, -1, input_dim)  # flatten the coordinates
        hidden = self.fourier_mapping(coord) if self.use_ff else coord

        for idx in range(self.config.n_layer):
            param_key = f"linear_wb{idx}"
            base_param = einops.repeat(self.params_dict[param_key], "n m -> b n m", b=batch_size)

            if (modulation_params_dict is not None) and (param_key in modulation_params_dict.keys()):
                modulation_param = modulation_params_dict[param_key]
            else:
                if self.config.use_bias:
                    modulation_param = torch.ones_like(base_param[:, :-1])
                else:
                    modulation_param = torch.ones_like(base_param)

            if self.config.use_bias:
                ones = torch.ones(*hidden.shape[:-1], 1, device=hidden.device)
                hidden = torch.cat([hidden, ones], dim=-1)

                base_param_w, base_param_b = base_param[:, :-1, :], base_param[:, -1:, :]

                if self.ignore_base_param_dict[param_key]: 
                    base_param_w = 1.
                param_w = base_param_w * modulation_param
                if self.normalize_weight:
                    param_w = F.normalize(param_w, dim=1)
                modulated_param = torch.cat([param_w, base_param_b], dim=1)
            else:
                if self.ignore_base_param_dict[param_key]:
                    base_param = 1.
                if self.normalize_weight:
                    modulated_param = F.normalize(base_param * modulation_param, dim=1)
                else:
                    modulated_param = base_param * modulation_param

            hidden = torch.bmm(hidden, modulated_param)

            if idx < (self.config.n_layer - 1):
                hidden = self.activation(hidden)

        outputs = hidden + self.output_bias
        outputs = outputs.view(batch_size, *coord_shape, -1)
        return outputs

    @torch.no_grad()
    def compute_modulated_params_dict(self, modulation_params_dict):
        """Computes the modulated parameters from the modulation parameters.

        Args:
            modulation_params_dict (dict[str, torch.Tensor]): The dictionary of modulation parameters.

        Returns:
            modulated_params_dict (dict[str, torch.Tensor]): The dictionary of modulated parameters.
                Contains keys identical to the keys of `self.params_dict` and corresponding per-instance params.
        """
        self.check_valid_param_keys(modulation_params_dict)

        batch_size = list(modulation_params_dict.values())[0].shape[0]

        modulated_params_dict = {}

        for idx in range(self.config.n_layer):
            param_key = f"linear_wb{idx}"
            base_param = einops.repeat(self.params_dict[param_key], "n m -> b n m", b=batch_size)

            if (modulation_params_dict is not None) and (param_key in modulation_params_dict.keys()):
                modulation_param = modulation_params_dict[param_key]
            else:
                if self.config.use_bias:
                    modulation_param = torch.ones_like(base_param[:, :-1])
                else:
                    modulation_param = torch.ones_like(base_param)

            if self.config.use_bias:
                base_param_w, base_param_b = base_param[:, :-1, :], base_param[:, -1:, :]
                if self.ignore_base_param_dict[param_key]:
                    base_param_w = 1.0
                param_w = base_param_w * modulation_param
                if self.normalize_weight:
                    param_w = F.normalize(param_w, dim=1)
                modulated_param = torch.cat([param_w, base_param_b], dim=1)
            else:
                if self.ignore_base_param_dict[param_key]:
                    base_param = 1.0
                if self.normalize_weight:
                    modulated_param = F.normalize(base_param * modulation_param, dim=1)
                else:
                    modulated_param = base_param * modulation_param

            modulated_params_dict[param_key] = modulated_param

        return modulated_params_dict

    def forward_with_params(self, coord, params_dict):
        """Computes the value for each coordinate, according to INRs with given modulated parameters.

        Args:
            coord (torch.Tensor): Input coordinates in shape (B, ...).
            params_dict (dict[str, torch.Tensor]): The dictionary of modulated parameters.
                Each parameter in `params_dict` must be per-instance (must be in shape (B, fan_in, fan_out)).

        Returns:
            outputs (torch.Tensor): Evaluated values by INRs with per-instance params `params_dict`.
        """
        self.check_valid_param_keys(params_dict)

        batch_size, coord_shape, input_dim = coord.shape[0], coord.shape[1:-1], coord.shape[-1]
        coord = coord.view(batch_size, -1, input_dim)  # flatten the coordinates
        hidden = self.fourier_mapping(coord) if self.use_ff else coord

        for idx in range(self.config.n_layer):
            param_key = f"linear_wb{idx}"

            modulated_param = params_dict[param_key]
            assert batch_size == modulated_param.shape[0]  # params_dict must contain per-sample params!!

            if self.config.use_bias:
                ones = torch.ones(*hidden.shape[:-1], 1, device=hidden.device)
                hidden = torch.cat([hidden, ones], dim=-1)

            hidden = torch.bmm(hidden, modulated_param)

            if idx < (self.config.n_layer - 1):
                hidden = self.activation(hidden)

        outputs = hidden + self.output_bias
        outputs = outputs.view(batch_size, *coord_shape, -1)

        return outputs
