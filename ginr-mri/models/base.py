import torch.nn as nn
import torch
from dataclasses import dataclass
from omegaconf import OmegaConf, MISSING
from typing import Any

from .backbones import build_backbone
from .inr import build_inr

@dataclass
class BaseModelConfig:
    name: str = "base"
    backbone: str = "nvidia2018"
    inr: Any = MISSING

    @classmethod
    def create(cls, config):
        defaults = OmegaConf.structured(cls())
        config = OmegaConf.merge(defaults, config)
        return config

class BaseModel(nn.Module):
    def __init__(self, config: BaseModelConfig) -> None:
        super().__init__()
        self.backbone = build_backbone(config.backbone)
        self.inr = build_inr(config.inr.name, config.inr.args)

    def forward(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, target = batch
        z = self.backbone(x)
        self.inr(torch.tensor([0.1, 0.2, 0.3]).unsqueeze(0).repeat(2, 1).to(x.device), z, target)
        return None
