import torch.nn as nn
import torch
from dataclasses import dataclass
from omegaconf import OmegaConf, MISSING

from .backbones import build_backbone

@dataclass
class BaseModelConfig:
    name: str = "base"
    backbone: str = "nvidia2018"
    inr = MISSING

    @classmethod
    def create(cls, config):
        defaults = OmegaConf.structured(cls())
        config = OmegaConf.merge(defaults, config)
        return config

class BaseModel(nn.Module):
    def __init__(self, config: BaseModelConfig) -> None:
        super().__init__()
        self.backbone = build_backbone(config.backbone)

    def forward(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, target = batch
        z = self.backbone(x)
        return None
