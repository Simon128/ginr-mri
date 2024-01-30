from dataclasses import dataclass, field
import torch

from .inr import INROutput

@dataclass
class ModelOutput:
    backbone_out: list[torch.Tensor] 
    inr_out: INROutput 
    loss: torch.Tensor | None = None
    subsampled_coords: torch.Tensor | None = None
    subsampled_targets: torch.Tensor | None = None
    additional: dict = field(default_factory=dict)
