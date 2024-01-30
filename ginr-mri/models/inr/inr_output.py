from dataclasses import dataclass
import torch

@dataclass
class INROutput:
    prediction: torch.Tensor
    loss: torch.Tensor | None = None
