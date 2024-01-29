from torchvision.transforms.v2 import Transform
from typing import Any, Dict
import torch


class CenterCrop3D(Transform):
    def __init__(self, size: tuple[int, int, int]) -> None:
        super().__init__()
        self.size = size

    def _transform(self, inpt: torch.Tensor, params: Dict[str, Any]) -> Any:
        '''
        expecting (C, D, H, W)
        '''
        if len(inpt.shape) != 4:
            raise ValueError(f"3D object of shape {inpt.shape} not supported")

        _, D, H, W = inpt.shape

        d_diff = (D - self.size[0]) // 2
        h_diff = (H - self.size[1]) // 2
        w_diff = (W - self.size[2]) // 2

        return inpt[:, d_diff:self.size[0]+d_diff, h_diff:self.size[1]+h_diff, w_diff:self.size[2]+w_diff]
