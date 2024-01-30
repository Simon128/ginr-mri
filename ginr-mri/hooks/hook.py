from torch.optim import Optimizer
from typing import TYPE_CHECKING
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset

from ..models import ModelOutput

if TYPE_CHECKING:
    from ..engine import Engine

class Hook:
    def __init__(self, priority: int = 0) -> None:
        self.priority = priority

    def pre_fit(
        self, 
        engine: 'Engine',
        model: nn.Module,
        train_dataset: Dataset, 
        val_dataset: Dataset,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: Optimizer,
        **kwargs
    ) -> dict | None:
        pass

    def post_fit(
        self, 
        engine: 'Engine',
        model: nn.Module,
        train_dataset: Dataset, 
        val_dataset: Dataset,
        optimizer: Optimizer,
        **kwargs
    ) -> dict | None:
        pass

    def pre_validation_epoch(
        self, 
        engine: 'Engine', 
        epoch: int,
        **kwargs
    ) -> dict | None:
        pass

    def post_validation_epoch(
        self, 
        engine: 'Engine', 
        epoch: int,
        **kwargs
    ) -> dict | None:
        pass

    def pre_training_epoch(
        self, 
        engine: 'Engine', 
        epoch: int,
        **kwargs
    ) -> dict | None:
        pass

    def post_training_epoch(
        self, 
        engine: 'Engine', 
        epoch: int,
        **kwargs
    ) -> dict | None:
        pass

    def pre_model_step(
        self, 
        engine: 'Engine', 
        iteration_step: int, 
        epoch: int,
        batch: dict, 
        stage: str,
        **kwargs
    ) -> dict | None:
        pass

    def post_model_step(
        self, 
        engine: 'Engine', 
        iteration_step: int, 
        epoch: int,
        output: ModelOutput, 
        stage: str,
        **kwargs
    ) -> dict | None:
        pass
