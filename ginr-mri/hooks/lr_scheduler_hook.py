from typing import TYPE_CHECKING
import logging
import torch.nn as nn
import torch.optim.lr_scheduler as torch_scheduler
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from .hook import Hook

if TYPE_CHECKING:
    from ..engine import Engine

logger = logging.getLogger(__name__)

OPTIONS = ['LambdaLR', 'MultiplicativeLR', 'StepLR', 'MultiStepLR', 'ConstantLR', 'LinearLR',
           'ExponentialLR', 'SequentialLR', 'CosineAnnealingLR', 'ChainedScheduler', 'ReduceLROnPlateau',
           'CyclicLR', 'CosineAnnealingWarmRestarts', 'OneCycleLR', 'PolynomialLR', 'LRScheduler']

def build_scheduler(scheduler_name: str, **kwargs):
    if scheduler_name not in OPTIONS:
        err_msg = f"{scheduler_name} is not a supported LR scheduler"
        logger.error(err_msg)
        raise ValueError(err_msg)

    return getattr(torch_scheduler, scheduler_name)(**kwargs)

class LRSchedulerHook(Hook):
    def __init__(self, priority: int = 0, scheduler_name: str = 'ReduceLROnPlateau', **kwargs) -> None:
        super().__init__(priority)
        self.scheduler_name = scheduler_name
        self.kwargs = kwargs

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
        self.scheduler = build_scheduler(self.scheduler_name, **{"optimizer": optimizer, **self.kwargs})

    def post_training_epoch(
        self, 
        engine: 'Engine', 
        epoch: int,
        **kwargs
    ) -> dict | None:
        self.scheduler.step()

