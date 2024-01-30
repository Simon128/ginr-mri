from typing import TYPE_CHECKING
import torch
import os
from torch.optim import Optimizer
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter
import pathlib
from datetime import datetime

from ..models import ModelOutput
from .hook import Hook

if TYPE_CHECKING:
    from ..engine import Engine

class TensorboardHook(Hook):
    def __init__(self, priority: int = 0, directory: str | None = None, batch_log_freq: int = 50) -> None:
        super().__init__(priority)
        if directory:
            self.directory = directory
        else:
            self.directory = os.path.join(pathlib.Path(__file__).parent.resolve(), "runs", datetime.now().isoformat())
        self.train_writer = SummaryWriter(log_dir=os.path.join(self.directory, "training"))
        self.val_writer = SummaryWriter(log_dir=os.path.join(self.directory, "validation"))
        self.batch_log_freq = batch_log_freq

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
        self.train_size = len(train_dataloader)

    def post_model_step(
        self, 
        engine: 'Engine', 
        iteration_step: int, 
        epoch: int,
        output: ModelOutput, 
        stage: str,
        **kwargs
    ) -> dict | None:
        for k, v in output.loss_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.clone().detach().cpu().item()
            if stage == "train":
                global_step = epoch * self.train_size + iteration_step
                if global_step % self.batch_log_freq == 0:
                    self.train_writer.add_scalar(k, v, global_step)

    def post_validation_epoch(
        self, 
        engine: 'Engine', 
        epoch: int,
        **kwargs
    ) -> dict | None:
        metrics = kwargs.get("metric")
        if metrics is None:
            return
        for k, v in metrics.items():
            self.val_writer.add_scalar(f"epoch/{k}", v, epoch)

    def post_training_epoch(
        self, 
        engine: 'Engine', 
        epoch: int,
        **kwargs
    ) -> dict | None:
        metrics = kwargs.get("metric")
        if metrics is None:
            return
        for k, v in metrics.items():
            self.train_writer.add_scalar(f"epoch/{k}", v, epoch)
