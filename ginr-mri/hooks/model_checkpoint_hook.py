from datetime import datetime
import pathlib
from torch.optim import Optimizer
from typing import TYPE_CHECKING
import torch
import torch.nn as nn
import os
import torch.distributed as torchdist
from torch.utils.data import Dataset

from ..utils.dist import Rank0Barrier
from .hook import Hook

if TYPE_CHECKING:
    from ..engine import Engine

class ModelCheckpointHook(Hook):
    def __init__(self, priority: int = 0, directory: str | None = None, frequency_per_epoch: int = 1) -> None:
        super().__init__(priority)
        if directory:
            self.directory = directory
        else:
            self.directory = os.path.join(pathlib.Path(__file__).parent.resolve(), "runs", datetime.now().isoformat())
        self.frequency_per_epoch = frequency_per_epoch

    def pre_fit(
        self, 
        engine: 'Engine',
        model: nn.Module,
        train_dataset: Dataset, 
        val_dataset: Dataset,
        optimizer: Optimizer,
        **kwargs
    ) -> dict | None:
        self.model = model
        self.optimizer = optimizer

    def post_training_epoch(
        self, 
        engine: 'Engine', 
        epoch: int,
        **kwargs
    ) -> dict | None:
        if epoch % self.frequency_per_epoch != 0: return None
        if torchdist.is_initialized() and torchdist.get_world_size() > 1:
            # we will have DDP wrapped model => we need to do model.module.state_dict()
            if torchdist.get_rank() == 0:
                self.model.eval()
                torch.save({
                    "model": self.model.module.state_dict(), 
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch
                }, os.path.join(self.directory, f"{epoch}.ckpt"))
                self.model.train()
                return {"model_checkpoint_path" : os.path.join(self.directory, f"{epoch}.ckpt")}
        else:
            self.model.eval()
            torch.save({
                "model": self.model.state_dict(), 
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch
            }, os.path.join(self.directory, f"{epoch}.ckpt"))
            self.model.train()
            return {"model_checkpoint_path" : os.path.join(self.directory, f"{epoch}.ckpt")}
