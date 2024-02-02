import torch
from typing import TYPE_CHECKING

from .hook import Hook
from ..models import ModelOutput
from ..metrics.inr_metrics import *

if TYPE_CHECKING:
    from ..engine import Engine

class INRMetricsHook(Hook):
    def __init__(self, priority: int = 0) -> None:
        super().__init__(priority)

    def pre_validation_epoch(
        self, 
        engine: 'Engine', 
        epoch: int,
        **kwargs
    ) -> dict | None:
        self.val_loss_dict_stack = {}
        self.val_metrics_stack = {}
        self.current_val_batch = None

    def pre_training_epoch(
        self, 
        engine: 'Engine', 
        epoch: int,
        **kwargs
    ) -> dict | None:
        self.train_loss_dict_stack = {}
        self.train_metrics_stack = {}
        self.current_train_batch = None

    def pre_model_step(
        self, 
        engine: 'Engine', 
        iteration_step: int, 
        epoch: int,
        batch: dict, 
        stage: str,
        **kwargs
    ) -> dict | None:
        if stage == "val":
            self.current_val_batch = batch
        if stage == "train":
            self.current_train_batch = batch

    def post_model_step(
        self, 
        engine: 'Engine', 
        iteration_step: int, 
        epoch: int,
        output: ModelOutput, 
        stage: str,
        **kwargs
    ) -> dict | None:
        loss = output.inr_out.loss
        loss = loss.clone().detach().cpu().item()
        if stage == "train":
            psnr = compute_psnr(output.inr_out.prediction, output.subsampled_targets)
            self.train_loss_dict_stack.setdefault("total_loss", [])
            self.train_loss_dict_stack["total_loss"].append(loss)
            self.train_metrics_stack.setdefault("psnr", [])
            self.train_metrics_stack["psnr"].append(psnr.clone().detach().cpu().item())
        elif stage == "val":
            psnr = compute_psnr(output.inr_out.prediction, output.subsampled_targets)
            self.val_loss_dict_stack.setdefault("total_loss", [])
            self.val_loss_dict_stack["total_loss"].append(loss)
            self.val_metrics_stack.setdefault("psnr", [])
            self.val_metrics_stack["psnr"].append(psnr.clone().detach().cpu().item())

    def post_validation_epoch(
        self, 
        engine: 'Engine', 
        epoch: int,
        **kwargs
    ) -> dict | None:
        return {
            "inr_metric": {
                **{k: sum(v) / len(v) for k, v in self.val_loss_dict_stack.items()},
                **{k: sum(v) / len(v) for k, v in self.val_metrics_stack.items()}
            }
        }

    def post_training_epoch(
        self, 
        engine: 'Engine', 
        epoch: int,
        **kwargs
    ) -> dict | None:
        return {
            "inr_metric": {
                **{k: sum(v) / len(v) for k, v in self.train_loss_dict_stack.items()},
                **{k: sum(v) / len(v) for k, v in self.train_metrics_stack.items()}
            }
        }
