import os
from typing import TYPE_CHECKING
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .hook import Hook
from ..metrics.inr_metrics import compute_psnr
from ..utils import save_tensor_as_nifti

if TYPE_CHECKING:
    from ..engine import Engine

class VisualizationHook(Hook):
    def __init__(self, priority: int = 0, directory: str | None = None, frequency: int = 100, num_slices: int = 8) -> None:
        super().__init__(priority)
        self.directory = directory
        self.frequency = frequency
        self.num_slices = num_slices

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
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model

    def post_validation_epoch(
        self, 
        engine: 'Engine', 
        epoch: int,
        **kwargs
    ) -> dict | None:
        if epoch % self.frequency == 0 and (not dist.is_initialized() or dist.is_initialized() and dist.get_rank() == 0):
            batch = next(iter(self.val_dataloader))
            was_train = self.model.training
            self.model.eval()
            with torch.inference_mode():
                if isinstance(self.model, DDP):
                    prediction = self.model.module.full_prediction(batch).inr_out.prediction
                else:
                    prediction = self.model.full_prediction(batch).inr_out.prediction
            self.model.train(mode=was_train)
            psnr = compute_psnr(prediction, batch[1])
            # prediction shape:
            # B, C, D, H, W
            # case no channels:
            if len(prediction.shape) == 4:
                prediction = prediction.unsqueeze(1)

            # saggital: sliced along depth dimension 
            # axial: sliced along width dimension
            # coronal: sliced along height dimension
            saggital_pred = []
            axial_pred = []
            coronal_pred = []
            saggital_target = []
            axial_target = []
            coronal_target = []
            
            # saggital
            dim_size = prediction.shape[-3]
            slice_step = dim_size // self.num_slices
            for idx in range(0, dim_size, slice_step):
                saggital_pred.append(torch.rot90(prediction.select(-3, idx), dims=[-2, -1]))
                saggital_target.append(torch.rot90(batch[1].select(-3, idx), dims=[-2, -1]))

            # axial
            dim_size = prediction.shape[-1]
            slice_step = dim_size // self.num_slices
            for idx in range(0, dim_size, slice_step):
                axial_pred.append(torch.rot90(prediction.select(-1, idx), dims=[-2, -1]))
                axial_target.append(torch.rot90(batch[1].select(-1, idx), dims=[-2, -1]))

            # coronal
            dim_size = prediction.shape[-2]
            slice_step = dim_size // self.num_slices
            for idx in range(0, dim_size, slice_step):
                coronal_pred.append(torch.rot90(prediction.select(-2, idx), dims=[-2, -1]))
                coronal_target.append(torch.rot90(batch[1].select(-2, idx), dims=[-2, -1]))

            saggital_pred = torch.stack(saggital_pred, dim=1)
            saggital_target = torch.stack(saggital_target, dim=1)
            axial_pred = torch.stack(axial_pred, dim=1)
            axial_target = torch.stack(axial_target, dim=1)
            coronal_pred = torch.stack(coronal_pred, dim=1)
            coronal_target = torch.stack(coronal_target, dim=1)

            if self.directory is not None:
                sub_dir = os.path.join(self.directory, "validation", "examples")
                os.makedirs(sub_dir, exist_ok=True)
                for c in range(prediction.shape[1]):
                    file_name = os.path.join(sub_dir, f"pred_epoch_{epoch}_channel_{c}.nii")
                    save_tensor_as_nifti(prediction[0][c], file_name)
                    file_name = os.path.join(sub_dir, f"target_epoch_{epoch}_channel_{c}.nii")
                    save_tensor_as_nifti(batch[1][0][c], file_name)

        if dist.is_initialized():
            dist.barrier()
        if epoch % self.frequency == 0 and (not dist.is_initialized() or dist.is_initialized() and dist.get_rank() == 0):
            return {
                "visualization": {
                    "pred_psnr": psnr,
                    "pred": prediction,
                    "target": batch[1],
                    "slices_saggital": (saggital_pred, saggital_target),
                    "slices_axial": (axial_pred, axial_target),
                    "slices_coronal": (coronal_pred, coronal_target)
                }
            }

    def post_training_epoch(
        self, 
        engine: 'Engine', 
        epoch: int,
        **kwargs
    ) -> dict | None:
        if epoch % self.frequency == 0 and (not dist.is_initialized() or dist.is_initialized() and dist.get_rank() == 0):
            batch = next(iter(self.val_dataloader))
            was_train = self.model.training
            self.model.eval()
            with torch.inference_mode():
                if isinstance(self.model, DDP):
                    prediction = self.model.module.full_prediction(batch).inr_out.prediction
                else:
                    prediction = self.model.full_prediction(batch).inr_out.prediction
            self.model.train(mode=was_train)
            psnr = compute_psnr(prediction, batch[1])
            # prediction shape:
            # B, C, D, H, W
            # case no channels:
            if len(prediction.shape) == 4:
                prediction = prediction.unsqueeze(1)

            # saggital: sliced along depth dimension
            # axial: sliced along width dimension
            # coronal: sliced along height dimension 
            saggital_pred = []
            axial_pred = []
            coronal_pred = []
            saggital_target = []
            axial_target = []
            coronal_target = []
            
            # saggital
            dim_size = prediction.shape[-3]
            slice_step = dim_size // self.num_slices
            for idx in range(0, dim_size, slice_step):
                saggital_pred.append(torch.rot90(prediction.select(-3, idx), dims=[-2, -1]))
                saggital_target.append(torch.rot90(batch[1].select(-3, idx), dims=[-2, -1]))

            # axial
            dim_size = prediction.shape[-1]
            slice_step = dim_size // self.num_slices
            for idx in range(0, dim_size, slice_step):
                axial_pred.append(torch.rot90(prediction.select(-1, idx), dims=[-2, -1]))
                axial_target.append(torch.rot90(batch[1].select(-1, idx), dims=[-2, -1]))

            # coronal
            dim_size = prediction.shape[-2]
            slice_step = dim_size // self.num_slices
            for idx in range(0, dim_size, slice_step):
                coronal_pred.append(torch.rot90(prediction.select(-2, idx), dims=[-2, -1]))
                coronal_target.append(torch.rot90(batch[1].select(-2, idx), dims=[-2, -1]))

            saggital_pred = torch.stack(saggital_pred, dim=1)
            saggital_target = torch.stack(saggital_target, dim=1)
            axial_pred = torch.stack(axial_pred, dim=1)
            axial_target = torch.stack(axial_target, dim=1)
            coronal_pred = torch.stack(coronal_pred, dim=1)
            coronal_target = torch.stack(coronal_target, dim=1)

            if self.directory is not None:
                sub_dir = os.path.join(self.directory, "training", "examples")
                os.makedirs(sub_dir, exist_ok=True)
                for c in range(prediction.shape[1]):
                    file_name = os.path.join(sub_dir, f"pred_epoch_{epoch}_channel_{c}.nii")
                    save_tensor_as_nifti(prediction[0][c], file_name)
                    file_name = os.path.join(sub_dir, f"target_epoch_{epoch}_channel_{c}.nii")
                    save_tensor_as_nifti(batch[1][0][c], file_name)

        if dist.is_initialized():
            dist.barrier()
        if epoch % self.frequency == 0 and (not dist.is_initialized() or dist.is_initialized() and dist.get_rank() == 0):
            return {
                "visualization": {
                    "pred_psnr": psnr,
                    "pred": prediction,
                    "target": batch[1],
                    "slices_saggital": (saggital_pred, saggital_target),
                    "slices_axial": (axial_pred, axial_target),
                    "slices_coronal": (coronal_pred, coronal_target)
                }
            }
