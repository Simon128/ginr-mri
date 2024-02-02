from typing import TYPE_CHECKING
import wandb
import omegaconf
import torch
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
import torchvision

from .hook import Hook

if TYPE_CHECKING:
    from ..engine import Engine

class WandbHook(Hook):
    def __init__(self, priority: int = 0, full_cfg = None, project="NA") -> None:
        super().__init__(priority)
        wandb.init(
            project=project,
            config=omegaconf.OmegaConf.to_container(full_cfg, resolve=True)
        )

    def post_validation_epoch(
        self, 
        engine: 'Engine', 
        epoch: int,
        **kwargs
    ) -> dict | None:
        wandb_logs = {}
        inr_metrics = kwargs.get("inr_metric")
        if inr_metrics is not None:
            for k, v in inr_metrics.items():
                wandb_logs[f"val/inr/{k}"] = v
        backbone_metrics = kwargs.get("backbone_metric")
        if backbone_metrics is not None:
            for k, v in backbone_metrics.items():
                wandb_logs[f"val/backbone/{k}"] = v
        latent_transformation_metrics = kwargs.get("latent_transformation_metric")
        if inr_metrics is not None:
            for k, v in latent_transformation_metrics.items():
                wandb_logs[f"val/latent_transformation/{k}"] = v

        visualization = kwargs.get("visualization")
        if visualization is not None:
            wandb_logs[f"val/example/psnr"] = visualization["pred_psnr"]
            # B, num_slices, C, **spatial
            num_batches = len(visualization["slices_axial"][0])
            num_channels = visualization["slices_axial"][0][0].shape[1]
            for b in range(num_batches):
                for c in range(num_channels):
                    p, t = visualization["slices_saggital"]
                    slice_grid = torchvision.utils.make_grid(p[b, :, c, ...].unsqueeze(1), nrow=4, normalize=True)
                    wandb_logs[f"val/saggital_slices/prediction/batch_{b}/channel_{c}"] = wandb.Image(slice_grid)
                    slice_grid = torchvision.utils.make_grid(t[b, :, c, ...].unsqueeze(1), nrow=4, normalize=True)
                    wandb_logs[f"val/saggital_slices/target/batch_{b}/channel_{c}"] = wandb.Image(slice_grid)
                    p, t = visualization["slices_axial"]
                    slice_grid = torchvision.utils.make_grid(p[b, :, c, ...].unsqueeze(1), nrow=4, normalize=True)
                    wandb_logs[f"val/axial_slices/prediction/batch_{b}/channel_{c}"] = wandb.Image(slice_grid)
                    slice_grid = torchvision.utils.make_grid(t[b, :, c, ...].unsqueeze(1), nrow=4, normalize=True)
                    wandb_logs[f"val/axial_slices/target/batch_{b}/channel_{c}"] = wandb.Image(slice_grid)
                    p, t = visualization["slices_coronal"]
                    slice_grid = torchvision.utils.make_grid(p[b, :, c, ...].unsqueeze(1), nrow=4, normalize=True)
                    wandb_logs[f"val/coronal_slices/prediction/batch_{b}/channel_{c}"] = wandb.Image(slice_grid)
                    slice_grid = torchvision.utils.make_grid(t[b, :, c, ...].unsqueeze(1), nrow=4, normalize=True)
                    wandb_logs[f"val/coronal_slices/target/batch_{b}/channel_{c}"] = wandb.Image(slice_grid)

        wandb.log(wandb_logs, step=epoch)

    def post_training_epoch(
        self, 
        engine: 'Engine', 
        epoch: int,
        **kwargs
    ) -> dict | None:
        wandb_logs = {}
        inr_metrics = kwargs.get("inr_metric")
        if inr_metrics is not None:
            for k, v in inr_metrics.items():
                wandb_logs[f"train/inr/{k}"] = v
        backbone_metrics = kwargs.get("backbone_metric")
        if backbone_metrics is not None:
            for k, v in backbone_metrics.items():
                wandb_logs[f"train/backbone/{k}"] = v
        latent_transformation_metrics = kwargs.get("latent_transformation_metric")
        if inr_metrics is not None:
            for k, v in latent_transformation_metrics.items():
                wandb_logs[f"train/latent_transformation/{k}"] = v

        visualization = kwargs.get("visualization")
        if visualization is not None:
            wandb_logs[f"train/example/psnr"] = visualization["pred_psnr"]
            # B, num_slices, C, **spatial
            num_batches = len(visualization["slices_axial"][0])
            num_channels = visualization["slices_axial"][0][0].shape[1]
            for b in range(num_batches):
                for c in range(num_channels):
                    p, t = visualization["slices_saggital"]
                    slice_grid = torchvision.utils.make_grid(p[b, :, c, ...].unsqueeze(1), nrow=4, normalize=True)
                    wandb_logs[f"train/saggital_slices/prediction/batch_{b}/channel_{c}"] = wandb.Image(slice_grid)
                    slice_grid = torchvision.utils.make_grid(t[b, :, c, ...].unsqueeze(1), nrow=4, normalize=True)
                    wandb_logs[f"train/saggital_slices/target/batch_{b}/channel_{c}"] = wandb.Image(slice_grid)
                    p, t = visualization["slices_axial"]
                    slice_grid = torchvision.utils.make_grid(p[b, :, c, ...].unsqueeze(1), nrow=4, normalize=True)
                    wandb_logs[f"train/axial_slices/prediction/batch_{b}/channel_{c}"] = wandb.Image(slice_grid)
                    slice_grid = torchvision.utils.make_grid(t[b, :, c, ...].unsqueeze(1), nrow=4, normalize=True)
                    wandb_logs[f"train/axial_slices/target/batch_{b}/channel_{c}"] = wandb.Image(slice_grid)
                    p, t = visualization["slices_coronal"]
                    slice_grid = torchvision.utils.make_grid(p[b, :, c, ...].unsqueeze(1), nrow=4, normalize=True)
                    wandb_logs[f"train/coronal_slices/prediction/batch_{b}/channel_{c}"] = wandb.Image(slice_grid)
                    slice_grid = torchvision.utils.make_grid(t[b, :, c, ...].unsqueeze(1), nrow=4, normalize=True)
                    wandb_logs[f"train/coronal_slices/target/batch_{b}/channel_{c}"] = wandb.Image(slice_grid)

        wandb.log(wandb_logs, step=epoch)