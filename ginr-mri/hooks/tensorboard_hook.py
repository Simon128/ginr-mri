from typing import TYPE_CHECKING
import torch
import os
from torch.utils.tensorboard.writer import SummaryWriter
import pathlib
from datetime import datetime
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
import torchvision

from .hook import Hook

if TYPE_CHECKING:
    from ..engine import Engine

class TensorboardHook(Hook):
    def __init__(self, priority: int = 0, directory: str | None = None) -> None:
        super().__init__(priority)
        if directory:
            self.directory = directory
        else:
            self.directory = os.path.join(pathlib.Path(__file__).parent.resolve(), "runs", datetime.now().isoformat())
        self.train_writer = SummaryWriter(log_dir=os.path.join(self.directory, "training"))
        self.val_writer = SummaryWriter(log_dir=os.path.join(self.directory, "validation"))

    def post_validation_epoch(
        self, 
        engine: 'Engine', 
        epoch: int,
        **kwargs
    ) -> dict | None:
        inr_metrics = kwargs.get("inr_metric")
        if inr_metrics is not None:
            for k, v in inr_metrics.items():
                self.val_writer.add_scalar(f"inr/{k}", v, epoch)
        backbone_metrics = kwargs.get("backbone_metric")
        if backbone_metrics is not None:
            for k, v in backbone_metrics.items():
                self.val_writer.add_scalar(f"backbone/{k}", v, epoch)
        latent_transformation_metrics = kwargs.get("latent_transformation_metric")
        if inr_metrics is not None:
            for k, v in latent_transformation_metrics.items():
                self.val_writer.add_scalar(f"latent_transformation/{k}", v, epoch)

        visualization = kwargs.get("visualization")
        if visualization is not None:
            self.val_writer.add_scalar(f"example/psnr", visualization["pred_psnr"], epoch)
            _example = visualization["pred"].moveaxis(2, -1)
            _target = visualization["target"].moveaxis(2, -1)
            for c in range(_example.shape[1]):
                plot_2d_or_3d_image(torch.rot90(_example[:, c].unsqueeze(1), dims=[-3, -2]), step=epoch, writer=self.val_writer, tag=f"saggital/prediction/channel{c}", frame_dim=-1)
                plot_2d_or_3d_image(torch.rot90(_target[:, c].unsqueeze(1), dims=[-3, -2]), step=epoch, writer=self.val_writer, tag=f"saggital/target/channel_{c}", frame_dim=-1)
                plot_2d_or_3d_image(_example[:, c].unsqueeze(1), step=epoch, writer=self.val_writer, tag=f"axial/prediction/channel_{c}", frame_dim=-2)
                plot_2d_or_3d_image(_target[:, c].unsqueeze(1), step=epoch, writer=self.val_writer, tag=f"axial/target/channel_{c}", frame_dim=-2)
                plot_2d_or_3d_image(torch.rot90(_example[:, c].unsqueeze(1), dims=[-2, -1], k=2), step=epoch, writer=self.val_writer, tag=f"coronal/prediction/channel{c}", frame_dim=-3)
                plot_2d_or_3d_image(torch.rot90(_target[:, c].unsqueeze(1), dims=[-2, -1], k=2), step=epoch, writer=self.val_writer, tag=f"coronal/target/channel_{c}", frame_dim=-3)

            # B, num_slices, C, **spatial
            num_batches = len(visualization["slices_axial"][0])
            num_channels = visualization["slices_axial"][0][0].shape[1]
            for b in range(num_batches):
                for c in range(num_channels):
                    p, t = visualization["slices_saggital"]
                    slice_grid = torchvision.utils.make_grid(p[b, :, c, ...].unsqueeze(1), nrow=4, normalize=True)
                    self.val_writer.add_image(f"saggital_slices/prediction/batch_{b}/channel_{c}", slice_grid, epoch)
                    slice_grid = torchvision.utils.make_grid(t[b, :, c, ...].unsqueeze(1), nrow=4, normalize=True)
                    self.val_writer.add_image(f"saggital_slices/target/batch_{b}/channel_{c}", slice_grid, epoch)
                    p, t = visualization["slices_axial"]
                    slice_grid = torchvision.utils.make_grid(p[b, :, c, ...].unsqueeze(1), nrow=4, normalize=True)
                    self.val_writer.add_image(f"axial_slices/prediction/batch_{b}/channel_{c}", slice_grid, epoch)
                    slice_grid = torchvision.utils.make_grid(t[b, :, c, ...].unsqueeze(1), nrow=4, normalize=True)
                    self.val_writer.add_image(f"axial_slices/target/batch_{b}/channel_{c}", slice_grid, epoch)
                    p, t = visualization["slices_coronal"]
                    slice_grid = torchvision.utils.make_grid(p[b, :, c, ...].unsqueeze(1), nrow=4, normalize=True)
                    self.val_writer.add_image(f"coronal_slices/prediction/batch_{b}/channel_{c}", slice_grid, epoch)
                    slice_grid = torchvision.utils.make_grid(t[b, :, c, ...].unsqueeze(1), nrow=4, normalize=True)
                    self.val_writer.add_image(f"coronal_slices/target/batch_{b}/channel_{c}", slice_grid, epoch)

    def post_training_epoch(
        self, 
        engine: 'Engine', 
        epoch: int,
        **kwargs
    ) -> dict | None:
        inr_metrics = kwargs.get("inr_metric")
        if inr_metrics is not None:
            for k, v in inr_metrics.items():
                self.train_writer.add_scalar(f"inr/{k}", v, epoch)
        backbone_metrics = kwargs.get("backbone_metric")
        if backbone_metrics is not None:
            for k, v in backbone_metrics.items():
                self.train_writer.add_scalar(f"backbone/{k}", v, epoch)
        latent_transformation_metrics = kwargs.get("latent_transformation_metric")
        if inr_metrics is not None:
            for k, v in latent_transformation_metrics.items():
                self.train_writer.add_scalar(f"latent_transformation/{k}", v, epoch)

        visualization = kwargs.get("visualization")
        if visualization is not None:
            self.train_writer.add_scalar(f"example/psnr", visualization["pred_psnr"], epoch)
            _example = visualization["pred"].moveaxis(2, -1)
            _target = visualization["target"].moveaxis(2, -1)
            for c in range(_example.shape[1]):
                plot_2d_or_3d_image(torch.rot90(_example[:, c].unsqueeze(1), dims=[-3, -2]), step=epoch, writer=self.train_writer, tag=f"saggital/prediction/channel{c}", frame_dim=-1)
                plot_2d_or_3d_image(torch.rot90(_target[:, c].unsqueeze(1), dims=[-3, -2]), step=epoch, writer=self.train_writer, tag=f"saggital/target/channel_{c}", frame_dim=-1)
                plot_2d_or_3d_image(_example[:, c].unsqueeze(1), step=epoch, writer=self.train_writer, tag=f"axial/prediction/channel_{c}", frame_dim=-2)
                plot_2d_or_3d_image(_target[:, c].unsqueeze(1), step=epoch, writer=self.train_writer, tag=f"axial/target/channel_{c}", frame_dim=-2)
                plot_2d_or_3d_image(torch.rot90(_example[:, c].unsqueeze(1), dims=[-2, -1], k=2), step=epoch, writer=self.train_writer, tag=f"coronal/prediction/channel{c}", frame_dim=-3)
                plot_2d_or_3d_image(torch.rot90(_target[:, c].unsqueeze(1), dims=[-2, -1], k=2), step=epoch, writer=self.train_writer, tag=f"coronal/target/channel_{c}", frame_dim=-3)

            # B, num_slices, C, **spatial
            num_batches = len(visualization["slices_axial"][0])
            num_channels = visualization["slices_axial"][0][0].shape[1]
            for b in range(num_batches):
                for c in range(num_channels):
                    p, t = visualization["slices_saggital"]
                    slice_grid = torchvision.utils.make_grid(p[b, :, c, ...].unsqueeze(1), nrow=4, normalize=True)
                    self.train_writer.add_image(f"saggital_slices/prediction/batch_{b}/channel_{c}", slice_grid, epoch)
                    slice_grid = torchvision.utils.make_grid(t[b, :, c, ...].unsqueeze(1), nrow=4, normalize=True)
                    self.train_writer.add_image(f"saggital_slices/target/batch_{b}/channel_{c}", slice_grid, epoch)
                    p, t = visualization["slices_axial"]
                    slice_grid = torchvision.utils.make_grid(p[b, :, c, ...].unsqueeze(1), nrow=4, normalize=True)
                    self.train_writer.add_image(f"axial_slices/prediction/batch_{b}/channel_{c}", slice_grid, epoch)
                    slice_grid = torchvision.utils.make_grid(t[b, :, c, ...].unsqueeze(1), nrow=4, normalize=True)
                    self.train_writer.add_image(f"axial_slices/target/batch_{b}/channel_{c}", slice_grid, epoch)
                    p, t = visualization["slices_coronal"]
                    slice_grid = torchvision.utils.make_grid(p[b, :, c, ...].unsqueeze(1), nrow=4, normalize=True)
                    self.train_writer.add_image(f"coronal_slices/prediction/batch_{b}/channel_{c}", slice_grid, epoch)
                    slice_grid = torchvision.utils.make_grid(t[b, :, c, ...].unsqueeze(1), nrow=4, normalize=True)
                    self.train_writer.add_image(f"coronal_slices/target/batch_{b}/channel_{c}", slice_grid, epoch)
