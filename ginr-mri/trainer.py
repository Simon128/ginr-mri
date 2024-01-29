from omegaconf import OmegaConf, MISSING
from dataclasses import dataclass
import torch
import math
from torch.optim._multi_tensor import Adam
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
import logging

from .data import build_data
from .models import build_model
from .utils import Timer

logger = logging.getLogger(__name__)

@dataclass
class TrainerConfig:
    epochs: int = MISSING
    batch_size: int = 4
    amp: bool = False
    validation_frequency: float = 1.0

    @classmethod
    def create(cls, config):
        defaults = OmegaConf.structured(cls())
        config = OmegaConf.merge(defaults, config)
        return config

class Trainer:
    def __init__(self, config_path) -> None:
        if dist.is_initialized():
            self.rank = dist.get_rank()
        else:
            self.rank = 0
        conf = OmegaConf.load(config_path)
        self.train_dataset, self.val_dataset = build_data(conf.data.name, conf.data, self.rank)
        self.conf = TrainerConfig.create(conf.trainer)
        self.model = build_model(conf.model.name, conf.model).to(self.rank)
        self.optimizer = Adam(self.model.parameters(), lr=1e-4) # todo: make configurable
        self.hooks = [] # todo
        sorted(self.hooks, key=lambda h: h.priority, reverse=True)
        self.resume_epoch = 0 # todo
        self.epochs = self.conf.epochs - self.resume_epoch

    def run_hooks(self, func: str, **kwargs):
        current_kwargs = kwargs
        for h in self.hooks:
            temp = getattr(h, func)(**current_kwargs)
            if temp is not None:
                current_kwargs = {
                    **current_kwargs,
                    **temp
                }

    def get_dataloader(self, dataset: Dataset, shuffle = True, batch_size: int | None = None):
        if dist.is_initialized():
            trainsampler = DistributedSampler(dataset, shuffle=shuffle) # type:ignore
            dataloader = DataLoader(
                dataset=dataset, # type:ignore
                sampler=trainsampler,
                batch_size=self.conf.batch_size if batch_size is None else batch_size,
                num_workers=4,
                pin_memory=True
            )
        else:
            dataloader = DataLoader(
                dataset=dataset, # type:ignore
                batch_size=self.conf.batch_size if batch_size is None else batch_size,
                shuffle=shuffle
            )
        return dataloader

    def fit(self):
        torch.backends.cudnn.benchmark = True
        trainloader = self.get_dataloader(self.train_dataset, shuffle=False)
        valloader = self.get_dataloader(self.val_dataset, shuffle=False)
        self.run_hooks(
            "pre_fit", engine=self, model=self.model, 
            train_dataset=self.train_dataset, val_dataset=self.val_dataset, 
            train_dataloader=trainloader, val_dataloader=valloader, 
            optimizer=self.optimizer
        )
        train_size = len(trainloader) 
        val_step = math.floor(train_size / self.conf.validation_frequency)
        log_step = train_size // 10
        self.model.train()
        epoch_timer = Timer()
        if self.conf.amp:
            scaler = torch.cuda.amp.grad_scaler.GradScaler()

        for e in range(self.epochs):
            train_loss = 0
            if dist.is_initialized():
                trainloader.sampler.set_epoch(e) # type:ignore
                valloader.sampler.set_epoch(e) # type:ignore
            if e == 0:
                logger.info(f"Running Epoch {e + self.resume_epoch}/{self.epochs} on rank {self.rank} -- elapsed: {epoch_timer.get_elapsed()} -- eta: {epoch_timer.get_eta(self.epochs - e)}")
            train_iter = iter(trainloader)
            self.run_hooks("pre_training_epoch", engine=self, epoch=e + self.resume_epoch)
            train_timer = Timer()
            for it in range(train_size):
                if it % log_step == 0:
                    logger.info(f"Training {it}/{train_size} in epoch {e + self.resume_epoch} on rank {self.rank} -- elapsed {train_timer.get_elapsed()} -- eta: {train_timer.get_eta(train_size - it)}")
                self.optimizer.zero_grad()
                batch = next(train_iter)
                self.run_hooks("pre_model_step", engine=self, iteration_step=it, epoch=e + self.resume_epoch, batch=batch, stage="train")

                if self.conf.amp:
                    with torch.cuda.amp.autocast_mode.autocast():
                        output = self.model(batch)
                    scaler.scale(output.loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    output = self.model(batch)
                    output.loss.backward()
                    self.optimizer.step()
                
                train_loss += output.loss.item()
                self.run_hooks("post_model_step", engine=self, iteration_step=it, epoch=e + self.resume_epoch, output=output, stage="train")
                train_timer.step()
                if it == train_size - 1:
                    logger.info(f"Training {it + 1}/{train_size} in epoch {e + self.resume_epoch} on rank {self.rank} -- elapsed {train_timer.get_elapsed()} -- eta: {train_timer.get_eta(train_size - it)}")
                if (it + 1) % val_step == 0:
                    self.validate(valloader, e + self.resume_epoch)

            self.run_hooks("post_training_epoch", engine=self, epoch=e + self.resume_epoch)
            epoch_timer.step()
            logger.info(f"Running Epoch {e + self.resume_epoch + 1}/{self.epochs} on rank {self.rank} -- elapsed: {epoch_timer.get_elapsed()} -- eta: {epoch_timer.get_eta(self.epochs - e)}")
    
        self.run_hooks("post_fit", engine=self, model=self.model, train_dataset=self.train_dataset, val_dataset=self.val_dataset, optimizer=self.optimizer)

    def validate(self, valloader, epoch: int):
        self.run_hooks("pre_validation_epoch", engine=self, epoch=epoch)
        val_size = len(valloader)
        val_iter = iter(valloader)
        log_step = val_size // 10
        val_timer = Timer()
        self.model.eval()
        val_loss = 0
        with torch.inference_mode():
            for it in range(val_size):
                if it % log_step == 0:
                    logger.info(f"Validating {it}/{val_size} in epoch {epoch} on rank {self.rank} -- elapsed {val_timer.get_elapsed()} -- eta: {val_timer.get_eta(val_size - it)}")
                batch = next(val_iter)
                self.run_hooks("pre_model_step", engine=self, iteration_step=it, epoch=epoch, batch=batch, stage="val")
                output = self.model(batch)
                val_loss += output.loss.item()
                self.run_hooks("post_model_step", engine=self, iteration_step=it, epoch=epoch, output=output, stage="val")
                val_timer.step()
                if it == val_size - 1:
                    logger.info(f"Validating {it + 1}/{val_size} in epoch {epoch} on rank {self.rank} -- elapsed {val_timer.get_elapsed()} -- eta: {val_timer.get_eta(val_size - it)}")
        self.run_hooks("post_validation_epoch", engine=self, epoch=epoch)
        self.model.train()
                
