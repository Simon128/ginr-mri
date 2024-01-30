from argparse import ArgumentParser
import os
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .engine import Engine
from .data import build_data
from .models import build_model
from .hooks import build_hooks
from .optimizer import build_optimizer

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # required for dist gather ops
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def parser():
    parser = ArgumentParser(
        prog="ginr-mri",
        description="Train and run generalizable INRs on MRI scans"
    )
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("config", type=str)
    parser.add_argument("--ranks", nargs="+", default=[0])
    return parser

def run_wrapper(
        rank, 
        world_size, 
        args
    ):
    if world_size > 1:
        setup(rank, world_size)
    conf = OmegaConf.load(args.config)
    hooks = build_hooks(conf.hooks)
    model = build_model(conf.model.name, conf.model).to(rank)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    engine = Engine(conf.engine, hooks)
    if args.mode == "train":
        train_ds, val_ds = build_data(conf.data.name, conf.data, rank)
        optimizer = build_optimizer(model, conf.optimizer)
        engine.fit(model, train_ds, val_ds, optimizer)
    if world_size > 1:
        cleanup()

if __name__ == "__main__":
    _parser = parser()
    args = _parser.parse_args()
    world_size = len(args.ranks)
    if world_size > 1:
        mp.spawn(run_wrapper, # type:ignore
             args=(args),
             nprocs=world_size,
             join=True
        )
    else:
        run_wrapper(0, 1, args)
