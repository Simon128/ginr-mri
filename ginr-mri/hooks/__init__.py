import enum
from dataclasses import dataclass
import logging

from .hook import Hook
from .lr_scheduler_hook import LRSchedulerHook
from .model_checkpoint_hook import ModelCheckpointHook
from .tensorboard_hook import TensorboardHook

logger = logging.getLogger(__name__)

class HooksEnum(enum.StrEnum):
    early_stop = "early_stop"
    model_checkpoint = "model_checkpoint"
    tensorboard_hook = "tensorboard"
    lr_scheduler = "lr_scheduler"

@dataclass
class HookCFG:
    hook: HooksEnum
    priority: int
    cfg: dict

def build_hook(cfg: HookCFG) -> Hook:
    if HooksEnum(cfg.hook) == HooksEnum.early_stop:
        return Hook(cfg.priority)
    elif HooksEnum(cfg.hook) == HooksEnum.model_checkpoint:
        return ModelCheckpointHook(cfg.priority, **cfg.cfg)
    elif HooksEnum(cfg.hook) == HooksEnum.tensorboard_hook:
        return TensorboardHook(cfg.priority, **cfg.cfg)
    elif HooksEnum(cfg.hook) == HooksEnum.lr_scheduler:
        return LRSchedulerHook(cfg.priority, **cfg.cfg)
    else:
        err_msg = f"hook {cfg.hook} not supported"
        logger.error(err_msg)
        raise ValueError(err_msg)

def build_hooks(cfgs: list[HookCFG]):
    hooks = []
    for hcfg in cfgs:
        hooks.append(build_hook(hcfg))
    return hooks
