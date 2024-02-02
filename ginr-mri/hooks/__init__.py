import enum
from dataclasses import dataclass
import logging

from .wandb_hook import WandbHook
from .hook import Hook
from .lr_scheduler_hook import LRSchedulerHook
from .model_checkpoint_hook import ModelCheckpointHook
from .tensorboard_hook import TensorboardHook
from .inr_metrics_hook import INRMetricsHook
from .visualization_hook import VisualizationHook

logger = logging.getLogger(__name__)

class HooksEnum(enum.StrEnum):
    early_stop = "early_stop"
    model_checkpoint = "model_checkpoint"
    tensorboard_hook = "tensorboard"
    lr_scheduler = "lr_scheduler"
    inr_metrics = "inr_metrics"
    visualization = "visualization"
    wandb = "wandb"

@dataclass
class HookCFG:
    hook: HooksEnum
    priority: int
    cfg: dict

def build_hook(cfg: HookCFG, full_cfg) -> Hook:
    match HooksEnum(cfg.hook):
        case HooksEnum.early_stop: return Hook(cfg.priority)
        case HooksEnum.model_checkpoint: return ModelCheckpointHook(cfg.priority, **cfg.cfg)
        case HooksEnum.tensorboard_hook: return TensorboardHook(cfg.priority, **cfg.cfg)
        case HooksEnum.lr_scheduler: return LRSchedulerHook(cfg.priority, **cfg.cfg)
        case HooksEnum.inr_metrics: return INRMetricsHook(cfg.priority, **cfg.cfg)
        case HooksEnum.visualization: return VisualizationHook(cfg.priority, **cfg.cfg)
        case HooksEnum.wandb: return WandbHook(cfg.priority, full_cfg, **cfg.cfg)
        case _:
            err_msg = f"hook {cfg.hook} not supported"
            logger.error(err_msg)
            raise ValueError(err_msg)

def build_hooks(cfgs: list[HookCFG], full_cfg):
    hooks = []
    for hcfg in cfgs:
        hooks.append(build_hook(hcfg, full_cfg))
    return hooks
