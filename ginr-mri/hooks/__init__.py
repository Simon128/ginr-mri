import enum
from dataclasses import dataclass
import logging
from omegaconf import OmegaConf

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
    tensorboard = "tensorboard"
    lr_scheduler = "lr_scheduler"
    inr_metrics = "inr_metrics"
    visualization = "visualization"
    wandb = "wandb"
    none= "none"

@dataclass
class HookCFG:
    hook: HooksEnum
    priority: int
    cfg: dict | None = None

    @classmethod
    def create(cls, config):
        defaults = OmegaConf.structured(cls(HooksEnum.none, priority=100))
        config = OmegaConf.merge(defaults, config)
        return config

def build_hook(cfg: HookCFG, full_cfg) -> Hook:
    if cfg.cfg is None:
        cfg.cfg = {}
    match HooksEnum(cfg.hook):
        case HooksEnum.early_stop: return Hook(cfg.priority)
        case HooksEnum.model_checkpoint: return ModelCheckpointHook(cfg.priority, **cfg.cfg)
        case HooksEnum.tensorboard: return TensorboardHook(cfg.priority, **cfg.cfg)
        case HooksEnum.lr_scheduler: return LRSchedulerHook(cfg.priority, **cfg.cfg)
        case HooksEnum.inr_metrics: return INRMetricsHook(cfg.priority)
        case HooksEnum.visualization: return VisualizationHook(cfg.priority, **cfg.cfg)
        case HooksEnum.wandb: return WandbHook(cfg.priority, full_cfg, **cfg.cfg)
        case _:
            err_msg = f"hook {cfg.hook} not supported"
            logger.error(err_msg)
            raise ValueError(err_msg)

def build_hooks(cfgs: list[HookCFG], full_cfg):
    hooks = []
    for hcfg in cfgs:
        _hcfg = HookCFG.create(hcfg)
        hooks.append(build_hook(_hcfg, full_cfg))
    return hooks
