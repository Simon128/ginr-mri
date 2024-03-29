from .base import BaseModel, BaseModelConfig
from .model_output import ModelOutput

def build_model(name: str, config):
    if name == "base":
        cfg = BaseModelConfig.create(config)
        return BaseModel(cfg)
    else:
         raise ValueError(f"model {name} not supported")
