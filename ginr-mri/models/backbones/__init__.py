import torch

from .nvidia2018 import Nvidia2018, Nvidia2018Config
from .medical_net import ResNet, MedicalNetConfig

def build_backbone(name, config):
    if name == "nvidia2018":
        cfg = Nvidia2018Config.create(config)
        return Nvidia2018(cfg)
    if name == "medical_net":
        cfg = MedicalNetConfig.create(config)
        model = ResNet(cfg)
        if cfg.pretrained_path is not None:
            pt = torch.load(cfg.pretrained_path)
            state_dict = model.state_dict()
            pretrain_dict = {k: v for k, v in pt['state_dict'].items() if k in state_dict.keys()}
            state_dict.update(pretrain_dict)
            model.load_state_dict(state_dict)
            for param in model.parameters():
                param.requires_grad = False
        return model
    else:
        raise ValueError(f"backbone {name} is not supported")

