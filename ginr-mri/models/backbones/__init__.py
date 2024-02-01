from .nvidia2018 import Nvidia2018, Nvidia2018Config

def build_backbone(name, config):
    if name == "nvidia2018":
        cfg = Nvidia2018Config.create(config)
        return Nvidia2018(cfg)
    else:
        raise ValueError(f"backbone {name} is not supported")

