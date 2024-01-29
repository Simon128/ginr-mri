from .nvidia2018 import Nvidia2018

def build_backbone(name):
    if name == "nvidia2018":
        return Nvidia2018()
    else:
        raise ValueError(f"backbone {name} is not supported")

