from .centercrop3d import CenterCrop3D


def build_transform(name, config):
    if name == "centercrop3d":
        return CenterCrop3D(config.size)
    else:
        raise ValueError(f"transform {name} not supported")

