from .brats import BraTS, BraTSDataPrepConfig
from ..transforms import build_transform
from torchvision.transforms.v2 import Compose

def build_data(name: str, config, device):
    if hasattr(config, "source_transforms"):
        src_transform = Compose([build_transform(t.name, t.args) for t in config.source_transforms])
    else:
        src_transform = None
    if hasattr(config, "target_transforms"):
        trgt_transform = Compose([build_transform(t.name, t.args) for t in config.target_transforms])
    else:
        trgt_transform = None
    if name == "brats":
        data_prep_config = BraTSDataPrepConfig.create(config.data_prep)
        train_ds = BraTS(data_prep_config, "train", src_transform, trgt_transform, device)
        val_ds = BraTS(data_prep_config, "val", src_transform, trgt_transform, device)
        return train_ds, val_ds
    else:
        raise ValueError(f"dataset {name} not supported")
