import nibabel as nib
from pathlib import Path
from torch.utils.data import Dataset
import re
from enum import IntFlag, auto, StrEnum 
from omegaconf import OmegaConf
import math
from dataclasses import dataclass, field
import os
import random
import torch

class BraTSScanTypes(IntFlag):
    NA = auto()
    T1 = auto()
    T2 = auto()
    FLAIR = auto()
    T1CE = auto()
    BRAINMASK = auto()
    SEG = auto()

def str_list_to_int_flag(str_list: list[str], target_int_flag: type[IntFlag]):
    selection = BraTSScanTypes.NA

    for name in str_list:
        selection = selection | target_int_flag[name]

    return selection

class BraTSDataPrepOption(StrEnum):
    STACK = "STACK" 
    RANDOM_LEAVE_ONE_OUT = "RANDOM_LEAVE_ONE_OUT" 

@dataclass
class BraTSDataPrepConfig:
    use_lgg: bool = True
    use_hgg: bool = True
    use_data: list[str] = field(default_factory=lambda: ["T1", "T2", "FLAIR", "T1CE"])
    data_prep_option: BraTSDataPrepOption = BraTSDataPrepOption.STACK
    input_data: list[str] = field(default_factory=lambda: ["T1", "T2", "FLAIR", "T1CE"])
    output_data: list[str] = field(default_factory=lambda: ["T1", "T2", "FLAIR", "T1CE"])

    @classmethod
    def create(cls, config):
        defaults = OmegaConf.structured(cls())
        config = OmegaConf.merge(defaults, config)
        return config

class BraTS(Dataset):
    root = Path(__file__).parent.parent.parent.joinpath("data_folder/brats")
    lgg_path = root.joinpath("LGG")
    hgg_path = root.joinpath("HGG")
    split_seed = 12

    _scan_regex_map: dict[BraTSScanTypes, re.Pattern] = {
        BraTSScanTypes.BRAINMASK: re.compile(r"^.*brainmask\.nii\.gz"),
        BraTSScanTypes.FLAIR: re.compile(r"^.*flair\.nii\.gz"),
        BraTSScanTypes.T1: re.compile(r"^.*t1\.nii\.gz"),
        BraTSScanTypes.T1CE: re.compile(r"^.*t1ce\.nii\.gz"),
        BraTSScanTypes.T2: re.compile(r"^.*t2\.nii\.gz"),
        BraTSScanTypes.SEG: re.compile(r"^.*seg\.nii\.gz")
    }

    def __init__(
            self, 
            data_prep_cfg : BraTSDataPrepConfig = BraTSDataPrepConfig(),
            split="train", 
            src_transform=None,
            trgt_transform=None,
            device="cpu"
        ) -> None:
        super().__init__()
        assert split in ["train", "val"]
        self.device = device
        self.src_transform = src_transform
        self.trgt_transform = trgt_transform
        self.split = split
        self.data_prep_cfg = data_prep_cfg
        self.use_patterns = self._get_usable_patterns(data_prep_cfg.use_data) 
        self.use_options = str_list_to_int_flag(data_prep_cfg.use_data, BraTSScanTypes)
        self.input_options = str_list_to_int_flag(data_prep_cfg.input_data, BraTSScanTypes)
        self.output_options = str_list_to_int_flag(data_prep_cfg.output_data, BraTSScanTypes)
        self.items = []

        if data_prep_cfg.use_lgg:
            self.items += self._discover_data(str(self.lgg_path))
        if data_prep_cfg.use_hgg:
            self.items += self._discover_data(str(self.hgg_path))

        random.seed(self.split_seed)
        random.shuffle(self.items)
        train_num = math.ceil(0.8 * len(self))

        if split == "train":
            self.items = self.items[:train_num]
        else:
            self.items = self.items[train_num:]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        return self._prepare_item(item)

    def _prepare_item(self, item):
        opt =  BraTSDataPrepOption(self.data_prep_cfg.data_prep_option)
        if opt == BraTSDataPrepOption.STACK:
            return self._prepare_item_stack(item)
        elif opt == BraTSDataPrepOption.RANDOM_LEAVE_ONE_OUT:
            return self._prepare_item_random_leave_one_out(item)
        else:
            raise ValueError(f"{opt} not supported")

    def _prepare_item_stack(self, item):
        input_list = []
        output_list = []
        useable = {}

        for opt in self.use_patterns:
            if opt == BraTSScanTypes.NA: continue
            path = item[opt]
            t = torch.tensor(nib.load(path).get_fdata(), device=self.device, dtype=torch.float32) # type:ignore
            # normalize 
            t = (t - torch.min(t)) * 2 / (torch.max(t) - torch.min(t)) - 1
            useable[opt] = t

        for opt in self.input_options:
            if opt == BraTSScanTypes.NA: continue
            input_list.append(useable[opt]) 

        for opt in self.output_options:
            if opt == BraTSScanTypes.NA: continue
            output_list.append(useable[opt])

        input = torch.stack(input_list)
        if self.src_transform:
            input = self.src_transform(input)

        output = torch.stack(output_list)
        if self.trgt_transform:
            output = self.trgt_transform(output)

        return input, output

    def _prepare_item_random_leave_one_out(self, item):
        input_list = []

        for opt in self.input_options:
            path = item[opt]
            t = torch.tensor(nib.load(path).get_fdata(), device=self.device, dtype=torch.float32) # type:ignore
            # normalize 
            t = (t - torch.min(t)) * 2 / (torch.max(t) - torch.min(t)) - 1
            input_list.append(t)

        target_idx = random.randrange(0, len(input_list))
        target = torch.tensor(input_list.pop(target_idx))

        input = torch.stack(input_list)
        if self.src_transform:
            input = self.src_transform(input)

        if self.trgt_transform:
            target = self.trgt_transform(target)

        return input, target

    def _discover_data(self, folder_path: str): 
        output = {}
        instance_folders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

        for instance in instance_folders:
            for f in os.scandir(instance):
                usable = self._get_usable_file(f.path)
                if usable:
                    output.setdefault(instance, {})
                    output[instance][usable[0]] = usable[1]

        return [v for v in output.values()]
        
    def _get_usable_file(self, path):
        for option, pattern in self.use_patterns.items():
            if re.match(pattern, path): 
                return option, path
        return None

    def _get_usable_patterns(self, option_str_list: list[str]):
        patterns = {}
        selected_scan_options = str_list_to_int_flag(option_str_list, BraTSScanTypes)

        for k, v in self._scan_regex_map.items():
            if k in selected_scan_options:
                patterns[k] = v

        return patterns
