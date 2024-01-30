from .weight_modulated_inr import WeightModulatedINRConfig, WeightModulatedINR
from .inr_output import INROutput

def build_inr(name, config):
    if name == "weight_modulated_inr":
        conf = WeightModulatedINRConfig.create(config)
        return WeightModulatedINR(conf)
    else:
        raise ValueError(f"inr {name} not supported")
