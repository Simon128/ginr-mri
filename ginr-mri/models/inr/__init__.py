from .weight_modulated_inr import WeightModulatedINRConfig, WeightModulatedINR
from .locality_aware_inr import LocalityAwareINR, LocalityAwareINRConfig

from .inr_output import INROutput

def build_inr(name, config):
    if name == "weight_modulated_inr":
        conf = WeightModulatedINRConfig.create(config)
        return WeightModulatedINR(conf)
    if name == "locality_aware_inr":
        conf = LocalityAwareINRConfig.create(config)
        return LocalityAwareINR(conf)
    else:
        raise ValueError(f"inr {name} not supported")
