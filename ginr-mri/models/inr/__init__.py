from .weight_modulated_inr import WeightModulatedINRConfig, WeightModulatedINR

def build_inr(name, config):
    if name == "weight_modulated_inr":
        conf = WeightModulatedINRConfig.create(config)
        return WeightModulatedINR(conf)
