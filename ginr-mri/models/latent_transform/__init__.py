from .convolutional import ConvolutionalLTConfig, ConvolutionalLT

def build_latent_transform(name: str, config):
    if name == "convolutional":
        cfg = ConvolutionalLTConfig.create(config)
        return ConvolutionalLT(cfg)
    else:
        raise ValueError(f"latent transform {name} not supported")
