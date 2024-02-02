import torch

def compute_psnr(prediction: torch.Tensor, target: torch.Tensor):
    sample_mses = torch.reshape((prediction - target) ** 2, (1, -1)).mean(dim=-1)
    psnr = (-10 * torch.log10(sample_mses)).mean()
    return psnr
