import nibabel as nib
import torch
import numpy as np

def save_tensor_as_nifti(tensor: torch.Tensor, filename: str):
    ndarr = tensor.detach().clone().cpu().numpy()
    image = nib.nifti1.Nifti1Image(ndarr, affine=np.eye(4))
    nib.save(image, filename)
