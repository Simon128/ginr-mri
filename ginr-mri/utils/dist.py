import torch.distributed as torchdist

class Rank0Barrier:
    def __enter__(self):
        if torchdist.is_initialized():
            torchdist.barrier()

            if torchdist.get_rank() == 0:
                return True
            else:
                return False
        else:
            return True

    def __exit__(self, exctype, value, traceback):
        if torchdist.is_initialized():
            torchdist.barrier()

def rank0only():
    if torchdist.is_initialized():
        return torchdist.get_rank() == 0
    else:
        # no distributed training
        return True
