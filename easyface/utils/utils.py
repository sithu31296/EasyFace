import os
import pickle
import torch
import time
import numpy as np
from torch import distributed as dist


def l2_norm(x, dim=1):
    norm = torch.norm(x, 2, dim, keepdim=True)
    output = torch.div(x, norm)
    return output, norm

def l2_norm_np(x: np.ndarray, axis=1):
    norm = np.sqrt(np.sum(x**2, axis=axis, keepdims=True))
    output = x / norm
    return output, norm


def cosine_similarity(x1: np.ndarray, x2: np.ndarray):
    x1 = l2_norm_np(x1)
    x2 = l2_norm_np(x2)
    return x1 @ x2.T


def fuse_feats_with_norm(feats, norms):
    assert feats.ndim == 3  # (N_features, BS, C)
    assert norms.ndim == 3  # (N_features, BS, 1)
    fused = (feats * norms).sum(dim=0)
    fused, fused_norm = l2_norm(fused, dim=1)
    return fused, fused_norm


def is_dist_avail_and_is_initalized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_is_initalized():
        return 1
    return dist.get_world_size()


def get_local_rank():
    if not is_dist_avail_and_is_initalized():
        return 0
    return int(os.environ['LOCAL_RANK'])


class Timer:
    def __init__(self) -> None:
        self.clear()

    def tic(self):
        self.start_time = time.time()

    def toc(self):
        self.total_time += time.time() - self.start_time
        self.calls += 1
        return self.total_time / self.calls
    
    def clear(self):
        self.total_time = 0.
        self.start_time = 0.
        self.calls = 0.
        










def all_gather(data):
    world_size = get_world_size()
    local_rank = get_local_rank()

    if world_size == 1:
        return [data]

    device = torch.device('cuda', local_rank)
    
    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(local_rank)

    # obtain tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device=device)
    size_list = [torch.tensor([0], device=device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving tensor from all ranks
    # pad the tensor because dist.all_gather does not support gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device=device))
    
    if local_size != max_size:
        padding = torch.empty((max_size - local_size,), dtype=torch.uint8, device=device)
        tensor = torch.cat((tensor, padding), dim=0)

    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))
    return data_list