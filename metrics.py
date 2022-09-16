import os.path
from typing import Tuple
import torch
from torch import Tensor
import numpy as np

def accuracy(output, target, topk=(1,)):
    """
        Computes the accuracy over the k top predictions for the specified values of k
        ..source: https://github.com/pytorch/examples/blob/main/imagenet/main.py
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



def bytesize(inp: Tensor | str):
    """
    Args:
        inp (Tensor or String): Input tensor or path to file with compressed content
    Returns:
        Integer: size of tensor or file in bytes
    """

    if isinstance(inp, Tensor):
        return inp.element_size() * inp.nelement()

    if isinstance(inp, bytes):
        return len(inp)

    if isinstance(inp, np.ndarray):
        return inp.nbytes

    if isinstance(inp, str):
        if os.path.exists(inp):
            return os.path.getsize(inp)

    raise ValueError("input must be either a torch.Tensor or a valid file path")



def bitsize(inp: Tensor | str):
    """
    Args:
        inp (Tensor or String): Input tensor or path to file with compressed content
    Returns:
        Integer: size of tensor or file in bits
    """
    return bytesize(inp) * 8



def bpp(bsize: int, res: Tuple[int, int]):
    """
    Args:
        bsize (Integer): size of an object in bits
        res (Tuple[Integer, Integer]): resolution (height, width)
    Returns:
        Integer: bits per pixel
    """
    return bsize / (res[0] * res[1])



def CR(bpp_compressed):
    """
        Compression ratio:

        bpp(uncompresed)/bpp(compressed)

        Original image color depth is 24 bits, hence bpp(uncompressed) is always 24.
    """
    return 24 / bpp_compressed
