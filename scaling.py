import torch
from torch import Tensor
from typing import Tuple
from collections import deque

class Scaler:

    def __init__(self):
        self.stats = deque()
        self.nstats = deque()


    def reset(self):
        self.stats.clear()
        self.nstats.clear()


    def scale(self, t: Tensor, interval: Tuple[float, float]):
        self.stats.appendleft((torch.min(t).item(), torch.max(t).item()))
        return rescale(t, interval)


    def unscale(self, t: Tensor):
        return rescale(t, self.stats.pop())


    def normalize(self, t: Tensor):
        _mean = torch.mean(t).item()
        _std = torch.std(t, unbiased=False).item()
        self.nstats.appendleft((_mean, _std))
        return (t - _mean) / _std


    def denormalize(self, t: Tensor):
        _mean, _std = self.nstats.pop()
        return t * _std + _mean



def rescale(t: Tensor, interval: Tuple[float, float]):
    """
    Args:
        t (Tensor): Input tensor
        interval (Tuple[float, float]): interval [a, b]
    Returns:
        Tensor: Input after rescaling values to interval [a,b]

    .. source: https://stackoverflow.com/questions/5294955/how-to-scale-down-a-range-of-numbers-with-a-known-min-and-max-value
    """
    a = interval[0]
    b = interval[1]
    _min = torch.min(t).item()
    _max = torch.max(t).item()
    t = t.clamp(min=_min, max=_max)
    t = (b - a) * (t - _min) / (_max - _min) + a
    return t


if __name__ == "__main__":

    x = torch.tensor([-0.124, -1.456, -0.003, 0.0001, 0.496, 1.446, 5.01033])
    amin = torch.min(x).item()
    amax = torch.max(x).item()

    print(rescale(x, (-1, 1)))
    print(rescale(x, (amin, amax)))
    print(rescale(x, (amin, amax)))
