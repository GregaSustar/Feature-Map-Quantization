import torch
from torch import Tensor
import scaling


class Compander:

    def __init__(self, mu):
        self.mu = mu


    def mu_law_compress(self, t: Tensor):
        return mu_law_compress(t, self.mu)


    def mu_law_expand(self, t_mu: Tensor):
        return mu_law_expand(t_mu, self.mu)



def mu_law_compress(t: Tensor, mu: int):
    """
    Args:
        t (Tensor): Input tensor
        mu (int): degree of non-uniformity (always set to number of quantization bins - 1)
    Returns:
        Tensor: Input after applying mu-law function

    This algorithm expects that the input tensor has been scaled to between -1 and 1

    .. source: https://en.wikipedia.org/wiki/%CE%9C-law_algorithm#cite_note-mulaw-equation-1
    .. source: https://pytorch.org/audio/stable/_modules/torchaudio/functional/functional.html#mu_law_encoding
    """
    if not t.is_floating_point():
        raise TypeError(
            "The input Tensor must be of floating type."
        )
    mu = torch.tensor(mu, dtype=t.dtype)
    t_mu = torch.sign(t) * torch.log1p(mu * torch.abs(t)) / torch.log1p(mu)
    return t_mu


def mu_law_expand(t_mu: Tensor, mu: int):
    """
    Args:
        t_mu (Tensor): Input tensor
        mu (int): degree of non-uniformity (always set to number of quantization bins - 1)
    Returns:
        Tensor: Input after applying inverse mu-law function

    This algorithm expects that the input tensor has been scaled to between -1 and 1

    .. source: https://en.wikipedia.org/wiki/%CE%9C-law_algorithm#cite_note-mulaw-equation-1
    .. source: https://pytorch.org/audio/stable/_modules/torchaudio/functional/functional.html#mu_law_decoding
    """
    if not t_mu.is_floating_point():
        t_mu = t_mu.to(torch.float)
    mu = torch.tensor(mu, dtype=t_mu.dtype)
    t = torch.sign(t_mu) * (torch.exp(torch.abs(t_mu) * torch.log1p(mu)) - 1.0) / mu
    return t


if __name__ == "__main__":
    r1 = -1
    r2 = 1
    a = 5
    b = 5
    _qbins = 8
    _mu = _qbins - 1

    x = scaling.rescale(torch.rand(a, b), (r1, r2))
    qx = mu_law_compress(x, _mu)
    dqx = mu_law_expand(qx, _mu)

    print(x)
    print(qx)
    print(dqx)

    print(torch.isclose(x, dqx))
