import torch
from torch import Tensor

import scaling


class Quantizer:
    """
    Uniform Quantizer
    """

    def __init__(self, qbins):
        self.qbins = qbins


    def quantize(self, t: Tensor):
        return quantize(t, self.qbins)


    def dequantize(self, t: Tensor):
        return dequantize(t)



def quantize(t: Tensor, qbins: int):
    """
    Args:
        t (Tensor): Input tensor
        qbins (int): Number of quantization bins
    Returns:
        Tensor: quantized uint8 tensor

    Uniform quantization by rounding to nearest integer 'round(x)' - [Round half up ( floor(x + 1/2) )]
    """
    t = scaling.rescale(t, (0., qbins - 1)) + 0.5
    return t.byte()




def dequantize(t: Tensor):
    """
    Args:
        t (Tensor): Input tensor
    Returns:
        Tensor: dequantized float32 tensor

    Maps quantized values to appropriate values between -1 and 1 and converts them to float32
    """
    t = scaling.rescale(t, (-1, 1))
    return t.float()



if __name__ == "__main__":
    r1 = -3
    r2 = 18
    a = 5
    b = 5
    _qbins = 8

    x = scaling.rescale(torch.rand(a, b), (r1, r2))
    qx = quantize(x, _qbins)
    dqx = dequantize(qx)

    print(x)
    print(x.dtype)
    print(qx)
    print(qx.dtype)
    print(dqx)
    print(dqx.dtype)

