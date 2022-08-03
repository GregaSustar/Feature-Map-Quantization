import torch
import torch.nn as nn
from torch import Tensor
import companding
import quantization
import scaling
from newresnet import resnet101_blockn_split


class QuantResNet(nn.Module):
    """
    Full quantized resnet101 architecture
    """

    def __init__(self,
                 split: int,
                 compander: companding.Compander | None,
                 quantizer: quantization.Quantizer,
                 pretrained: bool = False,
                 cifar: bool = True,
                 num_classes: int = 1000):

        super().__init__()

        # create split models
        self.model_edge, self.model_cloud = resnet101_blockn_split(split, num_classes=num_classes,
                                                                   pretrained=pretrained, cifar=cifar)
        self.scaler = scaling.Scaler()
        self.compander = compander
        self.quantizer = quantizer


    def edgepass(self, x: Tensor):
        x = self.model_edge(x)
        return x


    def cloudpass(self, x: Tensor):
        x = self.model_cloud(x)
        return x


    def scale(self, x: Tensor):
        out = []
        for s in x:
            s = self.scaler.scale(s, (-1., 1.))
            out.append(s)
        out = torch.stack(out)
        return out


    def unscale(self, x: Tensor):
        out = []
        for s in x:
            s = self.scaler.unscale(s)
            out.append(s)
        out = torch.stack(out)
        return out


    def compress(self, x: Tensor):
        x = self.compander.mu_law_compress(x)
        return x


    def expand(self, x: Tensor):
        x = self.compander.mu_law_expand(x)
        return x


    def quantize(self, x: Tensor):
        x = self.quantizer.quantize(x)
        return x


    def dequantize(self, x: Tensor):
        x = self.quantizer.dequantize(x)
        return x


    def analysis(self, x: Tensor):
        x = self.edgepass(x)
        x = self.scale(x)
        if self.compander:
            x = self.compress(x)
        x = self.quantize(x)
        return x


    def synthesis(self, x: Tensor):
        x = self.dequantize(x)
        if self.compander:
            x = self.expand(x)
        x = self.unscale(x)
        x = self.cloudpass(x)
        return x


    def _forward_train(self, x):
        x = self.edgepass(x)
        x = _generate_noise(x)
        x = self.cloudpass(x)
        return x


    def _forward_infer(self, x):
        x = self.edgepass(x)
        x = self.scale(x)
        if self.compander:
            x = self.compress(x)

        x = self.quantize(x)
        x = self.dequantize(x)

        if self.compander:
            x = self.expand(x)
        x = self.unscale(x)
        x = self.cloudpass(x)
        return x


    def _forward_impl(self, x):
        if self.training:
            return self._forward_train(x)
        else:
            return self._forward_infer(x)


    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)



def _generate_noise(t: Tensor):
    r = 1 / 2
    noise = (r + r) * torch.rand_like(t) - r
    return t + noise



if __name__ == "__main__":
    h = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5],
                     [0.6, 0.7, 0.8, 0.9, 1.0]])
    print(_generate_noise(h))
