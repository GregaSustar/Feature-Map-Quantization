"""
Code Adjusted from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Union, List, Any
from torch.hub import load_state_dict_from_url


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.downsample(x)
        out = self.relu(out)
        return out



class ResNet(nn.Module):

    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 num_blocks: List[int],
                 init_conv: bool = True,
                 fin_fc: bool = True,
                 start_block_ix: int = 0,
                 cifar: bool = True,
                 num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = None

        self.init_conv = init_conv
        self.fin_fc = fin_fc
        self.start_block_ix = start_block_ix
        self.cifar = cifar

        conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        avgpool = nn.AvgPool2d(4)
        if not self.cifar:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        if self.init_conv:
            self.conv1 = conv1
            self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.layer2 = self.layer3 = self.layer4 = nn.Identity()
        if num_blocks[0]:
            self.in_planes = self.in_planes or 64
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)

        if num_blocks[1]:
            self.in_planes = self.in_planes or 256
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)

        if num_blocks[2]:
            self.in_planes = self.in_planes or 512
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        if num_blocks[3]:
            self.in_planes = self.in_planes or 1024
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        if self.fin_fc:
            self.fc = nn.Linear(512*block.expansion, num_classes)
            self.avgpool = avgpool


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        if self.start_block_ix:
            strides[0] = 1
            self.in_planes = planes * block.expansion
            for _ in range(self.start_block_ix):
                layers.append(
                    nn.Identity()
                )
            self.start_block_ix = 0


        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        if self.init_conv:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            if not self.cifar:
                x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.fin_fc:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


def _resnet(arch: str,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            pretrained: bool,
            progress: bool,
            **kwargs: Any,
            ) -> ResNet:

    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def _split_resnet(arch: str,
                  block: Type[Union[BasicBlock, Bottleneck]],
                  layers: (List[int], List[int]),
                  pretrained: bool,
                  progress: bool,
                  **kwargs: Any,
                  ) -> (ResNet, ResNet):

    split_layer = [l0 > 0 and l1 > 0 for l0, l1 in zip(layers[0], layers[1])]
    try:
        skip_layer = split_layer.index(True)
        start_block_ix = layers[0][skip_layer]
    except ValueError:
        start_block_ix = 0

    model_edge = ResNet(
        block, layers[0], fin_fc=False, **kwargs
    )
    model_cloud = ResNet(
        block, layers[1], init_conv=False, start_block_ix=start_block_ix, **kwargs
    )
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model_edge.load_state_dict(state_dict, strict=False)
        model_cloud.load_state_dict(state_dict, strict=False)

    return model_edge, model_cloud



def _split_list(n: int, lst: List[int]):
    i = 0
    while n > 0:
        n = n - lst[i]
        i += 1

    lst0 = lst[:i]
    lst1 = lst[i-1:]
    lst0 = lst0 + [0]*(4 - len(lst0))
    lst1 = [0]*(4 - len(lst1)) + lst1
    lst0[i-1] += n
    lst1[i-1] = -n

    return lst0, lst1



def resnet101_blockn_split(n: int, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> (ResNet, ResNet):
    if n < 0 or n > 34:
        raise ValueError(
            "Invalid block split for ResNet101. n needs to be between 0 and 34."
        )
    layers = _split_list(n, [3, 4, 23, 3])
    return _split_resnet(
        "resnet101", Bottleneck, layers, pretrained, progress, **kwargs
    )


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)
