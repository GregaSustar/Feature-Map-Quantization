import os

import matplotlib
import torch
import shutil
import numpy as np
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt

from quantresnet import QuantResNet
from newresnet import ResNet


def save_checkpoint(state, is_best, dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = 'last_checkpoint.ckpt.tar'
    path = os.path.join(dir_path, filename)
    torch.save(state, path)
    if is_best:
        filename = 'model_best.ckpt.tar'
        shutil.copyfile(path, os.path.join(dir_path, filename))



def load_state_dict(checkpoint, model):
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    if checkpoint["model"] == str(ResNet) and str(model.__class__) == str(QuantResNet):
        model.model_edge.load_state_dict(checkpoint['state_dict'], strict=False)
        model.model_cloud.load_state_dict(checkpoint['state_dict'], strict=False)



def kaiming_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, mode="fan_out", nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']




def show(t: Tensor, split: int, q: bool = False):
    t = t.detach().cpu().numpy()
    plt.figure(figsize=(20, 20))

    if t.shape[0] < 4:
        plt.imshow(t.T)
        plt.axis("off")
        plt.savefig(f"results/split[{split}]_original_image.png")
        plt.close()
        return

    for i, f in enumerate(t):
        if i == 64:  # we will visualize only 8x8 blocks from each layer
            break
        plt.subplot(8, 8, i + 1)
        plt.imshow(f.T, cmap='gray')
        plt.axis("off")

    if q:
        plt.savefig(f"results/split[{split}]_quant_featmaps.png")
    else:
        plt.savefig(f"results/split[{split}]_featmaps.png")
    plt.close()
