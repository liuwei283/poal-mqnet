import torch
import torch.nn as nn
import time
import numpy as np
import random
from transform_layers import *

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count

def normalize(x, dim=1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_shift_module(P, eval=False):

    if P['shift_trans_type'] == 'rotation':
        shift_transform = Rotation()
        K_shift = 4
    elif P['shift_trans_type'] == 'cutperm':
        shift_transform = CutPerm()
        K_shift = 4
    else:
        shift_transform = nn.Identity()
        K_shift = 1

    return shift_transform, K_shift

def get_simclr_augmentation(image_size):
    # parameter for resizecrop
    resize_scale = (0.08, 1.0) # resize scaling factor

    # Align augmentation
    color_jitter = ColorJitterLayer(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8)
    color_gray = RandomColorGrayLayer(p=0.2)
    resize_crop = RandomResizedCropLayer(scale=resize_scale, size=image_size)

    transform = nn.Sequential(
        color_jitter,
        color_gray,
        resize_crop,
    )
    return transform