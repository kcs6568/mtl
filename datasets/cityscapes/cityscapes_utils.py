import os 
import numpy as np

import torch.nn as nn
from torchvision.transforms.functional import normalize


def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std)


# def get_transforms(type):
#     if type == 'train':
        
    