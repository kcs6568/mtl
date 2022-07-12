import torch
from torchvision.transforms import functional as tv_F


def get_origin_size(data):
        return {k: v.size()[-2:] for k, v in data.items()}
    
    
def get_sharing_size(origin_size, type='mean'):
    if type == 'mean':
        return torch.stack([torch.tensor(s[-2:]).float() for s in origin_size.values()
            ]).transpose(0, 1).mean(dim=1)
        
    
def resize_features(data, h_for_resize, w_for_resize):
    return tv_F.resize(data, (int(h_for_resize), int(w_for_resize)))