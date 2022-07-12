from collections import OrderedDict
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as tv_F

from ...apis.transforms_in_forward import (
    get_origin_size, get_sharing_size, resize_features)
from ..modules.se_module import ChannelSE

class NDDRLayer(nn.Module):
    def __init__(self, datasets, channels, alpha, beta, use_se=False):
        super(NDDRLayer, self).__init__()
        self.datasets = datasets
        in_channel_length = 1 if use_se else len(self.datasets)
        self.layer = nn.ModuleDict({dset: nn.Sequential( # Momentum set as NDDR-CNN repo
                                        nn.Conv2d(in_channel_length * channels, channels, 1, 1, 0, bias=False), 
                                        nn.BatchNorm2d(channels, momentum=0.05), 
                                        nn.ReLU()) for dset in self.datasets}) 
        self.se = ChannelSE(channels, r=len(self.datasets))
        
        # Initialize
        for i, dset in enumerate(self.datasets):
            layer = self.layer[dset]
            t_alpha = torch.diag(torch.FloatTensor([alpha for _ in range(channels)])) # C x C
            t_beta = torch.diag(torch.FloatTensor([beta for _ in range(channels)])).repeat(1, in_channel_length) # C x (C x T)
            
            t_alpha = t_alpha.view(channels, channels, 1, 1)
            t_beta = t_beta.view(channels, channels * in_channel_length, 1, 1)

            layer[0].weight.data.copy_(t_beta)
            if use_se:
                layer[0].weight.data[:,:channels].copy_(t_alpha)
            else:
                layer[0].weight.data[:,int(i*channels):int((i+1)*channels)].copy_(t_alpha)
            layer[1].weight.data.fill_(1.0)
            layer[1].bias.data.fill_(0.0)

    def forward(self, x):
        if self.training:
            x = torch.cat([self.se(x[dset]) for dset in self.datasets], 1) # Use self.tasks to retain order!
            output = {dset: self.layer[dset](x) for dset in self.datasets}
        else:
            output = {dset: self.layer[dset](feats) for dset, feats in x.items()}
        
        return output


class NDDRCNN(nn.Module):
    def __init__(self, datasets, 
                 models: nn.ModuleDict, 
                 fpn_task,
                 return_layers: list, 
                 stages: list, channels: dict, 
                 alpha: float, beta: float,
                 use_se):
        super(NDDRCNN, self).__init__()
        
        # Tasks, backbone and heads
        self.datasets = datasets
        self.models = models
        self.fpn_task = fpn_task
        self.return_layers = return_layers
        self.stages = stages

        # NDDR-CNN units
        self.nddr = nn.ModuleDict({stage: NDDRLayer(datasets, channels[stage], alpha, beta, use_se) for stage in stages})
        
    # def _get_origin_size(self, data):
    #     return {k: v.size()[-2:] for k, v in data.items()}
    
    
    # def _get_mean_size(self, origin_size):
    #     return torch.stack([torch.tensor(s[-2:]).float() for s in origin_size.values()
    #         ]).transpose(0, 1).mean(dim=1)
        
        
    # def _resize_features(self, data, h_for_resize, w_for_resize):
    #     return tv_F.resize(data, (int(h_for_resize), int(w_for_resize)))


    def _forward_train(self, data_dict, tasks):
        data = {dset: self.models[dset].stem(data_dict[dset][0]) for dset in self.datasets}
        
        
        return_features = {k: {} for k in self.datasets}
        for i, stage in enumerate(self.stages):
            for dset in self.datasets:
                layer = getattr(self.models[dset].backbone.body, stage)
                data[dset] = layer(data[dset])
            
            origin_size = get_origin_size(data)
            mean_h, mean_w = get_sharing_size(origin_size)
            reshape_features = {dset: resize_features(stage_feats, mean_h, mean_w)
                                for dset, stage_feats in data.items()}
            
            nddr_out = self.nddr[stage](reshape_features)
            
            data = {dset: resize_features(nddr_out[dset], o_size_h, o_size_w)
                for dset, (o_size_h, o_size_w) in origin_size.items()}
            
            # return_features[dset].update({str(i): data[dset] for dset in self.datasets if stage in self.return_layers[dset]})
            
            for dset in self.datasets:
                if stage in self.return_layers[dset]:
                    return_features[dset].update({str(i): data[dset]})
        
        
        
        return_features.update({
            dset: self.models[dset].backbone.fpn(return_features[dset]) for dset in self.fpn_task})
        
        total_losses = OrderedDict()
        for dset, task in tasks.items():
            head = self.models[dset].head
            backbone_features = return_features[dset]
            
            if task == 'det':
                losses = head(data_dict[dset][0], backbone_features, 
                                       origin_targets=data_dict[dset][1], 
                                       trs_fn=self.models[dset].stem.transform)
                
            else:
                losses = head(backbone_features, data_dict[dset][1])
            
            
            losses = {f"{dset}_{k}": l for k, l in losses.items()}
            total_losses.update(losses)
        
        return total_losses


    def _forward_val(self, images, kwargs):
        dset = list(kwargs.keys())[0]
        task = list(kwargs.values())[0]
        data = {dset: self.models[dset].stem(images)}
        
        return_features = OrderedDict()
        for i, stage in enumerate(self.stages): # layer1, layer2, ...
            layer = getattr(self.models[dset].backbone.body, stage)
            data[dset] = layer(data[dset])
            
            data = self.nddr[stage](data)

            if stage in self.return_layers[dset]:
                return_features.update({str(i): data[dset]})
    
        if dset in self.fpn_task:
            return_features = self.models[dset].backbone.fpn(return_features)
        
        
        if task == 'det':
            predictions = self.models[dset].head(
                images, return_features,
                trs_fn=self.models[dset].stem.transform)
            
            return predictions
            
        else:
            if task == 'seg':
                predictions = self.models[dset].head(
                    return_features, input_shape=images.shape[-2:])
            
            else:
                predictions = self.models[dset].head(
                    return_features)
            
            return dict(outputs=predictions)


    def forward(self, data_dict, kwargs):
        if self.training:
            return self._forward_train(data_dict, kwargs)

        else:
            return self._forward_val(data_dict, kwargs)
