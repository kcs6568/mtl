from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as tv_F

from ...apis.transforms_in_forward import (
    get_origin_size, get_sharing_size, resize_features)


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.constant_(m.bias, 0)


class ChannelWiseMultiply(nn.Module):
    def __init__(self, num_channels):
        super(ChannelWiseMultiply, self).__init__()
        self.param = nn.Parameter(torch.FloatTensor(num_channels), requires_grad=True)

    def init_value(self, value):
        with torch.no_grad():
            self.param.data.fill_(value)

    def forward(self, x):
        return torch.mul(self.param.view(1,-1,1,1), x)


class CrossStitchUnit(nn.Module):
    def __init__(self, tasks, num_channels, alpha, beta):
        super(CrossStitchUnit, self).__init__()
        self.cross_stitch_unit = nn.ModuleDict({t: nn.ModuleDict({t: ChannelWiseMultiply(num_channels) for t in tasks}) for t in tasks})
        
        for t_i in tasks:
            for t_j in tasks:
                if t_i == t_j:
                    self.cross_stitch_unit[t_i][t_j].init_value(alpha)
                else:
                    self.cross_stitch_unit[t_i][t_j].init_value(beta)


    def forward(self, task_features):
        out = {}
        for t_i in task_features.keys():
            prod = torch.stack([self.cross_stitch_unit[t_i][t_j](task_features[t_j]) for t_j in task_features.keys()])
            out[t_i] = torch.sum(prod, dim=0)
                
        return out


class CrossStitchNetwork(nn.Module):
    def __init__(self,
                 datasets,
                 models,
                 stages, channels, alpha, beta,
                 fpn_task, return_layers,
                 **kwargs) -> None:
        super().__init__()
        self.datasets = datasets
        self.models = models
        self.stages = stages 
        self.fpn_task = fpn_task
        self.return_layers = return_layers
        
        self.cross_stitch = nn.ModuleDict(
            {stage: CrossStitchUnit(self.datasets, channels[stage], alpha, beta) for stage in stages}
        )
        
        self.backbone_size = kwargs['backbone_size']
    
    def _foward_train(self, data_dict, tasks):
        # image tensor ordering: [batch size, channels, height, width]
        data = {dset: self.models[dset].stem(data_dict[dset][0]) for dset in self.datasets}
        
        stage_idx = 0
        feature_idx = {k: 0 for k in self.datasets}
        return_features = {k: {} for k in self.datasets}
        while stage_idx < self.backbone_size:
            for dset in self.datasets:
                data[dset] = self.models[dset].backbone.body[str(stage_idx)](data[dset])
            
            if str(stage_idx) in self.stages:
                origin_size = get_origin_size(data)
                mean_h, mean_w = get_sharing_size(origin_size)
                reshape_features = {dset: resize_features(stage_feats, mean_h, mean_w)
                                    for dset, stage_feats in data.items()}
                
                stitch_out = self.cross_stitch[str(stage_idx)](reshape_features)
                
                data = {dset: resize_features(stitch_out[dset], o_size_h, o_size_w)
                        for dset, (o_size_h, o_size_w) in origin_size.items()}
                
            for dset in self.datasets:
                if str(stage_idx) in self.return_layers[dset]:
                    return_features[dset].update({str(feature_idx[dset]): data[dset]})
                    feature_idx[dset] += 1

            stage_idx += 1
            
        # print(return_features['minicoco'])
        # print(stage_idx)
        # print(self.stages)
        # print(self.return_layers)
        # exit()
        
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
        # detection_fpn_feats = {k: {} for k in self.fpn_task}
        data = {dset: self.models[dset].stem(images)}
        return_features = OrderedDict()
        
        stage_idx = 0
        feature_idx = 0
        while stage_idx < self.backbone_size:
            data[dset] = self.models[dset].backbone.body[str(stage_idx)](data[dset])
            
            if str(stage_idx) in self.stages:
                data = self.cross_stitch[str(stage_idx)](data)
                
            if str(stage_idx) in self.return_layers[dset]:
                return_features.update({str(feature_idx): data[dset]})
                feature_idx += 1

            stage_idx += 1
        
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
            return self._foward_train(data_dict, kwargs)

        else:
            return self._forward_val(data_dict, kwargs)