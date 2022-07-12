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


# class CrossStitchNetwork(nn.Module):
#     def __init__(self,
#                  datasets,
#                  stem: nn.ModuleDict,
#                  backbones: nn.ModuleDict, 
#                  heads: nn.ModuleDict,
#                  stages, channels, alpha, beta,
#                  fpn_task, return_layers
#                  ) -> None:
#         super().__init__()
#         self.datasets = datasets
#         self.stem = stem
#         self.backbones = backbones
#         self.heads = heads
#         self.stages = stages
#         self.fpn_task = fpn_task
#         self.return_layers = return_layers
        
#         self.cross_stitch = nn.ModuleDict(
#             {stage: CrossStitchUnit(self.datasets, channels[stage], alpha, beta) for stage in stages}
#         )
        
    
#     def _get_origin_size(self, data):
#         return {k: v.size()[-2:] for k, v in data.items()}
    
    
#     def _get_mean_size(self, origin_size):
#         return torch.stack([torch.tensor(s[-2:]).float() for s in origin_size.values()
#             ]).transpose(0, 1).mean(dim=1)
        
        
#     def _resize_features(self, data, h_for_resize, w_for_resize):
#         # return {dset: tv_F.resize(stage_feats, (int(h_for_resize), int(w_for_resize)))
#         #         for dset, stage_feats in data.items()}
        
#         return tv_F.resize(data, (int(h_for_resize), int(w_for_resize)))
        
        
#     def _foward_train(self, data_dict, tasks):
#         # image tensor ordering: [batch size, channels, height, width]
#         data = {dset: self.stem[dset](data_dict[dset][0]) for dset in self.datasets}
        
#         for dset, stem_f in data.items():
#             print(dset, stem_f.requires_grad)
        
#         detection_fpn_feats = {k: {} for k in self.fpn_task}
#         return_features = {k: {} for k in self.datasets}
#         # Backbone
#         for i, stage in enumerate(self.stages): # layer1, layer2, ...
    
#             # Forward through next stage of task-specific network
#             for dset in self.datasets:
#                 layer = getattr(self.backbones[dset].body, stage)
#                 data[dset] = layer(data[dset])
#                 print(dset)
#                 for n, p in layer.named_parameters():
#                     print(n, p.requires_grad)
#                 print("---"*40)
#             exit()
                
            
#             origin_size = self._get_origin_size(data)
#             mean_h, mean_w = self._get_mean_size(origin_size)
            
#             # print(data['cifar10'])
#             # a = self._resize_features(
#             #     data['cifar10'], mean_h, mean_w
#             # )
            
#             print(stage)
#             for dset, f in data.items():
#                 print(dset, f.requires_grad)
#             print()
            
#             reshape_features = {dset: self._resize_features(stage_feats, mean_h, mean_w)
#                                 for dset, stage_feats in data.items()}
            
#             stitch_out = self.cross_stitch[stage](reshape_features)
            
#             for dset, f in stitch_out.items():
#                 print(dset, f.requires_grad)
#             print()
            
#             data = {dset: self._resize_features(stitch_out[dset], o_size_h, o_size_w)
#                     for dset, (o_size_h, o_size_w) in origin_size.items()}
            
#             # return_features.update(
#             #     {for dset, layer in self.return_layers.items() if layer == stage}
#             # )
            
#             # origin_size = {k: v.size()[-2:] for k, v in data.items()}
#             # mean_h, mean_w  = torch.stack([
#             #     torch.tensor(s[-2:]).float() for s in origin_size.values()
#             #     ]).transpose(0, 1).mean(dim=1)
            
#             # reshape_features = {dset: tv_F.resize(stage_feats, (int(mean_h), int(mean_w)))
#             #                     for dset, stage_feats in data.items()}
            
#             # origin_size = self._get_origin_size(data)
#             # mean_h, mean_w = self._get_mean_size(origin_size)
#             # reshape_features = self._resize_features(data, mean_h, mean_w)
            
#             # stitch_out = self.cross_stitch[stage](reshape_features)
            
#             # data = {dset: tv_F.resize(stitch_out[dset], (int(o_size_h), int(o_size_w)))
#             #                     for dset, (o_size_h, o_size_w) in origin_size.items()}
            
#             # data = {dset: self._resize_features(stitch_out[dset], o_size_h, o_size_w)
#             #         for dset, (o_size_h, o_size_w) in origin_size.items()}
            
            
#             # for k in self.fpn_task:
#             #     detection_fpn_feats[k].update({str(i): data[k]})
            
#             for dset in self.datasets:
#                 if stage in self.return_layers[dset]:
#                     return_features[dset].update({str(i): data[dset]})
        
#         return_features.update({
#             dset: self.backbones[dset].fpn(return_features[dset]) for dset in self.fpn_task})
            
            
#         # data.update({k: self.backbones[k].fpn(return_features[k]) for k in self.fpn_task})
#         # return_features.update({k: self.backbones[k].fpn(return_features[k]) for k in self.fpn_task})
        
#         total_losses = OrderedDict()
#         for dset, task in tasks.items():
#             head = self.heads[dset]
#             backbone_features = return_features[dset]
#             print(backbone_features.requires_grad)
            
#             if task == 'det':
#                 losses = head(data_dict[dset][0], backbone_features, 
#                                        origin_targets=data_dict[dset][1], 
#                                        trs_fn=self.stem[dset].transform)
                
#             else:
#                 losses = head(backbone_features, data_dict[dset][1])
            
#             total_losses.update(losses)
        
#         # print(total_losses)
#         # exit()
#         return total_losses
            
    
#     def _forward_val(self, images, kwargs):
#         dtype = kwargs['dtype']
#         # task = kwargs['task']
#         detection_fpn_feats = {k: {} for k in self.fpn_task}
#         data = {dtype: self.stem[dtype](images)}
        
#         for i, stage in enumerate(self.stages): # layer1, layer2, ...
#             layer = getattr(self.backbones[dtype].body, stage)
#             data[dtype] = layer(data[dtype])
            
#             # origin_size = self._get_origin_size(data)
#             # mean_h, mean_w = self._get_mean_size(origin_size)
#             # reshape_features = {dset: self._resize_features(stage_feats, (int(mean_h), int(mean_w)))
#             #                     for dset, stage_feats in data.items()}
            
#             data = self.cross_stitch[stage](data)
            
#             # data = {dset: self._resize_features(stitch_out[dset], o_size_h, o_size_w)
#             #         for dset, (o_size_h, o_size_w) in origin_size.items()}

#             if dtype in self.fpn_task:
#                 for k in self.fpn_task:
#                     detection_fpn_feats[k].update({str(i): data[k]})
        
#         if dtype in self.fpn_task:
#             data.update({k: self.backbones[k].fpn(detection_fpn_feats[k]) for k in self.fpn_task})

#     def forward(self, data_dict, kwargs):
#         if self.training:
#             return self._foward_train(data_dict, kwargs)

#         else:
#             return self._forward_val(data_dict, kwargs)

class CrossStitchNetwork(nn.Module):
    def __init__(self,
                 datasets,
                 models,
                 stages, channels, alpha, beta,
                 fpn_task, return_layers
                 ) -> None:
        super().__init__()
        self.datasets = datasets
        self.models = models
        self.stages = stages
        self.fpn_task = fpn_task
        self.return_layers = return_layers
        
        self.cross_stitch = nn.ModuleDict(
            {stage: CrossStitchUnit(self.datasets, channels[stage], alpha, beta) for stage in stages}
        )
        
    
    def _foward_train(self, data_dict, tasks):
        # image tensor ordering: [batch size, channels, height, width]
        data = {dset: self.models[dset].stem(data_dict[dset][0]) for dset in self.datasets}
        
        return_features = {k: {} for k in self.datasets}
        for i, stage in enumerate(self.stages): # layer1, layer2, ...
    
            for dset in self.datasets:
                layer = getattr(self.models[dset].backbone.body, stage)
                data[dset] = layer(data[dset])
            
            origin_size = get_origin_size(data)
            mean_h, mean_w = get_sharing_size(origin_size)
            reshape_features = {dset: resize_features(stage_feats, mean_h, mean_w)
                                for dset, stage_feats in data.items()}
            
            stitch_out = self.cross_stitch[stage](reshape_features)
            
            data = {dset: resize_features(stitch_out[dset], o_size_h, o_size_w)
                    for dset, (o_size_h, o_size_w) in origin_size.items()}
            
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
        # detection_fpn_feats = {k: {} for k in self.fpn_task}
        data = {dset: self.models[dset].stem(images)}
        return_features = OrderedDict()
        
        # seg = True if task == 'seg' else False
        # if seg:
        #     print(images.size())
        for i, stage in enumerate(self.stages): # layer1, layer2, ...
            layer = getattr(self.models[dset].backbone.body, stage)
            data[dset] = layer(data[dset])
            
            # origin_size = self._get_origin_size(data)
            # mean_h, mean_w = self._get_mean_size(origin_size)
            # reshape_features = {dset: self._resize_features(stage_feats, (int(mean_h), int(mean_w)))
            #                     for dset, stage_feats in data.items()}
            
            data = self.cross_stitch[stage](data)
            
            # if seg:
            #     print(data[dset].size())
            
            
            # data = {dset: self._resize_features(stitch_out[dset], o_size_h, o_size_w)
            #         for dset, (o_size_h, o_size_w) in origin_size.items()}

            # if dset in self.fpn_task:
            #     for k in self.fpn_task:
            #         return_features[k].update({str(i): data[k]})

            if stage in self.return_layers[dset]:
                return_features.update({str(i): data[dset]})
    
        if dset in self.fpn_task:
            # data.update({dset: self.models[dset].backbone.fpn(return_features)})
            return_features = self.models[dset].backbone.fpn(return_features)
        
        # print(return_features)
        
        # if seg:
        #     s = {k: v.size() for k, v in return_features.items()}
        #     print(s)
        
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