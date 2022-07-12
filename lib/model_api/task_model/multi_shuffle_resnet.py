from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.get_detector import build_detector, DetStem
from ..modules.get_backbone import build_backbone
from ..modules.get_segmentor import build_segmentor, SegStem
from ..modules.get_classifier import build_classifier, ClfStem
from ...apis.loss_lib import AutomaticWeightedLoss
from ...apis.transforms_in_forward import (
    get_origin_size, get_sharing_size, resize_features)
from ..backbones.resnet import Bottleneck


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


def channel_shuffle(x, groups):    
    N,C,H,W = x.size()
    return x.view(N,groups,C//groups,H,W).permute(0,2,1,3,4).reshape(N,C,H,W)



class TaskEnhanceLayer(nn.Module):
    def __init__(self, in_channel, relu_type='relu', kernel_size = 1, stride = 1, padding = 0) -> None:
        super().__init__()
        mid_channel = in_channel // 4
        
        if relu_type == 'relu':
            activation = (nn.ReLU(inplace=True))
        elif relu_type == 'leaky':
            activation = (nn.LeakyReLU(inplace=True))
        
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(mid_channel),
            activation,
            nn.Conv2d(mid_channel, in_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(in_channel)
        )
        
    
    def forward(self, x):
        return self.layer(x)


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).reshape(N,C,H,W)


class FeatureShuffle(nn.Module):
    def __init__(self, in_channel, groups, relu_type='relu') -> None:
        super().__init__()
        mid_channel = in_channel//4
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.shuffle= ShuffleBlock(groups=groups)
        self.conv2 = nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1, groups=mid_channel, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channel)
        self.conv3 = nn.Conv2d(mid_channel, in_channel, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channel)
        
        if relu_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif relu_type == 'leaky':
            self.activation = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.shuffle(out)
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        return out


class MultiShuffleNetwork(nn.Module):
    def __init__(self,
                 backbone,
                 detector,
                 segmentor,
                 task_cfg,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.backbone = build_backbone(
            backbone, detector, segmentor, kwargs)
        self.channel_cfg = kwargs['channel_cfg']
        self.stages = kwargs['stages']
        self.task_groups = kwargs['task_groups']
        self.datasets = list(task_cfg.keys())
        relu_type = kwargs['relu_type']
        self.aggr_type = kwargs['aggr_type']
        
        self.return_layers = kwargs['return_layers']
        self.fpn_task = kwargs['fpn_task']
        
        self.stem_dict = nn.ModuleDict()
        self.head_dict = nn.ModuleDict()
        self.task_enhance_layer = nn.ModuleDict()
        # self.task_squeeze_layer = nn.ModuleDict()
        # self.task_excitation_layer = nn.ModuleDict()
        self.shuffle_layer = nn.ModuleDict()
        self.bn_relu_before_shuffle = nn.ModuleDict()
        
        stem_weight = kwargs['state_dict']['stem']
        for data, cfg in task_cfg.items():
            task = cfg['task']
            num_classes = cfg['num_classes']
            if task == 'clf':
                stem = ClfStem(**cfg['stem'])
                head = build_classifier(
                    backbone, num_classes, cfg['head'])
                stem.apply(init_weights)
                
            elif task == 'det':
                stem = DetStem(**cfg['stem'])
                
                head_kwargs = {'num_anchors': len(self.backbone.body.return_layers)+1}
                head = build_detector(
                    backbone, detector, 
                    self.backbone.fpn_out_channels, num_classes, **head_kwargs)
                if stem_weight is not None:
                    ckpt = torch.load(stem_weight)
                    stem.load_state_dict(ckpt, strict=False)
                    print("!!!Load weights for detection stem layer!!!")
            
            elif task == 'seg':
                stem = SegStem(**cfg['stem'])
                head = build_segmentor(segmentor, num_classes=num_classes, cfg_dict=cfg['head'])
                if stem_weight is not None:
                    ckpt = torch.load(stem_weight)
                    stem.load_state_dict(ckpt, strict=False)
                    print("!!!Load weights for segmentation stem layer!!!")
            
            head.apply(init_weights)
            self.stem_dict.update({data: stem})
            self.head_dict.update({data: head})
            
        if relu_type == 'relu':
            self.last_activation = F.relu
        elif relu_type == 'leaky':
            self.last_activation = F.leaky_relu
        
        for stage, channel in zip(self.stages, self.channel_cfg):
            # squeeze = {
            #     dataset: nn.Sequential(
            #         nn.Conv2d(channel, channel // 4, 1, 1, 0),
            #         nn.BatchNorm2d(channel // 4),
            #         nn.ReLU(inplace=True)) for dataset in task_cfg.keys()
            # }
            # self.task_squeeze_layer[stage] = nn.ModuleDict(squeeze)
            
            enhance = {
                dataset: TaskEnhanceLayer(channel, relu_type=relu_type) for dataset in task_cfg.keys()
            }
            self.task_enhance_layer[stage] = nn.ModuleDict(enhance)
            self.shuffle_layer[stage] = FeatureShuffle(channel, self.task_groups, relu_type=relu_type)
            
            self.bn_relu_before_shuffle[stage] = nn.Sequential(
                nn.BatchNorm2d(channel),
                # nn.ReLU(inplace=True)
            )
            
            # excitation = {
            #     dataset: nn.Sequential(
            #         nn.Conv2d(channel // 4, channel, 1, 1, 0),
            #         nn.BatchNorm2d(channel),
            #         nn.ReLU(inplace=True)) for dataset in task_cfg.keys()
            # }
            
            # self.task_excitation_layer[stage] = nn.ModuleDict(excitation)
        
    
    def _forward_train(self, data_dict, tasks):
        data = {dset: self.stem_dict[dset](data_dict[dset][0]) for dset in self.datasets}
        return_features = {k: {} for k in self.datasets}
        for i, stage in enumerate(self.stages): # layer1, layer2, ...
            task_feats = {}
            for dset in self.datasets:
                batch_size = len(data_dict[dset][0])
                layer = getattr(self.backbone.body, stage)
                layer_out = layer(data[dset])
                enhance_feats = self.task_enhance_layer[stage][dset](layer_out)
                task_feats.update({dset: enhance_feats})
                
                # squeeze_feats = self.task_squeeze_layer[stage][dset](enhance_feats)
                # task_feats.update({dset: self.task_enhance_layer[stage][dset](layer_out)})
            
            origin_size = get_origin_size(task_feats)
            mean_h, mean_w = get_sharing_size(origin_size)
            # reshape_features = {dset: resize_features(stage_feats, mean_h, mean_w)
            #                     for dset, stage_feats in task_feats.items()}
            reshape_features = [resize_features(stage_feats, mean_h, mean_w) for stage_feats in task_feats.values()]
            
            all_task_feats = torch.cat(reshape_features, dim=0)
            
            # if self.aggr_type == 'sum':
            #     all_task_feats = torch.sum(torch.stack(reshape_features, dim=0), dim=0)
            # elif self.aggr_type == 'mul':
            #     all_task_feats = None
            #     for f in reshape_features:
            #         if all_task_feats is None:
            #             all_task_feats = f
            #         else:
            #             all_task_feats *= f
                        
            
            bn_relu_feats = self.bn_relu_before_shuffle[stage](all_task_feats)
            after_shuffle_and_unshuffle = self.shuffle_layer[stage](bn_relu_feats)
            splited = torch.split(after_shuffle_and_unshuffle, batch_size, dim=0)
            origin_feats = {dset: resize_features(splited[i], o_size_h, o_size_w)
                    for i, (dset, (o_size_h, o_size_w)) in enumerate(origin_size.items())}
            
            # origin_feats = {dset: resize_features(after_shuffle_and_unshuffle, o_size_h, o_size_w)
            #         for dset, (o_size_h, o_size_w) in origin_size.items()}
            # feature_for_shuffle = [all_task_feats*feats for feats in task_feats.values()]
            # concat_for_shuffle = torch.cat(list(reshape_features.values()), dim=1)
            # after_shuffle_and_unshuffle = self.shuffle_layer[stage](concat_for_shuffle)
            # splited = torch.split(after_shuffle_and_unshuffle, self.channel_cfg[i] // 4, dim=1)
            # print("splited", splited[0].size(), splited[1].size(), splited[2].size(), splited[3].size())
            # origin_feats = {dset: resize_features(splited[i], o_size_h, o_size_w)
            #         for i, (dset, (o_size_h, o_size_w)) in enumerate(origin_size.items())}
            # print(origin_feats['cifar10'].size(), origin_feats['stl10'].size(), origin_feats['minicoco'].size(), origin_feats['voc'].size())
            
            data = {dset: self.last_activation(feats + task_feats[dset]) for dset, feats in origin_feats.items()}
            
            for dset in self.datasets:
                if stage in self.return_layers[dset]:
                    return_features[dset].update({str(i): data[dset]})
        
        return_features.update({
            dset: self.backbone.fpn(return_features[dset]) for dset in self.fpn_task})
        
        total_losses = OrderedDict()
        for dset, task in tasks.items():
            head = self.head_dict[dset]
            backbone_features = return_features[dset]
            targets = data_dict[dset][1]
            
            if task == 'clf':
                losses = head(backbone_features, targets)
                
            elif task == 'det':
                losses = head(data_dict[dset][0], backbone_features, 
                                        origin_targets=targets, 
                                        trs_fn=self.stem_dict[dset].transform)
            elif task == 'seg':
                losses = head(
                    backbone_features, targets, input_shape=data_dict[dset][0].shape[-2:])

            total_losses.update({f"{dset}_{k}": l for k, l in losses.items()})
        
        return total_losses
        
        
    def _forward_val(self, images, kwargs):
        dset = list(kwargs.keys())[0]
        task = list(kwargs.values())[0]
        data = self.stem_dict[dset](images)
        return_features = {dset: {}}
        for i, stage in enumerate(self.stages): # layer1, layer2, ...
            # task_feats = {}
            layer = getattr(self.backbone.body, stage)
            layer_out = layer(data)
            enhance_feats = self.task_enhance_layer[stage][dset](layer_out)
                
            # data[dset] = self.shuffle_layer[stage](enhance_feats)
            bn_relu_feats = self.bn_relu_before_shuffle[stage](enhance_feats)
            after_shuffle_and_unshuffle = self.shuffle_layer[stage](bn_relu_feats)
            data = self.last_activation(enhance_feats + after_shuffle_and_unshuffle)
            
            # data = self.task_excitation_layer[stage][dset](after_shuffle_and_unshuffle)
            
            if stage in self.return_layers[dset]:
                return_features[dset].update({str(i): data})
        
        if dset in self.fpn_task:
            return_features.update({
                dset: self.backbone.fpn(return_features[dset])})
        
        head = self.head_dict[dset]
        backbone_feat = return_features[dset]
        if task == 'clf':
            out = head(backbone_feat)
            
            return dict(outputs=out)
            
        elif task == 'det':
            out = head(images, backbone_feat, 
                                trs_fn=self.stem_dict[dset].transform)
            return out
            
        elif task == 'seg':
            out = head(
                backbone_feat, input_shape=images.shape[-2:])
            return dict(outputs=out)
        
        
    def forward(self, data_dict, kwargs):
        if self.training:
            return self._forward_train(data_dict, kwargs)

        else:
            return self._forward_val(data_dict, kwargs)
        
