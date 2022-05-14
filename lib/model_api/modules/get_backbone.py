import torch.nn as nn
from torchvision.ops import misc as misc_nn_ops
# from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.ops.feature_pyramid_network import (FeaturePyramidNetwork, 
                                                     LastLevelP6P7, LastLevelMaxPool)
# from torchvision.models._utils import IntermediateLayerGetter
# from torchvision.models.detection.backbone_utils import BackboneWithFPN

from collections import OrderedDict
from typing import Dict, Optional

from ..backbones.resnet import get_resnet


class IntermediateLayerGetter(nn.ModuleDict):
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        
        layers = OrderedDict()
        
        for name, module in model.named_children():
            if isinstance(module, nn.ModuleDict):
                for n, m in module.items():
                    if n in return_layers:
                        new_k = n
                        layers[new_k] = m
                        del return_layers[n]
            
            else:
                layers[name] = module                        
            
            if not return_layers:
                break
        
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers


    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            # print(name, x.size())
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        
        return out
    

class BackboneWithFPN(nn.Module):
    def __init__(self, backbone, return_layers, in_channels_list, out_channels, 
                 extra_blocks=None, use_fpn=True, backbone_type='origin'):
        super(BackboneWithFPN, self).__init__()
        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()
        
        if backbone_type == 'origin':
            self.body = backbone
        elif backbone_type == 'intermediate':
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.last_out_channel = backbone.last_out_channel
        
        if use_fpn:
            self.fpn = FeaturePyramidNetwork(
                in_channels_list=in_channels_list,
                out_channels=out_channels,
                extra_blocks=extra_blocks,
            )
            self.fpn_out_channels = out_channels

        self.use_fpn = use_fpn
    def forward(self, x):
        x = self.body(x)
        
        if self.use_fpn:
            if self.fpn:
                x = self.fpn(x)
                
        return x
    
    
    # def forward_stage(self, x, stage):
    #     assert(stage in ['layer1','layer2','layer3','layer4', 'layer1_without_conv'])
        
    #     if stage == 'layer1':
    #         return self.body.layer1(x)
        
    #     else: # Stage 2, 3 or 4
    #         layer = getattr(self.body, stage)
    #         return layer(x)


def resnet_fpn_backbone(
    backbone_name,
    backbone_args,
    weight_path=None,
    norm_layer=None,
    trainable_layers=3,
    returned_layers=None,
    extra_blocks=None,
    use_fpn=True,
    backbone_type='origin'
):
    
    # if 'res' in backbone_name:
    #     if dilation_type == 'fft':
    #         replace_stride_with_dilation = [False, False, True]
            
    #     elif dilation_type == 'fff':
    #         replace_stride_with_dilation = None
            
    #     elif dilation_type == 'ftt':
    #         replace_stride_with_dilation = [False, True, True]
        
    #     # print(replace_stride_with_dilation)
    #     kwargs = {
    #         # 'replace_stride_with_dilation': [False, False, False],
    #         'replace_stride_with_dilation': replace_stride_with_dilation,
    #         'norm_layer': norm_layer
    #     }
        
        # print(kwargs)
        
        # if backbone_setting == 'det':
        #     kwargs = {
        #         'norm_layer': norm_layer
        #     }
            
        # elif backbone_setting == 'seg':
        #     kwargs = {
        #         'replace_stride_with_dilation': [False, True, True]
        #     }
            
        # elif backbone_setting == 'clf':
        #     kwargs = None
        
        # else:
        #     kwargs = backbone_setting
        
    backbone = get_resnet(backbone_name, weight_path, **backbone_args)
    
    # select layers that wont be frozen
    assert 0 <= trainable_layers <= 4
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1'][:trainable_layers]
    
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)
    
    # for name, parameter in backbone.named_parameters():
    #     print(name, parameter.requires_grad)
    
    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
    
    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, 
                           extra_blocks=extra_blocks, use_fpn=use_fpn, backbone_type=backbone_type)


def build_backbone(arch, detector=None,
                   segmentor=None,
                   model_args=None):
                #    weight_path=None,
                #    use_fpn=True,
                #    freeze_backbone=False,
                #    freeze_bn=False,
                #    train_allbackbone=False,
                #    dilation_type='fft',
                #    backbone_type='intermediate'):
    
    
    returned_layer = None
    if not detector and not segmentor: # single-clf task
        returned_layer = [4]
        
    elif (detector and not segmentor) \
        or (not detector and segmentor): # train detection task or segemtation task 
        if segmentor:
            returned_layer = [3, 4]
    
    freeze_backbone = model_args['freeze_backbone']
    train_allbackbone = model_args['train_allbackbone']
    dilation_type = model_args['dilation_type']
    weight_path = model_args['state_dict']['backbone']
    backbone_type = model_args['backbone_type']
    freeze_bn = model_args['freeze_bn']
    use_fpn = model_args['use_fpn']
    
    if freeze_backbone:
        trainable_backbone_layers = 0
    elif train_allbackbone:
        trainable_backbone_layers = 4
    else:
        if detector and not segmentor:
            trainable_backbone_layers = 3
        else:
            trainable_backbone_layers = 4
            
    norm_layer = misc_nn_ops.FrozenBatchNorm2d if freeze_bn else None
    
    if dilation_type == 'fft':
            replace_stride_with_dilation = [False, False, True]
        
    elif dilation_type == 'fff':
        replace_stride_with_dilation = None
        
    elif dilation_type == 'ftt':
        replace_stride_with_dilation = [False, True, True]
    
    # print(replace_stride_with_dilation)
    backbone_kwargs = {
        # 'replace_stride_with_dilation': [False, False, False],
        'replace_stride_with_dilation': replace_stride_with_dilation,
        'norm_layer': norm_layer
    }
    
    shared_args = {
       'weight_path': weight_path,
       'returned_layers': returned_layer,
       'norm_layer': norm_layer,
       'trainable_layers': trainable_backbone_layers,
       'backbone_type': backbone_type
    }
    
    # print(detector, segmentor)
    # print(trainable_backbone_layers)
    # print(norm_layer)
    # print(dilation_type)
    # print(returned_layer)
    # print()
    
    if 'resnet' in arch:
        # if backbone_type == 'origin':
        #     backbone = get_resnet(arch, weight_path, **backbone_kwargs)
        #     if detector is not None:
                
        #         backbone.fpn_out_channels = 256
                
            
        # else:
        if detector is not None: 
            if 'faster' in detector:
                backbone = resnet_fpn_backbone(
                    arch,
                    backbone_kwargs,
                    # weight_path,
                    # returned_layers=returned_layer,
                    # norm_layer=norm_layer, # under experiment
                    # trainable_layers=trainable_backbone_layers,
                    **shared_args
                    )
            
            elif 'retina' in detector:
                backbone = resnet_fpn_backbone(
                    arch,
                    weight_path,
                    backbone_kwargs,
                    trainable_layers=trainable_backbone_layers,
                    returned_layers=[2, 3, 4],
                    extra_blocks=LastLevelP6P7(256, 256)
                                        )
            else:
                ValueError("The detector name {} is not supported detector.".format(detector))
        
        elif not detector and segmentor:
            assert not use_fpn
            shared_args.update({'use_fpn': use_fpn})
            backbone = resnet_fpn_backbone(
                arch,
                backbone_kwargs,
                # weight_path,
                # returned_layers=returned_layer,
                # use_fpn=use_fpn,
                # norm_layer=norm_layer, # under experiment
                # trainable_layers=trainable_backbone_layers
                **shared_args
                )
        
        elif not detector and not segmentor:
            assert not use_fpn
            shared_args.update({'use_fpn': use_fpn})
            backbone = resnet_fpn_backbone(
                arch,
                backbone_kwargs,
                # weight_path,
                # returned_layers=returned_layer,
                # use_fpn=use_fpn,
                # norm_layer=norm_layer, # under experiment
                # trainable_layers=trainable_backbone_layers,
                **shared_args
                )
        
    elif 'mobile' in arch:
        pass
    
    else:
        raise ValueError("The backbone name should be required.")

    
    return backbone





# class BackboneWithNoFPN(nn.Module):
#     def __init__(self, backbone, return_layers):
#         super(BackboneWithNoFPN, self).__init__()
#         self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
#         self.body_out_channel = backbone.last_out_channel


#     def forward(self, x):
#         x = self.body(x)
        
#         return x


# def resnet_backbone(
#     backbone_name,
#     use_segmentor=None,
#     weight_path=None,
#     pretrained=True,
#     norm_layer=torchvision.ops.misc.FrozenBatchNorm2d,
#     trainable_layers=4,
#     returned_layers=4,
# ):
#     if 'res' in backbone_name:
#         if use_segmentor:
#             replace_stride_with_dilation = [False, True, True]
#         else:
#             replace_stride_with_dilation = None
            
#         kwargs = {
#                 'replace_stride_with_dilation': replace_stride_with_dilation
#             }
        
#         backbone = get_resnet(backbone_name, weight_path, **kwargs)
    
#     assert 0 <= trainable_layers <= 4
#     layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1'][:trainable_layers]
    
#     for name, parameter in backbone.named_parameters():
#         if all([not name.startswith(layer) for layer in layers_to_train]):
#             parameter.requires_grad_(False)
    
#     # for n, p in backbone.named_parameters():
#     #     print(n, p.requires_grad)
        
#     if returned_layers is None:
#         returned_layers = [1, 2, 3, 4]
#     assert min(returned_layers) > 0 and max(returned_layers) < 5
#     return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
    
    
#     return BackboneWithNoFPN(backbone, return_layers)