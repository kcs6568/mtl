from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import functional as F


class SegStem(nn.Module):
    def __init__(self,
                 init_channels=64,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 stem_weight=None) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, init_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=False)
        self.bn = nn.BatchNorm2d(init_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        if stem_weight:
            ckpt = torch.load(stem_weight)
            self.load_state_dict(ckpt)
        
        
    def forward(self, x):
        if self.training:
            assert x.size()[2] == x.size()[3]
            
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        return x



class FCNHead(nn.Module):
    def __init__(self, in_channels=2048, num_classes=21, use_aux=True) -> None:
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.fcn_head = self._make_fcn_head(in_channels, inter_channels, num_classes)
        if use_aux:
            aux_inchannels = in_channels // 2
            self.aux_head = self._make_fcn_head(aux_inchannels, aux_inchannels//4, num_classes)
        

    def _make_fcn_head(self, in_channels, inter_channels, num_classes):
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, num_classes, 1)
        ]
        
        return nn.Sequential(*layers)


    def criterion(self, inputs, target):
        losses = {}
        for name, x in inputs.items():
            losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)
        
        if "seg_aux_loss" in losses:
            losses["seg_aux_loss"] *= 0.5
         
        return losses
        

    def forward(self, feats, target=None, input_shape=480):
        assert isinstance(feats, dict) or isinstance(feats, OrderedDict)
        
        results = OrderedDict()
        _, x = feats.popitem()
        # print(x.size())
        x = self.fcn_head(x)
        # print(x.size())
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        # print(x.size())
        results["seg_out_loss"] = x
        
        if self.aux_head is not None:
            _, x = feats.popitem()
            # print(x.size())
            x = self.aux_head(x)
            # print(x.size())
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            # print(x.size())
            results["seg_aux_loss"] = x
        # exit()
        if self.training:
            return self.criterion(results, target)

        else:
            return results["seg_out_loss"]
        

def build_segmentor(
    segmentor_name,
    cfg_dict=None,
    detector=None,
    pretrained=False,
    ):
    
    segmentor_name = segmentor_name.lower()
    
    if 'maskrcnn' in segmentor_name:
        from torchvision.ops import MultiScaleRoIAlign
        from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
        
        mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=14,
            sampling_ratio=2)

        mask_layers = (256, 256, 256, 256)
        mask_dilation = 1
        mask_head = MaskRCNNHeads(cfg_dict['out_channels'], mask_layers, mask_dilation)
        
        mask_predictor_in_channels = 256  # == mask_layers[-1]
        mask_dim_reduced = 256
        mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels,
                                            mask_dim_reduced, cfg_dict['num_classes'])
        

        detector.roi_heads.mask_roi_pool = mask_roi_pool
        detector.roi_heads.mask_head = mask_head
        detector.roi_heads.mask_predictor = mask_predictor
        
        return detector

    else:
        if 'fcn' in segmentor_name:
            head = FCNHead(**cfg_dict)
        
        elif 'deeplab' in segmentor_name:
            head = None    
        
        return head
    
        
