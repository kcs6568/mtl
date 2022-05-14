from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..get_detector import build_detector, DetStem
from ..get_backbone import build_backbone
from ..get_segmentor import build_segmentor, SegStem
from ..get_classifier import build_classifier, ClfStem
from ...apis.loss_lib import AutomaticWeightedLoss

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

        
class SKBackbone(nn.Module):
    def __init__(self,
                 backbone_name,
                 detector_name,
                 segmentor_name,
                 backbone_weight=None,
                 freeze_backbone=False,
                 loss_reduction_rate=None,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.backbone = build_backbone(
            backbone_name, detector_name, segmentor_name, weight_path=backbone_weight,
            train_allbackbone=kwargs['train_allbackbone'],
            use_fpn=kwargs['use_fpn'],
            freeze_all_backbone_layers=freeze_backbone,
            freeze_bn=kwargs['freeze_bn'])
        
    
    def _foward_train(self, data_dict, kwargs={}):
        total_losses = OrderedDict()
        
            
        return total_losses
    
    
    def _forward_val(self, images, kwargs):
        pass
    
    def forward(self, data_dict, kwargs={}):
        if self.training:
            return self._foward_train(data_dict, kwargs)

        else:
            return self._forward_val(data_dict, kwargs)
        
        
class SKClfStem(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.clf_stem = ClfStem()
        self.clf_stem.apply(init_weights)
        
        
    def forward(self, data):
        pass
    
    
class SKClfHead(nn.Module):
    def __init__(self, body_out_channel, num_classes) -> None:
        super().__init__()
        self.clf_head = build_classifier(body_out_channel, num_classes)
        self.clf_head.apply(init_weights)
        
        
    def forward(self, data):
        pass
    
    
class SKDetStem(nn.Module):
    def __init__(self, stem_weight=None) -> None:
        super().__init__()
        self.det_stem = DetStem(stem_weight=stem_weight)
        
        if not stem_weight:
            self.det_stem.apply(init_weights)
        
        
    def forward(self, data):
        pass
    
    
class SKDetHead(nn.Module):
    def __init__(self, detector_name, fpn_out_channels, num_classes) -> None:
        super().__init__()
        self.det_head = build_detector(
            detector_name, fpn_out_channels, num_classes
        )
        self.det_head.apply(init_weights)
        
        
    def forward(self, data):
        pass
    
    
class SKSegStem(nn.Module):
    def __init__(self, stem_weight=None) -> None:
        super().__init__()
        self.seg_stem = SegStem(stem_weight=stem_weight)
        if not stem_weight:
            self.seg_stem.apply(init_weights)
        
        
    def forward(self, data):
        pass
    
    
class SKSegHead(nn.Module):
    def __init__(self, segmentor_name, backbone_outc, num_classes) -> None:
        super().__init__()
        head_cfg = {
            'in_channels': backbone_outc,
            'num_classes': num_classes
        }
        
        self.seg_head = build_segmentor(segmentor_name, head_cfg)
        self.seg_head.apply(init_weights)
        
        
    def forward(self, data):
        pass