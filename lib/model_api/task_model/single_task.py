from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.get_detector import build_detector, DetStem
from ..modules.get_backbone import build_backbone
from ..modules.get_segmentor import build_segmentor, SegStem
from ..modules.get_classifier import build_classifier, ClfStem
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


def make_stem(task, cfg, stem_weight=None):
    if task == 'clf':
        stem = ClfStem(**cfg['stem'])
        stem.apply(init_weights)
    
    elif task == 'det':
        stem = DetStem()
        if stem_weight is not None:
                ckpt = torch.load(stem_weight)
                stem.load_state_dict(ckpt)
    
    elif task == 'seg':
        stem = SegStem(**cfg['stem'])
        if stem_weight is not None:
            ckpt = torch.load(stem_weight)
            stem.load_state_dict(ckpt)
    
    return stem
    
    
def make_head(task, cfg, backbone, dense_task):
    if task == 'clf':
        head = build_classifier(backbone.last_out_channel, cfg['num_classes'])
    
    elif task == 'det':
        head = build_detector(
            dense_task, 
            backbone.fpn_out_channels, 
            cfg['num_classes'])
    
    elif task == 'seg':
        head = build_segmentor(dense_task, cfg['head'])
    
    head.apply(init_weights)
    return head


class SingleTaskNetwork(nn.Module):
    def __init__(self,
                 backbone,
                 detector,
                 segmentor,
                 dataset,
                 task_cfg,
                 **kwargs
                 ) -> None:
        super().__init__()
        # self.backbone = build_backbone(
        #     backbone, detector, segmentor, weight_path=kwargs['state_dict']['backbone'],
        #     train_allbackbone=kwargs['train_allbackbone'],
        #     use_fpn=kwargs['use_fpn'],
        #     freeze_all_backbone_layers=kwargs['freeze_backbone'],
        #     freeze_bn=kwargs['freeze_bn'],
        #     dilation_type=kwargs['dilation_type'],
        #     backbone_type=kwargs['backbone_type'])
        
        self.backbone = build_backbone(
            backbone, detector, segmentor, kwargs)
        
        self.dset = dataset
        self.task = task_cfg['task']
        
        stem_weight = kwargs['state_dict']['stem']
        self.stem = make_stem(self.task, task_cfg, stem_weight)
        
        dense_task = detector if detector else segmentor
        self.head = make_head(self.task, task_cfg, self.backbone, dense_task)
        
        if self.task == 'clf':
            self.task_forward = self._forward_clf
        elif self.task == 'det':
            self.task_forward = self._forward_coco
        elif self.task == 'seg':
            self.task_forward = self._forward_voc
        
    
    def _forward_clf(self, images, targets=None):
        stem_feats = self.stem(images)
        backbone_features = self.backbone(stem_feats)
        
        if self.training:
            losses = self.head(backbone_features, targets)
            return losses
        
        else:
            return self.head({'a': backbone_features})
    
    
    def _forward_coco(self, images, targets):
        feats, targets = self.stem(images, targets=targets)
        features = self.backbone(feats)
        
        if self.training:
            losses = self.head(images, features, 
                                       trs_targets=targets, 
                                       trs_fn=self.det_stem.transform)
            return losses
        
        else:
            return self.head(images, features,                                       
                                       trs_fn=self.det_stem.transform)
            
    
    def _forward_voc(self, images, targets=None):
        feats = self.stem(images)
        features = self.backbone(feats)
        
        if self.training:
            losses = self.head(features, targets,
                               input_shape=targets.shape[-2:])
            return losses
        
        else:
            predictions = self.head(
                    features, input_shape=images.shape[-2:])
            
            # print(predictions.size())
            return predictions
            

    def _foward_train(self, data_dict):
        images, targets = data_dict[self.dset]
        loss_dict = self.task_forward(images, targets)
        
        return loss_dict
        
    
    def _forward_val(self, images):
        prediction_results = self.task_forward(images)
        
        return dict(outputs=prediction_results)
    
    
    def forward(self, data_dict, kwargs=None):
        if self.training:
            return self._foward_train(data_dict)

        else:
            return self._forward_val(data_dict)
        
