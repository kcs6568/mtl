from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modules.get_detector import build_detector, DetStem
from ...modules.get_backbone import build_backbone
from ...modules.get_segmentor import build_segmentor, SegStem
from ...modules.get_classifier import build_classifier, ClfStem
from ....apis.loss_lib import AutomaticWeightedLoss


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


class SKNet(nn.Module):
    def __init__(self,
                 backbone_name,
                 detector_name,
                 segmentor_name,
                 task_cfg,
                 state_dict=None,
                 freeze_backbone=False,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.backbone = build_backbone(
            backbone_name, detector_name, segmentor_name, weight_path=state_dict['backbone'],
            train_allbackbone=kwargs['train_allbackbone'],
            use_fpn=kwargs['use_fpn'],
            freeze_all_backbone_layers=freeze_backbone,
            freeze_bn=kwargs['freeze_bn'],
            dilation_type=kwargs['dilation_type'])
        
        
        self.stem_dict = nn.ModuleDict()
        self.head_dict = nn.ModuleDict()
        
        for data, cfg in task_cfg.items():
            if cfg['task'] == 'clf':
                stem = ClfStem(**cfg['arch_cfg']['stem'])
                head = build_classifier(self.backbone.body_out_channel, cfg['num_classes'])
                stem.apply(init_weights)
                
                
            elif cfg['task'] == 'det':
                stem = DetStem()
                head = build_detector(detector_name, self.backbone.fpn_out_channels, 
                                      cfg['num_classes'])
                if state_dict['stem'] is not None:
                    ckpt = torch.load(state_dict['stem'])
                    stem.load_state_dict(ckpt)
            
            elif cfg['task'] == 'seg':
                stem = SegStem(**cfg['arch_cfg']['stem'])
                head = build_segmentor(segmentor_name, cfg['arch_cfg']['head'])
                if state_dict['stem'] is not None:
                    ckpt = torch.load(state_dict['stem'])
                    stem.load_state_dict(ckpt)
            
            head.apply(init_weights)
            self.stem_dict.update({data: stem})
            self.head_dict.update({data: head})
        
            
        
        
    def freeze_seperate_layers(self, return_task=False):
        assert self.seperate_task is not None
        
        for n, p in self.named_parameters():
            if 'backbone' in n:
                continue
            
            if return_task:
                if self.seperate_task in n:
                    p.requires_grad_(True)
                else:
                    p.requires_grad_(False)
            
            else:    
                if self.seperate_task in n:
                    p.requires_grad_(False)
                else:
                    p.requires_grad_(True)
            
   
    def _forward_clf(self, images, targets=None):
        stem_feats = self.clf_stem(images)
        backbone_features = self.backbone.body(stem_feats)
        
        if self.training:
            losses = self.clf_head(backbone_features, targets)
            return losses
        
        else:
            return self.self.clf_head(images)
    
    
    def _forward_voc(self, task, images, targets=None):
        if task == 'seg':
            feats = self.seg_stem(images)
            features = self.backbone.body(feats)
            
            if self.training:
                losses = self.seg_head(features, targets)
                return losses
            
            else:
                return self.seg_head(features)
            
    
    def _forward_coco(self, task, images, targets):
        feats, targets = self.det_stem(images, targets=targets)
        features = self.backbone(feats)
        
        if self.training:
            losses = self.detector(images, features, 
                                       trs_targets=targets, 
                                       trs_fn=self.det_stem.transform)
            return losses
        
        else:
            return self.detector(images, features,                                       
                                       trs_fn=self.det_stem.transform)
        
            
    def _extract_stem_feats(self, data_dict):
        stem_feats = OrderedDict()
        
        for task, (images, targets) in data_dict.items():
            if task == 'clf':
                feats = self.clf_stem(images)
                
            elif task == 'det':
                feats, targets = self.det_stem(images, targets=targets)
                
            elif task == 'seg':
                feats = self.seg_stem(images)
                
            else:
                raise KeyError("Not supported task was entered.")
                
            stem_feats.update({task: (feats, targets)})
        
        return stem_feats
    
    
    def _extract_backbone_feats(self, stem_feats):
        backbone_feats = OrderedDict()
        
        for task, (feats, targets) in stem_feats.items():
            if task == 'clf':
                features = self.backbone(feats, get_fpn=False)
            
            elif task == 'det':
                features = self.backbone(feats)
            
            elif task == 'seg':
                features = self.backbone(feats, get_fpn=False)
            
            backbone_feats.update({task: (features, targets)})
        
        return backbone_feats

    def _foward_train(self, data_dict, tasks):
        total_losses = OrderedDict()
        
        for dset, (images, targets) in data_dict.items():
            stem, head = self.stem_dict[dset], self.head_dict[dset]
            task = tasks[dset]
            
            if task == 'clf':
                stem_feats = stem(images)
                back_feats = self.backbone(stem_feats, get_fpn=False)
                losses = head(back_feats, targets)
                
            elif task == 'det':
                stem_feats, trs_targets = stem(images, targets=targets)
                back_feats = self.backbone(stem_feats)
                losses = head(data_dict[dset][0], back_feats, 
                                       trs_targets=trs_targets, 
                                       trs_fn=self.stem_dict[dset].transform)
                
            elif task == 'seg':
                stem_feats = stem(images)
                back_feats = self.backbone(stem_feats, get_fpn=False)
                losses = head(back_feats, targets)
                
            losses = {f"{dset}_{k}": l for k, l in losses.items()}
            total_losses.update(losses)
            
        return total_losses
    
    
    def _forward_val(self, images, kwargs):
        dtype = kwargs['dtype']
        task = kwargs['task']
        
        fpn_task = 'det'
        get_fpn = True
        if task != fpn_task:
            get_fpn = False
        
        stem, head = self.stem_dict[dtype], self.head_dict[dtype]
        stem_feats = stem(images)
        back_feats = self.backbone(stem_feats, get_fpn=get_fpn)
        
        if task == 'det':
            predictions = head(images, back_feats, trs_fn=stem.transform)
            return predictions
        
        else:
            if task == 'seg':
                predictions = head(
                    back_feats, input_shape=images.shape[-2:])
        
            else:
                predictions = head(back_feats)
            
            return dict(outputs=predictions)
        
    
    def forward(self, data_dict, kwargs):
        if self.training:
            return self._foward_train(data_dict, kwargs)

        else:
            return self._forward_val(data_dict, kwargs)
        
