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


class MultiTaskNetwork(nn.Module):
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
        
        self.stem_dict = nn.ModuleDict()
        self.head_dict = nn.ModuleDict()
        
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
        
        if 'use_awl' in kwargs:
            if kwargs['use_awl']:
                self.awl = AutomaticWeightedLoss(len(task_cfg))    
        
            
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
        
            
    # def _extract_stem_feats(self, data_dict):
    #     stem_feats = OrderedDict()
        
    #     for task, (images, targets) in data_dict.items():
    #         if task == 'clf':
    #             feats = self.clf_stem(images)
                
    #         elif task == 'det':
    #             feats, targets = self.det_stem(images, targets=targets)
                
    #         elif task == 'seg':
    #             feats = self.seg_stem(images)
                
    #         else:
    #             raise KeyError("Not supported task was entered.")
                
    #         stem_feats.update({task: (feats, targets)})
        
    #     return stem_feats
    
    
    # def _extract_backbone_feats(self, stem_feats):
    #     backbone_feats = OrderedDict()
        
    #     for task, (feats, targets) in stem_feats.items():
    #         if task == 'clf':
    #             features = self.backbone(feats)
            
    #         elif task == 'det':
    #             features = self.backbone(feats)
            
    #         elif task == 'seg':
    #             features = self.backbone(feats)
            
    #         backbone_feats.update({task: (features, targets)})
        
    #     return backbone_feats
    
    
    def _extract_stem_feats(self, data_dict, tasks):
        stem_feats = OrderedDict()
        
        for dset, (images, _) in data_dict.items():
            task = tasks[dset]
            
            if task == 'clf':
                feats = self.stem_dict[dset](images)
                
            elif task == 'det':
                feats = self.stem_dict[dset](images)
                
            elif task == 'seg':
                feats = self.stem_dict[dset](images)
                
            else:
                raise KeyError("Not supported task was entered.")
            
            stem_feats.update({dset: feats})
        return stem_feats
    
    
    def _extract_backbone_feats(self, stem_feats, tasks):
        backbone_feats = OrderedDict()
        
        for dset, feats in stem_feats.items():
            task = tasks[dset]
            if task == 'clf':
                features = self.backbone.body(feats)
            
            elif task == 'det':
                features = self.backbone(feats)
            
            elif task == 'seg':
                features = self.backbone.body(feats)
            
            backbone_feats.update({dset: features})
            
        return backbone_feats
    
    
    def _foward_train(self, data_dict, tasks):
        total_losses = OrderedDict()
        
        stem_feats = self._extract_stem_feats(data_dict, tasks)
        backbone_feats = self._extract_backbone_feats(stem_feats, tasks)
        
        for dset, back_feats in backbone_feats.items():
            # print(dset)
            # for k, v in back_feats.items():
            #     print(k, v.size())
            # print()
            # continue
            
            
            task = tasks[dset]
            targets = data_dict[dset][1]
            
            if task == 'clf':
                losses = self.head_dict[dset](back_feats, targets)
                
            elif task == 'det':
                losses = self.head_dict[dset](data_dict[dset][0], back_feats,
                                        self.stem_dict[dset].transform, 
                                       origin_targets=targets)
                
            elif task == 'seg':
                losses = self.head_dict[dset](
                    back_feats, targets, input_shape=targets.shape[-2:])
                
            losses = {f"{dset}_{k}": l for k, l in losses.items()}
            total_losses.update(losses)
        
        if hasattr(self, 'awl'):
            total_losses = self.awl(total_losses)
        
        return total_losses
    

    # def _foward_train(self, data_dict, tasks):
    #     total_losses = OrderedDict()
        
    #     for dset, (images, targets) in data_dict.items():
    #         task = tasks[dset]
    #         dset_task = f"{dset}_{task}"
    #         stem, head = self.stem_dict[dset], self.head_dict[dset]
            
    #         if task == 'clf':
    #             stem_feats = stem(images)
    #             back_feats = self.backbone.body(stem_feats)
    #             losses = head(back_feats, targets)
                
    #         elif task == 'det':
    #             stem_feats = stem(images)
    #             back_feats = self.backbone(stem_feats)
    #             losses = head(images, back_feats, stem.transform, origin_targets=targets)
                
    #         elif task == 'seg':
    #             stem_feats = stem(images)
    #             back_feats = self.backbone.body(stem_feats)
    #             losses = head(back_feats, targets, input_shape=targets.shape[-2:])
                
    #         losses = {f"{dset}_{k}": l for k, l in losses.items()}
    #         total_losses.update(losses)
            
    #     return total_losses
    
    
    def _forward_val(self, images, kwargs):
        dset = list(kwargs.keys())[0]
        task = list(kwargs.values())[0]
        
        stem, head = self.stem_dict[dset], self.head_dict[dset]
        stem_feats = stem(images)
        
        if task == 'det':
            back_feats = self.backbone(stem_feats)
            predictions = head(images, back_feats, stem.transform)
            return predictions
        
        else:
            back_feats = self.backbone.body(stem_feats)
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
        
