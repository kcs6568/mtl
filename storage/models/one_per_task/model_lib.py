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


       
class GeneralNet(nn.Module):
    def __init__(self,
                 backbone_name,
                 detector_name,
                 segmentor_name,
                 task_cfg,
                 arch_cfg=None,
                 state_dict=None,
                 freeze_backbone=False,
                 loss_reduction_rate=None,
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
        
        if task_cfg['clf'] is not None:
            if arch_cfg and arch_cfg['stem'] and arch_cfg['stem']['clf']:
                self.clf_stem = ClfStem(**arch_cfg['stem']['clf'])
            else:
                self.clf_stem = ClfStem()
                
            self.clf_head = build_classifier(self.backbone.body_out_channel, 
                                    task_cfg['clf']['num_classes'])
            self.clf_stem.apply(init_weights)
            self.clf_head.apply(init_weights)
            
        if task_cfg['det'] is not None:
            self.det_stem = DetStem(stem_weight=state_dict['stem'], 
                                    freeze_bn=kwargs['freeze_bn'])
            
            
            self.det_head = build_detector(detector_name, self.backbone.fpn_out_channels, 
                                           task_cfg['det']['num_classes'])
            
            if not state_dict['stem']:
                self.det_stem.apply(init_weights)
            self.det_head.apply(init_weights)
            
        if task_cfg['seg'] is not None:
            self.seg_stem = SegStem(stem_weight=state_dict['stem'])
            
            head_cfg = {
                'in_channels': self.backbone.body_out_channel,
                'num_classes': task_cfg['seg']['num_classes']
            }
            self.seg_head = build_segmentor(segmentor_name, head_cfg)
            
            if not state_dict['stem']:
                self.seg_stem.apply(init_weights)
            self.seg_head.apply(init_weights)
        
        
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
            
   
    def freeze_layer(self, freeze_task):
        assert isinstance(freeze_task, list)
        assert freeze_task in [
            ['clf', 'det'], ['clf', 'seg'], ['det', 'seg']]
        
        for task in freeze_task:
            for n, p in self.named_parameters():
                    if 'backbone' in n:
                        continue
                    
                    if task in n:
                        p.requires_grad_(False)
                    
                    
    def unfreeze_layer(self, unfreeze_task):
        assert isinstance(unfreeze_task, list)
        assert unfreeze_task in [
            ['clf', 'det'], ['clf', 'seg'], ['det', 'seg']]
        
        for task in unfreeze_task:
            for n, p in self.named_parameters():
                    if 'backbone' in n:
                        continue
                    
                    if task in n:
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
            
    
    '''
    TODO: minicoco segmentation 데이터 전처리 부분 & 입력 후 처리 부분 수정
    '''
    def _forward_coco(self, task, images, targets):
        feats, targets = self.det_stem(images, targets=targets)
        features = self.backbone(feats)
        
        if self.training:
            losses = self.det_head(images, features, 
                                       trs_targets=targets, 
                                       trs_fn=self.det_stem.transform)
            return losses
        
        else:
            return self.det_head(images, features,                                       
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

    
    def _foward_train(self, data_dict, kwargs={}):
        total_losses = OrderedDict()
        
        stem_feats = self._extract_stem_feats(data_dict)
        backbone_feats = self._extract_backbone_feats(stem_feats)

        for task, (feats, targets) in backbone_feats.items():
            if task == 'clf':
                losses = self.clf_head(feats, targets)
                
            elif task == 'det':
                losses = self.det_head(data_dict['det'][0], feats, 
                                       trs_targets=targets, 
                                       trs_fn=self.det_stem.transform)

            elif task == 'seg':
                losses = self.seg_head(feats, targets)
                
            else:
                raise KeyError("Not supported task was entered.")
            assert isinstance(losses, dict)
            
            # if 'load_count' in kwargs:
            #     losses = {k: v*(1.0/kwargs['load_count'][task]) for k, v in losses.items()}
            
            total_losses.update(losses)
            
        return total_losses
    
    
    def _forward_val(self, images, kwargs):
        task = kwargs['task']
        if 'clf' == task:
            stem_feats = self.clf_stem(images)
            classification_feats = self.backbone(stem_feats, get_fpn=False)
            predictions = self.clf_head(classification_feats)
            
            return dict(outputs=predictions)
        
        elif 'det' == task:
            stem_feats, _ = self.det_stem(images)
            backbone_feats = self.backbone(stem_feats)
            predictions = self.det_head(images, backbone_feats, trs_fn=self.det_stem.transform)
            
            return predictions
        
        elif 'seg' == task:
            stem_feats = self.seg_stem(images)
            backbone_feats = self.backbone(stem_feats, get_fpn=False)
            predictions = self.seg_head(
                backbone_feats,
                input_shape=images.shape[-2:])
            
            return dict(outputs=predictions)
        
    
    def forward(self, data_dict, kwargs={}):
        if self.training:
            return self._foward_train(data_dict, kwargs)

        else:
            return self._forward_val(data_dict, kwargs)
        
        
class LossbalNet(nn.Module):
    def __init__(self,
                 backbone_name,
                 detector_name,
                 segmentor_name,
                 task_cfg,
                 state_dict=None,
                 freeze_backbone=False,
                 loss_reduction_rate=None,
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
        
        if task_cfg['clf'] is not None:
            self.clf_stem = ClfStem()
            self.clf_head = build_classifier(self.backbone.body_out_channel, 
                                    task_cfg['clf']['num_classes'])
            self.clf_stem.apply(init_weights)
            self.clf_head.apply(init_weights)
            
        if task_cfg['det'] is not None:
            self.det_stem = DetStem(stem_weight=state_dict['stem'])
            
            self.detector = build_detector(detector_name, self.backbone.fpn_out_channels, 
                                        task_cfg['det']['num_classes'])
            self.detector.apply(init_weights)
            
        if task_cfg['seg'] is not None:
            self.seg_stem = SegStem(stem_weight=state_dict['stem'])
            
            head_cfg = {
                'in_channels': self.backbone.body_out_channel,
                'num_classes': task_cfg['seg']['num_classes']
            }
            self.seg_head = build_segmentor(segmentor_name, head_cfg)
            self.seg_head.apply(init_weights)
        
        
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
            
    
    '''
    TODO: minicoco segmentation 데이터 전처리 부분 & 입력 후 처리 부분 수정
    '''
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

    
    def _foward_train(self, data_dict, kwargs={}):
        total_losses = OrderedDict()
        
        stem_feats = self._extract_stem_feats(data_dict)
        backbone_feats = self._extract_backbone_feats(stem_feats)

        for task, (feats, targets) in backbone_feats.items():
            if task == 'clf':
                losses = self.clf_head(feats, targets)
                
            elif task == 'det':
                losses = self.detector(data_dict['det'][0], feats, 
                                       trs_targets=targets, 
                                       trs_fn=self.det_stem.transform)

            elif task == 'seg':
                losses = self.seg_head(feats, targets)
                
            else:
                raise KeyError("Not supported task was entered.")
            assert isinstance(losses, dict)
            
            # if 'load_count' in kwargs:
            #     losses = {k: v*(1.0/kwargs['load_count'][task]) for k, v in losses.items()}
            
            total_losses.update(losses)
            
        return total_losses
    
    
    def _forward_val(self, images, kwargs):
        
        if isinstance(images, dict):
            cls_stem_feats = self.clf_stem(images['clf'])
            classification_feats = self.backbone(cls_stem_feats, get_fpn=False)
            predictions = self.clf_head(classification_feats)
            
            det_stem_feats, _ = self.det_stem(images['det'])
            det_backbone_feats = self.backbone(det_stem_feats)
            predictions = self.detector(images['det'], det_backbone_feats, trs_fn=self.det_stem.transform)
            
            seg_stem_feats = self.seg_stem(images['seg'])
            seg_backbone_feats = self.backbone(seg_stem_feats, get_fpn=False)
            predictions = self.seg_head(
                seg_backbone_feats,
                input_shape=images['seg'].shape[-2:])
            
            return
            
        else:
            task = kwargs.task
            if 'clf' == task:
                stem_feats = self.clf_stem(images)
                classification_feats = self.backbone(stem_feats, get_fpn=False)
                # _, last_feature = classification_feats.popitem()
                predictions = self.clf_head(classification_feats)
                
                return dict(outputs=predictions)
            
            elif 'det' == task:
                stem_feats = self.det_stem(images)
                backbone_feats = self.backbone(stem_feats)
                predictions = self.detector(images, backbone_feats, trs_fn=self.det_stem.transform)
                
                return predictions
            
            elif 'seg' == task:
                stem_feats = self.seg_stem(images)
                backbone_feats = self.backbone(stem_feats, get_fpn=False)
                predictions = self.seg_head(
                    backbone_feats,
                    input_shape=images.shape[-2:])
                
                return dict(outputs=predictions)
        
    
    def forward(self, data_dict, kwargs={}):
        if self.training:
            return self._foward_train(data_dict, kwargs)

        else:
            return self._forward_val(data_dict, kwargs)
        
