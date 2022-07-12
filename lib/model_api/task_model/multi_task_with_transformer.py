from typing import Optional, List
from collections import OrderedDict

import torch
import torch.nn as nn, Tensor
import torch.nn.functional as F

from ..modules.get_detector import build_detector, DetStem
from ..modules.get_backbone import build_backbone
from ..modules.get_segmentor import build_segmentor, SegStem
from ..modules.get_classifier import build_classifier, ClfStem
from ...apis.loss_lib import AutomaticWeightedLoss
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

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


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)
    
    
class TransformerEncoderLayer(nn.Module):
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)    



class MultiTaskNetwork(nn.Module):
    def __init__(self,
                 backbone,
                 detector,
                 segmentor,
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
        
        self.stem_dict = nn.ModuleDict()
        self.head_dict = nn.ModuleDict()
        
        stem_weight = kwargs['state_dict']['stem']
        # relu_type = kwargs['relu_type']
        for data, cfg in task_cfg.items():
            task = cfg['task']
            if task == 'clf':
                stem = ClfStem(**cfg['stem'])
                head = build_classifier(self.backbone.last_out_channel, cfg['num_classes'])
                stem.apply(init_weights)
                
                
            elif task == 'det':
                stem = DetStem()
                head = build_detector(detector, self.backbone.fpn_out_channels, 
                                      cfg['num_classes'])
                if stem_weight is not None:
                    ckpt = torch.load(stem_weight)
                    stem.load_state_dict(ckpt, strict=False)
            
            elif task == 'seg':
                stem = SegStem(**cfg['stem'])
                # cfg['head'].update({'relu': relu_type})
                head = build_segmentor(segmentor, cfg['head'])
                if stem_weight is not None:
                    ckpt = torch.load(stem_weight)
                    stem.load_state_dict(ckpt, strict=False)
            
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
            print(dset)
            for k, v in back_feats.items():
                print(k, v.size())
            print("---"*60)
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
        
        exit()    
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
        
