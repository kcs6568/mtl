from collections import OrderedDict
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.get_detector import build_detector, DetStem
from ..modules.get_backbone import build_backbone
from ..modules.get_segmentor import build_segmentor, SegStem
from ..modules.get_classifier import build_classifier, ClfStem
from ..backbones.resnet import Bottleneck, conv1x1, conv3x3
# from ...apis.loss_lib import AutomaticWeightedLoss


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


class MTAN(nn.Module):
    def __init__(self,
                 backbone,
                 detector,
                 segmentor,
                 task_cfg,
                 **kwargs
                 ) -> None:
        super().__init__()
        backbone_net = build_backbone(
            backbone, detector, segmentor, kwargs)
        
        self.dense_task = kwargs['dense_task']
        self.before_att_layers = nn.ModuleDict()
        self.att_layers = nn.ModuleDict()
        self.backbone_size = 0
        
        before = []
        self.before_stage = 0
        for k, layer in backbone_net.body.named_children():
            if k in kwargs['attention_stages']:
                self.before_att_layers[str(self.before_stage)] = nn.Sequential(*before)
                
                self.before_stage += 1
                before.clear()
                
                print("att layer:",k)
                self.att_layers[k] = layer
            else:
                print("before layer:",k)
                before.append(layer)
                
                if k == str(len(backbone_net.body) - 1):
                    self.before_att_layers[str(self.before_stage)] = nn.Sequential(*before)
                    self.before_stage += 1
                    del before
                
            self.backbone_size += 1
        
        self.fpn =backbone_net.fpn
        
        self.task_per_dset = kwargs['task_per_dset']
        self.attention_stages = kwargs['attention_stages'] # 5, 11, 14
        self.before_attention_stage = [str(int(s) - 1) for s in self.attention_stages]
        
        self.concat_stage  = [False] + [True for _ in range(len(self.attention_stages)-1)]
        
        self.det_return_idx = kwargs['det_idx']
        self.seg_return_idx = kwargs['seg_idx']
        
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
                
                head_kwargs = {'num_anchors': len(backbone_net.body.return_layers)+1}
                head = build_detector(
                    backbone, detector, 
                    backbone_net.fpn_out_channels, num_classes, **head_kwargs)
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
        
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        
        attention_ch = kwargs['attention_channels'] # this channel must be same the before_attention_stages
        
        ch_expansion = [1] + [2]*len(self.before_attention_stage[1:])
        
        self.att_encoder_group = nn.ModuleList()
        for i in range(len(self.before_attention_stage)):
            self.att_encoder_group.append(nn.ModuleDict())
            for k in task_cfg.keys():
                self.att_encoder_group[i][k] = self.att_layer(attention_ch[i]*ch_expansion[i], attention_ch[i]//4, attention_ch[i])
        
        shared_att_group = [
            self.att_block_layer(attention_ch[i], attention_ch[i+1] // 4) for i in range(len(self.attention_stages[:-1]))
        ]
        shared_att_group.append(None)
        self.shared_att_group = nn.Sequential(*shared_att_group)
        
        self.last_att = self.att_block_layer(attention_ch[-2], attention_ch[-1] // 4)
        
        
    def att_layer(self, in_channel, intermediate_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=intermediate_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(intermediate_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=intermediate_channel, out_channels=out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid())
                

    def att_block_layer(self, in_channel, out_channel):
        downsample = nn.Sequential(conv1x1(in_channel, 4 * out_channel, stride=1),
                                   nn.BatchNorm2d(4 * out_channel))
        return Bottleneck(in_channel, out_channel, downsample=downsample)

    
    def _generate_features(self, data_dict, tasks):
        det_feats = OrderedDict()
        seg_feats = OrderedDict()
        
        data = OrderedDict(
            {dset: self.stem_dict[dset](data[0]) for dset, data in data_dict.items()}
        )
        
        att_feats = {}
        try:
            for i in range(self.before_stage):
                att_stage = self.attention_stages[i]
                
                ################## backbone feature generation ##################                
                data = {dset: self.before_att_layers[str(i)][0](feats) for dset, feats in data.items()}
                
                ############## save dense feature #############
                if self.det_return_idx[0] in data_dict:
                    if str(i) in self.det_return_idx:
                        det_feats[str(i)] = data[self.det_return_idx[0]]
                
                if self.seg_return_idx[0] in data_dict:
                    if str(i) in self.seg_return_idx:
                        seg_feats[str(i)] = data[self.seg_return_idx[0]]
                ################################################
                
                before_feats = {dset: self.before_att_layers[str(i)][1:](feats) for dset, feats in data.items()}
                att_feats = {dset: self.att_layers[att_stage](feats) for dset, feats in before_feats.items()}
                #################################################################
                
                
                ######### attention feature generation #########
                if self.concat_stage[i]:
                    before_att_masks = {
                        dset: self.att_encoder_group[i][dset](
                            torch.cat((be_feat, att_masks[dset]), dim=1)) for dset, be_feat in before_feats.items()}    
                else:
                    before_att_masks = {
                        dset: self.att_encoder_group[i][dset](be_feat) for dset, be_feat in before_feats.items()}
                
                att_masks = {
                    dset: be_mask * att_feats[dset] for dset, be_mask in before_att_masks.items()
                }
                ################################################
                
                if self.shared_att_group[i] is not None:
                    att_masks = {
                        dset: self.down_sampling(self.shared_att_group[i](att_mask)) for dset, att_mask in att_masks.items()
                    }
                    
                data = att_feats
                
                
        except IndexError as e:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            att_masks = {dset: self.last_att(feats) for dset, feats in att_masks.items()}
            final_data = {dset: (self.before_att_layers[str(i)](feats) * att_masks[dset]) for dset, feats in data.items()}
            
            ############## save dense feature #############
            if self.det_return_idx[0] in data_dict:
                det_feats[str(i)] = self.down_sampling(final_data[self.det_return_idx[0]])
            if self.seg_return_idx[0] in data_dict:
                seg_feats[str(i)] = final_data[self.seg_return_idx[0]]
            ################################################
        
        total_losses = OrderedDict()  
        for dset, feats in final_data.items():
            task = tasks[dset]
            head = self.head_dict[dset]
            targets = data_dict[dset][1]
            
            if task == 'clf':
                out = head(feats, targets)
                
                if not self.training:
                    return dict(outputs=out)
                
            elif task == 'det':
                fpn_feats = self.fpn(det_feats)
                out = head(data_dict[dset][0], fpn_feats, 
                                       origin_targets=targets, 
                                       trs_fn=self.stem_dict[dset].transform)
                
                if not self.training:
                    return out
                
            elif task == 'seg':
                out = head(
                    seg_feats, targets, input_shape=data_dict[dset][0].shape[-2:])
                
                if not self.training:
                    return dict(outputs=out)
            
            total_losses.update({f"{dset}_{k}": l for k, l in out.items()})
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        return total_losses
        
    
    def forward(self, data_dict, kwargs):
        if not self.training:
            if not hasattr(data_dict, 'items'):
                data_dict = {list(kwargs.keys())[0]: [data_dict, None]}
        
        return self._generate_features(data_dict, kwargs)
    

    