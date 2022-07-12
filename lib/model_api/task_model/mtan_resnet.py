from collections import OrderedDict
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
        attention_ch = kwargs['attention_channels']
        self.task_per_dset = kwargs['task_per_dset']
        
        self.shared_layer1_b = backbone_net.body.layer1[:-1] 
        self.shared_layer1_t = backbone_net.body.layer1[-1]

        self.shared_layer2_b = backbone_net.body.layer2[:-1]
        self.shared_layer2_t = backbone_net.body.layer2[-1]

        self.shared_layer3_b = backbone_net.body.layer3[:-1]
        self.shared_layer3_t = backbone_net.body.layer3[-1]

        self.shared_layer4_b = backbone_net.body.layer4[:-1]
        self.shared_layer4_t = backbone_net.body.layer4[-1]
        
        self.fpn = backbone_net.fpn
        
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
            
                # stem = DetStem()
                # head = build_detector(detector, backbone.fpn_out_channels, 
                #                       cfg['num_classes'])
                # if stem_weight is not None:
                #     ckpt = torch.load(stem_weight)
                #     stem.load_state_dict(ckpt)
            
            elif task == 'seg':
                stem = SegStem(**cfg['stem'])
                head = build_segmentor(segmentor, num_classes=num_classes, cfg_dict=cfg['head'])
                if stem_weight is not None:
                    ckpt = torch.load(stem_weight)
                    stem.load_state_dict(ckpt, strict=False)
                    print("!!!Load weights for segmentation stem layer!!!")
                    
                # stem = SegStem(**cfg['stem'])
                # head = build_segmentor(segmentor, cfg['head'])
                # if stem_weight is not None:
                #     ckpt = torch.load(stem_weight)
                #     stem.load_state_dict(ckpt)
            
            head.apply(init_weights)
            self.stem_dict.update({data: stem})
            self.head_dict.update({data: head})
        
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        
        self.att_encoder1 = nn.ModuleDict({k: self.att_layer(attention_ch[0], attention_ch[0]//4, attention_ch[0]) for k in task_cfg.keys()})
        self.att_encoder2 = nn.ModuleDict({k: self.att_layer(2 * attention_ch[1], attention_ch[1]//4, attention_ch[1]) for k in task_cfg.keys()})
        self.att_encoder3 = nn.ModuleDict({k: self.att_layer(2 * attention_ch[2], attention_ch[2]//4, attention_ch[2]) for k in task_cfg.keys()})
        self.att_encoder4 = nn.ModuleDict({k: self.att_layer(2 * attention_ch[3], attention_ch[3]//4, attention_ch[3]) for k in task_cfg.keys()})
        
        self.encoder_block_att1 = self.att_block_layer(attention_ch[0], attention_ch[1] // 4)
        self.encoder_block_att2 = self.att_block_layer(attention_ch[1], attention_ch[2] // 4)
        self.encoder_block_att3 = self.att_block_layer(attention_ch[2], attention_ch[3] // 4)
        
    
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
        mode = 'train' if self.training else 'val'
        det_feats = OrderedDict()
        seg_feats = OrderedDict()
        
        # ttt = True if 'voc' in data_dict else False
        
        stem_feats = OrderedDict(
            {dset: self.stem_dict[dset](data[0]) for dset, data in data_dict.items()}
        )
        
        shared_backbone_feat1 = OrderedDict(
            {dset: self.shared_layer1_b(data) for dset, data in stem_feats.items()}
        )
        shared_last_feat1 = OrderedDict(
            {dset: self.shared_layer1_t(data) for dset, data in shared_backbone_feat1.items()}
        )
        
        shared_backbone_feat2 = OrderedDict(
            {dset: self.shared_layer2_b(data) for dset, data in shared_last_feat1.items()}
        )
        shared_last_feat2 = OrderedDict(
            {dset: self.shared_layer2_t(data) for dset, data in shared_backbone_feat2.items()}
        )
        
        shared_backbone_feat3 = OrderedDict(
            {dset: self.shared_layer3_b(data) for dset, data in shared_last_feat2.items()}
        )
        shared_last_feat3 = OrderedDict(
            {dset: self.shared_layer3_t(data) for dset, data in shared_backbone_feat3.items()}
        )
        
        shared_backbone_feat4 = OrderedDict(
            {dset: self.shared_layer4_b(data) for dset, data in shared_last_feat3.items()}
        )
        shared_last_feat4 = OrderedDict(
            {dset: self.shared_layer4_t(data) for dset, data in shared_backbone_feat4.items()}
        ) 
        
        a_1_mask = {dset: self.att_encoder1[dset](back_feat) for dset, back_feat in shared_backbone_feat1.items()}  
        a_1 = {dset: a_1_mask_i * shared_last_feat1[dset] for dset, a_1_mask_i in a_1_mask.items()}  
        det_feats.update({'0': f for t, f in a_1.items() if self.task_per_dset[t] == 'det'})
        a_1 = {dset: self.down_sampling(self.encoder_block_att1(a_1_i)) for dset, a_1_i in a_1.items()}
        
        
        a_2_mask = {dset: self.att_encoder2[dset](torch.cat(
            (back_feat, a_1[dset]), dim=1)) for dset, back_feat in shared_backbone_feat2.items()}    
        a_2 = {dset: a_2_mask_i * shared_last_feat2[dset] for dset, a_2_mask_i in a_2_mask.items()}  
        det_feats.update({'1': f for t, f in a_2.items() if self.task_per_dset[t] == 'det'})
        a_2 = {dset: self.down_sampling(self.encoder_block_att2(a_2_i)) for dset, a_2_i in a_2.items()}
        
        
        a_3_mask = {dset: self.att_encoder3[dset](torch.cat(
            (back_feat, a_2[dset]), dim=1)) for dset, back_feat in shared_backbone_feat3.items()} 
        a_3 = {dset: a_3_mask_i * shared_last_feat3[dset] for dset, a_3_mask_i in a_3_mask.items()}  
        det_feats.update({'2': f for t, f in a_3.items() if self.task_per_dset[t] == 'det'})
        seg_feats.update({'2': f for t, f in a_3.items() if self.task_per_dset[t] == 'seg'})
        a_3 = {dset: self.encoder_block_att3(a_3_i) for dset, a_3_i in a_3.items()}
        
        a_4_mask = {dset: self.att_encoder4[dset](torch.cat(
            (back_feat, a_3[dset]), dim=1)) for dset, back_feat in shared_backbone_feat4.items()} 
        a_4 = {dset: a_4_mask_i * shared_last_feat4[dset] for dset, a_4_mask_i in a_4_mask.items()}  
        det_feats.update({'3': f for t, f in a_4.items() if self.task_per_dset[t] == 'det'})
        seg_feats.update({'3': f for t, f in a_4.items() if self.task_per_dset[t] == 'seg'})
        
        total_losses = OrderedDict()
        
        for dset, att_feats in a_4.items():
            task = tasks[dset]
            head = self.head_dict[dset]
            targets = data_dict[dset][1]
            
            if task == 'clf':
                out = head(att_feats, targets)
                
                if mode == 'val':
                    return dict(outputs=out)
                
            elif task == 'det':
                fpn_feats = self.fpn(det_feats)
                
                out = head(data_dict[dset][0], fpn_feats, 
                                       origin_targets=targets, 
                                       trs_fn=self.stem_dict[dset].transform)
                
                if mode == 'val':
                    return out
                
            elif task == 'seg':
                out = head(
                    seg_feats, targets, input_shape=data_dict[dset][0].shape[-2:])
                
                if mode == 'val':
                    return dict(outputs=out)
            
            total_losses.update({f"{dset}_{k}": l for k, l in out.items()})
        
        return total_losses
        
    
    def forward(self, data_dict, kwargs):
        if not self.training:
            if not hasattr(data_dict, 'items'):
                data_dict = {list(kwargs.keys())[0]: [data_dict, None]}
        
        return self._generate_features(data_dict, kwargs)
    

    
    
