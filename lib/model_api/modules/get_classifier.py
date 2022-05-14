from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClfHead(nn.Module):
    def __init__(self,
                 in_channel,
                 num_classes,
                 use_avgpool=True) -> None:
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1,1)) if use_avgpool else None
        self.fc = nn.Linear(in_channel, num_classes)
        self.criterion = nn.CrossEntropyLoss().cuda()

        
    def forward(self, feats, targets=None):
        assert isinstance(feats, OrderedDict) or isinstance(feats, dict)
        
        _, x = feats.popitem()
    
        if self.avg:
            x = self.avg(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        
        if self.training:
            losses = self.criterion(out, targets)
            
            losses = dict(clf_loss=losses)
            
            return losses
        
        else:
            return out
    
    
class ClfStem(nn.Module):
    def __init__(self,
                 out_channels=64,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 maxpool=None) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        # self.conv = nn.Conv2d(3, out_channels, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = maxpool
        
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        if self.maxpool:
            x = self.maxpool(x)
        
        
        return x
    
    
def build_classifier(in_channel, num_classes):
    return ClfHead(in_channel=in_channel, num_classes=num_classes)
    