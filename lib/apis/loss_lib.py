from collections import OrderedDict

import torch
import torch.nn as nn


class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=4):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    
    def __str__(self) -> str:
        delimeter = " | "
        params = [f"param_{i}: {p}" for i, (p) in enumerate(self.params.data)]
        return delimeter.join(params)
    
    
    def forward(self, total_losses):
        awl_dict = OrderedDict()
        
        for i, (k, v) in enumerate(total_losses.items()):
            losses = sum(list(v.values()))
            awl_dict['awl_'+k] = \
                0.5 / (self.params[i] ** 2) * losses + torch.log(1 + self.params[i] ** 2)
        
        # awl_dict['auto_params'] = str(self)
        return awl_dict


class DiceLoss(nn.Module):
    def __init__(self, num_classes, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets, smooth=1, eps=1e-7):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        losses = {}
        # targets = targets.view(-1)
        
        # for name, x in inputs.items():
        #     print(name, x.size(), targets.size())
        #     continue
        #     inputs = torch.sigmoid(x)       
        #     #flatten label and prediction tensors
        #     inputs = inputs.view(-1)
            
        #     intersection = (inputs * targets).sum()                            
        #     dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
            
        #     losses[name] = 1 - dice
        #     # return 1 - dice
        
        # # exit()
        # # return losses
    

        for name, x in inputs.items():
            true_1_hot = torch.eye(self.num_classes)[targets]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = nn.functional.softmax(x, dim=1)
            true_1_hot = true_1_hot.type(x.type())
            dims = (0,) + tuple(range(2, targets.ndimension()))
            intersection = torch.sum(probas * true_1_hot, dims)
            cardinality = torch.sum(probas + true_1_hot, dims)
            dice_loss = (2. * intersection / (cardinality + eps)).mean()
        
            losses[name] = 1 - dice_loss
        
        return losses
    
    
    

