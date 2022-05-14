from collections import OrderedDict

import torch
import torch.nn as nn


class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=3):
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
