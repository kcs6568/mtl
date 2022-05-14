from collections import OrderedDict
import torch

def get_optimizer(args, model):
    method = args.method
    
    if method == 'baseline':
        params = params = [p for p in model.parameters() if p.requires_grad]
        
        # optimizer = optimizer(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        
    elif method == 'cross_stitch':
        cross_stitch_params = [param for name, param in model.named_parameters() if 'cross_stitch' in name]
        single_task_params = [param for name, param in model.named_parameters() if not 'cross_stitch' in name]
        params = [
            {'params': single_task_params},
            {'params': cross_stitch_params, 'lr': 100*args.lr},
        ]
        # print(optimizer)
        # optimizer = optimizer(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'nesterov':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif args.opt =='adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt =='adamw':
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    
    
    return optimizer