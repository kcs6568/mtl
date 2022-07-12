from collections import OrderedDict

import torch
from torch.optim.lr_scheduler import _LRScheduler


def get_optimizer(args, model):
    method = args.method
    setup = args.setup

    # cls_ = [k for k, v in args.task_per_dset.items() if v == 'clf']
    # det_ = [k for k, v in args.task_per_dset.items() if v == 'det']
    # seg_ = [k for k, v in args.task_per_dset.items() if v == 'seg']
    
    
    '''
    TODO
    - Method별 parameter setting 수정하기
    '''
    
    if method == 'baseline':
        if args.setup == 'multi_task':
            backbone_params = [param for name, param in model.named_parameters() if ('backbone' in name) and (param.requires_grad)]
            cls_params = [param for name, param in model.named_parameters() if ('clf' in name) and (param.requires_grad)]
            det_params = [param for name, param in model.named_parameters() if ('det' in name) and (param.requires_grad)]
            seg_params = [param for name, param in model.named_parameters() if ('seg' in name) and (param.requires_grad)]

            # params = [
            #     {'params': backbone_params, 'lr': args.lr},
            #     {'params': cls_params, 'lr': args.lr*10},
            #     {'params': det_params, 'lr': args.lr*10},
            #     {'params': seg_params, 'lr': args.lr*10}
            # ]

            params = [p for p in model.parameters() if p.requires_grad]
        
        else:
            if setup == 'single_task':
                # if args.segmentor is not None:
                #     params = [
                #         {'params': [p for name, p in model.named_parameters() if (not 'aux' in name) and (p.requires_grad)], 'out': 'out'},
                #         {'params': [p for name, p in model.named_parameters() if ('aux' in name) and (p.requires_grad)],  'lr': args.lr*10, 'aux': 'aux'}
                #     ]
                # else:
                #     params = [p for p in model.parameters() if p.requires_grad]
                params = [p for p in model.parameters() if p.requires_grad]

    elif method == 'cross_stitch':
        # cross_stitch_params = [param for name, param in model.named_parameters() if 'cross_stitch' in name]
        # single_task_params = [param for name, param in model.named_parameters() if not 'cross_stitch' in name]
        # params = [
        #     {'params': single_task_params},
        #     {'params': cross_stitch_params, 'lr': 100*args.lr},
        # ]
        # print(optimizer)
        # optimizer = optimizer(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        params = [p for p in model.parameters() if p.requires_grad]
    
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'nesterov':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif args.opt =='adam':
        if 'eps' in args:
            optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay, 
                                         eps=float(args.eps))
        else:
            optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
            
    elif args.opt =='adamw':
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    
    
    return optimizer