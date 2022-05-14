import torch


def build_model(args):
    # model_instance = get_model(args)
    
    
    model_args = {
        'use_awl': args.use_awl,
        'train_allbackbone': args.allbackbone_train,
        'freeze_bn': args.freeze_bn,
        'dilation_type': args.dilation_type,
        'num_datasets': args.num_datasets,
        'freeze_backbone': args.freeze_backbone,
        'state_dict': args.state_dict
    }
    # args.use_fpn = False
    # if args.detector is not None:
    #     args.use_fpn = True
    # model_args.update({'use_fpn': args.use_fpn})
    
    model = None
    if args.setup == 'single_task':
        from .task_model.single_task import SingleTaskNetwork
        
        # use_fpn = args.task_cfg[args.dataset]['use_fpn'] 
        model_args.update({'use_fpn': args.task_cfg[args.dataset]['use_fpn']})
        model_args.update({'backbone_type': 'intermediate'})    
        model_args.update({k: v for k, v in args.task_cfg[args.dataset]['backbone'].items()})
        
        model = SingleTaskNetwork(
            args.backbone,
            args.detector,
            args.segmentor,
            args.dataset,
            args.task_cfg[args.dataset],
            **model_args
        )
    
    elif args.setup == 'multi_task':
        if args.method == 'baseline':
            from .task_model.multi_task import MultiTaskNetwork
            
            model_args.update({'use_fpn': True})
            model_args.update({'backbone_type': 'intermediate'})
            model = MultiTaskNetwork(
                args.backbone, 
                args.detector,
                args.segmentor, 
                args.task_cfg, 
                **model_args
            )
            
            model.to(args.device)
        
        elif args.method == 'cross_stitch':
            from .task_model.cross_stitch import CrossStitchNetwork
            from .task_model.single_task import SingleTaskNetwork
            
            stem_dict, backbone_dict, head_dict = {}, {}, {}
            return_layers = {}
            model_dict = {}
            fpn_task = []
            for dset, cfg in args.task_cfg.items():
                task = cfg['task']
                detector = None
                segmentor = None
                backbone_type = 'origin'

                if task == 'det':
                    detector = args.detector
                    backbone_type = 'intermediate'
                    
                elif task == 'seg':
                    segmentor = args.segmentor
                
                model_args.update({k: v for k, v in cfg['backbone'].items()})
                model_args.update({'use_fpn': cfg['use_fpn']})
                model_args.update({'backbone_type': backbone_type})
            #     single_model = SingleTaskNetwork(
            #         args.backbone,
            #         detector,
            #         segmentor,
            #         dset,
            #         cfg,
            #         **model_args
            #     )
                
            #     stem_dict[dset] = single_model.stem
            #     backbone_dict[dset] = single_model.backbone
            #     head_dict[dset] = single_model.head
            #     return_layers[dset] = cfg['backbone']['return_layer']
                
            #     if use_fpn:
            #         fpn_task.append(dset)
            
            # args.cross_stitch_kwargs.update({'fpn_task': fpn_task})
            # args.cross_stitch_kwargs.update({'return_layers': return_layers})
            
            # model = CrossStitchNetwork(
            #     args.task,
            #     torch.nn.ModuleDict(stem_dict),
            #     torch.nn.ModuleDict(backbone_dict),
            #     torch.nn.ModuleDict(head_dict),
            #     **args.cross_stitch_kwargs
            # )
                single_model = SingleTaskNetwork(
                        args.backbone,
                        detector,
                        segmentor,
                        dset,
                        cfg,
                        **model_args
                    )
                
                model_dict[dset] = single_model
                return_layers[dset] = cfg['backbone']['return_layer']
                
                if cfg['use_fpn']:
                    fpn_task.append(dset)
                
                args.cross_stitch_kwargs.update({'fpn_task': fpn_task})
                args.cross_stitch_kwargs.update({'return_layers': return_layers})
            
            model = CrossStitchNetwork(
                args.task,
                torch.nn.ModuleDict(model_dict),
                **args.cross_stitch_kwargs
            )
            
    
    # data = torch.rand(1, 64, 32, 32)
    
    # print(backbone_dict)
    # # layer1 = getattr(backbone_dict['minicoco'].body, 'layer1')
    # # out = layer1(data)
    # out = backbone_dict['minicoco'].forward_stage(data, 'layer1')
    # print(out.size())
    # # print(backbone_dict)
    
    # exit()
    
    assert model is not None
    return model
        


