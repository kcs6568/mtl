import torch
from .backbones.resnet import setting_resnet_args
from .backbones.mobilenet_v3 import setting_mobilenet_args


def build_model(args):
    # model_instance = get_model(args)
    
    # if 'resnet' in args.backbone:
    #     model_args = setting_resnet_args(args)
    # elif 'mobilenet' in args.backbone:
    #     model_args = setting_mobilenet_args(args)
    
    model_args = {
        'state_dict': args.state_dict,
    }
    
    model = None
    if args.setup == 'single_task':
        from .task_model.single_task import SingleTaskNetwork
        
        if 'resnet' in args.backbone:
                model_args = setting_resnet_args(args, model_args)
        elif 'mobilenet' in args.backbone:
            model_args = setting_mobilenet_args(args, model_args)
        
        model_args.update({
            'train_allbackbone': args.train_allbackbone,
            'freeze_bn': args.freeze_bn,
            'freeze_backbone': args.freeze_backbone,
        })
        
        model_args.update({k: v for k, v in args.task_cfg[args.dataset]['backbone'].items()})
        
        model = SingleTaskNetwork(
            args.backbone,
            args.detector,
            args.segmentor,
            args.dataset,
            args.task_cfg[args.dataset],
            **model_args
        )
    
    else:
        if 'resnet' in args.backbone:
                model_args = setting_resnet_args(args, model_args)
        elif 'mobilenet' in args.backbone:
            model_args = setting_mobilenet_args(args, model_args)
            
        if args.setup == 'multi_task':
            if args.method == 'baseline':
                from .task_model.multi_task import MultiTaskNetwork
                
                model_args.update({
                    'train_allbackbone': args.train_allbackbone,
                    'freeze_bn': args.freeze_bn,
                    'freeze_backbone': args.freeze_backbone,
                })
                model_args.update({'use_fpn': True})
                model_args.update({'backbone_type': 'intermediate'})
                
                model = MultiTaskNetwork(
                    args.backbone, 
                    args.detector,
                    args.segmentor, 
                    args.task_cfg, 
                    **model_args,
                )
                
            
            elif args.method == 'multi_shuffle':
                if 'resnet' in args.backbone:
                    from .task_model.multi_shuffle_resnet import MultiShuffleNetwork
                # elif 'mobilenet' in args.backbone:
                #     from .task_model.multi_shuffle_mobilev3 import MultiShuffleNetwork
                
                
                model_args.update({
                    'train_allbackbone': args.train_allbackbone,
                    'freeze_bn': args.freeze_bn,
                    'freeze_backbone': args.freeze_backbone,
                })
                model_args.update({'use_fpn': True})
                model_args.update({'backbone_type': 'intermediate'})
                
                model_args.update({k: v for k, v in args.shuffle_info.items()})
            
                model = MultiShuffleNetwork(
                    args.backbone, 
                    args.detector,
                    args.segmentor, 
                    args.task_cfg, 
                    **model_args,
                )
                
                
            elif args.method == 'cross_stitch':
                if 'resnet' in args.backbone:
                    from .task_model.cross_stitch_resnet import CrossStitchNetwork
                elif 'mobilenet' in args.backbone:
                    from .task_model.cross_stitch_mobilev3 import CrossStitchNetwork
                
                from .task_model.single_task import SingleTaskNetwork
                
                return_layers = {}
                model_dict = {}
                fpn_task = []
                kwargs = {}
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
                    
                    # model_args.update({'use_fpn': cfg['use_fpn']})
                    # model_args.update({'backbone_type': backbone_type})
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
                    
                    if model_args['use_fpn']:
                        fpn_task.append(dset)
                
                args.cross_stitch_kwargs.update({'fpn_task': fpn_task})
                args.cross_stitch_kwargs.update({'return_layers': return_layers})
                
                if 'mobilenet' in args.backbone:
                        kwargs.update({'backbone_size': len(single_model.backbone.body)})
                
                model = CrossStitchNetwork(
                    args.task,
                    torch.nn.ModuleDict(model_dict),
                    **args.cross_stitch_kwargs,
                    **kwargs
                )

            elif args.method == 'mtan':
                model_args.update({
                    'train_allbackbone': args.train_allbackbone,
                    'freeze_bn': args.freeze_bn,
                    'freeze_backbone': args.freeze_backbone,
                })
                model_args.update({'use_fpn': True})
                model_args.update({'backbone_type': 'intermediate'})
                model_args.update({'task_per_dset': args.task_per_dset})
                model_args.update({k: v for k, v in args.mtan_kwargs.items()})
                
                if 'resnet' in args.backbone:
                    from .task_model.mtan_resnet import MTAN
                elif 'mobilenet' in args.backbone:
                    from .task_model.mtan_mobilev3 import MTAN
                
                model = MTAN(
                    args.backbone, 
                    args.detector,
                    args.segmentor, 
                    args.task_cfg, 
                    **model_args
                )
                
            # elif args.method == 'nddr_cnn':
            #     from .task_model.nddr_cnn import NDDRCNN
            #     from .task_model.single_task import SingleTaskNetwork
                
            #     stem_dict, backbone_dict, head_dict = {}, {}, {}
            #     return_layers = {}
            #     model_dict = {}
            #     fpn_task = []
            #     for dset, cfg in args.task_cfg.items():
            #         task = cfg['task']
            #         detector = None
            #         segmentor = None
            #         backbone_type = 'origin'

            #         if task == 'det':
            #             detector = args.detector
            #             backbone_type = 'intermediate'
                        
            #         elif task == 'seg':
            #             segmentor = args.segmentor
                    
            #         model_args.update({k: v for k, v in cfg['backbone'].items()})
            #         model_args.update({'use_fpn': cfg['use_fpn']})
            #         model_args.update({'backbone_type': backbone_type})
                    
            #         single_model = SingleTaskNetwork(
            #                 args.backbone,
            #                 detector,
            #                 segmentor,
            #                 dset,
            #                 cfg,
            #                 **model_args
            #             )
                    
            #         model_dict[dset] = single_model
            #         return_layers[dset] = cfg['backbone']['return_layer']
                    
            #         if cfg['use_fpn']:
            #             fpn_task.append(dset)
                    
            #         args.nddr_cnn_kwargs.update({'fpn_task': fpn_task})
            #         args.nddr_cnn_kwargs.update({'return_layers': return_layers})
                    
            #     model = NDDRCNN(
            #         args.task,
            #         torch.nn.ModuleDict(model_dict),
            #         **args.nddr_cnn_kwargs
            #     )
            
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
        


