from collections import OrderedDict

import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


from datasets.coco.coco_utils import get_coco, coco_collate_fn
from datasets.voc.voc_utils import *
from datasets.data_utils import *


def load_datasets(args, only_val=True):
    import os
    download=False
    
    
    def load_cifar10(path):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        if args.no_hflip:
            transform=transforms.Compose([
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize
            ])
            
        else:
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize
            ])
        
        train_dataset = datasets.CIFAR10(
            path,
            transform=transform,
            download=download
        )
        
        test_dataset = datasets.CIFAR10(
            path,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize
            ]),
            download=download,
            train=False
        )
        
        train_sampler = None
        test_sampler = None
        
        if args.distributed:
            train_sampler, test_sampler = return_sampler(train_dataset, test_dataset, args.world_size, args.gpu)

        train_loader, test_loader = get_dataloader(train_dataset, test_dataset, train_sampler, test_sampler, 
                                                   args, args.batch_size)
        
        return train_loader, test_loader
    
    
    def load_cifar100(path):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        if args.no_hflip:
            transform=transforms.Compose([
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize
            ])
            
        else:
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize
            ])
        
        train_dataset = datasets.CIFAR100(
           path,
            transform=transform,
            download=download
        )
        
        test_dataset = datasets.CIFAR100(
            path,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize
            ]),
            download=download,
            train=False
        )

        train_sampler = None
        test_sampler = None
        
        if args.distributed:
            train_sampler, test_sampler = return_sampler(train_dataset, test_dataset, args.world_size, args.gpu)

        train_loader, test_loader = get_dataloader(train_dataset, test_dataset, train_sampler, test_sampler, 
                                                   args, args.batch_size)
        
        return train_loader, test_loader
        
    
    def load_stl10(path, input_size=96):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        transform=transforms.Compose([
            transforms.RandomCrop(input_size, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        
        train_dataset = datasets.STL10(
            root=path,
            split="train",
            transform=transform,
            download=download
        )
        
        test_dataset = datasets.STL10(
            root=path,
            split="test",
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize
            ]),
            download=download
        )
        
        train_sampler = None
        test_sampler = None
        
        if args.distributed:
            train_sampler, test_sampler = return_sampler(train_dataset, test_dataset, args.world_size, args.gpu)

        train_loader, test_loader = get_dataloader(train_dataset, test_dataset, train_sampler, test_sampler, 
                                                   args, args.batch_size)
        
        return train_loader, test_loader
    
    
    def load_imagenet1k(path, img_size=224, only_val=True):
        ratio = 224.0 / float(img_size)
        
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        traindir = os.path.join(path, 'ILSVRC2012_img_train')
        valdir = os.path.join(path, 'ILSVRC2012_img_val')
        
        train_sampler = None
        val_sampler = None
        
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # ColorAugmentation(),
                normalize,
            ]))
        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(int(256 * ratio)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize,
        ]))
        
        train_loader, val_loader = get_dataloader(train_dataset, val_dataset, train_sampler, val_sampler, 
                                                  args, args.batch_size)

        # train_loader = torch.utils.data.DataLoader(
        #     train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        #     num_workers=args.workers, pin_memory=(train_sampler is None), sampler=train_sampler)

        # val_loader = torch.utils.data.DataLoader(
        #     val_dataset, batch_size=args.batch_size, shuffle=False,
        #     num_workers=args.workers, pin_memory=True, sampler=val_sampler)
        
        if only_val:
            return None, val_loader
        else:
            return train_loader, val_loader
        
        
    def load_coco(data_path, trs_type='det', multiple_trs=False, loader_type='det'):
        import lib.utils.metric_utils as metric_utils
        from lib.apis.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
        
        train_type = 'train'
        if args.use_minids:
            train_type = 'mini' + train_type
        
        train_transforms = get_transforms(multiple_trs, trs_type, True, args)
        val_transforms = get_transforms(multiple_trs, trs_type, False, args)
        
        train_ds = get_coco(data_path, train_type, train_transforms, multiple_trs)
        val_ds = get_coco(data_path, 'val', val_transforms, multiple_trs)
        
        # train_ds = get_coco(data_path, train_type, get_transform(True, args))
        # val_ds = get_coco(data_path, 'val', get_transform(False, args))
            
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds)
        else:
            train_sampler = torch.utils.data.RandomSampler(train_ds)
            val_sampler = torch.utils.data.SequentialSampler(val_ds)
        
        if args.aspect_ratio_group_factor >= 0:
            group_ids = create_aspect_ratio_groups(train_ds, k=args.aspect_ratio_group_factor)
            train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
        else:
            train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
        
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_sampler=train_batch_sampler, num_workers=args.workers, collate_fn=coco_collate_fn,
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=1, sampler=val_sampler, num_workers=args.workers, collate_fn=coco_collate_fn
        )
        
        test_loader = None
        if args.use_testset:
            test_transforms = get_transforms(multiple_trs, trs_type, False, args)
            test_ds = get_coco(data_path, 'test', test_transforms, multiple_trs)
            
            if args.distributed:
                test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds)
            else:
                test_sampler = torch.utils.data.SequentialSampler(test_ds)
                
            test_loader = torch.utils.data.DataLoader(
                test_ds, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=metric_utils.collate_fn
            )

        return train_loader, val_loader, test_loader
    
    
    
    def load_voc(task_type, task_cfg, multiple_trs=False):
        assert task_type in ['det', 'seg', 'aug']
        
        task_cfg['train'].update(dict(transforms=get_transforms(multiple_trs, task_type, True, args))) 
        task_cfg['test'].update(dict(transforms=get_transforms(multiple_trs, task_type, False, args))) 
        
        train_ds = get_voc_dataset(task_type, task_cfg['train'])
        val_ds = get_voc_dataset(task_type, task_cfg['test'])
        
        if args.distributed:
            train_sampler, val_sampler = return_sampler(train_ds, val_ds, args.world_size, args.gpu)
        else:
            train_sampler = torch.utils.data.RandomSampler(train_ds)
            val_sampler = torch.utils.data.SequentialSampler(val_ds)

        test_loader = None
        
        args.pin_memory = False
        if len(args.task_bs) == 1:
            args.pin_memory = True
        
        # args.pin_memory = True  
        train_loader, val_loader = get_dataloader(
            train_ds, val_ds, train_sampler, val_sampler, args, args.batch_size, collate_fn=voc_collate_fn(task_type))
        
        return train_loader, val_loader, test_loader
        
        
        
    train_loaders = OrderedDict()
    val_loaders = OrderedDict()
    test_loaders = OrderedDict()
    
    multiple = is_multiple_dataset(args.task_cfg)
    multiple = False
    for task, cfg in args.task_cfg.items():
        if cfg is None:
            continue
        
        args.batch_size = args.task_bs[task]
        
        if 'clf' in task:
            if cfg['type'] == 'cifar10':
                test_ld = None
                train_ld, val_ld = load_cifar10('/root/data/pytorch_datasets')
            
            elif cfg['type'] == 'cifar100':
                test_ld = None
                train_ld, val_ld = load_cifar100('/root/data/pytorch_datasets')
            
            if cfg['type'] == 'stl10':
                test_ld = None
                train_ld, val_ld = load_stl10('/root/data/pytorch_datasets', input_size=cfg['input_size'])
            
            elif cfg['type'] == 'imagenet1k':
                test_ld = None
                train_ld, val_ld = load_imagenet1k(path='/root/data/img_type_datasets/ImageNet-1K', only_val=only_val)
            
            train_loaders['clf'] = train_ld
            val_loaders['clf'] = val_ld
            
            if test_ld:
                test_loaders['clf'] = test_ld
                
        else:
            if multiple:
                if 'multiple' in train_loaders:
                    continue
                
                if cfg['type'] == 'coco':
                    train_ld, val_ld, test_ld = load_coco("/root/data/mmdataset/coco", multiple_trs=multiple)
                
                elif cfg['type'] == 'voc':
                    train_ld, val_ld, test_ld = load_voc(cfg['data_cfg'], multiple_trs=multiple)
                    
                train_loaders['multiple'] = train_ld
                val_loaders['multiple'] = val_ld
                
                if test_ld:
                    test_loaders['multiple'] = test_ld
                    
            else:
                if 'det' in task:
                    if cfg['type'] == 'coco':
                        train_ld, val_ld, test_ld = load_coco("/root/data/mmdataset/coco", trs_type='det')
                        
                    elif cfg['type'] == 'voc':
                        train_ld, val_ld, test_ld = load_voc(cfg['task_cfg'], multiple_trs=multiple)
                    
                    train_loaders['det'] = train_ld
                    val_loaders['det'] = val_ld
                    
                    if test_ld:
                        test_loaders['det'] = test_ld
                        
                elif 'seg' in task:
                    if cfg['type'] == 'coco':
                        train_ld, val_ld, test_ld = load_coco("/root/data/mmdataset/coco", trs_type='det', loader_type='seg')
                        
                    elif cfg['type'] == 'voc':
                        train_ld, val_ld, test_ld = load_voc(task, cfg['task_cfg'], multiple_trs=multiple)
                    
                    train_loaders['seg'] = train_ld
                    val_loaders['seg'] = val_ld
                    
                    if test_ld:
                        test_loaders['seg'] = test_ld
                        
                        
    return train_loaders, val_loaders, test_loaders

