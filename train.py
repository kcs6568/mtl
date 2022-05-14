import os
import time
import math
import datetime

import torch
import torch.utils.data

from engines import engines
from datasets.load_datasets import load_datasets

import lib.utils.metric_utils as metric_utils
from lib.utils.dist_utils import init_distributed_mode, get_rank
from lib.utils.parser import TrainParser
from lib.utils.sundries import set_args, count_params
from lib.utils.logger import TextLogger, TensorBoardLogger
# from lib.models.model_lib import general_model
# from lib.models.get_origin_models import get_origin_model
from lib.apis.warmup import get_warmup_scheduler
from lib.apis.optimizer import get_optimizer
from lib.model_api.build_model import build_model

# try:
#     from torchvision import prototype
# except ImportError:
#     prototype = None


def main(args):
    args = set_args(args)
    
    metric_utils.mkdir(args.output_dir) # master save dir
    metric_utils.mkdir(os.path.join(args.output_dir, 'ckpts')) # checkpoint save dir
    
    init_distributed_mode(args)
    seed = metric_utils.set_random_seed(args.seed)
    print(torch.cuda.max_memory_allocated()*1e-6)
    # exit()
    
    log_dir = os.path.join(args.output_dir, 'logs')
    metric_utils.mkdir(log_dir)
    logger = TextLogger(log_dir, print_time=False)
    
    if args.resume:
        logger.log_text("{}\n\t\t\tResume training\n{}".format("###"*40, "###"*45))
    logger.log_text(f"Experiment Case: {args.exp_case}")
    
    tb_logger = None
    if args.distributed and get_rank() == 0:
        tb_logger = TensorBoardLogger(
            log_dir = os.path.join(log_dir, 'tb_logs'),
            filename_suffix=f"_{args.exp_case}"
        )
    
    if args.seperate and args.freeze_backbone:
        logger.log_text(
        f"he seperate task is applied. But the backbone network will be freezed. Please change the value to False one of the two.",
        level='error')
        logger.log_text("Terminate process", level='error')
        
        raise AssertionError

    
    metric_utils.save_parser(args, path=log_dir)
    logger.log_text(f"Set seed: {seed}")
    logger.log_text("Loading data")
    
    train_loaders, val_loaders, test_loaders = load_datasets(args)
    args.data_cats = {k: v['task'] for k, v in args.task_cfg.items() if v is not None}
        
    ds_size = [len(dl) for dl in train_loaders.values()]
    logger.log_text("Task list that will be trained:\n\t" \
        "Training Order: {}\n\t" \
        "Data Size: {}".format(
            list(train_loaders.keys()),
            ds_size
            )
        )
    
    logger.log_text("Creating model")
    logger.log_text(f"Freeze Shared Backbone Network: {args.freeze_backbone}")
    
    model = build_model(args)
    
    # print(model)
    # exit()
    
    # for n, p in model.named_parameters():
    #     print(n, p.requires_grad)
    # exit()
    
    # if len(args.flop_size) > 0:
    # from ptflops import get_model_complexity_info
    # from pthflops import count_ops
    # with torch.cuda.device(0):
    #     input_size = {}
    #     for data, cfg in args.task_cfg.items():
    #         if 'cifar' in data:
    #             size = (3, 32, 32)
    #         elif 'stl' in data:
    #             size = (3, 96, 96)
    #         elif 'coco' and cfg['task'] == 'det':
    #             size = (3, 800, 1333)
    #         elif 'voc' and cfg['task'] == 'seg':
    #             size = (3, 512, 512)
                
    #         input_size.update({data: size})
        
    #     model = model.to('cuda:0')
    #     inp = {
    #         task: torch.ones(()).new_empty((1, *res)).to('cuda:0') for task, res in input_size.items()
    #     }
    #     count_ops(model, inp)
    # exit()
    #     exit()
    #     #     macs, params = get_model_complexity_info(model, input_size, as_strings=False,
    #     #                                             print_per_layer_stat=True, verbose=True,
    #     #                                             ignore_modules=[], is_multi=True)
    #     # print(macs)
    #     # print(f"Computational complexity: {round(macs * 1e-9, 2)}GFLOPs")
    #     # print(f"Number of parameters: {round(params * 1e-6, 2) }M")
    
    metric_utils.get_params(model, logger, False)
    # exit()
    # model.cuda()
    model.to(args.device)
    print(torch.cuda.max_memory_allocated()*1e-6)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                          device_ids=[args.gpu])
        model_without_ddp = model.module
    
    optimizer = get_optimizer(args, model)
    
    # print(len(optimizer.param_groups))
    # exit()
    
    # print(torch.cuda.max_memory_allocated()*1e-6)
    # params = [p for p in model.parameters() if p.requires_grad]
    
    # params = []
    # for n, p in model.named_parameters():
    #     print(n, p.requires_grad)
    #     if p.requires_grad:
    #         params.append(p)
    
    
    # print(params)
    # exit()
    
    # if args.opt == 'sgd':
    #     optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # elif args.opt == 'nesterov':
    #     optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    # elif args.opt =='adam':
    #     optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    # elif args.opt =='adamw':
    #     optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    # print(optimizer)
    # exit()
    
    logger.log_text(f"Optimizer:\n{optimizer}")
    logger.log_text(f"Apply AMP: {args.amp}")
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )
    # print(torch.cuda.max_memory_allocated()*1e-6)
    best_results = {data: 0. for data in list(train_loaders.keys())}
    last_results = {data: 0. for data in list(train_loaders.keys())}
    best_epoch = {data: 0 for data in list(train_loaders.keys())}
    if args.resume or args.resume_tmp:
        logger.log_text("Load checkpoints")
        if args.resume_tmp:
            print("here")
            ckpt = os.path.join(args.output_dir, 'ckpts', "tmp_checkpoint.pth")
        # if args.resume_file:
            # ckpt = os.path.join(args.output_dir, 'ckpts', args.resume_file)
        else:
            ckpt = os.path.join(args.output_dir, 'ckpts', "checkpoint.pth")
            
        checkpoint = torch.load(ckpt, map_location="cuda")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        logger.log_text(f"Checkpoint List: {checkpoint.keys()}")
        if 'last_results' in checkpoint:
            last_results = checkpoint['last_results']
            logger.log_text(f"Performance of last epoch: {last_results}")
        
        if 'best_results' in checkpoint:
            best_results = checkpoint['best_results']
            logger.log_text(f"Best Performance so far: {best_results}")
            
        if 'best_epoch' in checkpoint:
            best_epoch = checkpoint['best_epoch']
            logger.log_text(f"Best epoch per data previous exp.:", best_epoch)
        
        if args.amp:
            logger.log_text("Load Optimizer Scaler for AMP")
            scaler.load_state_dict(checkpoint["scaler"])

    logger.log_text(f"First Validation: {args.validate}")
    if args.validate:
        logger.log_text("Evaluate First")
        results = engines.evaluate(model, val_loaders, args.data_cats, logger, args.num_classes)
        # logger.log_text("First Evaluation Results:\n\t{}".format(results))
        
        line="<First Evaluation Results>\n"
        for data, v in results.items():
            line += '\t{}: Current Perf. || Last Perf.: {} || {}\n'.format(
                data.upper(), v, last_results[data]
            )
        logger.log_text(line)
        
        
        
    logger.log_text("Multitask Learning Start!\n{} --> Method: {}".format("***"*60, args.method))
    start_time = time.time()
    
    
    # task_ratio = {k: float(r/10) for k, r in zip(list(args.data_cats.keys()), args.loss_ratio)}
    # args.loss_ratio = task_ratio
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            for i, (dset, loader) in enumerate(train_loaders.items()):
                if 'coco' in dset:
                    loader.batch_sampler.sampler.set_epoch(epoch)
                
                else:
                    loader.sampler.set_epoch(epoch)
                    
        
        logger.log_text("Training Start")    
        
        if args.num_datasets > 1:
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            time.sleep(3)
        
        warmup_fn = get_warmup_scheduler if epoch == 0 else None
        engines.training(
            model, 
            optimizer, 
            train_loaders, 
            epoch, 
            logger,
            tb_logger, 
            scaler,
            args,
            warmup_fn=warmup_fn)
        logger.log_text("Training Finish\n{}".format('---'*60))
        lr_scheduler.step()
        
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch
            }
        
        metric_utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'ckpts', f"tmp_checkpoint.pth"))    
        
        if args.amp:
            checkpoint["scaler"] = scaler.state_dict()
        
        coco_patient = 0
        # evaluate after every epoch
        logger.log_text("Validation Start")
        time.sleep(2)
        results = engines.evaluate(model, val_loaders, args.data_cats, logger, args.num_classes)
        logger.log_text("Validation Finish\n{}".format('---'*60))
        
        if get_rank() == 0:
            line = '<Compare with Best>\n'
            for data, v in results.items():
                line += '\t{}: Current Perf. || Previous Best: {} || {}\n'.format(
                    data.upper(), v, best_results[data]
                )
                
                if not math.isfinite(v):
                    logger.log_text(f"Performance of data {data} is nan.")
                    v == 0.
                    
                if v > best_results[data]:
                    best_results[data] = round(v, 2)
                    best_epoch[data] = epoch
                    
                else:
                    if 'coco' in data:
                        coco_patient += 1
                
            
            if epoch == args.epochs // 2:
                if coco_patient == 2:
                    logger.log_text(
                        "Training process will be terminated because the COCO patient is max value.", 
                        level='error')      
                    
                    import sys
                    sys.exit(1)
                
            
            logger.log_text(line)  
            logger.log_text(f"Best Epcoh per data: {best_epoch}")
            checkpoint['best_results'] = best_results
            checkpoint['last_results'] = results
            checkpoint['best_epoch'] = best_epoch
            # for data, e in best_epoch.items():
            #     if e == epoch:
            #         exist_file = glob.glob(os.path.join(args.output_dir, 'ckpts', f"best_{data}*"))        
                    
            #         if len(exist_file) == 1:
            #             logger.log_text(f"Previous best model for {data.upper()} will be deleted.")
            #             os.remove(exist_file[0])
                        
            #         logger.log_text("Save best model of {} at the {} epoch.".format(data.upper(), best_epoch[data]))
            #         metric_utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'ckpts', f"best_{data}_{best_epoch[data]}e.pth"))
        
            if tb_logger:
                tb_logger.update_scalars(
                    results, epoch, proc='val'
                )    
            
            
            logger.log_text("Save model checkpoint...")
            metric_utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'ckpts', "checkpoint.pth"))
            logger.log_text("Complete {} epoch\n{}\n\n".format(epoch, "###"*30))
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time.sleep(2)
    # End Training -----------------------------------------------------------------------------
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    logger.log_text("Best Epoch for each task: {}".format(best_epoch))
    logger.log_text("Final Results: {}".format(best_results))
    logger.log_text(f"Exp Case: {args.exp_case}")
    logger.log_text(f"Save Path: {args.output_dir}")
    if get_rank() == 0:
        os.remove(os.path.join(args.output_dir, 'ckpts', 'tmp_checkpoint.pth'))
        logger.log_text("Temporal checkpoint was removed.")    
    logger.log_text(f"Training time {total_time_str}")


if __name__ == "__main__":
    # args = get_args_parser().parse_args()
    args = TrainParser().args
    main(args)



# def get_args_parser(add_help=True):
#     import argparse

#     parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

#     parser.add_argument("--exp-case", default="0", type=str, help="exp case")
#     parser.add_argument("--data-path", default="/root/data/mmdataset/coco", type=str, help="dataset path")
#     parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
#     parser.add_argument("--model", default="maskrcnn_resnet50_fpn", type=str, help="model name")
#     parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
#     parser.add_argument(
#         "-b", "--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
#     )
#     parser.add_argument("--epochs", default=26, type=int, metavar="N", help="number of total epochs to run")
#     parser.add_argument(
#         "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
#     )
#     parser.add_argument(
#         "--lr",
#         default=0.02,
#         type=float,
#         help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu",
#     )
#     parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
#     parser.add_argument(
#         "--wd",
#         "--weight-decay",
#         default=1e-4,
#         type=float,
#         metavar="W",
#         help="weight decay (default: 1e-4)",
#         dest="weight_decay",
#     )
#     parser.add_argument(
#         "--lr-scheduler", default="multisteplr", type=str, help="name of lr scheduler (default: multisteplr)"
#     )
#     parser.add_argument(
#         "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
#     )
#     parser.add_argument(
#         "--lr-steps",
#         default=[16, 22],
#         nargs="+",
#         type=int,
#         help="decrease lr every step-size epochs (multisteplr scheduler only)",
#     )
#     parser.add_argument(
#         "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
#     )
#     parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
#     parser.add_argument("--output-dir", default="/root/volume/exp", type=str, help="path to save outputs")
#     # parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
#     parser.add_argument("--resume", action='store_true')
#     parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
#     parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
#     parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
#     parser.add_argument(
#         "--trainable-backbone-layers", default=None, type=int, help="number of trainable layers of backbone"
#     )
#     parser.add_argument(
#         "--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip)"
#     )
#     parser.add_argument(
#         "--sync-bn",
#         dest="sync_bn",
#         help="Use sync batch norm",
#         action="store_true",
#     )
#     parser.add_argument(
#         "--validate",
#         help="validate the model using validation datasets",
#         action="store_true",
#     )
    
#     parser.add_argument(
#         "--use_testset",
#         action="store_true",
#     )
    
#     parser.add_argument(
#         "--test-only",
#         dest="test_only",
#         action="store_true",
#     )
    
#     parser.add_argument(
#         "--pretrained",
#         dest="pretrained",
#         help="Use pre-trained models from the modelzoo",
#         action="store_true",
#     )

#     # distributed training parameters
#     parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
#     parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

#     # Prototype models only
#     parser.add_argument(
#         "--prototype",
#         dest="prototype",
#         help="Use prototype model builders instead those from main area",
#         action="store_true",
#     )
#     parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")

#     # Mixed precision training parameters
#     parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
#     parser.add_argument("--use-origin", action="store_true", help="Select model type between torchvision and custom model")
#     parser.add_argument("--use-minids", action="store_true", help="Use COCO minitrain dataset")
#     parser.add_argument("--task", nargs='+', default=['clf', 'det'])
#     parser.add_argument("--cfg", type=str, default=None)
#     parser.add_argument("--loss-alpha", type=float, default=None)
#     parser.add_argument("--alpha-task", nargs='+', type=str, default=None)
#     parser.add_argument("--opt", type=str, default='sgd')
#     parser.add_argument("--auto-aug", action='store_true')
#     parser.add_argument("--miles", nargs='+', default=[8, 11], type=int)
#     parser.add_argument("--loss-reduction-rate", default=0.9, type=float)
    

#     return parser





# def get_detection_dataset(task_list, name, image_set, transform, data_path):
#     paths = {
#         "coco": (data_path, get_coco, 91), 
#         "coco_kp": (data_path, get_coco_kp, 2)
#         }
#     p, ds_fn, num_classes = paths[name]

#     ds = ds_fn(p, image_set=image_set, transforms=transform)
#     return ds, num_classes


# def get_transform(train, args):
#     if train:
#         return presets.DetectionPresetTrain(args.data_augmentation)
#     elif not args.prototype:
#         return presets.DetectionPresetEval()
#     else:
#         if args.weights:
#             weights = prototype.models.get_weight(args.weights)
#             return weights.transforms()
#         else:
#             return prototype.transforms.CocoEval()




# train_type = 'train'
# if args.use_minids:
#     train_type = 'mini' + train_type
# dataset, num_classes = get_detection_dataset(args.dataset, train_type, get_transform(True, args), args.data_path)
# dataset_val, _ = get_detection_dataset(args.dataset, "val", get_transform(False, args), args.data_path)

# logger.log_text("Creating data loaders")
# if args.distributed:
#     train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
#     val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val)
# else:
#     train_sampler = torch.utils.data.RandomSampler(dataset)
#     val_sampler = torch.utils.data.SequentialSampler(dataset_val)
    
# if args.aspect_ratio_group_factor >= 0:
#     group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
#     train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
# else:
#     train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

# data_loader = torch.utils.data.DataLoader(
#     dataset, batch_sampler=train_batch_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
# )

# data_loader_val = torch.utils.data.DataLoader(
#     dataset_val, batch_size=1, sampler=val_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
# )

# if args.use_testset:
#     dataset_test, _ = get_detection_dataset(args.dataset, "test", get_transform(False, args), args.data_path)
#     test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
#     data_loader_test = torch.utils.data.DataLoader(
#         dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
#     )


  # images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
    # labels = torch.randint(1, 91, (4, 11))
    # images = list(image for image in images)
    # targets = []
    # for i in range(len(images)):
    #     d = {}
    #     d['boxes'] = boxes[i]
    #     d['labels'] = labels[i]
    #     targets.append(d)
    
    # model.train()
    # data = dict(
    #     clf=[
    #         torch.rand(1, 3, 32, 32), torch.tensor([1])
    #     ],
    #     det=[images, targets],
    #     seg=[torch.rand(1, 3, 480, 480), torch.rand(1, 480, 480)
    #     ],
    #     reload_clf=0
    # )
    
    # out = model(data)
    
    # exit()