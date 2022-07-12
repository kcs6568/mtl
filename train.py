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


def main(args):
    args = set_args(args)
    
    metric_utils.mkdir(args.output_dir) # master save dir
    metric_utils.mkdir(os.path.join(args.output_dir, 'ckpts')) # checkpoint save dir
    
    init_distributed_mode(args)
    
    metric_utils.set_random_seed(args.seed)
    
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
    # logger.log_text(f"Set seed: {seed}")
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
    
    metric_utils.get_params(model, logger, False)
    
    model.to(args.device)
    
    # print(model)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                          device_ids=[args.gpu])
        model_without_ddp = model.module
    
    logger.log_text(f"Model Configuration:\n{model}")

    optimizer = get_optimizer(args, model)
    
    logger.log_text(f"Optimizer:\n{optimizer}")
    logger.log_text(f"Apply AMP: {args.amp}")
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == "multi":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.gamma)
    elif args.lr_scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == 'lambda':
        lr_scheduler =  torch.optim.lr_scheduler.LambdaLR(
           optimizer, 
           lambda x: (1 - x / (max(ds_size) * args.epochs)) ** 0.9)
        
    #    lr_scheduler =  torch.optim.lr_scheduler.LambdaLR(
    #        optimizer, 
    #        lambda x: (1 - x / (iters_per_epoch * (args.epochs - args.lr_warmup_epochs))) ** 0.9)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )

    logger.log_text(f"Scheduler:\n{lr_scheduler}")

    best_results = {data: 0. for data in list(train_loaders.keys())}
    last_results = {data: 0. for data in list(train_loaders.keys())}
    best_epoch = {data: 0 for data in list(train_loaders.keys())}
    total_time = 0.
    
    if args.resume or args.resume_tmp or args.resume_file:
        logger.log_text("Load checkpoints")
        
        if args.resume_tmp:
            ckpt = os.path.join(args.output_dir, 'ckpts', "tmp_checkpoint.pth")
        elif args.resume_file is not None:
            # ckpt = os.path.join(args.output_dir, 'ckpts', args.resume_file)
            ckpt = args.resume_file
        else:
            ckpt = os.path.join(args.output_dir, 'ckpts', "checkpoint.pth")

        try:
            checkpoint = torch.load(ckpt, map_location=f'cuda:{torch.cuda.current_device()}')
            model_without_ddp.load_state_dict(checkpoint["model"])
            
            # model.module.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            
            if args.lr_scheduler == 'step':
                if checkpoint['lr_scheduler']['step_size'] != args.step_size:
                    checkpoint['lr_scheduler']['step_size'] = args.step_size
                    
                if checkpoint['lr_scheduler']['gamma'] != args.gamma:
                    checkpoint['lr_scheduler']['gamma'] = args.gamma
                # checkpoint['lr_scheduler']['_last_lr'] = args.lr * args.gamma
                optimizer.param_groups[0]['lr'] = checkpoint['lr_scheduler']['_last_lr'][0]
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"]
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
            
            if 'total_time' in checkpoint:
                total_time = checkpoint['total_time']
                logger.log_text(f"Previous Total Training Time:", total_time)
            
            if args.amp:
                logger.log_text("Load Optimizer Scaler for AMP")
                scaler.load_state_dict(checkpoint["scaler"])
                
        except:
            logger.log_text("The resume file is not exist")

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
        
    logger.log_text("Multitask Learning Start!\n{}\n --> Method: {}".format("***"*60, args.method))
    start_time = time.time()
    
    # if args.epochs < args.warmup_epoch:
    #     if args.warmup_epoch > 1:
    #         args.warmup_ratio = 1
    #     biggest_size = len(list(train_loaders.values())[0])
    #     warmup_sch = get_warmup_scheduler(optimizer, args.warmup_ratio, biggest_size * args.warmup_epoch)
    # else:
    #     warmup_sch = None
    
    if args.warmup_epoch > 1:
        args.warmup_ratio = 1
    biggest_size = len(list(train_loaders.values())[0])
    warmup_sch = get_warmup_scheduler(optimizer, args.warmup_ratio, biggest_size * args.warmup_epoch)
    
    logger.log_text(f"Parser Arguments:\n{args}")

    if args.find_epoch is not None:
        logger.log_text("Finding Hyper Parameter Process")
        
    for epoch in range(args.start_epoch, args.epochs):
        if (args.find_epoch is not None) and (epoch == args.find_epoch):
            logger.log_text("Finish Process on finding hyp.")
            break

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
        
        if warmup_sch is not None:
            if epoch >= args.warmup_epoch:
                warmup_sch = None
                
        # warmup_fn = get_warmup_scheduler if epoch == 0 else None
        one_training_time = engines.training(
            model, 
            optimizer, 
            train_loaders, 
            epoch, 
            logger,
            tb_logger, 
            scaler,
            args,
            warmup_sch=warmup_sch)
        total_time += one_training_time 
        logger.log_text("Training Finish\n{}".format('---'*60))

        if warmup_sch is None:
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
            checkpoint['total_time'] = total_time
        
            if tb_logger:
                tb_logger.update_scalars(
                    results, epoch, proc='val'
                )    
            
            logger.log_text("Save model checkpoint...")
            metric_utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'ckpts', "checkpoint.pth"))
            logger.log_text("Complete {} epoch\n{}\n\n".format(epoch+1, "###"*30))
        
        
        '''
        TODO
        !!!Warning!!!
        - Please do not write "exit()" code --> this will occur the gpu memory
        '''
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time.sleep(2)
    # End Training -----------------------------------------------------------------------------
    
    all_train_val_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(all_train_val_time)))
    
    logger.log_text("Best Epoch for each task: {}".format(best_epoch))
    logger.log_text("Final Results: {}".format(best_results))
    logger.log_text(f"Exp Case: {args.exp_case}")
    logger.log_text(f"Save Path: {args.output_dir}")
    
    if get_rank() == 0:
        os.remove(os.path.join(args.output_dir, 'ckpts', 'tmp_checkpoint.pth'))
        logger.log_text("Temporal checkpoint was removed.")    
    logger.log_text(f"Only Training Time: {str(datetime.timedelta(seconds=int(total_time)))}")
    logger.log_text(f"Training + Validation Time {total_time_str}")


if __name__ == "__main__":
    # args = get_args_parser().parse_args()
    args = TrainParser().args
    main(args)
    
    
    
# xmax = 1200
# ymax = 600
# images, boxes = torch.rand(4, 3, ymax, xmax), torch.rand(4, 11, 4)
# boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
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