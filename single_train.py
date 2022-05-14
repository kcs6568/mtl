import os
import time
import math
import glob
import datetime

import torch
import torch.utils.data

from engines import engines
from mtl_cl.datasets.load_datasets import load_datasets

import lib.utils.metric_utils as metric_utils
from lib.utils.dist_utils import init_distributed_mode, get_rank
from lib.utils.parser import TrainParser
from lib.utils.sundries import set_args, count_params
from lib.utils.logger import TextLogger, TensorBoardLogger
# from lib.models.model_lib import general_model
# from lib.models.get_origin_models import get_origin_model
from lib.apis.warmup import get_warmup_scheduler
from lib.model_api.build_model import build_model


def main(args):
    args = set_args(args)
    
    metric_utils.mkdir(args.output_dir) # master save dir
    metric_utils.mkdir(os.path.join(args.output_dir, 'ckpts')) # checkpoint save dir
    
    init_distributed_mode(args)
    seed = metric_utils.set_random_seed(args.seed)
    
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
    ds_size = [len(dl) for dl in list(train_loaders.values())]
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
    model.to(args.device)
    
    # for n, p in model.named_parameters():
    #     print(n, p.requires_grad)
    # exit()
    
    '''
    TODO FLOPS 구하는 부분 내 모델에 맞게 수정
    '''
    if args.flop_size is not None:
        from ptflops import get_model_complexity_info
        # if len(args.flop_size) == 1:
        #     input_size = (3, args.flop_size[0], args.flop_size[0])
        # else:
        #     input_size = (3, args.flop_size[0], args.flop_size[1])
        input_size = (3, args.flop_size, args.flop_size)
        with torch.cuda.device(0):
            print("aaaa")
            macs, params = get_model_complexity_info(model, (args.dataset, input_size), as_strings=False,
                                                    print_per_layer_stat=False, verbose=True, is_multi=False,
                                                    ignore_modules=[])
        logger.log_text(f"Computational complexity: {round(macs * 1e-9, 2)}GFLOPs")
        logger.log_text(f"Number of parameters: {round(params * 1e-6, 2) }M")
        
    params, flops = count_params(model, 32, 'clf')
    # print(params, flops)
    # exit()
    
    
    metric_utils.get_params(model, logger, False)
    # exit()
    # model.cuda()
    # model.to(args.device)
    
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                          device_ids=[args.gpu])
        # model = torch.nn.parallel.DistributedDataParallel(model, 
        #                                                   device_ids=[args.gpu],
        #                                                   bucket_cap_mb=100.)
        model_without_ddp = model.module
        
    
    params = [p for p in model.parameters() if p.requires_grad]
    
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'nesterov':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif args.opt =='adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt =='adamw':
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    
    logger.log_text(f"Apply AMP: {args.amp}")
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
        if 'lr_decay_step' in args:
            args.lr_decay_step = None
            
    elif args.lr_scheduler == "cosine":
        if args.lr_decay_step is not None:
                T_max = int(args.epochs / args.lr_decay_step)
                assert T_max == 12
        else:
            T_max = args.epochs
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )
        
    best_results = {task: 0. for task in list(train_loaders.keys())}
    last_results = {task: 0. for task in list(train_loaders.keys())}
    best_epoch = {task: 0 for task in list(train_loaders.keys())}
    if args.resume:
        logger.log_text("Load checkpoints")
        
        if args.resume_tmp:
            ckpt = os.path.join(args.output_dir, 'ckpts', 'tmp_checkpoint.pth')
        
        else:
            if args.resume_file:
                ckpt = os.path.join(args.output_dir, 'ckpts', args.resume_file)
        
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
            logger.log_text(f"Best epoch per task previous exp.:", best_epoch)
        
        if args.amp:
            logger.log_text("Load Optimizer Scaler for AMP")
            scaler.load_state_dict(checkpoint["scaler"])

    logger.log_text(f"First Validation: {args.validate}")
    if args.validate:
        logger.log_text("Evaluate First")
        results = engines.evaluate(model, val_loaders, args.data_cats, logger)
        # logger.log_text("First Evaluation Results:\n\t{}".format(results))
        
        line="<First Evaluation Results>\n"
        for task, v in results.items():
            line += '\t{}: Current Perf. || Last Perf.: {} || {}\n'.format(
                task.upper(), v, last_results[task]
            )
        logger.log_text(line)
        
    logger.log_text("Multitask Learning Start!\n{}".format("***"*60))
    start_time = time.time()
    
    # train_fn = train_lib.general_train
    # train_fn = engines.training
    
    '''
     TODO single task에서 gpu util 늘리기
    '''
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            for i, (task, loader) in enumerate(train_loaders.items()):
                if task == 'clf':
                    loader.sampler.set_epoch(epoch)
                
                elif task == 'det' or task == 'seg':
                    if args.data_cats[task] == 'coco':
                        loader.batch_sampler.sampler.set_epoch(epoch) # detection loader
                    else:
                        loader.sampler.set_epoch(epoch)
                    
        
        logger.log_text("Training Start")    
        
        if len(args.task) > 1:
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
        
        if args.lr_scheduler == 'cosine':
            if args.lr_decay_step is not None:
                if epoch > 0 and (epoch+1) % (args.lr_decay_step) == 0:
                    lr_scheduler.step()
                    logger.log_text(f"Decay learning rate at the {epoch} epoch")
        else:
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
        
        # evaluate after every epoch
        logger.log_text("Validation Start")
        time.sleep(2)
        # results = evaluate_up_two.evaluate(model, val_loaders, args.data_cats, logger)
        results = engines.evaluate(model, val_loaders, args.data_cats, logger)
        
        logger.log_text("Validation Finish\n{}".format('---'*60))
        
        if get_rank() == 0:
            line = '<Compare with Best>\n'
            for task, v in results.items():
                line += '\t{}: Current Perf. || Previous Best: {} || {}\n'.format(
                    task.upper(), v, best_results[task]
                )
                
                if not math.isfinite(v):
                    logger.log_text(f"Performance of task {task} is nan.")
                    v == 0.
                    
                if v > best_results[task]:
                    best_results[task] = round(v, 2)
                    best_epoch[task] = epoch
            
            logger.log_text(line)  
            logger.log_text(f"Best Epcoh per task: {best_epoch}")
            checkpoint['best_results'] = best_results
            checkpoint['last_results'] = results
            checkpoint['best_epoch'] = best_epoch
            for task, e in best_epoch.items():
                if e == epoch:
                    exist_file = glob.glob(os.path.join(args.output_dir, 'ckpts', f"best_{task}*"))        
                    
                    if len(exist_file) == 1:
                        logger.log_text(f"Previous best model for {task.upper()} will be deleted.")
                        os.remove(exist_file[0])
                        
                    logger.log_text("Save best model of {} at the {} epoch.".format(task.upper(), best_epoch[task]))
                    metric_utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'ckpts', f"best_{task}_{best_epoch[task]}e.pth"))
        
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
    args = TrainParser().args
    main(args)


