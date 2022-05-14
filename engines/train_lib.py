import math
import sys
import time
import datetime
from collections import OrderedDict

import torch

import lib.utils.metric_utils as metric_utils
from lib.utils.dist_utils import get_rank

cifar10_classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# BREAK=True
BREAK=False

'''
TODO:
    - 초기에 생성한 DDP에서 parameter 바꿀 수 있는 방법 찾아보기
'''

def lossbal_train(model, optimizer, data_loaders, 
          epoch, logger, 
          tb_logger, scaler, args,
          warmup_fn=None):
    model.train()
    assert args.lossbal and args.loss_ratio
    
    metric_logger = metric_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", metric_utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    
    input_dicts = OrderedDict()
    
    keys = list(data_loaders.keys())
    loaders = list(data_loaders.values())
    
    biggest_key, biggest_dl = keys[0], loaders[0]
    biggest_size = len(biggest_dl)
    
    others_keys = [None] + keys[1:]
    others_size = [None] + [len(ld) for ld in loaders[1:]]
    others_iterator = [None] + [iter(dl) for dl in loaders[1:]]
    
    print(others_iterator)
    
    load_cnt = {k: 1 for k in keys}
    
    header = f"Epoch: [{epoch}]"
    iter_time = metric_utils.SmoothedValue(fmt="{avg:.4f}")
    metric_logger.largest_iters = biggest_size
    metric_logger.epohcs = args.epochs
    metric_logger.set_before_train(header)
    
    lr_scheduler = warmup_fn(
        optimizer, args.warmup_ratio, biggest_size) if warmup_fn  else None
    
    if lr_scheduler:
        logger.log_text(f"Warmup Iteration: {int(lr_scheduler.total_iters)}/{biggest_size}")
    else:
        logger.log_text("No Warmup Training")
    
    start_time = time.time()
    end = time.time()
    for i, b_data in enumerate(biggest_dl):
        input_dicts.clear()
        input_dicts[biggest_key] = b_data

        try:
            for n_task in range(1, len(others_iterator)):
                input_dicts[others_keys[n_task]] = next(others_iterator[n_task])
            
        except StopIteration:
            print("occur StopIteration")
            for i, (it, size) in enumerate(zip(others_iterator, others_size)):
                if it is None:
                    continue
                
                if it._num_yielded == size:
                    print("full iteration size:", it._num_yielded, size)
                    others_iterator[i] = iter(loaders[i])
                    load_cnt[keys[i]] += 1
                    logger.log_text(f"Reloading Count: {load_cnt}\n")
                    
            for n_task in range(1, len(others_iterator)):
                if not others_keys[n_task] in input_dicts.keys():
                    input_dicts[others_keys[n_task]] = next(others_iterator[n_task])
        
        # if torch.cuda.is_available:
        #     torch.cuda.synchronize()
        
        if args.return_count:
            input_dicts.update({'load_count': load_cnt})

        input_set, kwargs = metric_utils.preprocess_data(input_dicts, 
                                                         device=args.device, data_cats=args.data_cats)
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(input_set, kwargs)
        
        losses = 0.
        for task in keys:
            task_loss = sum(loss for k, loss in loss_dict.items() if task in k)
            task_loss *= args.loss_ratio[task]
            loss_dict.update({'bal_'+task: task_loss})
            losses += task_loss
            
        loss_dict_reduced = metric_utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for k, loss in loss_dict_reduced.items() if 'bal' in k)
        loss_value = losses_reduced.item()
        
        if not math.isfinite(loss_value):
            logger.log_text(f"Loss is {loss_value}, stopping training\n\t{loss_dict_reduced}", level='error')
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            
        else:
            losses.backward()
            optimizer.step()
            

        if lr_scheduler is not None:
            lr_scheduler.step()
        
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        iter_time.update(time.time() - end) 
        
        if (i % args.print_freq == 0 or biggest_size - 1 == 0) and get_rank() == 0:
            metric_logger.log_iter(
                iter_time.global_avg,
                args.epochs-epoch,
                logger,
                i
            )
            
            if tb_logger:
                tb_logger.update_scalars(loss_dict_reduced, i)   

            '''
            If the break block is in this block, the gpu memory will stuck (bottleneck)
            '''
            
        if args.bottleneck_test:
            if i == int(biggest_size/3):
                break
            
        if BREAK and i == args.print_freq:
            print("BREAK!!")
            torch.cuda.synchronize()
            break
            
        end = time.time()
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    biggest_size
    logger.log_text(f"{header} Total time: {total_time_str} ({total_time / biggest_size:.4f} s / it)")
    
    del data_loaders
    torch.cuda.empty_cache()
    time.sleep(3)


def seperate_train(model, optimizer, data_loaders, 
          epoch, logger, 
          tb_logger, scaler, args,
          warmup_fn=None):
    model.train()
    metric_logger = metric_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", metric_utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    
    input_dicts = OrderedDict()
    
    keys = list(data_loaders.keys())
    loaders = list(data_loaders.values())
    
    biggest_key, biggest_dl = keys[0], loaders[0]
    biggest_size = len(biggest_dl)
    
    others_keys = [None] + keys[1:]
    others_size = [None] + [len(ld) for ld in loaders[1:]]
    others_iterator = [None] + [iter(dl) for dl in loaders[1:]]
    
    load_cnt = {k: 1 for k in keys}
    
    header = f"Epoch: [{epoch}]"
    iter_time = metric_utils.SmoothedValue(fmt="{avg:.4f}")
    metric_logger.largest_iters = biggest_size
    metric_logger.epohcs = args.epochs
    metric_logger.set_before_train(header)
    
    lr_scheduler = warmup_fn(
        optimizer, args.warmup_ratio, biggest_size) if warmup_fn  else None
    
    if lr_scheduler:
        logger.log_text(f"Warmup Iteration: {int(lr_scheduler.total_iters)}/{biggest_size}")
    
    start_time = time.time()
    end = time.time()
    for i, b_data in enumerate(biggest_dl):
        input_dicts.clear()
        input_dicts[biggest_key] = b_data

        try:
            for n_task in range(1, len(others_iterator)):
                input_dicts[others_keys[n_task]] = next(others_iterator[n_task])
            
        except StopIteration:
            print("occur StopIteration")
            for i, (it, size) in enumerate(zip(others_iterator, others_size)):
                if it is None:
                    continue
                
                if it._num_yielded == size:
                    print("full iteration size:", it._num_yielded, size)
                    others_iterator[i] = iter(loaders[i])
                    load_cnt[keys[i]] += 1
                    logger.log_text(f"Reloading Count: {load_cnt}\n")
                    
            for n_task in range(1, len(others_iterator)):
                if not others_keys[n_task] in input_dicts.keys():
                    input_dicts[others_keys[n_task]] = next(others_iterator[n_task])
        
        if torch.cuda.is_available:
            torch.cuda.synchronize()
        
        if args.return_count:
            input_dicts.update({'load_count': load_cnt})

        input_set, kwargs = metric_utils.preprocess_data(input_dicts, 
                                                         device=args.device, data_cats=args.data_cats)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(input_set, kwargs)
            
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_reduced = metric_utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        
        if not math.isfinite(loss_value):
            logger.log_text(f"Loss is {loss_value}, stopping training\n\t{loss_dict_reduced}", level='error')
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            
        else:
            losses.backward()
            optimizer.step()
            

        if lr_scheduler is not None:
            lr_scheduler.step()
        
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        iter_time.update(time.time() - end) 
        
        if (i % args.print_freq == 0 or biggest_size - 1 == 0) and get_rank() == 0:
            metric_logger.log_iter(
                iter_time.global_avg,
                args.epochs-epoch,
                logger,
                i
            )
            
            if tb_logger:
                tb_logger.update_scalars(loss_dict_reduced, i)   

            '''
            If the break block is in this block, the gpu memory will stuck (bottleneck)
            '''
            
        if args.bottleneck_test:
            if i == int(biggest_size/3):
                break
            
        if BREAK and i == args.print_freq:
            print("BREAK!!")
            torch.cuda.synchronize()
            break
            
        end = time.time()
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    biggest_size
    logger.log_text(f"{header} Total time: {total_time_str} ({total_time / biggest_size:.4f} s / it)")
    
    del data_loaders
    torch.cuda.empty_cache()
    time.sleep(3)


def general_train(model, optimizer, data_loaders, 
          epoch, logger, 
          tb_logger, scaler, args,
          warmup_fn=None):
    model.train()
    metric_logger = metric_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", metric_utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    
    input_dicts = OrderedDict()
    
    keys = list(data_loaders.keys())
    loaders = list(data_loaders.values())
    
    biggest_key, biggest_dl = keys[0], loaders[0]
    biggest_size = len(biggest_dl)
    
    others_keys = [None] + keys[1:]
    others_size = [None] + [len(ld) for ld in loaders[1:]]
    others_iterator = [None] + [iter(dl) for dl in loaders[1:]]
    
    load_cnt = {k: 1 for k in keys}
    
    header = f"Epoch: [{epoch}]"
    iter_time = metric_utils.SmoothedValue(fmt="{avg:.4f}")
    metric_logger.largest_iters = biggest_size
    # metric_logger.epohcs = args.epochs
    metric_logger.set_before_train(header)
    
    lr_scheduler = warmup_fn(
        optimizer, args.warmup_ratio, biggest_size) if warmup_fn  else None
    
    if lr_scheduler:
        logger.log_text(f"Warmup Iteration: {int(lr_scheduler.total_iters)}/{biggest_size}")
    
    start_time = time.time()
    end = time.time()
    
    for i, b_data in enumerate(biggest_dl):
        input_dicts.clear()
        input_dicts[biggest_key] = b_data

        try:
            for n_task in range(1, len(others_iterator)):
                input_dicts[others_keys[n_task]] = next(others_iterator[n_task])
                
        except StopIteration:
            for i, (it, size) in enumerate(zip(others_iterator, others_size)):
                if it is None:
                    continue
                
                if it._num_yielded == size:
                    print("full iteration size:", it._num_yielded, size)
                    others_iterator[i] = iter(loaders[i])
                    load_cnt[keys[i]] += 1
                    logger.log_text(f"Reloading Count: {load_cnt}\n")
                    
            for n_task in range(1, len(others_iterator)):
                if not others_keys[n_task] in input_dicts.keys():
                    input_dicts[others_keys[n_task]] = next(others_iterator[n_task])
        
        if torch.cuda.is_available and len(loaders) > 1:
            torch.cuda.synchronize()
        
        if args.return_count:
            input_dicts.update({'load_count': load_cnt})

        input_set, kwargs = metric_utils.preprocess_data(input_dicts, 
                                                         device=args.device, data_cats=args.data_cats)
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(input_set, kwargs)
        
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_reduced = metric_utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        
        if not math.isfinite(loss_value):
            logger.log_text(f"Loss is {loss_value}, stopping training\n\t{loss_dict_reduced}", level='error')
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        iter_time.update(time.time() - end) 
        
        if (i % args.print_freq == 0 or biggest_size - 1 == 0) and get_rank() == 0:
            metric_logger.log_iter(
                iter_time.global_avg,
                args.epochs-epoch,
                logger,
                i
            )
            
            if tb_logger:
                tb_logger.update_scalars(loss_dict_reduced, i)   

            '''
            If the break block is in this block, the gpu memory will stuck (bottleneck)
            '''
            
        if args.bottleneck_test:
            if i == int(biggest_size/3):
                break
            
        if BREAK and i == args.print_freq:
            print("BREAK!!")
            torch.cuda.synchronize()
            break
            
        end = time.time()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    biggest_size
    logger.log_text(f"{header} Total time: {total_time_str} ({total_time / biggest_size:.4f} s / it)")
    
    del data_loaders
    torch.cuda.empty_cache()
    time.sleep(3)
    
    
    
def step_train(model, optimizer, data_loaders, 
          epoch, logger, 
          tb_logger, scaler, args,
          warmup_fn=None):
    model.train()
    
    total_time = 0
    for i, (task, loader) in enumerate(data_loaders.items()):
        current_task = task.upper()
        task_size = len(loader)
        logger.log_text(f"Order {i}: {current_task} DataLoader")
        logger.log_text(f"{current_task} Dataset Training Start\n")
        
        header = f"Step Training - {current_task} | Epoch: [{epoch}]"
        iter_time = metric_utils.SmoothedValue(fmt="{avg:.4f}")
        metric_logger = metric_utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", metric_utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.largest_iters = task_size
        metric_logger.set_before_train(header)
        
        freezing_task = [t for t in data_loaders.keys() if not t == task]
        model.module.freeze_layer(freezing_task)
        
        task_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = type(optimizer)(task_params, lr=optimizer.defaults['lr'])
        
        
        
        
        
        if warmup_fn:
            lr_scheduler = warmup_fn(optimizer, args.task_warmup_ratio[i], task_size)
            logger.log_text(f"{current_task} Warmup Iteration: {int(lr_scheduler.total_iters)}/{task_size}")
        
        # exit()
        # print("---"*60)
        # for n, p in model.named_parameters():
        #     print(n, p.requires_grad)
        # print("---"*30)
        # exit()
        
        start_time = time.time()
        end = time.time()
        time.sleep(2)
        for iters, data_batch in enumerate(loader):
            input_dict = {task: data_batch}
            input_set, kwargs = metric_utils.preprocess_data(input_dict, 
                                                            device=args.device, data_cats=args.data_cats)
            
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss_dict = model(input_set, kwargs)
            
            losses = sum(loss for loss in loss_dict.values())
            loss_dict_reduced = metric_utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_value = losses_reduced.item()
            
            if not math.isfinite(loss_value):
                logger.log_text(f"Loss is {loss_value}, stopping training\n\t{loss_dict_reduced}", level='error')
                sys.exit(1)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
                
            else:
                losses.backward()
                optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()
            
            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            iter_time.update(time.time() - end) 
            
            if (i % args.print_freq == 0 or task_size - 1 == 0) and get_rank() == 0:
                metric_logger.log_iter(
                    iter_time.global_avg,
                    args.epochs-epoch,
                    logger,
                    i
                )
                
                if tb_logger:
                    tb_logger.update_scalars(loss_dict_reduced, i)   

                '''
                If the break block is in this block, the gpu memory will stuck (bottleneck)
                '''
                
            if BREAK and i == args.print_freq:
                print("BREAK!!")
                torch.cuda.synchronize()
                break
            
        end = time.time()
        task_time = time.time() - start_time
        task_time_str = str(datetime.timedelta(seconds=int(task_time)))
        logger.log_text(f"Task {current_task} training time: {task_time_str} ({total_time / len(loader):.4f} s / it)")
        total_time += task_time

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        model.module.unfreeze_layer(freezing_task)
        # for n, p in model.named_parameters():
        #     print(n, p.requires_grad)
        # print("---"*60)
        time.sleep(2)
        
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.log_text(f"Total time on all tasks: {total_time_str}")
    
    del data_loaders
    torch.cuda.empty_cache()
    time.sleep(3)