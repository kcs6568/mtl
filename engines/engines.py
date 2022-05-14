import math
import sys
import time
import datetime
from collections import OrderedDict

import torch

import lib.utils.metric_utils as metric_utils
from lib.utils.dist_utils import get_rank
from datasets.coco.coco_eval import CocoEvaluator
from datasets.coco.coco_utils import get_coco_api_from_dataset

cifar10_classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# BREAK=True
BREAK=False

'''
TODO:
    - 초기에 생성한 DDP에서 parameter 바꿀 수 있는 방법 찾아보기
'''

class LossCalculator:
    def __init__(self, type, data_cats, loss_ratio=None, method='multi_task') -> None:
        self.type = type
        self.method = method
        self.data_cats = data_cats
        
        if self.type == 'balancing':
            assert loss_ratio is not None
            self.loss_ratio = loss_ratio
            
            self.loss_calculator = self.balancing_loss
        
        elif self.type == 'general':
            self.loss_calculator = self.general_loss
            
            
    def balancing_loss(self, output_losses):
        assert isinstance(output_losses, dict)
        losses = 0.
        balanced_losses = dict()
        for data in self.data_cats:
            data_loss = sum(loss for k, loss in output_losses.items() if data in k)
            data_loss *= self.loss_ratio[data]
            balanced_losses.update({f"bal_{self.data_cats[data]}_{data}": data_loss})
            losses += data_loss
        
        # balanced_losses.update(output_losses)
        
        return losses
        
    
    def general_loss(self, output_losses):
        assert isinstance(output_losses, dict)
        
        # if self.method == 'cross_stitch':
        #     losses = 0.
        #     losses = sum(sum(loss.values()) for loss in output_losses.values())
        # else:
        losses = sum(loss for loss in output_losses.values())
        
        # # losses = 0.
        # # for k, v in output_losses.items():
        #     print(k, v)
        # #     losses += v
        
        return losses


    # def task_loss(self, output_losses):
    #     assert isinstance(output_losses, dict)
    #     losses = {dset: sum(list(losses.values())) for dset, losses in output_losses.items()}
        
    #     return losses
    

def training(model, optimizer, data_loaders, 
          epoch, logger, 
          tb_logger, scaler, args,
          warmup_fn=None):
    model.train()
    
    metric_logger = metric_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", metric_utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    
    input_dicts = OrderedDict()
    
    datasets = list(data_loaders.keys())
    loaders = list(data_loaders.values())
    
    biggest_datasets, biggest_dl = datasets[0], loaders[0]
    biggest_size = len(biggest_dl)
    
    others_dsets = [None] + datasets[1:]
    others_size = [None] + [len(ld) for ld in loaders[1:]]
    others_iterator = [None] + [iter(dl) for dl in loaders[1:]]
    
    load_cnt = {k: 1 for k in datasets}
    
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
    
    if args.lossbal:
        type = 'balancing'
    elif args.general:
        type = 'general'
        
    loss_calculator = LossCalculator(
        type, args.task_per_dset, args.loss_ratio, method=args.method)
    
    start_time = time.time()
    end = time.time()
    for i, b_data in enumerate(biggest_dl):
        input_dicts.clear()
        input_dicts[biggest_datasets] = b_data

        try:
            for n_dset in range(1, len(others_iterator)):
                input_dicts[others_dsets[n_dset]] = next(others_iterator[n_dset])
            
        except StopIteration:
            print("occur StopIteration")
            for i, (it, size) in enumerate(zip(others_iterator, others_size)):
                if it is None:
                    continue
                
                if it._num_yielded == size:
                    print("full iteration size:", it._num_yielded, size)
                    others_iterator[i] = iter(loaders[i])
                    load_cnt[datasets[i]] += 1
                    logger.log_text(f"Reloading Count: {load_cnt}\n")
                    
            for n_task in range(1, len(others_iterator)):
                if not others_dsets[n_task] in input_dicts.keys():
                    input_dicts[others_dsets[n_task]] = next(others_iterator[n_task])
        
        if args.return_count:
            input_dicts.update({'load_count': load_cnt})
        
        input_set = metric_utils.preprocess_data(input_dicts, args.task_per_dset)
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(input_set, args.task_per_dset)
        
        # print(loss_dict)
        
        losses = loss_calculator.loss_calculator(loss_dict)
        # print(losses)
        # exit()
        
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
        
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if args.method == 'cross_stitch':
            metric_logger.update(stitch_lr=optimizer.param_groups[1]["lr"])
        metric_logger.update(loss=losses, **loss_dict_reduced)
        iter_time.update(time.time() - end) 
        
        if BREAK:
            args.print_freq = 10
        
        if (i % args.print_freq == 0 or i == (biggest_size - 1)) and get_rank() == 0:
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
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    biggest_size
    logger.log_text(f"{header} Total time: {total_time_str} ({total_time / biggest_size:.4f} s / it)")
    
    del data_loaders
    torch.cuda.empty_cache()
    time.sleep(3)
    
def _get_iou_types(task):
    iou_types = ["bbox"]
    if task == 'seg':
        iou_types.append("segm")

    return iou_types


@torch.inference_mode()
def evaluate(model, data_loaders, data_cats, logger, num_classes):
    assert isinstance(num_classes, dict) or isinstance(num_classes, OrderedDict)
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    
    def _validate_classification(outputs, targets, start_time):
        # print("######### entered clf validate")
        accuracy = metric_utils.accuracy(outputs['outputs'].data, targets, topk=(1, 5))
        eval_endtime = time.time() - start_time
        metric_logger.update(
            top1=accuracy[0],
            top5=accuracy[1],
            eval_time=eval_endtime)
        

    def _metric_classification():
        # print("######### entered clf metric")
        metric_logger.synchronize_between_processes()
        # print("######### synchronize finish")
        top1_avg = metric_logger.meters['top1'].global_avg
        top5_avg = metric_logger.meters['top5'].global_avg
        
        logger.log_text("<Current Step Eval Accuracy>\n --> Top1: {}% || Top5: {}%".format(
            top1_avg, top5_avg))
        torch.set_num_threads(n_threads)
        
        return top1_avg
        
        
    def _validate_detection(outputs, targets, start_time):
        # print("######### entered det validate")
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - start_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        
    
    def _metric_detection():
        # print("######### entered det metric")        
        metric_logger.synchronize_between_processes()
        coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        coco_evaluator.log_eval_summation(logger)
        torch.set_num_threads(n_threads)
        
        return coco_evaluator.coco_eval['bbox'].stats[0] * 100.
    
    
    def _validate_segmentation(outputs, targets, start_time=None):
        # print("######### entered seg validate")
        confmat.update(targets.flatten(), outputs['outputs'].argmax(1).flatten())
        
        
    def _metric_segmentation():
        # print("######### entered seg metirc")
        confmat.reduce_from_all_processes()
        logger.log_text("<Current Step Eval Accuracy>\n{}".format(confmat))
        return confmat.mean_iou
    
    
    def _select_metric_fn(task, datatype):
        if task == 'clf':
            return _metric_classification
        
        elif task == 'det':
            if 'coco' in datatype:
                return _metric_detection
            elif 'voc' in datatype:
                pass
            
        elif task == 'seg':
            if 'coco' in datatype:
                pass
            elif ('voc' in datatype) \
                or ('cityscapes' in datatype):
                    return _metric_segmentation

                
    def _select_val_fn(task, datatype):
        if task == 'clf':
            return _validate_classification
        elif task == 'det':
            if 'coco' in datatype:
                return _validate_detection
            elif datatype == 'voc':
                pass
            
        elif task == 'seg':
            if 'coco' in datatype:
                pass
            
            elif ('voc' in datatype) \
                or ('cityscapes' in datatype):
                return _validate_segmentation
    
    
    final_results = dict()
    
    for dataset, taskloader in data_loaders.items():
        task = data_cats[dataset]
        dset_classes = num_classes[dataset]
        
        # if not 'seg' in task:
        #     continue
        
        if 'coco' in dataset:
            coco = get_coco_api_from_dataset(taskloader.dataset)
            iou_types = _get_iou_types(task)
            coco_evaluator = CocoEvaluator(coco, iou_types)
        
        val_function = _select_val_fn(task, dataset)
        metric_function = _select_metric_fn(task, dataset)
        metric_logger = metric_utils.MetricLogger(delimiter="  ")
        
        assert val_function is not None
        assert metric_function is not None
        
        
        
        confmat = None
        if task == 'seg':
            # if 'coco' in dataset:
            #     num_classes = 91
            # elif 'voc' in dataset:
            #     num_classes = 21
            confmat = metric_utils.ConfusionMatrix(dset_classes)
        
        header = "Validation - " + dataset.upper() + ":"
        iter_time = metric_utils.SmoothedValue(fmt="{avg:.4f}")
        metric_logger.largest_iters = len(taskloader)
        # metric_logger.epohcs = args.epochs
        metric_logger.set_before_train(header)
        
        # taskloader = dict([taskloader])
        # for i, batch_set in enumerate(metric_logger.log_every(taskloader, 50, logger, 1, header, train_mode=False)):
        
        start_time = time.time()
        end = time.time()
        
        task_kwargs = {'dtype': dataset, 'task': task}
        for i, data in enumerate(taskloader):
            # print(data)
            batch_set = {dataset: data}
            '''
            batch_set: images(torch.cuda.tensor), targets(torch.cuda.tensor)
            '''
            # batch_set, _ = metric_utils.preprocess_data(batch_set, task_size=len(data_loaders),
            #                                             data_cats=data_cats)

            # start_time = time.time()
            # outputs = model(batch_set[dataset][0], task_kwargs)
            
            batch_set = metric_utils.preprocess_data(batch_set, data_cats)

            start_time = time.time()
            outputs = model(batch_set[dataset][0], task_kwargs)
            
            val_function(outputs, batch_set[dataset][1], start_time)
            iter_time.update(time.time() - end) 
            
            if (i % 50 == 0 or len(taskloader) - 1 == 0) and get_rank() == 0:
                metric_logger.log_iter(
                iter_time.global_avg,
                1,
                logger,
                i
            )
            
            # if tb_logger:
            #     tb_logger.update_scalars(loss_dict_reduced, i)   
            end = time.time()
            if BREAK and i == 2:
                print("BREAK!!!")
                break
        
        time.sleep(2)
        eval_result = metric_function()
        final_results[dataset] = eval_result
        
        del taskloader
        time.sleep(1)
        torch.cuda.empty_cache()
            
        
    time.sleep(3)        
    
    return final_results
    

# def general_train(model, optimizer, data_loaders, 
#           epoch, logger, 
#           tb_logger, scaler, args,
#           warmup_fn=None):
#     model.train()
#     metric_logger = metric_utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter("lr", metric_utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    
#     input_dicts = OrderedDict()
    
#     datasets = list(data_loaders.keys())
#     loaders = list(data_loaders.values())
    
#     biggest_datasets, biggest_dl = datasets[0], loaders[0]
#     biggest_size = len(biggest_dl)
    
#     others_dsets = [None] + datasets[1:]
#     others_size = [None] + [len(ld) for ld in loaders[1:]]
#     others_iterator = [None] + [iter(dl) for dl in loaders[1:]]
    
#     load_cnt = {k: 1 for k in datasets}
    
#     header = f"Epoch: [{epoch}]"
#     iter_time = metric_utils.SmoothedValue(fmt="{avg:.4f}")
#     metric_logger.largest_iters = biggest_size
#     metric_logger.epohcs = args.epochs
#     metric_logger.set_before_train(header)
    
#     lr_scheduler = warmup_fn(
#         optimizer, args.warmup_ratio, biggest_size) if warmup_fn  else None
    
#     if lr_scheduler:
#         logger.log_text(f"Warmup Iteration: {int(lr_scheduler.total_iters)}/{biggest_size}")
#     else:
#         logger.log_text("No Warmup Training")
    
#     start_time = time.time()
#     end = time.time()
#     for i, b_data in enumerate(biggest_dl):
#         input_dicts.clear()
#         input_dicts[biggest_datasets] = b_data

#         try:
#             for n_dset in range(1, len(others_iterator)):
#                 input_dicts[others_dsets[n_dset]] = next(others_iterator[n_dset])
            
#         except StopIteration:
#             print("occur StopIteration")
#             for i, (it, size) in enumerate(zip(others_iterator, others_size)):
#                 if it is None:
#                     continue
                
#                 if it._num_yielded == size:
#                     print("full iteration size:", it._num_yielded, size)
#                     others_iterator[i] = iter(loaders[i])
#                     load_cnt[datasets[i]] += 1
#                     logger.log_text(f"Reloading Count: {load_cnt}\n")
                    
#             for n_task in range(1, len(others_iterator)):
#                 if not others_dsets[n_task] in input_dicts.keys():
#                     input_dicts[others_dsets[n_task]] = next(others_iterator[n_task])
        
#         if args.return_count:
#             input_dicts.update({'load_count': load_cnt})
        
#         input_set, kwargs = metric_utils.preprocess_data(input_dicts, task_size=args.num_datasets,
#                                                          device=args.device, data_cats=args.data_cats)
        
#         with torch.cuda.amp.autocast(enabled=scaler is not None):
#             loss_dict = model(input_set, kwargs)
        
#         losses = sum(loss for loss in loss_dict.values())
#         loss_dict_reduced = metric_utils.reduce_dict(loss_dict)
#         losses_reduced = sum(loss for loss in loss_dict_reduced.values())
#         loss_value = losses_reduced.item()
        
#         if not math.isfinite(loss_value):
#             logger.log_text(f"Loss is {loss_value}, stopping training\n\t{loss_dict_reduced}", level='error')
#             sys.exit(1)

#         optimizer.zero_grad()
#         if scaler is not None:
#             scaler.scale(losses).backward()
#             scaler.step(optimizer)
#             scaler.update()
            
#         else:
#             losses.backward()
#             optimizer.step()

#         if lr_scheduler is not None:
#             lr_scheduler.step()
        
#         metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#         iter_time.update(time.time() - end) 
        
#         if (i % args.print_freq == 0 or biggest_size - 1 == 0) and get_rank() == 0:
#             metric_logger.log_iter(
#                 iter_time.global_avg,
#                 args.epochs-epoch,
#                 logger,
#                 i
#             )
            
#             if tb_logger:
#                 tb_logger.update_scalars(loss_dict_reduced, i)   

            
#         if BREAK and i == args.print_freq:
#             print("BREAK!!")
#             torch.cuda.synchronize()
#             break
            
#         end = time.time()
#     total_time = time.time() - start_time
#     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#     biggest_size
#     logger.log_text(f"{header} Total time: {total_time_str} ({total_time / biggest_size:.4f} s / it)")
    
#     del data_loaders
#     torch.cuda.empty_cache()
#     time.sleep(3)
    

