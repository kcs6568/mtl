import time

import torch

import lib.utils.metric_utils as metric_utils
from datasets.coco.coco_eval import CocoEvaluator
from datasets.coco.coco_utils import get_coco_api_from_dataset

from lib.utils.dist_utils import get_rank

# BREAK=True
BREAK=False

def _get_iou_types(task):
    iou_types = ["bbox"]
    if task == 'seg':
        iou_types.append("segm")

    # model_without_ddp = model
    # if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    #     model_without_ddp = model.module
    # iou_types = ["bbox"]
    # if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
    #     iou_types.append("segm")
    # if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
    #     iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loaders, data_cats, logger):
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
            if datatype == 'coco':
                return _metric_detection
            elif datatype == 'voc':
                pass
            
        elif task == 'seg':
            if datatype == 'coco':
                pass
            elif datatype == 'voc':
                return _metric_segmentation
            
                
    def _select_val_fn(task, datatype):
        if task == 'clf':
            return _validate_classification
        elif task == 'det':
            if datatype == 'coco':
                return _validate_detection
            elif datatype == 'voc':
                pass
            
        elif task == 'seg':
            if datatype == 'coco':
                pass
            
            elif datatype == 'voc':
                return _validate_segmentation
    
    
    final_results = dict()
    for task, taskloader in data_loaders.items():
        if data_cats[task] == 'coco':
            coco = get_coco_api_from_dataset(data_loaders[task].dataset)
            iou_types = _get_iou_types(task)
            coco_evaluator = CocoEvaluator(coco, iou_types)
        
        val_function = _select_val_fn(task, data_cats[task])
        metric_function = _select_metric_fn(task, data_cats[task])
        metric_logger = metric_utils.MetricLogger(delimiter="  ")
        
        confmat = None
        if task == 'seg':
            if data_cats[task] == 'coco':
                num_classes = 91
            elif data_cats[task] == 'voc':
                num_classes = 21
            confmat = metric_utils.ConfusionMatrix(num_classes)
        
        header = "Validation - " + task.upper() + ":"
        iter_time = metric_utils.SmoothedValue(fmt="{avg:.4f}")
        metric_logger.largest_iters = len(taskloader)
        # metric_logger.epohcs = args.epochs
        metric_logger.set_before_train(header)
        
        # taskloader = dict([taskloader])
        # for i, batch_set in enumerate(metric_logger.log_every(taskloader, 50, logger, 1, header, train_mode=False)):
        
        start_time = time.time()
        end = time.time()
        
        task_kwargs = {'task': task}
        for i, data in enumerate(taskloader):
            batch_set = {task: data}
            '''
            batch_set: images(torch.cuda.tensor), targets(torch.cuda.tensor)
            '''
            batch_set, _ = metric_utils.preprocess_data(batch_set, data_cats=data_cats, train_mode=False)

            start_time = time.time()
            outputs = model(batch_set[task][0], task_kwargs)
            
            val_function(outputs, batch_set[task][1], start_time)
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
            if BREAK and i == 50:
                print("BREAK!!!")
                break
        
        time.sleep(2)
        eval_result = metric_function()
        final_results[task] = eval_result
        
        del taskloader
        time.sleep(1)
        torch.cuda.empty_cache()
        
    time.sleep(3)        
    
    return final_results