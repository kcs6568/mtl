import os
import cv2
import json
from matplotlib.image import _ImageBase
from matplotlib.style import available
import yaml
import numpy as np
import pandas as pd
import colorsys
from PIL import Image
from cv2 import transform
import matplotlib.pyplot as plt
from matplotlib import colors
from copy import deepcopy

import torch
import torchvision.transforms.functional as tv_F
from torch.nn import functional as F
from torchvision import datasets, models, transforms


from lib.utils.parser import InferenceParser
from lib.model_api.build_model import build_model


cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
stl_classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

# 91 classes
coco_classes = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
    'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window',
    'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush'
)

voc_classes = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]


def visualize_classification(prediction, output_dir, save_name):
    results = F.softmax(prediction['outputs'], dim=1)
    print(results)
    if isinstance(results, torch.Tensor):
        results = results.cpu().detach().numpy()
        
    cmap = plt.cm.YlGn
    # norm = colors.Normalize(vmin=1.5, vmax=4.5)
    # c = np.random.rand(len(results))*3+1.5
    
    im = plt.imshow(results.reshape(1, 10), cmap=cmap)
    plt.xticks(np.arange(10), labels=cifar_classes, rotation=90, fontsize=23)
    ax = plt.gca()
    ax.get_yaxis().set_visible(False)
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    plt.colorbar(im, cax)
    plt.show()    
    plt.savefig(
        os.path.join(output_dir, f"{save_name}_prob.png"),
        dpi=600)
    

def visualize_detection(image, prediction, output_dir, save_name):
    def _nms(scores, threshold=0.8):
        available_idx = np.ndarray(scores.shape)
        
        for i, s in enumerate(scores):
            if s >= threshold:
                available_idx[i] = 1
            else:
                available_idx[i] = 0
                
        return available_idx
    
    
    from torchvision.utils import draw_bounding_boxes, save_image
    
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    boxes = prediction[0]['boxes'].cpu().detach().numpy()
    scores = prediction[0]['scores'].cpu().detach().numpy()
    labels = prediction[0]['labels'].cpu().detach().numpy()
    boxes = boxes.astype(np.int32)
    
    available_boxes_idx = _nms(prediction[0]['scores'])
    
    # boxes = prediction[0]['boxes'].to(torch.uint8)
    # labels = [str(l.cpu().detach()) for l in prediction['labels']]
    
    color = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (36,255,12)
    ]
    
    font = [
        cv2.FONT_ITALIC,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX
    ]
    
    print(prediction)
    # exit()
    for i, (xmin, ymin, xmax, ymax) in enumerate(boxes):
        if available_boxes_idx[i] == 1:
            # print(boxes[i], scores[i], labels[i], coco_classes[labels[i]-1])
            image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color[-1], 1)
            
            label = coco_classes[labels[i]-1]
            score = str(round(scores[i], 2))
            text = f"{label}|{score}"
            print(i, labels[i], label, score)
            print(text)
            cv2.putText(image, text, (xmin, ymin-10), font[1], 0.5, color[-2], 1)
    # exit()
    save_path = os.path.join(output_dir, f"{save_name}.png")
    cv2.imwrite(save_path, image)
    
    # drawn_boxes = draw_bounding_boxes(
    #     image, 
    #     boxes,
    #     fill=True)
    #     # labels)
    
    # print(drawn_boxes)
    # print(drawn_boxes.size())
    
    # drawn_boxes = drawn_boxes.permute(2, 1, 0)
    # print(drawn_boxes.size())
    # drawn_boxes = drawn_boxes.cpu().detach().numpy()
    # save_path = os.path.join(output_dir, f"{save_name}_drawn.png")
    # plt.imshow(drawn_boxes)
    # plt.savefig(
    #     save_path, dpi=600
    # )
    
    
    
    # save_image(drawn_boxes, save_path)
    
    
    
    pass

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    # fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = tv_F.to_pil_image(img)
        plt.imshow(np.asarray(img))
        plt.axis("off")
        plt.savefig(f"/root/test{i}.png")
        
        # axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        # exit()
        


def visualize_segmentation(image, torch_image, prediction, output_dir, save_name, threshold=0.8):
    from torchvision.utils import draw_segmentation_masks, save_image
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    prediction = prediction['outputs']
    print(prediction.size())
    
    masks = torch.nn.functional.softmax(prediction, dim=1)
    m = masks[0].cpu().detach().numpy()
    
    # image_copy_for_mask = deepcopy(image)
    for i, c in enumerate(VOC_COLORMAP):
        image_copy = deepcopy(image)
        image_copy_for_mask = deepcopy(image)
        m_ = cv2.threshold(m[i], 0.3, 255, cv2.THRESH_BINARY)[1]
        image_copy_for_mask[m_==255] = c
        # transparented_result = cv2.addWeighted(image_copy, 0.5, image_copy_for_mask, 0.7, 0)
    # bool_masks = [m > threshold for m in masks[0]]
    
        cv2.imwrite(f"/root/volume/seg_multi_res_{i}.png", image_copy_for_mask)
    
    # # mask = bool_masks[0].cpu().detach().numpy()
    # mask = masks[0][0].cpu().detach().numpy()
    # dst = cv2.addWeighted(image, 0.8, mask, 0.2, 0)
    
    # cv2.imwrite("ttt.png", dst)
    exit()
    
    
    # result = [m for m in masks[0]]
    # print(result[0].unsqueeze(0).size())
    # exit()
    # show(result)
    # exit()
    
    
    
    
    
    # masks = torch.nn.functional.softmax(prediction, dim=1)
    # print(masks.size())
    # exit()

    # torch_image = torch_image.to(torch.uint8)
    
    
    # re = prediction.argmax(1).flatten()
    # print(re.size())
    # exit()
    
    
    # a = masks[0][0][0]
    # print(a.size())
    # for p in masks:
    #     print(p)
    exit()
    
    
    cls = torch.argmax(prediction, dim=1)
    print(cls.size())
    print(max(cls))
    exit()
    
    
    
    cls = cls.to(torch.uint8)
    a = draw_segmentation_masks(torch_image[0], cls)
    exit()
    
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    prediction = prediction['outputs']
    cls = torch.argmax(prediction, dim=1)
    cls = cls.cpu().detach().numpy()
    # canvus = np.zeros(cls.shape, np.uint8)
    
    cv2.imwrite("./test.png", cls)
    
    # print(cls)
    prediction = prediction.cpu().detach().numpy()
    pred_imgs = [prediction[p] for p in cls]

    print(pred_imgs)
    print(pred_imgs.shape)
    
    # for i, pred_img in enumerate(pred_imgs):
    #     plt.imshow(pred_img)
    #     plt.savefig(
    #         f"./test_{i}", dpi=600
    #     )
    
    exit()
    
    
    masks = torch.nn.functional.softmax(prediction, dim=1)
    
    
    bool_masks = masks > threshold
    # bool_masks = bool_masks.permute(2, 1, 0)
    # print(bool_masks.size())
    
    
    
    
    return
    color = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (36,255,12)
    ]
    
    font = [
        cv2.FONT_ITALIC,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX
    ]
    
    
    
    
    save_path = os.path.join(output_dir, f"{save_name}.png")
    cv2.imwrite(save_path, image)


def main(args):
    with open(args.yaml_cfg, 'r') as f:
        configs = yaml.safe_load(f)
    
    for i, j in configs.items():
        setattr(args, i, j)
        
    with open(args.cfg, 'r') as f:
        configs = yaml.safe_load(f)
    
    for i, j in configs.items():
        setattr(args, i, j)
    
    model = build_model(args)
    ckpt = os.path.join(args.ckpt)
    checkpoint = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    
    best_result = checkpoint['best_results']
    print("best result:", best_result)
    
    torch.cuda.set_device(int(args.gpu))
    model.cuda()
    model.eval()
    
    images = [
        Image.open(args.det_sample),
        Image.open(args.seg_sample)
    ]
    
    for i, image in enumerate(images):
        infer_transform = transforms.Compose([transforms.ToTensor()])
        torch_image = infer_transform(image).unsqueeze(0).cuda()
        
        task_cfg = args.task[args.task_type]
        prediction = model(torch_image, task_cfg)
        
        dataset = list(task_cfg.keys())[0]
        task = list(task_cfg.values())[0]
        
        save_name = args.save_name + f"_{str(i)}"
        
        if task == 'clf':
            visualize_classification(prediction, args.outdir, save_name)
        elif task == 'det':
            visualize_detection(image, prediction, args.outdir, save_name)
        elif task == 'seg':
            visualize_segmentation(image, torch_image, prediction, args.outdir, save_name)
    
    print("Inference Finish\n\n")

if __name__ == "__main__":
    args = InferenceParser().args
    main(args)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # x = np.arange(10)
    # y = np.random.rand(len(x))
    # c = np.random.rand(len(x))*3+1.5
    # df = pd.DataFrame({"x":x,"y":y,"c":c})

    # cmap = plt.cm.YlGn
    # norm = colors.Normalize(vmin=1.5, vmax=4.5)

    # # plt.barh(y, x, color=cmap(norm(df.c.values)))
    
    # # plt.yticks([0, 0.5, 1])
    # # plt.xticks(['a', 'b', 'c','d','e','f','g','h','i','j'])
    
    # fig, ax = plt.subplots()
    # hbars = ax.barh(y, x, color=cmap(norm(c)))
    # # ax.set_xticks([0, 0.5, 1])
    # ax.set_yticks(y, labels=classes)
    # ax.invert_yaxis()
    # ax.set_xlim(right=1)

    # # 
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])  # only needed for matplotlib < 3.1
    # fig.colorbar(sm)


    # cmap = plt.cm.YlGn
    # norm = colors.Normalize(vmin=1.5, vmax=4.5)
    # people = classes
    # y_pos = np.arange(len(people))
    # c = np.random.rand(len(people))*3+1.5
    # performance = np.array([9.9990e-01, 3.2414e-10, 8.3237e-05, 2.4076e-06, 5.4104e-08, 4.2238e-09,
    #      1.4520e-08, 1.2475e-10, 1.9153e-05, 7.9884e-10])
    
    # fig, ax = plt.subplots()

    # hbars = ax.barh(y_pos, performance, align='center', color=cmap(norm(c)))
    # ax.set_yticks(y_pos, labels=people)
    # ax.invert_yaxis()  # labels read top-to-bottom
    # ax.set_xlabel('Performance')
    # ax.set_xticks([0, 0.5, 1])
    # ax.set_title('How fast do you want to go today?')

    # # Label with specially formatted floats
    # # ax.bar_label(hbars, fmt='%.2f')
    # ax.set_xlim(right=0.1)  # adjust xlim to fit labels


    # c = ax1.pcolor(Z, edgecolors='k', linewidths=4)
    # ax1.set_title('thick edges')

    
    
    # color = [192, 64, 1]
    # ratio = np.array([1/3, 1/2, 1/1, 1/5, 1/7, 1/8, 1/2, 1/2, 1/1, 1/3])
    
    # r, g, b = 192, 64, 1
    # r, g, b = [x/255.0 for x in [r, g, b]]
    # h, l, s = colorsys.rgb_to_hls(r, g*1/3, b)
    # r, g, b = colorsys.hls_to_rgb(h, l, s)
    # r, g, b = [x*255.0 for x in [r, g, b]]
    
    # print(r, g, b)
    # exit()
    
    
    # data = [scale_lightness(color, 1/scale) for scale in ratio]
    # print(data)
    
    
    
    
    # # exit()
    
    
    # # rgb_ratio = colors.hsv_to_rgb(hsv_ratio)
    # # print(rgb_ratio)
    # # # print(hsv_ratio)
    # # exit()
    # # colors = [255, 255, 0]
    
    # # data = (np.array(ratio) * 60).round().astype(int)
    # ratios_int = (np.array(ratio) * 60).round().astype(int)
    # plt.imshow(
    #     np.repeat(np.arange(len(data)), ratios_int).reshape(1, -1),
    #     cmap=colors.ListedColormap(data),
    #     aspect=ratios_int.sum()/10
    # )
    
    # plt.axis('off')
    
    
    # ax = plt.subplots()
    # im = ax.imshow(np.array(10).reshape(1,10), cmap=)
    

    
    
    
