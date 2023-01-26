"""
# Description: 
# Author: Taojj
# Date: 2020-09-04 10:24:06
# LastEditTime: 2021-01-15 18:06:16
# FilePath: /FCOSLite/test_img.py
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from datasets import *
from model.detector import Detector
from datasets.voc import VOCJsonAnnotationTransform, VOCDataset
import torch.utils.data as data
from utils import get_device,get_parameter_number
from datasets.voc import VOC_CLASSES, VOCDataset
import numpy as np
import cv2
import time
from decimal import *
from configs import *
from model.backbone.repvgg import repvgg_model_convert
USE_DISK = True

VOC_ROOT = '../datasets/VOCdevkit/'
SIZE = 512
def get_img(file_path,file_name):
    img = cv2.imread(file_path + file_name)
    
    height, width, channels = img.shape
    
    # to rgb
    img = cv2.resize(img, (SIZE, SIZE))
    _img = img[:, :, (2, 1, 0)].astype(np.float32)
    
    #img = img_[:, :, (2, 1, 0)]
    # to tensor
    return  torch.from_numpy(_img).permute(2, 0, 1).unsqueeze(0), img
def test_model(model, device, img, size,class_color):
    bbox_list, score_list, cls_list = model(img.to(device))
    for bboxes, scores, cls_indexes in zip(bbox_list, score_list, cls_list):
        img = img.squeeze(0)
        img = img.permute(1,2,0).numpy()[:, :, (2, 1, 0)].astype(np.uint8).copy()
        for bbox, score, cls_index in zip(bboxes, scores, cls_indexes):
                cls_name = VOC_CLASSES[cls_index] 
                xmin, ymin, xmax, ymax = bbox
                xmin *= size[0]
                ymin *= size[1]
                xmax *= size[0]
                ymax *= size[1]
                print(xmin,ymin,xmax,ymax)
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color[cls_index], 1)
                cv2.rectangle(img, (int(xmin), int(abs(ymin)+10)), (int(xmax), int(ymin)), class_color[cls_index], -1)  
                img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color[cls_index], 2)
    return img
        

def init_model(voc):
    num_classes = len(VOC_CLASSES)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        device = "cuda"
    else:
        device = "cpu"

    model = Detector(device, input_size=voc['input_size'], num_cls=20, strides = voc['strides'],  cfg=REPA0_RDB_VOC, boost=True)
    checkpoint = torch.load('./weights/REPA0_RDB_VOC.pth',map_location=device)
    # NOTE: uncomment below if optimizer parameters in checkpoint
    # model.load_state_dict(checkpoint["model"],strict=False)
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    model = repvgg_model_convert(model)
    model.eval()
    print(get_parameter_number(model))
    print('Finished loading model!')
    return model, device
def get_all_files(dir):
    files_ = []
    list_ = os.listdir(dir)
    for i in range(0, len(list_)):
        path = os.path.join(dir, list_[i])
        if os.path.isdir(path):
            files_.extend(get_all_files(path))
        if os.path.isfile(path):
            files_.append(path)
    return files_
def plot_bbox_labels(img, bbox, label, cls_color, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    # plot title bbox
    cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
    # put the test on the title bbox
    cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img


def visualize(img, bboxes, scores, cls_inds, vis_thresh, class_colors, class_names, class_indexs=None, dataset='voc'):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        if scores[i] >= vis_thresh:
            if dataset in ('coco-val', 'coco-test'):
                cls_color = class_colors[int(cls_inds[i])]
                cls_id = class_indexs[int(cls_inds[i])]
            else:
                cls_id = int(cls_inds[i])
                cls_color = class_colors[cls_id]
            mess = '%s: %.2f' % (class_names[cls_id], scores[i])
            color = (int(cls_color[0]),int(cls_color[1]),int(cls_color[2]))
            img = plot_bbox_labels(img, bbox, mess, color, text_scale=ts)

    return img

    
if __name__ == "__main__":
    class_color = []
    class_color = np.load('class_color.npy', allow_pickle=True)
    voc = voc_512
    scale = np.array([[SIZE, SIZE, SIZE, SIZE]])
    model, device = init_model(voc)
    model = repvgg_model_convert(model)
    count = 0
    with open("../datasets/VOCdevkit/VOC2007/ImageSets/Main/test.txt",'r') as f:
        for line in f:
            filename = line.strip("\n")
            count += 1
            img_tensor, img = get_img('../datasets/VOCdevkit/VOC2007/JPEGImages/',filename+".jpg")
            img_tensor = img_tensor.to(device)
            gpu_t, all_list = model(img_tensor)
            bbox_list, score_list, cls_list = all_list[:]
            print(gpu_t)
            bbox_list = bbox_list * scale
            img_processed = visualize(img.astype(np.uint8).copy(),bbox_list[0],score_list[0],cls_list[0],0.2,class_color,VOC_CLASSES)
            cv2.imshow('test', img_processed)
            cv2.waitKey(0)
