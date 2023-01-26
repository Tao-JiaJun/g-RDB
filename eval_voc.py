import sys
import glob
import time
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from datasets import *
from utils import get_parameter_number
import torch.utils.data as data
MINOVERLAP = 0.5 
from datasets.voc import VOC_CLASSES, VOCDataset
from datasets.config import voc_384,voc_512
from model.detector import Detector
from configs import *
from tqdm import tqdm
from contextlib import redirect_stdout, redirect_stderr
from model.backbone.repvgg import repvgg_model_convert
def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre

def sort_by_score(detec_result):
    """
    # Description: Sort the results from high to low by score
    # Author: Taojj
    """
    for item in detec_result:
        item.sort(key=lambda x:float(x['score']), reverse=True)
        
def count_iou(rec_a, rec_b):
    """
    # Description: count IoU
    # Author: Bubbliiiing
    """
    xA = max(rec_a[0], rec_b[0])
    yA = max(rec_a[1], rec_b[1])
    xB = min(rec_a[2], rec_b[2])
    yB = min(rec_a[3], rec_b[3])
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    recA_area = (rec_a[2] - rec_a[0] + 1) * (rec_a[3] - rec_a[1] + 1)
    recB_area = (rec_b[2] - rec_b[0] + 1) * (rec_b[3] - rec_b[1] + 1)
    iou = inter_area / float(recA_area + recB_area - inter_area)
    return iou

def calculate_ap(ground_truth, detec_result, gt_counter_per_class):
    sum_AP = 0
    count_true_positives = {}
    bbox_final = {}
    for key in ground_truth.keys():
       bbox_final[key] = []
    for cls_index, (dr_data, cls_name) in enumerate(zip(detec_result, VOC_CLASSES)):
        count_true_positives[cls_name] = 0
        nd = len(dr_data)
        tp = [0] * nd
        fp = [0] * nd
        for index, detection in enumerate(dr_data):
            iou_max = -1
            gt_match = -1
            file_path = detection["file_path"]
            file_name = detection["file_name"]
            bb_dr = [ float(x) for x in detection["bbox"].split() ]
            for obj in ground_truth[file_name]:
                if obj["cls_name"] == cls_name:
                    bb_gt = [ float(x) for x in obj["bbox"].split() ]
                    iou = count_iou(bb_dr, bb_gt)
                    if iou > iou_max:
                        iou_max = iou
                        gt_match = obj
      
            min_overlap = MINOVERLAP
            if iou_max >= min_overlap:
                if gt_match["difficult"] == 0:
                    if not bool(gt_match["used"]):
                        bbox_final[file_name].append(bb_dr)
                        # true positive
                        tp[index] = 1
                        gt_match["used"] = True
                        count_true_positives[cls_name] += 1
                    else:
                        # false positive (multiple detection)
                        fp[index] = 1
            else:
                # false positive
                fp[index] = 1
        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        rec = tp[:]
        for idx, val in enumerate(tp):
            if cls_name in gt_counter_per_class:
                rec[idx] = float(tp[idx]) / gt_counter_per_class[cls_name]
            else:
                rec[idx] = -1
        #print(rec)
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        #print(prec)
        if -1 in rec:
            ap = -1
        else:
            ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap
        text = "{0:.2f}%".format(ap*100) + " = " + cls_name + " AP " 
        print(text)    
    mAP = sum_AP  / len(VOC_CLASSES)
    text = "mAP = {0:.2f}%".format(mAP*100)
    #print(text)
    return mAP
 
               
def eval_model(model, device, dataloader, num_images):
    num_images = num_images
    image_names = []
    scale_list = []
    ground_truth = {}
    gt_counter_per_class = {}
    # ground_truth = np.load('./eval/gt.npy', allow_pickle=True).item()
    # gt_counter_per_class = np.load('./eval/gt_counter.npy', allow_pickle=True).item()
    detec_result = [[] for _ in range(len(VOC_CLASSES)) ]
    counter_images_per_class = {}
    gpu_time = 0
    detec_time = 0
    print("===> Detecting")
    # with tqdm(total=len(dataloader)) as pbar:
    for i, (img_info, img, targets) in enumerate(dataloader):
        img = img.to(device)
        tic = time.time()
        gpu_t, all_list = model(img)
        bbox_list, score_list, cls_list = all_list[:]
        detec_time += time.time() - tic
        gpu_time += gpu_t
        for j, (info, gts, bboxes, scores, cls_indexes) in enumerate(zip(img_info, targets, bbox_list, score_list, cls_list)):
            # get image info
            file_path = info[0]
            file_name = info[1]
            w = info[2]
            h = info[3]
            # NOTE: If do not have both ./eval/gt.npy and ./eval/gt_counter.py, uncomment below
            img_gt = []
            for gt in gts:
                gt *= np.array([w, h, w, h, 1])
                cls_index = int(gt[4])
                cls_name = VOC_CLASSES[cls_index]
                bbox = "{:.1f} {:.1f} {:.1f} {:.1f}".format(round(gt[0])+1,round(gt[1])+1,round(gt[2])+1,round(gt[3])+1)
                img_gt.append({"cls_name":cls_name, "bbox":bbox, "difficult": 0, "used":False})

                if cls_name in gt_counter_per_class:
                    gt_counter_per_class[cls_name] += 1
                else:
                    gt_counter_per_class[cls_name] = 1
            ground_truth[file_name] = img_gt
            # NOTE: If do not have both ./eval/gt.npy and ./eval/gt_counter.py, uncomment the above
            img_bbox = []
            scale = np.array([[w, h, w, h]])
            bboxes *= scale
            for bbox, score, cls_index in zip(bboxes, scores, cls_indexes):
                cls_name = VOC_CLASSES[cls_index] 
                box = "{:.1f} {:.1f} {:.1f} {:.1f}".format(bbox[0]+1,bbox[1]+1,bbox[2]+1,bbox[3]+1)
                detec_result[cls_index].append({"file_path":file_path,"file_name":file_name,"score":score,"bbox":box})
            
            # NOTE: If you want use official evaluator for mAP, uncomment below and run ./eval/get_map.py
            # with open("./eval/input/detection-results/"+file_name+".txt", "w+") as new_f:
            #     for bbox, score, cls_index in zip(bboxes, scores, cls_indexes):
            #         cls_name = VOC_CLASSES[cls_index] 
            #         new_f.write("%s %s %s %s %s %s\n" % (cls_name,score,str(int(bbox[0]+1)), str(int(bbox[1]+1)), str(int(bbox[2]+1)), str(int(bbox[3]+1))))
            # NOTE: If you want use official evaluator for mAP, uncomment the above and run ./eval/get_map.py
            # pbar.update(1)
    # 按score对结果排序
    sort_by_score(detec_result)
    # np.save("./eval/2012_val_all_results",detec_result)
    print("===> Calculate")
    mAP = calculate_ap(ground_truth, detec_result, gt_counter_per_class)
    FPS = num_images / detec_time
    print("gpu_time=",gpu_time)
    return mAP, FPS, detec_time



if __name__ == '__main__':

    num_classes = len(VOC_CLASSES)
    if  torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        device = "cuda"
    else:
        device = "cpu"

    voc = voc_512
    model = Detector(device, input_size=voc['input_size'], num_cls=20, strides = voc['strides'], 
                     cfg=REPA0_RDB_VOC, boost=True)
    print('Let us test MSKPD-RDBCA on the VOC0712 dataset ......')

    # load checkpoint
    checkpoint = torch.load('./weights/REPA0_RDB_VOC.pth',map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    # NOTE: uncomment below if optimizer parameters in checkpoint
    # model.load_state_dict(checkpoint["model"],strict=False)
    model = model.to(device)
    model = repvgg_model_convert(model)
    model.eval()
    print(get_parameter_number(model))
    print('Finished loading model!')
    # load data
    input = torch.randn(8, 3, 512, 512).to(device)
    output = model(input)
    print("warm up over")
    dataset = VOCDataset(root=voc["root"], 
                         img_size=voc["input_size"][0],
                         image_sets=[('2007', 'test')],
                         transform=BaseTransform(model.input_size,MEANS),
                         is_test=True)
 
    batch_size = 1

    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=detection_collate)
    
    
    # evaluation
    tic = time.time()
    mAP, FPS, detec_time = eval_model(model, device, dataloader, len(dataset))
    total_time = time.time() - tic
    print("[mAP:%.3f%%][FPS:%.3f][detec_time:%.2f][total_time:%.2f]" % (mAP*100, FPS, detec_time, total_time))

