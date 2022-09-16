"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import jpeg4py as jpeg
import json
import numpy as np
import time
import random
sys.path.append("./")
from datasets.augmentations import SSDAugmentation
from datasets import BaseTransform
VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# note: if you used our download scripts, this should be right
# path_to_dir = osp.dirname(osp.abspath(__file__))
# VOC_ROOT = path_to_dir + "/      /"

VOC_ROOT = '../datasets/VOCdevkit/'

class VOCJsonAnnotationTransform(object):
    """
    json style
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, json_data, width, height):
        objects = json_data["object"]
        res = []
        for obj in objects:
            difficult = int(obj["difficult"]) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj["name"].lower().strip()
            bbox = obj["bndbox"]
            xmin = (int(bbox["xmin"])-1) / width
            xmax = (int(bbox["xmax"])-1) / width
            ymin = (int(bbox["ymin"])-1) / height
            ymax = (int(bbox["ymax"])-1) / height
          
            bndbox = [xmin, ymin, xmax, ymax]
            # class name to class index
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            # calculate square
            bndbox.append((bndbox[2]-bndbox[0])*(bndbox[3]-bndbox[1]))
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        # 面积由小到大排序
        res.sort(key=lambda x:float(x[-1]))
        # 删除面积结果
        res = [s[:-1] for s in res]
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

class VOCDataset(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, 
                 root,
                 image_sets,
                 img_size,
                 transform=None,
                 target_transform=VOCJsonAnnotationTransform(),
                 dataset_name='VOC0712',
                 mosaic = False,
                 mixup=False,
                 is_test = False):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.is_test = is_test
        self.mosaic = mosaic
        self.mixup = mixup
        self.img_size = img_size
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        self.json_data = dict()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))
            with open("./voc" + year + "_" + name + ".json", 'r') as f:
                print("load voc" + year + "_" + name + ".json")
                self.json_data.update(json.load(f))
        print("use mosaic:", mosaic)
        print("use mixup:", mixup)

    def __getitem__(self, index):
        img_id, im, gt = self.pull_item(index)

        return img_id, im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        if self.mosaic and random.random() < 0.3:
            img_id = self.ids[index]
            img_info = [img_id[0], img_id[1], self.img_size, self.img_size]
            # mosaic
            img, target, height, width = self.load_mosaic(index)
            # mixup
            if self.mixup and random.random() < 0.3:
                id2 = random.randint(0, len(self.ids)-1)
                img2, _, _, _ = self.load_mosaic(id2)
                r = np.random.beta(4.0, 4.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                #target = np.concatenate((target, target2), axis=0)
        else:
            img_id = self.ids[index]
            img, target, height, width = self.load_img_targets(img_id)
        if not self.is_test:
            # augment
            img, target = self.zero_padding(img, target, height, width)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            img_info = [img_id[0], img_id[1], self.img_size, self.img_size]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        else:
            # resize without padding
            if len(target) == 0:
                target = np.zeros([1, 5])
            else:
                target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            img_info = [img_id[0], img_id[1], width, height]
        return img_info, torch.from_numpy(img).permute(2, 0, 1), target
    def zero_padding(self,img, target, height, width):
        #zero padding
        if height > width:
            img_ = np.zeros([height, height, 3])
            img_[:, :, 0] = np.mean(img[:,:,0])
            img_[:, :, 1] = np.mean(img[:,:,1])
            img_[:, :, 2] = np.mean(img[:,:,2])
            delta_w = height - width
            left = delta_w // 2
            img_[:, left:left+width, :] = img
            offset = np.array([[ left / height, 0.,  left / height, 0.]])
            scale =  np.array([[width / height, 1., width / height, 1.]])
            width = height
        elif height < width:
            img_ = np.zeros([width, width, 3])
            img_[:, :, 0] = np.mean(img[:,:,0])
            img_[:, :, 1] = np.mean(img[:,:,1])
            img_[:, :, 2] = np.mean(img[:,:,2])
            delta_h = width - height
            top = delta_h // 2
            img_[top:top+height, :, :] = img
            offset = np.array([[0.,    top / width, 0.,    top / width]])
            scale =  np.array([[1., height / width, 1., height / width]])
            height = width
        else:
            img_ = img
            scale =  np.array([[1., 1., 1., 1.]])
            offset = np.zeros([1, 4])
        if len(target) == 0:
            target = np.zeros([1, 5])
        else:
            target = np.array(target)
            target[:, :4] = target[:, :4] * scale + offset
        return img_, target
    def load_mosaic(self, index):
        ids_list_ = self.ids[:index] + self.ids[index+1:]
        # random sample other indexs
        id1 = self.ids[index]
        id2, id3, id4 = random.sample(ids_list_, 3)
        ids = [id1, id2, id3, id4]

        img_lists = []
        tg_lists = []
        # load image and target
        for id_ in ids:
            img_i, target_i, _, _ = self.load_img_targets(id_)
            img_lists.append(img_i)
            tg_lists.append(target_i)

        mosaic_img = np.zeros([self.img_size*2, self.img_size*2, img_i.shape[2]], dtype=np.uint8)
        # mosaic center
        yc, xc = [int(random.uniform(-x, 2*self.img_size + x)) for x in [-self.img_size // 2, -self.img_size // 2]]
        # yc = xc = self.img_size

        mosaic_tg = []
        for i in range(4):
            img_i, target_i = img_lists[i], tg_lists[i]
            h0, w0, _ = img_i.shape

            # resize
            r = self.img_size / max(h0, w0)
            if r != 1: 
                img_i = cv2.resize(img_i, (int(w0 * r), int(h0 * r)))
            h, w, _ = img_i.shape

            # place img in img4
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.img_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.img_size * 2), min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            mosaic_img[y1a:y2a, x1a:x2a] = img_i[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            # labels
            target_i = np.array(target_i)
            target_i_ = target_i.copy()
            if len(target_i) > 0:
                # a valid target, and modify it.
                target_i_[:, 0] = (w * (target_i[:, 0]) + padw)
                target_i_[:, 1] = (h * (target_i[:, 1]) + padh)
                target_i_[:, 2] = (w * (target_i[:, 2]) + padw)
                target_i_[:, 3] = (h * (target_i[:, 3]) + padh)     
                
                mosaic_tg.append(target_i_)
        # check target
        if len(mosaic_tg) == 0:
            mosaic_tg = np.zeros([1, 5])
        else:
            mosaic_tg = np.concatenate(mosaic_tg, axis=0)
            # Cutout/Clip targets
            np.clip(mosaic_tg[:, :4], 0, 2 * self.img_size, out=mosaic_tg[:, :4])
            # normalize
            mosaic_tg[:, :4] /= (self.img_size * 2) 

        return mosaic_img, mosaic_tg, self.img_size, self.img_size
    def load_img_targets(self, img_id):
        target = self.json_data[img_id[1]]
        img = jpeg.JPEG(self._imgpath % img_id).decode()
        height, width, _ = img.shape
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        return img, target, height, width
if __name__ == "__main__":
    

    # dataset
    size = 384
    dataset = VOCDataset(VOC_ROOT, [('2007', 'test'),('2012', 'trainval')],
                            img_size=size,
                            #transform=SSDAugmentation([size,size]),
                            transform=BaseTransform([size,size]),
                            mosaic=True,
                            mixup=False,
                            is_test=False)
    #tic = time.time()
    for i in range(len(dataset)):
        img_id, im, gt = dataset.pull_item(i)
        h, w = img_id[2:]
        img = im.permute(1,2,0).numpy()[:, :, (2, 1, 0)].astype(np.uint8).copy()
        for box in gt:
            xmin, ymin, xmax, ymax, _ = box
            xmin *= w
            ymin *= h
            xmax *= w
            ymax *= h
            img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,255), 2)
        cv2.imshow('gt', img)
        cv2.waitKey(0)
    #print("time=", time.time()-tic)
