# from .voc import VOCDataset, VOCAnnotationTransform, VOC_CLASSES
from .config import *
import torch
import cv2
import numpy as np

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    image_names = []
    targets = []
    imgs = []
    for sample in batch:
        image_names.append(sample[0])
        imgs.append(sample[1])
        targets.append(sample[2])
    return image_names, torch.stack(imgs, 0), targets


def base_transform(image, size, mean):
    x = cv2.resize(image, (size[1], size[0])).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean=(0,0,0)):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels
