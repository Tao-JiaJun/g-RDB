import json
import tempfile
import torch
import numpy as np
from pycocotools.cocoeval import COCOeval
from torch.autograd import Variable

from datasets import detection_collate
from datasets.coco import COCODataset
from pycocotools.coco import COCO
from model.detector import Detector
from datasets.config import coco_512 as coco_cf
from datasets import *
from configs import *
from utils import get_device, get_parameter_number
import time
from model.backbone.repvgg import repvgg_model_convert
class COCOAPIEvaluator():
    """
    COCO AP Evaluation class.
    All the data in the val2017 dataset are processed \
    and evaluated by COCO API.
    """
    def __init__(self, root, img_size,batch_size, device, testset=False, transform=None):
        """
        Args:
            root (str): dataset root directory
            img_size (int): image size after preprocess. images are resized \
                to squares whose shape is (img_size, img_size).
            confthre (float):
                confidence threshold ranging from 0 to 1, \
                which is defined in the config file.
            nmsthre (float):
                IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.testset = testset
        if self.testset:
            json_file='image_info_test-dev2017.json'
            name = 'test2017'
        else:
            json_file='instances_val2017.json'
            name='val2017'

        self.dataset = COCODataset(
                                   root=root,
                                   img_size=img_size,
                                   json_file=json_file,
                                   transform=None,
                                   name=name,
                                   is_test=True)
        self.dataloader = torch.utils.data.DataLoader(
                                    self.dataset, 
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    collate_fn=detection_collate,
                                    num_workers=0)
        self.img_size = img_size
        self.transform = transform
        self.device = device

    def evaluate(self, model):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        Args:
            model : model object
        Returns:
            ap50_95 (float) : calculated COCO AP for IoU=50:95
            ap50 (float) : calculated COCO AP for IoU=50
        """
        model.eval()
        ids = []
        data_dict = []
        num_images = len(self.dataset)
        print('total number of images: %d' % (num_images))
        detec_time = 0
        FPS = 0
        # start testing
        for index in range(num_images): # all the data in val2017
            if index % 500 == 0:
                print('[Eval: %d / %d]'%(index, num_images))

            img, id_ = self.dataset.pull_image(index)  # load a batch
            if self.transform is not None:
                x = torch.from_numpy(self.transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
                x = x.unsqueeze(0).to(self.device)
            scale = np.array([[img.shape[1], img.shape[0],
                            img.shape[1], img.shape[0]]])
            
            id_ = int(id_)
            ids.append(id_)
            
            with torch.no_grad():
                tic = time.time()
                bbox_list, score_list, cls_list = model(x)
                detec_time += time.time() - tic
                for j, (bboxes, scores, cls_indexes) in enumerate(zip(bbox_list, score_list, cls_list)):
                    bboxes *= scale
                for i, box in enumerate(bboxes):
                    x1 = float(box[0])
                    y1 = float(box[1])
                    x2 = float(box[2])
                    y2 = float(box[3])
                    label = self.dataset.class_ids[int(cls_indexes[i])]
                    
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    score = float(scores[i]) # object score * class score
                    A = {"image_id": id_, "category_id": label, "bbox": bbox,
                        "score": score} # COCO json format
                    data_dict.append(A)
        FPS = num_images / detec_time
        print("FPS=",FPS )
        annType = ['segm', 'bbox', 'keypoints']

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            print('evaluating ......')
            cocoGt = self.dataset.coco
            json.dump(data_dict, open('zzz.json', 'w'))
            # workaround: temporarily write data to json file because pycocotools can't process dict in py36.
            if self.testset:
                json.dump(data_dict, open('zzz.json', 'w'))
                cocoDt = cocoGt.loadRes('zzz.json')
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, 'w'))
                cocoDt = cocoGt.loadRes(tmp)
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.params.imgIds = ids
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            ap50_95, ap50 = cocoEval.stats[0], cocoEval.stats[1]
            print('ap50_95 : ', ap50_95)
            print('ap50 : ', ap50)
            return ap50_95, ap50, FPS
        else:
            return -1, -1, -1


def coco_test(model, device, input_size, batch_size, test=False):
    if test:
        # test-dev
        print('test on test-dev 2017')
        evaluator = COCOAPIEvaluator(
                        root=coco_cf["root"],
                        img_size=coco_cf["input_size"][0],
                        batch_size = batch_size,
                        device=device,
                        testset=True,
                        transform=BaseTransform(coco_cf["input_size"],MEANS)
                        )

    else:
        # eval
        evaluator = COCOAPIEvaluator(
                        root=coco_cf["root"],
                        img_size=coco_cf["input_size"][0],
                        batch_size = batch_size,
                        device=device,
                        testset=False,
                        transform=BaseTransform(coco_cf["input_size"], MEANS)
                        )

    # COCO evaluation
    return evaluator.evaluate(model)

if __name__ == '__main__':
    # dataset
    device = get_device()
    # input size
    input_size = coco_cf["input_size"]
    strides = [8, 16]
    scales  = [0, 48, 96, 192, 1e10]

    model = Detector(device, input_size=input_size, num_cls=coco_cf["num_cls"], strides = strides, cfg=REPA0_RDB_COCO, boost=True)
    # load net
    checkpoint = torch.load('./weights/REPA0_RDB_COCO.pth',map_location=device)
    model.load_state_dict(checkpoint,strict=False)
    # NOTE: uncomment below if optimizer parameters in checkpoint
    # model.load_state_dict(checkpoint["model"],strict=False)
    model = model.to(device)
    model = repvgg_model_convert(model)
    print(get_parameter_number(model))
    model.eval()
    batch_size = 1
    print('Finished loading model!')
    
    # evaluation
    coco_test(model, device, input_size, batch_size, test=False) # coco-val
    #coco_test(model, device, input_size,batch_size, test=True) # coco-test
