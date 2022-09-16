import collections
import math
import os
import time

import numpy as np
import torch
import torch.cuda as cuda
import torch.optim as optim
import torch.utils.data as data
from datasets.augmentations import SSDAugmentation
from datasets import BaseTransform, detection_collate
from datasets.config import coco_512, MEANS
from datasets.coco import COCODataset
from configs import *
from model.detector import Detector
from model.ground_truth_maker import GTMaker
from model.loss import Loss
from utils import get_device, get_parameter_number
from eval_coco import coco_test
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Detector')
    # optimizer
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('--step_epoch', default=88, type=int, 
                        help='initial learning rate')
    # training strategy
    parser.add_argument('--use_warm_up', action='store_true', default=False,
                        help='use warm up')
    # dataloader
    parser.add_argument('--num_workers', default=4, type=int, 
                        help='Number of workers used in dataLoader')
    # other
    parser.add_argument('--eval_epoch', type=int, default=3,
                         help='interval between evaluations')
    parser.add_argument('--iou_loss', type=str,
                            default='iou', help='select iou loss')
    # cuda
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use cuda.')
    parser.add_argument('--cuda_device', default='0', type=str, 
                        help='CUDA_VISIBLE_DEVICES')
    # batch size
    parser.add_argument('--batch_size', default=16, type=int, 
                        help='Batch size for training')
    # resume
    parser.add_argument('-r', '--resume', action='store_true', default=False, 
                        help='keep training')
    parser.add_argument('--save_weights', default='./weights/REPA0_RDB_COCO/', type=str, 
                        help='floder where save model weights')
    parser.add_argument('--model_weight', default='', type=str, 
                        help='model weight name')
    # config files
    parser.add_argument('--config_file', default=REPA0_RDB_COCO, type=dict, 
                        help='floder where save model weights')
    return parser.parse_args()


def train_model():
    args = parse_args()
    if args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
        device = get_device()

    cocoset = coco_512
    print('==> Loading the model...', flush=True)
    model = Detector(device,
                     input_size=cocoset["input_size"],
                     num_cls=cocoset["num_cls"],
                     strides=cocoset["strides"],
                     cfg=args.config_file)
    model.to(device)
    print(get_parameter_number(model))
   
    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    # each part of loss weight
    iteration = 0
    start_epoch = 0
    path_to_save = args.save_weights
    os.makedirs(path_to_save, exist_ok=True)
    if args.resume: 
        model_path = os.path.join(path_to_save, args.model_weight)
        assert os.path.isfile(model_path)
        checkpoint = torch.load(model_path)
        best_map = checkpoint['best_map']
        start_epoch = checkpoint['epoch'] + 1
        iteration = checkpoint['iteration']
        tmp_lr = checkpoint["lr"]
        model.load_state_dict(checkpoint['model'],strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Learning rate is {}.'.format(tmp_lr))
        print('Load checkpoint at epoch {}.'.format(start_epoch))
        print('Load checkpoint at iteration {}.'.format(iteration))
        print('Best map so far {}.'.format(best_map))

    print('==> Loading datasets...', flush=True)
    batch_size = args.batch_size
    train_dataset = COCODataset(root=cocoset["root"],
                                json_file='instances_train2017.json',
                                img_size=cocoset["input_size"][0],
                                transform=SSDAugmentation(cocoset["input_size"]))
    train_dataloader = data.DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       num_workers=args.num_workers,
                                       shuffle=True,
                                       collate_fn=detection_collate,
                                       pin_memory=True,
                                       drop_last=True)

    epoch_size = len(train_dataset) // batch_size
    print('The train dataset size:', len(train_dataset))
    print("--------------------Object Detection--------------------")
    print("Start to train...!")
    print("--------------------------------------------------------", flush=True)
    num_hint = 200
    loss_hint = collections.deque(maxlen=num_hint)
    cls_hint = collections.deque(maxlen=num_hint)
    reg_hint = collections.deque(maxlen=num_hint)
    nks_hint = collections.deque(maxlen=num_hint)
    epochs = cocoset["max_epoch"]
    optimizer.zero_grad()
    for epoch in range(start_epoch, epochs):
        model.train()
        set_cos_lr(epoch, args.step_epoch, cocoset["max_epoch"], optimizer, base_lr)
        tik = time.time()
        for iter_i, (_, images, gt_list) in enumerate(train_dataloader):
            if args.use_warm_up:
                set_warm_up(epoch, epoch_size, iter_i, optimizer, base_lr)
            iteration += 1
            # reset gradient
            optimizer.zero_grad()
            # inference and compute loss
            cls_loss, reg_loss, nks_loss = model(images.float().to(device), gt_list)
            total_loss = cls_loss + reg_loss + nks_loss
            # backward
            total_loss.backward()
            # update parameters of net
            optimizer.step()
            # record
            loss_hint.append(total_loss.item())
            cls_hint.append(cls_loss.item())
            reg_hint.append(reg_loss.item())
            nks_hint.append(nks_loss.item())
            # print
            if (iter_i+1) % num_hint == 0:
                tmp_lr = get_lr(optimizer)
                time_diff = time.time() - tik
                print('[Epoch: %d/%d][Iter: %d/%d][cls: %3.4f][reg: %3.4f][nks: %3.4f][lr: %.7f][time: %3.4f]' %
                      (epoch+1, epochs, iter_i+1, epoch_size,
                       np.mean(cls_hint), np.mean(reg_hint),np.mean(nks_hint), tmp_lr, time_diff), flush=True)
                tik = time.time()
        # eval
        if (epoch+1) % args.eval_epoch == 0 :
            tmp_lr = get_lr(optimizer)
            if (epoch+1) % (args.eval_epoch*1) == 0 or epoch > epochs - 10:
                model.eval()
                t_eval_s = time.time()
                print("===> Starting evaluation", flush=True)
                ap50_95, ap50, FPS = coco_test(model, device, cocoset["input_size"],batch_size=batch_size, test=False)
                eval_time_diff = time.time() - t_eval_s
                print("[Iter: %dk][AP50_95:%3.4f%%][AP50:%3.4f%%][FPS:%3.2f][total_time: %3.4f]" % (
                    iteration*0.001, ap50_95*100,ap50*100, FPS , eval_time_diff), flush=True)
                save_model(path_to_save, model, optimizer, epoch, iteration,  tmp_lr, ap50_95*100)
            else:
                save_model(path_to_save, model, optimizer, epoch, iteration,  tmp_lr, 0*100)

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def set_cos_lr(epoch, step_epoch, total_epoch, optimizer, base_lr):
    if epoch <=step_epoch:
        epochs = step_epoch
        min_lr = 1e-4
        start_epoch = 24
        end_epoch = 0
    else:
        epoch = epoch - step_epoch
        epochs = total_epoch - step_epoch
        min_lr = 5e-6
        base_lr = base_lr * 0.1
        start_epoch = 0
        end_epoch = 0

    if epoch > start_epoch and epoch <= epochs - end_epoch:
        # use cos lr
        tmp_lr = min_lr + 0.5*(base_lr-min_lr) * \
            (1+math.cos(math.pi*(epoch-start_epoch)*1. / (epochs-start_epoch-end_epoch)))
        set_lr(optimizer, tmp_lr)
    elif epoch > epochs - end_epoch:
        tmp_lr = min_lr
        set_lr(optimizer, tmp_lr)
    else:
        pass


def set_warm_up(epoch, epoch_size, iter_i, optimizer, base_lr):
    # WarmUp strategy for learning rate
    epoch_limit = 1
    if epoch < epoch_limit:
        tmp_lr = base_lr * pow((iter_i+epoch*epoch_size)
                                * 1. / (epoch_limit*epoch_size), 4)
        set_lr(optimizer, tmp_lr)
    elif epoch == epoch_limit and iter_i == 0:
        tmp_lr = base_lr
        set_lr(optimizer, tmp_lr)

def get_tensorboard_writer():
    print('==> use tensorboard')
    c_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    log_path = 'log/' + c_time
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    return writer


def save_model(path_to_save, model, optimizer, epoch, iteration, lr, mAP=None):
    checkpoint = {
        'best_map': 0 if mAP == None else mAP,
        'epoch': epoch,
        'iteration': iteration,
        'model': model.state_dict(),
        "lr": lr,
        'optimizer': optimizer.state_dict(),
    }
    if mAP == None:
        path = os.path.join(path_to_save, 'checkpoint_{:d}_{:.6f}.pth.tar'.format(
            epoch, lr))
    else:
        path = os.path.join(path_to_save, 'checkpoint_{:d}_{:.2f}.pth.tar'.format(
            epoch, mAP))
    torch.save(checkpoint, path)
    print("===> Saving model", flush=True)



if __name__ == "__main__":
    train_model()
