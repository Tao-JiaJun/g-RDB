"""
# Description: 
# Author: Taojj
# Date: 2020-08-07 17:54:28
# LastEditTime: 2020-10-03 00:29:17
# FilePath: /FCOSLite/utils/__init__.py
"""

import torch
import torch.cuda as cuda
import time
import datetime
def get_device():
    """
    # Description: 有gpu则在gpu上运行
    # Author: Taojj
    """
    if cuda.is_available():
        print('==> use cuda')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        print("==> use cpu")
        device = torch.device("cpu")
    return device

def run_time(func):
    """
    Calculate the running time of the function
    :param func: the function need to been calculated
    :return:
    """
    def call_fun(*args, **kwargs):
        start_time = time.time()
        f = func(*args, **kwargs)
        end_time = time.time()
        print('%s() run time：%s s' % (func.__name__, int(end_time - start_time)))
        return f
    return call_fun
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}