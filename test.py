"""
# Description: 
# Author: Taojj
# Date: 2020-08-08 19:30:18
# LastEditTime: 2020-10-27 22:05:33
# FilePath: /FCOSLite/backbone/test.py
"""

import collections

import torch


def transfer_model(pretrained_file, model):
    '''
    只导入pretrained_file部分模型参数
    tensor([-0.7119,  0.0688, -1.7247, -1.7182, -1.2161, -0.7323, -2.1065, -0.5433,-1.5893, -0.5562]
    update:
        D.update([E, ]**F) -> None.  Update D from dict/iterable E and F.
        If E is present and has a .keys() method, then does:  for k in E: D[k] = E[k]
        If E is present and lacks a .keys() method, then does:  for k, v in E: D[k] = v
        In either case, this is followed by: for k in F:  D[k] = F[k]
    :param pretrained_file:
    :param model:
    :return:
    '''
    pretrained_dict = torch.load(pretrained_file)  # get pretrained dict
    model_dict = model.state_dict()  # get model dict
    # 在合并前(update),需要去除pretrained_dict一些不需要的参数
    pretrained_dict = transfer_state_dict(pretrained_dict, model_dict)
    model_dict.update(pretrained_dict)  # 更新(合并)模型的参数
    model.load_state_dict(model_dict)
    return model
 
 
def transfer_state_dict(pretrained_dict, model_dict):
    '''
    根据model_dict,去除pretrained_dict一些不需要的参数,以便迁移到新的网络
    url: https://blog.csdn.net/qq_34914551/article/details/87871134
    :param pretrained_dict:
    :param model_dict:
    :return:
    '''
    # state_dict2 = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            # state_dict.setdefault(k, v)
            state_dict[k] = v
        else:
            print("Missing key(s) in state_dict :{}".format(k))
    return state_dict

# if __name__ == "__main__":
#     model_name = 'EfficientNet-B0'
#     model = EfficientNet.from_name(model_name.lower())
#     print(model)
if __name__=='__main__':
    pretrained_file = "./weights/REPB0_RDB_VOC.tar"
    #empty_model_file = "./empty.pth"
    # model_name = 'EfficientNet-B0'
    model = torch.load(pretrained_file)["model"]
    #model = EfficientNet.from_name(model_name.lower())
    #model = transfer_model(pretrained_file, model)
    #empty_model = torch.load(empty_model_file)
  
    new_dict = collections.OrderedDict()

    for i, p in enumerate(model.items()):
            new_dict[p[0]] = p[1]


        #empty_model[p[0]] = p[1]
    # # para_optim = []
    # for i,(key ,value)  in enumerate(model.items()):
    #     name = key[:]
    #     new_dict[name] = value

    torch.save(new_dict,"./REPB0_RDB_VOC.pth")

