import os.path

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))
MEANS = (104, 117, 123)

voc_384 = {
    'root': '../datasets/VOCdevkit/',
    'num_cls': 20,
    'input_size': [384, 384],
    'strides': [8, 16],
    'max_epoch': 250,
    'name': 'VOC',
}
voc_512 = {
    'root': '../datasets/VOCdevkit/',
    'num_cls': 20,
    'input_size': [512, 512],
    'strides': [8, 16],
    'lr_epoch': (90, 120),
    'max_epoch': 250,
    'name': 'VOC',
}

coco_512 = {
    'root': '../datasets/COCO/',
    'num_cls': 80,
    'input_size': [512, 512],
    'strides': [8, 16,],
    'lr_epoch': (70, 140),
    'max_epoch': 110,
    'name': 'COCO',
}