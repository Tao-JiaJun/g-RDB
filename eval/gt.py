"""
# Description: 
# Author: Taojj
# Date: 2020-10-22 21:34:00
# LastEditTime: 2020-12-11 22:29:31
# FilePath: /FCOSLite/gt.py
"""
import sys
import os
import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm

image_ids = open('../datasets/VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()

if not os.path.exists("./eval/input"):
    os.makedirs("./eval/input")
if not os.path.exists("./eval/input/ground-truth"):
    os.makedirs("./eval/input/ground-truth")

for image_id in tqdm(image_ids):
    with open("./eval/input/ground-truth/"+image_id+".txt", "w") as new_f:
        root = ET.parse("../datasets/VOCdevkit/VOC2007/Annotations/"+image_id+".xml").getroot()
        for obj in root.findall('object'):
            difficult_flag = False
            if obj.find('difficult')!=None:
                difficult = obj.find('difficult').text
                if int(difficult)==1:
                    difficult_flag = True
            obj_name = obj.find('name').text
            bndbox = obj.find('bndbox')
            left = bndbox.find('xmin').text
            top = bndbox.find('ymin').text
            right = bndbox.find('xmax').text
            bottom = bndbox.find('ymax').text
            if difficult_flag:
                new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
            else:
                new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
            
print("Conversion completed!")