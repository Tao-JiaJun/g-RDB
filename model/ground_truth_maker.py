import numpy as np
import torch
import time
import cv2
import math
class GTMaker(object):
    def __init__(self, input_size, num_cls, strides):
        """
        # Author: 
        #   Taojj
        # Args: 
        #   input_size  : int 
        #   num_cls     : int  
        #   strides     : list 
        """
        self.num_cls = num_cls
        self.strides = strides
        self.w = input_size[0]
        self.h = input_size[1]
        self.feature_map_points = [(self.w // s) * (self.h // s) for s in self.strides]
        # Calculate the space needed for flattening
        self.total = sum(self.feature_map_points)
        # offsets
        self.regular_index = [0]
        for i,_ in enumerate(strides):
            self.regular_index.append(sum(self.feature_map_points[j] for j in range(i+1)))
        """
        # num_cls     = 20
        # l, t, r, b  =  4
        # nks         =  1
        # +           = 25
        """
        self.gt_num = num_cls + 4 + 1
        # Calculate how many detection points are in each scale
        self.w_detec_point = [self.get_step_pixel(self.w, s) for s in self.strides]
        self.h_detec_point = [self.get_step_pixel(self.h, s) for s in self.strides]
    def __call__(self, gt_list):
        forsake = False
        batch_size = len(gt_list)
        gt_np = np.zeros([batch_size, self.total, self.gt_num])
        # handle the data in each batch
        for batch_i, gts in enumerate(gt_list):
            # handle the bbox and label of each image
            for gt in gts:
                for i in range(0, len(self.strides)):
                    ret = self.label_assignment(gt_np, batch_i, gt, scale_i=i)
                    if ret == "labeled":
                        break
                    elif ret == "forsake":
                        forsake = True
                        continue
                    else:
                        break
        return gt_np
        
    def label_assignment(self, gt_np,batch_i, gt, scale_i):
        occupied = False
        box   = gt[:4]
        label = gt[-1]
        # Restore original size
        box = (np.array(box) * np.array([self.w,self.h,self.w,self.h])).tolist()
        # Check for dirty data
        if self.is_dirty_data(box):
            return "dirty"
        # TODO 1 By default fall into the shallow layer first
        num_w = len(self.h_detec_point[scale_i])
        # TODO 2 Calculate the center point
        ci = int(((box[2]+box[0]) / 2) // self.strides[scale_i])
        cj = int(((box[3]+box[1]) / 2) // self.strides[scale_i])
        # TODO 3 Determine whether the point is repeated
        # get offset Number of rows * row + column
        offset = int(num_w * cj + ci)
        # start offset
        start_index = self.regular_index[scale_i]
        # end offset
        end_index = self.regular_index[scale_i+1]
        index = start_index + offset
        if gt_np[batch_i, index, -1] == 1.0:
            occupied = True
            #print("It's occupied!")
        # TODO 4 count l, t, r, b
        # get l, t, r, b
        x = self.w_detec_point[scale_i][ci]
        y = self.h_detec_point[scale_i][cj]
        l, t, r, b = self.get_ltrb(x, y, box)
        if occupied:
            # TODO 3.1 Calculate iou from l, t, r, b
            iou_heatmap = self.get_iou(self.w_detec_point[scale_i], l, t, r, b, box)
            # TODO 3.2 heatmap from large to small
            iou_sorted_offset = np.argsort(-iou_heatmap)
            # TODO 3.3 Judging whether there is a conflict in turn, if iou<0.7 still cannot find a suitable point, put it in the deep layer
            new_offset = -1
            for bak_offset in iou_sorted_offset[1:]:
                bak_index = start_index + bak_offset
                if gt_np[batch_i, bak_index, -1] == 1.0:
                    #print("Stiil occupied!")
                    continue
                else:
                    if iou_heatmap[bak_offset] >= 0.7:
                        new_offset = bak_offset
                        #print("founded")
                        break
                    else:
                        if scale_i == 0:
                            #print("founded, but iou<0.7, try another map")
                            pass
                        else:
                            #print("sorry, forsake!")
                            pass
                        break
            if new_offset >= 0:
                # TODO 3.4 count x and y according to index
                ci,cj,x,y = self.get_point_pixel_by_offset(new_offset, scale_i)
                l, t, r, b = self.get_ltrb(x, y, box)
                offset = int(num_w * cj + ci)
                index = start_index + offset
            else:
                return "forsake"
        # TODO 4 Calculate Gaussian Radius
        box_w_s = (box[2] - box[0]) / self.strides[scale_i]
        box_h_s = (box[3] - box[1]) / self.strides[scale_i] 
        gaussian_r = max(0,self.gaussian_radius([box_w_s, box_h_s]))
        diameter = 2*gaussian_r + 1
        # TODO 5 Label positive samples
        gt_np[batch_i, index, int(label)] = 1.0
        gt_np[batch_i, index, self.num_cls:self.num_cls + 4] = np.array([l, t, r, b])
        gt_np[batch_i, index, -1] = 1.0
        # Generate heat map
        grid_x_mat, grid_y_mat = np.meshgrid(np.arange(num_w), np.arange(num_w))
        heatmap = np.exp(- (grid_x_mat - ci) ** 2 / (2*(diameter/3)**2) - (grid_y_mat - cj)**2 / (2*(diameter/3)**2)).reshape(-1)
        # Get previous records
        pre_v = gt_np[batch_i, start_index : end_index, int(label)]
        # Save the higher value of both
        gt_np[batch_i, start_index : end_index, int(label)] = np.maximum(heatmap, pre_v)
        # iou
        iou_heatmap = self.get_iou(self.w_detec_point[scale_i], l, t, r, b, box)
        pre_iou = gt_np[batch_i, start_index : end_index, -1]
        gt_np[batch_i, start_index : end_index, -1] = np.maximum(iou_heatmap, pre_iou)
        return "labeled"
    @staticmethod
    def gaussian_radius(det_size, min_overlap=0.5):
        height, width = det_size

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2 #
        #r1 = (b1 - sq1) / (2 * a1)

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2
        #r2 = (b2 - sq2) / (2 * a2)

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        # r3 = (b3 + sq3) / 2
        r3 = (b3 + sq3) / (2 * a3)
        return min(r1, r2, r3)

    def get_ltrb(self, x, y, box):
        xmin, ymin, xmax, ymax = box[:]
        l = max(1e-14, x - xmin)
        r = max(1e-14, xmax - x)
        t = max(1e-14, y - ymin)
        b = max(1e-14, ymax - y)
        return l, t, r, b
    def get_point_pixel_by_offset(self, offset, scale):
        # FIXME It is not considered here that the feature map is a rectangle
        i = offset %  len(self.w_detec_point[scale])
        j = offset // len(self.h_detec_point[scale])
        x = self.w_detec_point[scale][i]
        y = self.h_detec_point[scale][j]
        return i,j,x,y

    def get_a0_and_n(self,scale):
        a0 = math.pow(2, 2+scale)
        n = math.pow(2, 3+scale)
        return a0, n

    def get_centerness(self,grid_x_pixel, grid_y_pixel,scale,num_w,box):
        xmin, ymin, xmax, ymax = box[:]
        a0, n = self.get_a0_and_n(scale)
        grid_x_pixel = a0 + n * grid_x_pixel
        grid_y_pixel = a0 + n * grid_y_pixel
        l = np.maximum(grid_x_pixel - xmin, 1e-14)
        r = np.maximum(xmax - grid_x_pixel, 1e-14)
        t = np.maximum(grid_y_pixel - ymin, 1e-14)
        b = np.maximum(ymax - grid_y_pixel, 1e-14)
        centerness = np.sqrt((np.minimum(l,r) / (np.maximum(l,r)+1e-14)) * (np.minimum(t,b)/(np.maximum(l,b)+1e-14)))
        # nan to 0
        centerness = np.nan_to_num(centerness).reshape(-1)
        return centerness

    def get_iou(self, w_detec_point, l, t, r, b, box):
        num = len(w_detec_point)
        pixel_location = np.zeros([num*num, 4])

        # all points of a heatmap, including backgrounds
        detec_point = w_detec_point.copy()
        for idx, i in enumerate(w_detec_point):
            a  = np.array([w_detec_point,[i]*num,w_detec_point,[i]*num]).T
            pixel_location[idx*num:(idx+1)*num] = a

        # l,t,r,b is the location of positive sample
        pixel_location = pixel_location + np.array([l, t, r, b]) * [-1,-1,1,1]
        box = np.array(box).reshape(1,4)
        box = np.repeat(box,num*num,axis=0)
        iou_heatmap = self.count_iou(pixel_location,box)
        return iou_heatmap  

    @staticmethod    
    def count_iou(rec_a, rec_b):
        xA = np.maximum(rec_a[...,0], rec_b[...,0])
        yA = np.maximum(rec_a[...,1], rec_b[...,1])
        xB = np.minimum(rec_a[...,2], rec_b[...,2])
        yB = np.minimum(rec_a[...,3], rec_b[...,3])
       
        inter_area = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
   
        recA_area = (rec_a[...,2] - rec_a[...,0] + 1) * (rec_a[...,3] - rec_a[...,1] + 1)
        recB_area = (rec_b[...,2] - rec_b[...,0] + 1) * (rec_b[...,3] - rec_b[...,1] + 1)

        uion = np.array(recA_area + recB_area - inter_area).astype(np.float32)
        iou = np.true_divide(inter_area.astype(np.float32), uion)
        return iou
    def get_step_pixel(self, pixel, s):
        steps = pixel // s
        x = [i * s + s // 2 for i in range(steps)]
        return x

    @staticmethod
    def is_dirty_data(box):
        xmin, ymin, xmax, ymax = box[:]
        if xmax - xmin < 1e-10 or ymax - ymin < 1e-10:
            return True
        return False

def vis_heatmap(targets, size):
    # vis heatmap
    HW = targets.shape[0]
    h = int(np.sqrt(HW))
    heatmap = np.zeros([size,size])
    for c in range(-1,0):
        tmp_map = targets[:, c].reshape(h, h)
        #print(np.sum(tmp_map>=1e-1))
        if sum(sum(tmp_map)) == 0:
            continue
        tmp_map = cv2.resize(tmp_map,(size, size))
        heatmap += tmp_map
    return heatmap
if __name__ == "__main__":
    import sys
    sys.path.append("./")
    from datasets.augmentations import SSDAugmentation
    from datasets.voc import VOCDataset
    
    def base_transform(image, size, mean):
        x = cv2.resize(image, (size[1], size[0])).astype(np.float32)
        x -= mean
        x = x.astype(np.float32)
        return x

    class BaseTransform:
        def __init__(self, size, mean):
            self.size = size
            self.mean = np.array(mean, dtype=np.float32)

        def __call__(self, image, boxes=None, labels=None):
            return base_transform(image, self.size, self.mean), boxes, labels
    strides = [8,16]
    size = 640
    gt_maker = GTMaker([size,size], 20, strides)
    # dataset
    
    VOC_ROOT = '../datasets/VOCdevkit/'
    dataset = VOCDataset(VOC_ROOT, [('2007', 'trainval')],
                            img_size = size,
                            mosaic=True,
                            mixup=True,
                            #transform=SSDAugmentation([size,size]))
                            transform=BaseTransform([size,size],(0,0,0)))
    class_color = [(77,63,238),(230,235,248)]
    count = 0
    for i in range(len(dataset)):
        img_id, im, gt = dataset.pull_item(i)
        #print(img_id)
        res = gt_maker([gt])
        img = im.permute(1,2,0).numpy()[:, :, (2, 1, 0)].astype(np.uint8).copy()
        for box in gt:
            xmin, ymin, xmax, ymax, _ = box
            xmin *= size
            ymin *= size
            xmax *= size
            ymax *= size
       
            img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (26,106,64), 2)

            
        points_list = []
        # for resolution of 384
        # fm1 = res[0][0:2304]
        # fm2 = res[0][2304:2880]

        # for resolution of 640
        fm1 = res[0][0:6400]
        fm2 = res[0][6400:8000]
        fms = [fm1,fm2]
        heatmap = np.zeros([size,size])
        for scale_i, fm in enumerate(fms):
            hmap = vis_heatmap(fm, size)
            heatmap += hmap
            for offset, point in enumerate(fm):
                if point[-1] == 1:
                    i = offset %  len(gt_maker.w_detec_point[scale_i])
                    j = offset // len(gt_maker.h_detec_point[scale_i])
                    x = gt_maker.w_detec_point[scale_i][i]
                    y = gt_maker.h_detec_point[scale_i][j]
                    l,t,r,b = point[-5:-1]
                    print(l,t,r,b)
                    img = cv2.rectangle(img, (int(x-l), int(y-t)), (int(x+r), int(y+b)), class_color[scale_i], 2)
                    points_list.append([(x,y),class_color[scale_i]])
        for point in points_list:
            cv2.circle(img, point[0], radius=2,color=point[1], thickness=2)
        heatmap[heatmap>=1.0] = 1.0
        # convert heat map to RGB format
        heatmap = np.uint8(255 * heatmap)  
        # apply the heat map to the original image
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        img_add = cv2.addWeighted(heatmap, 0.5, img, 1.0, 1)
        # if forsake:
        print(img_id)
        cv2.imshow('gt', img_add)
        cv2.waitKey(0)
    print(count)