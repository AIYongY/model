# coding: utf-8

from __future__ import division, print_function

import numpy as np
import tensorflow as tf
import cv2


def parse_line(line):
    '''
    输入的 line 是    一个一个的    输入   图片名字  +  标签 +  框
    返回的是  图片的路径  边框的四个坐标  标签值    没有映射到和特这个图一样的形状  没有置信度
    return pic_path, boxes, labels
    Given a line from the training/test txt file, return parsed
    pic_path, boxes info, and label info.
    return:
        pic_path: string.
        boxes: shape [N, 4], N is the ground truth count, elements in the second
            dimension are [x_min, y_min, x_max, y_max]

             这里的 line 是    一个一个的    输入   图片名字  +  标签 +  框
    '''
    s = line.strip().split(' ')
    pic_path = s[0]  #取得图片文件名
    s = s[1:]        #取得标签
    box_cnt = len(s) // 5
    #一个框5个数   不包含置信度 label, x_min, y_min, x_max, y_max
    #box_cnt表示有几个框的置信度

    boxes = []
    labels = []
    for i in range(box_cnt):
        label, x_min, y_min, x_max, y_max = int(s[i*5]), float(s[i*5+1]), float(s[i*5+2]), float(s[i*5+3]), float(s[i*5+4])
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(label)
    boxes = np.asarray(boxes, np.float32)
    labels = np.asarray(labels, np.int64)
    return pic_path, boxes, labels


def resize_image_and_correct_boxes(img, boxes, img_size):
    #return img, boxes  一个 返回一个图片的内容416，416   所有在图片转换到416，416的边框
    #输入     img图片值  boxes边框坐标值   img_size=[416,416]
    #把读取过来的图象   把边框的值转换到 416，416的尺寸中

    # convert gray scale image to 3-channel fake RGB image
    if len(img) == 2:#应该是输入的数灰度图像的时候只有2维度  增加一维度
        img = np.expand_dims(img, -1)
    ori_height, ori_width = img.shape[:2]
    new_width, new_height = img_size
    # shape to (new_height, new_width)
    img = cv2.resize(img, (new_width, new_height))
    #转换到新的尺寸 416 416

    # convert to float
    img = np.asarray(img, np.float32)

    # boxes
    # xmin, xmax
    # =boxes[:, 0]*    new_width/ori_width   把原图自己打的标签值 转换到416，416
    boxes[:, 0] = boxes[:, 0] / ori_width * new_width#=boxes[:, 0]* new_width/ori_width
    boxes[:, 2] = boxes[:, 2] / ori_width * new_width
    # ymin, ymax
    boxes[:, 1] = boxes[:, 1] / ori_height * new_height
    boxes[:, 3] = boxes[:, 3] / ori_height * new_height

    return img, boxes


def data_augmentation(img, boxes, label):
    '''
    Do your own data augmentation here.
    param:
        img: a [H, W, 3] shape RGB format image, float32 dtype
        boxes: [N, 4] shape boxes coordinate info, N is the ground truth box number,
            4 elements in the second dimension are [x_min, y_min, x_max, y_max], float32 dtype
        label: [N] shape labels, int64 dtype (you should not convert to int32)
    '''
    return img, boxes, label


def process_box(boxes, labels, img_size, class_num, anchors):
    '''
    #输入416，416图片的boxes边框值  lable每个框的标签值
    # img_size=[416,416] class_num=80  anchors=  9个值

    y_true_13, y_true_26, y_true_52 = process_box(boxes, labels, img_size, class_num, anchors)


    一个个的输入  return  y_true_13, y_true_26, y_true_52

    '''
    anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]

    # convert boxes form:
    # shape: [N, 2]
    # (x_center, y_center)
    box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
    # (width, height)
    box_sizes = boxes[:, 2:4] - boxes[:, 0:2]

    # [13, 13, 3, 3+num_class]  img_size=416
    #定义标签形状
    y_true_13 = np.zeros((img_size[1] // 32, img_size[0] // 32, 3, 5 + class_num), np.float32)
    y_true_26 = np.zeros((img_size[1] // 16, img_size[0] // 16, 3, 5 + class_num), np.float32)
    y_true_52 = np.zeros((img_size[1] // 8, img_size[0] // 8, 3, 5 + class_num), np.float32)

    y_true = [y_true_13, y_true_26, y_true_52]

    # [N, 1, 2]  box_sizes是宽度值  # (width, height)
    box_sizes = np.expand_dims(box_sizes, 1)

#-------------------------------------  不知道  应该是求置信度
    # broadcast tricks [N, 1, 2]  3个值
    # [N, 1, 2] & [9, 2] ==> [N, 9, 2]
    #np.maximum  一个[1, 2] 值与  9个 [9, 2]值比较
    #就是比较哪个框在内
    #                  [N, 1, 2]        [9, 2]
    mins = np.maximum(- box_sizes / 2, - anchors / 2)#anchors=  9个值
    #就是比较边框中心  左边    哪一个边框值    比较偏右

    # 就是比较边框中心  右边    哪一个边框值    比较偏左
    maxs = np.minimum(box_sizes / 2, anchors / 2) #最小值
    # [N, 9, 2]
    whs = maxs - mins
    #whs 右边的 减去 -左边  就是 得到图像尺寸    其实就是求哪个框在里面

    #每一个y_true边框（N个）  与  9个anchors 求在里面的框  得到   [N, 9]


    # [N, 9]
    iou = (whs[:, :, 0] * whs[:, :, 1]) / (box_sizes[:, :, 0] * box_sizes[:, :, 1] + anchors[:, 0] * anchors[:, 1] - whs[:, :, 0] * whs[:, :, 1] + 1e-10)
    # 得到置信度标签   置信度 是真实的自己打标签的边框 与

    # 得到[N]   每一个值就是9个值当中最大的置信度  np.argmax 就是返回最大值的下标
    #[N]  的9  是3个尺度  的值
    best_match_idx = np.argmax(iou, axis=1)#best_match_idx是0到8的

    ratio_dict = {1.: 8., 2.: 16., 3.: 32.}

    # best_match_idx 是在 [N, 9]  每一列iou最大的索引
    for i, idx in enumerate(best_match_idx):#i   是从0开始
        # idx: 0,1,2 ==> 2;   3,4,5 ==> 1;   6,7,8 ==> 2
        #取整除 - 返回商的整数部分（向下取整）
        feature_map_group = 2 - idx // 3 #idx 0,1,2  feature_map_group =2
        #  idx是实际的标签索引得到的是表示在哪个框 ，同时也可表示在哪个维度

        # idx: 0,1,2 ==>ratio 8;   3,4,5 ==> 16;   6,7,8 ==> 32
        ratio = ratio_dict[np.ceil((idx + 1) / 3.)]

        x = int(np.floor(box_centers[i, 0] / ratio))#把中心坐标转换到其他的尺度
        y = int(np.floor(box_centers[i, 1] / ratio))


        # anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        # idx: 0,1,2 ==> feature_map_group2;   3,4,5 ==> 1;   6,7,8 ==> 2
        #idxs是具体里面的值anchors_mask， 返回的值的下标
        k = anchors_mask[feature_map_group].index(idx)
        #k=0,1,2  [3, 4, 5]  [6, 7, 8] 的其中一个  一次只能是一个

        c = labels[i]#labels是 0，1，2，3，4的类别表示哪个类别
        # print feature_map_group, '|', y,x,k,c

        #feature_map_group = 2,1,0
        #只是 在0矩阵里面 进行  填充
        #feature_map_group  是表示哪个特征图   y,x是转化尺度后的中心

        y_true[feature_map_group][y, x, k, :2] = box_centers[i]
        y_true[feature_map_group][y, x, k, 2:4] = box_sizes[i]
        y_true[feature_map_group][y, x, k, 4] = 1.#这里的1表示是否含有，待检测的物体在里面，表示背景还是前景
        y_true[feature_map_group][y, x, k, 5+c] = 1.
        # k=0,1,2  [3, 4, 5]  [6, 7, 8] 的其中一个  一次只能是一个
        # box_centers[i]  box_sizes[i] 都是在原图上的中心 和框的尺寸宽度值
        #自己打的标签 与框 ，通过与   9个锚点   进行计算IOU  筛选最大的填入对应的值
    return y_true_13, y_true_26, y_true_52


def parse_data(line, class_num, img_size, anchors, mode):
    '''
    这里的 line 是    一个一个的    输入   图片名字  +  标签
    #class_num = 80
    [x, args.class_num,args.img_size, args.anchors, 'val']   val 和 train
            args.img_size = [416，416]
            生成一个dataset，dataset中的每一个元素就对应了文件中的一行
            anchors  text文本  9个值
    param:
        line: a line from the training/test txt file
        args: args returned from the main program
        mode: 'train' or 'val'. When set to 'train', data_augmentation will be applied.
    '''
    # 输入的 line 是  一个一个的
    # 输入 图片名字 + 标签 + 框
    # 返回的是     图片的路径    边框的四个坐标  标签值  没有映射到和特这个图一样的形状    没有置信度
    pic_path, boxes, labels = parse_line(line)
    #返回的是  图片的路径   边框的四个坐标  标签值   没有置信度
    #line包含了这些信息  只是分开了

    img = cv2.imread(pic_path)#picture path
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #一个图片 矩形框   [416,416]

    img, boxes = resize_image_and_correct_boxes(img, boxes, img_size)
    # return img, boxes  一个 返回一个图片的内容416，416   所有在图片转换到416，416的边框
    # 把读取过来的图象 转换为  416，416  把边框的值转换到 416，416的尺寸中
    # 传进来是一张图片
    # 一个图片 矩形框   [416,416]
    # convert gray scale image to 3-channel fake RGB image

    # do data augmentation here
    if mode == 'train':
        img, boxes, labels = data_augmentation(img, boxes, labels)
        #这个函数是数据增加  需要自己加  里面是空的 当作没有

    # the input of yolo_v3 should be in range 0~1
    img = img / 255.
    #(boxes, labels, img_size, class_num, anchors)
    #输入416，416图片的boxes边框值  lable每个框的标签值
    # img_size=[416,416] class_num=80  anchors=  9个值
    y_true_13, y_true_26, y_true_52 = process_box(boxes, labels, img_size, class_num, anchors)

    return img, y_true_13, y_true_26, y_true_52
