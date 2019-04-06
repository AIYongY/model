#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from config import cfg
from test import im_detect
from nms_wrapper import nms

from timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from vgg16 import vgg16
from resnet_v1 import resnetv1

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes.

      #所有的框预测  21个类别  输入的是  所有的框预测一个类别的框与得分值
        # 进行NMS后 得到的框就是需要输出的框   他对应的英文标签是  cls
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
        """
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))#创建一个 Figure画图背景  默认是画一个图    尺寸是12cm*12cm
    #画图背景对象    ax是要画图的区域
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')#关闭轴显示
    plt.tight_layout()
    #tight_layout会自动调整子图参数，使之填充整个图像区域。
    # 这是个实验特性，可能在一些情况下不工作。它仅仅检查坐标轴标签、刻度标签以及标题的部分。
    plt.draw()

def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    # im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
    #             '001763.jpg', '004545.jpg']
    #一张张图片输入
    # net = vgg16()
    # demo(sess, net, im_name)

    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    #输入 net = vgg16()   im一张图片
    scores, boxes = im_detect(sess, net, im)

    # post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N#__C.TEST.RPN_POST_NMS_TOP_N = 300    非极大值抑制输出的 最大个数
    # nms_thresh = cfg[cfg_key].RPN_NMS_THRESH#__C.TEST.RPN_NMS_THRESH = 0.7

    # scores 是rpn scores = self._predictions['cls_prob']  =  每个类别的概率cls_score 讲过soft_max得到
    # pred_boxes
    # pred_boxes = bbox_transform_inv(boxes, box_deltas)
    # 做回归预测   两条路劲rpn得到的的 box_deltas 作为 dx dy dw dh 与 筛选出来的框做回归预测
    # pred_boxes  anchors回归预测后的值
    # return scores, pred_boxes



    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):#CLASSES就是英文标签
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        #得到抽取出来的 一个框  一个 boxes shape (?, 21*4)  得到 [?*21,4]
        cls_scores = scores[:, cls_ind]#scores    shape  (？，21)  #  得到[?*21]
        dets = np.hstack((cls_boxes,# [?*21,4]
                          cls_scores[:, np.newaxis])).astype(np.float32)#[?*21,1]


        #得到的是[?*21,x,y,w,h,scores]  0.3
        #所有的框预测 21个类别  输入的是  所有的框预测一个类别的框与得分值
        # 是一个一个类别输入NMS  假如类别1    则输入 [?,x,y,w,h,scores]
        keep = nms(dets, NMS_THRESH)#为什么要用   (cpu)gpu_nms.pys
        #nms 纯python语言实现：简介方便、速度慢
        #Cython是一个快速生成Python扩展模块的工具，从语法层面上来讲是Python语法和C语言语法的混血，
        # 当Python性能遇到瓶颈时，Cython直接将C的原生速度植入Python程序，这样使Python程序无需使用C重写
        # ，能快速整合原有的Python程序，这样使得开发效率和执行效率都有很大的提高，而这些中间的部分，
        # 都是Cython帮我们做了。

        #https://www.cnblogs.com/king-lps/p/9031568.html  解释
        #  之前的 非极大值抑制  作用实在rpn路径上的   rpn路径就是为了做推荐而已 推荐 VGG16特征出来的框，但是太多 所以需要筛选
        #这里 是筛选出来之后  再做  nms


        #CONF_THRESH = 0.8
        #NMS_THRESH = 0.3
        dets = dets[keep, :]
        #im输入的图像经过限制在600，1000
        #cls  是实际的英文标签
        # dets是 [?*21,x,y,w,h,scores]  经过nms 得到 的    0.3
        #for      cls_ind, cls         in enumerate(CLASSES[1:]):#CLASSES就是英文标签


        #所有的框预测  21个类别  输入的是  所有的框预测一个类别的框与得分值
        # 进行NMS后 得到的框就是需要输出的框   他对应的英文标签是  cls    #CONF_THRESH = 0.8
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
        #再次进行排除 让 概率值  大于0.8彩输出  并且画图 输出

def parse_args():#parse解析  args参数
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
        #class vgg16(Network): vgg16继承了class Network(object)
        #所以 VGG16能调用   Network 函数
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", 21,
                          tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg']
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        # im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
        #             '001763.jpg', '004545.jpg']
        #net = vgg16()
        demo(sess, net, im_name)

    plt.show()
