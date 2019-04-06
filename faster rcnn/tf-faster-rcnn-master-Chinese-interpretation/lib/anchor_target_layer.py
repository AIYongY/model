# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from config import cfg
import numpy as np
import numpy.random as npr
from cython_bbox import bbox_overlaps
from bbox_transform import bbox_transform


def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors):
  """
   #返回的是     特征图映射到原图的  所有的边框
  #把超出图像尺寸的 边框 置为-1
  #在输入的所有的边框与标签中  帅选出  小于等于256/2 个正负样本  总共 就是256 样本  正为1负为0 其他为-1

  #超出图像尺寸的边框的  label等置为-1  边框偏移量 0 边框权重0  边框权重的归一化参数0
  #    标签正样本1，负0，不关注-1  (1, 1, A * height, width)
  #    边框 偏移量  是偏移量 dx dy dw dh  是中心坐标与  边框长度的偏移量(1, height, width, A * 4)
  #    边框权重1             (1, height, width, A * 4)
  #    边框权重的归一化参数  (1, height, width, A * 4)
  return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


      rpn_cls_score  rpn 一条路径得到的  背景前景值 [:, :, :, 18]
      self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
      self._feat_stride = 16
      self._im_info = tf.placeholder(tf.float32, shape=[3])
      self._anchors   wgg特征图 对应原始坐标的所有 边框  [-1,4]
      self._num_anchors = 9
  """
  A = num_anchors  #=9
  total_anchors = all_anchors.shape[0]  #total_anchors得到 锚点的个数 N*9个
  K = total_anchors / num_anchors       #   得到N  就是得到VGG特征图有几个点

  # allow boxes to sit over the edge by a small amount
  _allowed_border = 0

  # map of shape (..., H, W)
  height, width = rpn_cls_score.shape[1:3]#rpn_cls_score [:, :, :, 18]

  # only keep anchors inside the image
  # np.where返回的是满足条件的  标索引 和类型  [0]意思是只返回索引
  inds_inside = np.where(# 所有archors边界可能超出图像，取在图像内部的archors的索引
    (all_anchors[:, 0] >= -_allowed_border) &#_allowed_border=0
    (all_anchors[:, 1] >= -_allowed_border) &
    (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
    (all_anchors[:, 3] < im_info[0] + _allowed_border))[0]  # height
  """
    i= np.array([1,1,1,2,3,4,5])
    inds_inside = np.where((i>= 2))
    print(inds_inside)      (array([3, 4, 5, 6], dtype=int64),)
  """
  # keep only inside anchors
  # 得到在图像内部archors的坐标
  anchors = all_anchors[inds_inside, :]

  # label: 1 is positive, 0 is negative, -1 is dont care
  labels = np.empty((len(inds_inside),), dtype=np.float32)
  # np.empty()
  # 返回一个随机元素的矩阵，大小按照参数定义
  labels.fill(-1)#把里面的值都变为-1

  # label: 1  正样本, 0  负样本, -1  不关注


#??????????????????????????????????????????????????????????？？？？？？？？？？？？？？？？？？？？？？？
  # overlaps between the anchors and the gt boxes
  # overlaps (ex, gt)
  # 计算每个anchors:n*4和每个真实位置   gt_boxes:m*4的重叠区域的比的矩阵:n*m
  overlaps = bbox_overlaps(
    np.ascontiguousarray(anchors, dtype=np.float),
    np.ascontiguousarray(gt_boxes, dtype=np.float))
  # ??????????????????????????????????????????????????????????？？？？？？？？？？？？？？？？？？？？？？？
  #这里不可以自己写？？？？？？？？？？？？？？？？？？

  # overlaps  n*m   是重叠区域   交并比   猜的  猜的  猜的



  #overlaps   n*m  argmax_overlaps  是下标
  argmax_overlaps = overlaps.argmax(axis=1)
  # 找到每行最大值的位置，即每个archors对应的正样本的位置，得到  [n]  1维  的行向量

  #首先 overlaps 得到预测的 预测与真实边框的置信度 是经过  inds_inside帅选所以  inds_inside的个数等于overlaps个数

  #得到每个预测的边框的 anchors  与gt_boxes   比最大的值max_overlaps
  max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]

  gt_argmax_overlaps = overlaps.argmax(axis=0)#索引
  #上面是 求得行的最大值  是一个预测与 所有真实  的最大
  # 这里求得列的最大值    是所有的预测 与  所有的真实边框的  一个一个一个  框的比值  的最大值索引

  #gt_argmax_overlaps [1,V]  遍历V     overlaps [n,m]                 m
  gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]

  #gt_argmax_overlaps 是预测的所有边框
  gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
  #

  #__C.TRAIN.RPN_CLOBBER_POSITIVES = False
  if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
    # assign bg labels first so that positive labels can clobber them
    # first set the negatives
    #label 是和 置信度 具有一样长度 值全为-1
    # labels = np.empty((len(inds_inside),), dtype=np.float32)
    # label: 1 正样本, 0 负样本, -1 不关注
    # labels.fill(-1)

    #__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
    # max_overlaps是列最大值的地方
    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    # 将archors对应的正样本的重叠区域中小于阈值的置0

  # fg label: for each gt, anchor with highest overlap
  labels[gt_argmax_overlaps] = 1
  #每个真实位置对应的archors置1

  # fg label: above threshold IOU  __C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
  labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
  #得到的是真实边框对应最适合的一个预测边框

  # __C.TRAIN.RPN_CLOBBER_POSITIVES = False
  if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
    # assign bg labels last so that negative labels can clobber positives
    #cfg.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    # 将archors对应的正样本的重叠区域中小于阈值的置0



# 限定得到的框在256/2  之内   小于256/2之内则不变
#限定得到的框在256/2  之内    小于256/2之内则不变

  # subsample positive labels if we have too many
  #__C.TRAIN.RPN_FG_FRACTION = 0.5
  #__C.TRAIN.RPN_BATCHSIZE = 256
  num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
  fg_inds = np.where(labels == 1)[0]
  if len(fg_inds) > num_fg:
    disable_inds = npr.choice(
      fg_inds, size=(len(fg_inds) - num_fg), replace=False)
    labels[disable_inds] = -1

  # subsample negative labels if we have too many
  #               256
  num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
  #得到等于1的个数 如果是大于256/2则返回256/2

  bg_inds = np.where(labels == 0)[0]
  if len(bg_inds) > num_bg:
    disable_inds = npr.choice(
      bg_inds, size=(len(bg_inds) - num_bg), replace=False)
    labels[disable_inds] = -1

# 限定得到的框在256/2  之内  小于256/2之内则不变



  # 所有archors边界可能超出图像，取在图像内部的archors的索引
  bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)

  #得到在图像内部archors的坐标 anchors = all_anchors[inds_inside, :]
  bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])
  #bbox_targets 得到移动的  dx dy dw dh

  #通过archors和archors对应的正样本计算坐标的偏移



  bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
  # only the positive ones have regression targets
  #cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS  [1.0, 1.0, 1.0, 1.0]
  bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)
  #     正样本   的四个坐标的权重均设置为1   只是正样本 其他都是0


  bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)



  if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:  #cfg.TRAIN.RPN_POSITIVE_WEIGHT = -1.0
    # uniform weighting of examples (given non-uniform sampling)
    num_examples = np.sum(labels >= 0)#正负样本的个数
    positive_weights = np.ones((1, 4)) * 1.0 / num_examples# 归一化的权重
    negative_weights = np.ones((1, 4)) * 1.0 / num_examples# 归一化的权重
  else:
    #cfg.TRAIN.RPN_POSITIVE_WEIGHT  默认是-1  如果是 在   0到1的值

    assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
            (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
    positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                        np.sum(labels == 1))
    negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                        np.sum(labels == 0))

  #这里不在  if里面了
  bbox_outside_weights[labels == 1, :] = positive_weights# 归一化的权重
  bbox_outside_weights[labels == 0, :] = negative_weights# 归一化的权重


  # map up to original set of anchors
  # total_anchors = all_anchors.shape[0]  #total_anchors得到 锚点的个数 N*9个
  # inds_inside  所有archors边界可能超出图像，取在图像内部的archors的索引
  #labels = np.empty((len(inds_inside),), dtype=np.float32)
  labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
  # 函数的作用是  在特征图映射到原图的所有框中   把超出边界的 边框  的label置为 - 1

  bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
  #得到在图像内部archors的坐标 anchors = all_anchors[inds_inside, :]
  # bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])
  #bbox_targets 得到移动的  dx dy dw dh

  #把超出边框的 dx dy dw dh 置为0

  bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
  # 所有archors中正样本的四个坐标的权重均设置为1，其他为0

  #归一化参数   把超出  边界的边框   的 归一化参数  置为0
  bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)


  # labels        A = num_anchors  #=9
  #height, width = rpn_cls_score.shape[1:3]  # rpn_cls_score [:, :, :, 18]
  # labels   这里 的label 是已经把  超出图像尺寸 与没超出的  尺寸  组合在一起
  labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
  labels = labels.reshape((1, 1, A * height, width))
  rpn_labels = labels

  #                           bbox_targets
  # bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
  # 得到在图像内部archors的坐标 anchors = all_anchors[inds_inside, :]
  # bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])
  # bbox_targets 得到移动的  dx dy dw dh
  bbox_targets = bbox_targets \
    .reshape((1, height, width, A * 4))

  rpn_bbox_targets = bbox_targets
  # bbox_inside_weights
  bbox_inside_weights = bbox_inside_weights \
    .reshape((1, height, width, A * 4))

  rpn_bbox_inside_weights = bbox_inside_weights
  # bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
  # 所有archors中正样本的四个坐标的权重均设置为1，其他为0

  # bbox_outside_weights
  bbox_outside_weights = bbox_outside_weights \
    .reshape((1, height, width, A * 4))

  rpn_bbox_outside_weights = bbox_outside_weights

  #返回的是     特征图映射到原图的  所有的边框
  #把超出图像尺寸的 边框 置为-1
  #在输入的所有的边框与标签中  帅选出  小于等于256/2 个正负样本  总共 就是256 样本  正为1负为0 其他为-1

  #超出图像尺寸的边框的  label等置为-1  边框偏移量 0 边框权重0  边框权重的归一化参数0
  #    标签正样本1，负0，不关注-1  (1, 1, A * height, width)
  #    边框 偏移量  是偏移量 dx dy dw dh  是中心坐标与  边框长度的偏移量(1, height, width, A * 4)
  #    边框权重1             (1, height, width, A * 4)
  #    边框权重的归一化参数  (1, height, width, A * 4)
  return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _unmap(data, count, inds, fill=0):
  """
  函数的作用是  在特征图映射到原图的所有框中 把超出边界的 边框 的label置为-1

  # map up to original set of anchors
  # total_anchors = all_anchors.shape[0]  #total_anchors得到 锚点的   个数 N*9个
  # inds_inside  所有archors边界可能超出图像，取在图像内部的archors的索引
  labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
       #labels = np.empty((len(inds_inside),), dtype=np.float32)

  """
  if len(data.shape) == 1:#判断label的维度
    ret = np.empty((count,), dtype=np.float32)
    ret.fill(fill)
    ret[inds] = data
  else:
    #这里构建的是   count行  data.shape[1:]列的   矩阵
    #元组(count,)是有逗号的   作用就完全不同 data.shape[1:]这样返回也是有逗号的
    ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
    ret.fill(fill)
    ret[inds, :] = data
  return ret


def _compute_targets(ex_rois, gt_rois):
  """Compute bounding-box regression targets for an image."""
  #真实边框与预测便边框最回归预测

  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 5


  return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
