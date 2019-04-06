# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick, Sean Bell and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from config import cfg
from bbox_transform import bbox_transform
from cython_bbox import bbox_overlaps


def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, _num_classes):
  """
   #labels  取128个  前面的fg_rois_per_image是正样本 小于等于32   是非极大值抑制之后 筛选最优的  128个
  #rois 是框 取128个 前面的fg_rois_per_image是正样本 小于等于32      是非极大值抑制之后 筛选最优的  128个
  #roi_scores  取128个 前面的fg_rois_per_image是正样本 小于等于32  是非极大值抑制之后 筛选最优的  128个
  # return labels, rois, roi_scores, bbox_targets, bbox_inside_weights

  # bbox_targets：256*(4*21)的矩阵，只有为正样本时，对应类别的坐标才不为0，其他类别的坐标全为0
  # bbox_inside_weights：256*(4*21)的矩阵，正样本时，对应类别四个坐标的权重为1，其他全为0

  # return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

   #rois   roi_scores  非极大值抑制后得到的   输入的是 特征图所有的值计算非极大值抑制

   def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
      返回非极大值抑制之后得到的 框 blob  shape=(V, 1+4)  1里面是0   有啥用？
      blob  scores  框值对应的 置信度 交并比  roi_scores  shape [-1,1]

      rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
        proposal_target_layer,
        [rois, roi_scores, self._gt_boxes, self._num_classes],
        [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
        name="proposal_target")

        #self._num_classes = num_classes #  21

         self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])

  """
  # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
  # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
  all_rois = rpn_rois
  all_scores = rpn_scores

  # Include ground-truth boxes in the set of candidate rois

#未看？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？//
  if cfg.TRAIN.USE_GT:#False
    zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
    all_rois = np.vstack(
      (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
    )
    # not sure if it a wise appending, but anyway i am not using it
    all_scores = np.vstack((all_scores, zeros))




  num_images = 1   # 该程序只能一次处理一张图片

  #cfg.TRAIN.BATCH_SIZE = 128
  rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images ## 每张图片中最终选择的rois
  #cfg.TRAIN.FG_FRACTION =     0.25              *128
  fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)


  # all_rois    all_scores非极大值抑制得到的框    输入all_rois是[-1,5] [-1,1]是框的得分值？？？
  # #rois  roi_scores 非极大值抑制后得到的 输入的是 特征图所有的值计算非极大值抑制
  # gt_boxes   0.25*128   128   21
  labels, rois, roi_scores, bbox_targets, bbox_inside_weights = _sample_rois(
    all_rois, all_scores, gt_boxes, fg_rois_per_image,
    rois_per_image, _num_classes)
  #labels  取128个  前面的fg_rois_per_image是正样本 小于等于32   是非极大值抑制之后 筛选最优的  128个
  #rois 是框 取128个 前面的fg_rois_per_image是正样本 小于等于32      是非极大值抑制之后 筛选最优的  128个
  #roi_scores  取128个 前面的fg_rois_per_image是正样本 小于等于32  是非极大值抑制之后 筛选最优的  128个
  # return labels, rois, roi_scores, bbox_targets, bbox_inside_weights

  # bbox_targets：256*(4*21)的矩阵，只有为正样本时，对应类别的坐标才不为0，其他类别的坐标全为0
  # bbox_inside_weights：256*(4*21)的矩阵，正样本时，对应类别四个坐标的权重为1，其他全为0

  # return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

  rois = rois.reshape(-1, 5)
  roi_scores = roi_scores.reshape(-1)
  labels = labels.reshape(-1, 1)
  bbox_targets = bbox_targets.reshape(-1, _num_classes * 4)
  bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)
  bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

  return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


def _get_bbox_regression_labels(bbox_target_data, num_classes):
  """Bounding-box regression targets (bbox_target_data) are stored in a
  compact form N x (class, tx, ty, tw, th)

  This function expands those targets into the 4-of-4*K representation used
  by the network (i.e. only one class has non-zero targets).

  Returns:
      bbox_target (ndarray): N x 4K blob of regression targets
      bbox_inside_weights (ndarray): N x 4K blob of loss weights

       #  输入  [标签 ， dx ,dy, dw ,dh]    21   #label不是one_hot
  """

  clss = bbox_target_data[:, 0]  #标签  #label不是one_hot
  bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32) #   256*(4*21)的矩阵    是256？？？
  bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
  inds = np.where(clss > 0)[0]#正样本的下标

  for ind in inds: # inds 正样本的索引
    cls = clss[ind]  # 正样本的类别
    start = int(4 * cls)
    end = start + 4
    bbox_targets[ind, start:end] = bbox_target_data[ind, 1:] #对应的坐标偏移  赋值   给对应的类别
    # [标签 ， dx ,dy, dw ,dh]  的 dx ,dy, dw ,dh  转换到   256*(4*21)的矩阵
    bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    #[1.0, 1.0, 1.0, 1.0]  # 对应的权重(1.0, 1.0, 1.0, 1.0)  赋值给对应的类别
  return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
  """
  #组合  [标签 ， dx ,dy, dw ,dh]
  return np.hstack(
    (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)
  bbox_target_data = _compute_targets(
    rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

   输入的都是  128个
    #rois = all_rois[keep_inds]#取128个 前面的fg_rois_per_image是正样本 小于等于32
  #gt_boxes[gt_assignment[keep_inds], :4]  labels 是经过筛选IOU得到的128个值  前面的fg_rois_per_image是正样本

  """
  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 4

  # targets = np.vstack(
  #   (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
  # return targets
  targets = bbox_transform(ex_rois, gt_rois)#输入的128个 得到 128个 dx dy dw dh


  if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED: # True
    # Optionally normalize targets by a precomputed mean and stdev
    #cfg.TRAIN.BBOX_NORMALIZE_MEANS = [0.0, 0.0, 0.0, 0.0]
    targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
               / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))#[0.1, 0.1, 0.2, 0.2]

  #组合  [标签 ， dx ,dy, dw ,dh]
  return np.hstack(
    (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
  """

    # bbox_targets：256*(4*21)的矩阵，只有为正样本时，对应类别的坐标才不为0，其他类别的坐标全为0
  # bbox_inside_weights：256*(4*21)的矩阵，正样本时，对应类别四个坐标的权重为1，其他全为0
  # labels  128 个  前面的fg_rois_per_image是正样本 小于等于32   是非极大值抑制之后 筛选最优的  128个
  # rois  取128个 前面的fg_rois_per_image是正样本 小于等于32      是非极大值抑制之后 筛选最优的  128个
  #roi_scores  取128个 前面的fg_rois_per_image是正样本 小于等于32  是非极大值抑制之后 筛选最优的  128个
  return labels, rois, roi_scores, bbox_targets, bbox_inside_weights

  examples.
   # all_rois    all_scores 非极大值抑制得到的框 与 交并比
  # #rois roi_scores 非极大值抑制后得到的 输入的是 特征图所有的值计算非极大值抑制
  # gt_boxes   0.25*128   128   21
  """
  # overlaps: (rois x gt_boxes)
  overlaps = bbox_overlaps(
    np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
    np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))

  #overlaps交并比    特征图中非极大值抑制之后  得到的框 与真实边框的 交并比值
  #[N.V]
  gt_assignment = overlaps.argmax(axis=1)#列  最大的下标[N]
  max_overlaps = overlaps.max(axis=1)#列  最大的值 [N]
  labels = gt_boxes[gt_assignment, 4]#label不是one_hot

  #cfg.TRAIN.FG_THRESH = 0.5
  fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
  # Guard against the case when an image has fewer than fg_rois_per_image
  # cfg.TRAIN.BG_THRESH_HI = 0.5
  # cfg.TRAIN.BG_THRESH_LO = 0.1
  bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                     (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]

  # Small modification to the original version where we ensure a fixed number of regions are sampled
  if fg_inds.size > 0 and bg_inds.size > 0:
    fg_rois_per_image = min(fg_rois_per_image, fg_inds.size)
    #fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # 0.25*128 fg_rois_per_image
    #比较 0.25*128  与
    #overlaps交并比    特征图中非极大值抑制之后  得到的框 与真实边框的 交并比值
    #列的最大值且大于 0.5的个数  与  0.25*128 取较小值

    #就是将 筛选overlaps交并比  得到的 框控制在 0.25*128  范围之内

    #fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    #在列的最大值且大于 0.5的个数  中随机抽取 fg_rois_per_image  小于等于32个
    fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_image), replace=False)
    #fg_inds  iou大于iou 0.5  小于等于32个 的下标
    #fg_rois_per_image  小于32  就是本来的数据

    #选取正正样本的下标

    #   rois_per_image =  128          -          fg_rois_per_image 小于等于32个
    bg_rois_per_image = rois_per_image - fg_rois_per_image
    #总共要得到  128个 正样本 和负样本  fg_rois_per_image正样本个数
    #bg_rois_per_image负样本个数

    #0.1=< bg_inds <0.5
    to_replace = bg_inds.size < bg_rois_per_image
    bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_image), replace=to_replace)
    #replace表示随机所选的元素中 ，是否可以重复 当为True 可以重复

    #bg_inds选取负样本的下标

  elif fg_inds.size > 0:#fg_inds  是大于0.5   ##0.1=< bg_inds <0.5为0
    to_replace = fg_inds.size < rois_per_image  # =128
    fg_inds = npr.choice(fg_inds, size=int(rois_per_image), replace=to_replace)
    fg_rois_per_image = rois_per_image
  elif bg_inds.size > 0:  #0.1=< bg_inds <0.5 #是大于  0.5为0
    to_replace = bg_inds.size < rois_per_image
    bg_inds = npr.choice(bg_inds, size=int(rois_per_image), replace=to_replace)
    fg_rois_per_image = 0
  else:
    #？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
    import pdb
    pdb.set_trace()

  #fg_inds正样本的下标 bg_inds负样本的下标  加起来  =128
  keep_inds = np.append(fg_inds, bg_inds)#一维的数据拼接在一起
  labels = labels[keep_inds] # 取128个

  #fg_rois_per_image 正样本的个数   bg_rois_per_image负样本的个数
  labels[int(fg_rois_per_image):] = 0 #正样本和后面的标签  设置为 0
  # labels 128 个
  rois = all_rois[keep_inds]#取128个 前面的fg_rois_per_image是正样本 小于等于32

  roi_scores = all_scores[keep_inds]#取128个 前面的fg_rois_per_image是正样本 小于等于32


  # gt_assignment = overlaps.argmax(axis=1)#列  最大的下标[N]
  bbox_target_data = _compute_targets(
    rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)
  #rois = all_rois[keep_inds]#取128个 前面的fg_rois_per_image是正样本 小于等于32
  #gt_boxes[gt_assignment[keep_inds], :4]  labels 是经过筛选IOU得到的128个值  前面的fg_rois_per_image是正样本

  # 返回   组合  [标签 ， dx ,dy, dw ,dh]
  #     bbox_target_data     =return np.hstack((labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

  #组合[标签 ， dx, dy, dw, dh]    21  # label不是one_hot
  bbox_targets, bbox_inside_weights = \
    _get_bbox_regression_labels(bbox_target_data, num_classes)
  # bbox_targets：256*(4*21)的矩阵，只有为正样本时，对应类别的坐标才不为0，其他类别的坐标全为0
  # bbox_inside_weights：256*(4*21)的矩阵，正样本时，对应类别四个坐标的权重为1，其他全为0

  # bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]  # 对应的坐标偏移  赋值   给对应的类别
  # # [标签 ， dx ,dy, dw ,dh]  的 dx ,dy, dw ,dh  转换到   256*(4*21)的矩阵
  # bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
  # [1.0, 1.0, 1.0, 1.0]  # 对应的权重(1.0, 1.0, 1.0, 1.0)  赋值给对应的类别
  # return bbox_targets, bbox_inside_weights

  # bbox_targets：256*(4*21)的矩阵，只有为正样本时，对应类别的坐标才不为0，其他类别的坐标全为0
  # bbox_inside_weights：256*(4*21)的矩阵，正样本时，对应类别四个坐标的权重为1，其他全为0
  # labels  128 个  前面的fg_rois_per_image是正样本 小于等于32   是非极大值抑制之后 筛选最优的  128个
  # rois  取128个 前面的fg_rois_per_image是正样本 小于等于32      是非极大值抑制之后 筛选最优的  128个
  #roi_scores  取128个 前面的fg_rois_per_image是正样本 小于等于32  是非极大值抑制之后 筛选最优的  128个
  return labels, rois, roi_scores, bbox_targets, bbox_inside_weights
