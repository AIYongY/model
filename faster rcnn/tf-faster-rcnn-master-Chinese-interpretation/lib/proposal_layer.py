# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from config import cfg
from bbox_transform import bbox_transform_inv, clip_boxes, bbox_transform_inv_tf, clip_boxes_tf
from nms_wrapper import nms

def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
  """A simplified version compared to fast/er RCNN
     For details please see the technical report
  """
  if type(cfg_key) == bytes:
      cfg_key = cfg_key.decode('utf-8')
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N# RPN_PRE_NMS_TOP_N = 6000
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  # __C.TEST.RPN_POST_NMS_TOP_N = 300    非极大值抑制输出的 最大个数

  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
  # __C.TEST.RPN_NMS_THRESH = 0.7

  # Get the scores and bounding boxes
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
  scores = scores.reshape((-1, 1))

  proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
  # shape = (length, 4)
  # proposals  就是真实预测的边框的四个坐标值
  # 特征图映射到原图的所有的框anchors   与特征图的值rpn_bbox_pred  组合    进行回归预测

  proposals = clip_boxes(proposals, im_info[:2])
  # 限制预测坐标在原始图像上  限制这预测 的坐标的  值  在一定的范围内


  # Pick the top region proposals
  order = scores.ravel().argsort()[::-1]
  if pre_nms_topN > 0:
    order = order[:pre_nms_topN]
  proposals = proposals[order, :]
  scores = scores[order]

  # Non-maximal suppression
  keep = nms(np.hstack((proposals, scores)), nms_thresh)

  # Pick th top region proposals after NMS
  if post_nms_topN > 0:
    keep = keep[:post_nms_topN]
  proposals = proposals[keep, :]
  scores = scores[keep]

  # Only support single image as input
  batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
  blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

  return blob, scores

# self._im_info = tf.placeholder(tf.float32, shape=[3])
#rpn_cls_prob   rpn_bbox_pred rpn  两条路径的值
def proposal_layer_tf(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
  """
  #返回非极大值抑制之后得到的 框 blob shape=(V, 1+4)  1里面是0   有啥用？
  #blob  scores  框值对应的 置信度
  :param rpn_cls_prob:
  :param rpn_bbox_pred:
  :param im_info:
  :param cfg_key:
  :param _feat_stride:
  :param anchors:
  :param num_anchors:
  :return:
  """
  #rpn_cls_prob是背景还是前景的置信度   rpn_cls_prob[:, :, :, num_anchors:] 就是得到前景的置信度
  if type(cfg_key) == bytes:
    cfg_key = cfg_key.decode('utf-8')
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N#RPN_PRE_NMS_TOP_N = 6000
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N#__C.TEST.RPN_POST_NMS_TOP_N = 300    非极大值抑制输出的 最大个数
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH#__C.TEST.RPN_NMS_THRESH = 0.7

  # Get the scores and bounding boxes

  #rpn_cls_prob    shape[-1,?, ?, 18]
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  scores = tf.reshape(scores, shape=(-1,))
  # rpn_bbox_pred    shape[-1,?, ?, 36]
  rpn_bbox_pred = tf.reshape(rpn_bbox_pred, shape=(-1, 4))

  proposals = bbox_transform_inv_tf(anchors, rpn_bbox_pred) #是原图所有的矩形框
  #shape = (length, 4)
  #proposals  就是真实预测的边框的四个坐标值
  #特征图映射到原图的所有的框anchors   与特征图的值rpn_bbox_pred  组合    进行回归预测

  proposals = clip_boxes_tf(proposals, im_info[:2])
  # 限制预测坐标在原始图像上  限制这预测 的坐标的  值  在一定的范围内

  # Non-maximal suppression   非极大值抑制函数
  #输入 proposals所有的 预测的边框
  # scores 前景 的所有的置信度  也就是proposals所有的 置信度
  #post_nms_topN   非极大值抑制输出的 最大个数
  #nms_thresh 超过这个重叠度  就去掉
  indices = tf.image.non_max_suppression(proposals, scores, max_output_size=post_nms_topN, iou_threshold=nms_thresh)
  #indices  所有的框   非极大值抑制得到的框   返回的是索引   用tf.gather取值
  # 通过nms得到分值最大的post_nms_topN个坐标的索引

  #proposals   shape=(length, 4)        boxes  shape=(V, 4)
  boxes = tf.gather(proposals, indices)
  boxes = tf.to_float(boxes)
  scores = tf.gather(scores, indices)
  scores = tf.reshape(scores, shape=(-1, 1))

  # Only support single image as input
  batch_inds = tf.zeros((tf.shape(indices)[0], 1), dtype=tf.float32)
  blob = tf.concat([batch_inds, boxes], 1)

  #返回非极大值抑制之后得到的 框 blob shape=(V, 1+4)  1里面是0   有啥用？
  #blob  scores  框值对应的 置信度
  # scores  = rpn_cls_prob[:, :, :, num_anchors:]  = [-1，1]
  return blob, scores


