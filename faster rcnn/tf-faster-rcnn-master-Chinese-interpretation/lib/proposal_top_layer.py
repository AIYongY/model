# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from config import cfg
from bbox_transform import bbox_transform_inv, clip_boxes, bbox_transform_inv_tf, clip_boxes_tf

import tensorflow as tf
import numpy as np
import numpy.random as npr
print(cfg.TEST.RPN_TOP_N)
def proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, im_info, _feat_stride, anchors, num_anchors):
  """A layer that just selects the top region proposals
     without using non-maximal suppression,
     For details please see the technical report

       self._im_info,
          # self._im_info = tf.placeholder(tf.float32, shape=[3])
          self._feat_stride,#16
          self._anchors, # 特征图的所有点的9个框对应原始坐标的  所有  坐标anchors  anchor_length和个数length
          self._num_anchors#9
            [tf.float32, tf.float32], name="proposal_top"
  """
  rpn_top_n = cfg.TEST.RPN_TOP_N
  # cfg.TEST.RPN_TOP_N = 5000

  #num_anchors 9
  scores = rpn_cls_prob[:, :, :, num_anchors:]

  rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
  scores = scores.reshape((-1, 1))

  length = scores.shape[0]
  if length < rpn_top_n:#  5000
    # Random selection, maybe unnecessary and loses good proposals
    # But such case rarely happens
    #choice() 方法返回一个列表，元组或字符串的随机项
    top_inds = npr.choice(length, size=rpn_top_n, replace=True)
    #  npr   random
  else:
    top_inds = scores.argsort(0)[::-1]
    #argsort函数返回的是数组值从小到大的索引值
    top_inds = top_inds[:rpn_top_n]#取5000个
    top_inds = top_inds.reshape(rpn_top_n, )

  # Do the selection here
  anchors = anchors[top_inds, :]
  #特征图映射到原图的所有框  top_inds 是5000个值  :是四个坐标值
  rpn_bbox_pred = rpn_bbox_pred[top_inds, :]
  scores = scores[top_inds]

  # Convert anchors into proposals via bbox transformations
  proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
  #输入的是5000个特征图上映射到原图的框坐标
  #输入的是5000个特征图上的框坐标

  # Clip predicted boxes to image#限定范围
  proposals = clip_boxes(proposals, im_info[:2])

  # Output rois blob
  # Our RPN implementation only supports a single input image, so all
  # batch inds are 0
  batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
  blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
  return blob, scores


def proposal_top_layer_tf(rpn_cls_prob, rpn_bbox_pred, im_info, _feat_stride, anchors, num_anchors):
  """
  #batch_inds 0,1,2,3 blob = tf.concat([batch_inds, proposals], 1)
  #blob, top_scores  5000个  特征图映射到原图
  #筛选 5000个 惊醒矫正后的框 blob和得分值top_scores

 A layer that just selects the top region proposals
     without using non-maximal suppression,
     For details please see the technical report
          self._im_info,
          # self._im_info = tf.placeholder(tf.float32, shape=[3])
          self._feat_stride,#16
          self._anchors, # 特征图的所有点的9个框对应原始坐标的  所有  坐标anchors  anchor_length和个数length
          self._num_anchors#9
  """
  rpn_top_n = cfg.TEST.RPN_TOP_N# cfg.TEST.RPN_TOP_N = 5000
  #self._num_anchors  9
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  rpn_bbox_pred = tf.reshape(rpn_bbox_pred, shape=(-1, 4))
  scores = tf.reshape(scores, shape=(-1,))

  # Do the selection here
  #为了找到输入的张量的最后的一个维度的最大的k个值和它的下标！
  top_scores, top_inds = tf.nn.top_k(scores, k=rpn_top_n)
  #得到5000个

  top_scores = tf.reshape(top_scores, shape=(-1, 1))

  top_anchors = tf.gather(anchors, top_inds)
  #从’params"中，按照axis坐标和indices标注的元素下标，
  # 把这些元素抽取出来组成新的tensor.
  # temp2 = tf.gather(temp, [1, 5, 9])
  # [1 11 21 31 41 51 61 71 81 91]
  # [11 51 91]

  top_rpn_bbox = tf.gather(rpn_bbox_pred, top_inds)

  #输入的是 5000个
  proposals = bbox_transform_inv_tf(top_anchors, top_rpn_bbox)
  #返回    就是预测的边框的四个坐标值
  # 特征图映射到原图的所有的框   与   特征图 的 值  进行回归  预测

  # Clip predicted boxes to image
  proposals = clip_boxes_tf(proposals, im_info[:2])
  # 限制预测坐标在原始图像上  clip剪辑
  # Output rois blob
  # Our RPN implementation only supports a single input image, so all
  # batch inds are 0
  proposals = tf.to_float(proposals)
  #rpn_top_n = 5000
  batch_inds = tf.zeros((rpn_top_n, 1))
  blob = tf.concat([batch_inds, proposals], 1)

  #batch_inds 0,1,2,3 blob = tf.concat([batch_inds, proposals], 1)
  #blob, top_scores  5000个  特征图映射到原图 筛选 5000个 惊醒矫正后的框 blob和得分值
  return blob, top_scores
