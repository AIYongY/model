# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from generate_anchors import generate_anchors

def generate_anchors_pre(height, width, feat_stride, anchor_scales=(8,16,32), anchor_ratios=(0.5,1,2)):
  """ A wrapper function to generate anchors given different scales
    Also return the number of anchors in variable 'length'
  """
  anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
  """
           9*4矩阵
           直接调用这个函数 就可以 返回 VGG15 conv5_2第一个点 对应再原始图像的
           9个矩形框
           """
  A = anchors.shape[0]
  shift_x = np.arange(0, width) * feat_stride
  shift_y = np.arange(0, height) * feat_stride
  shift_x, shift_y = np.meshgrid(shift_x, shift_y)
  shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
  K = shifts.shape[0]
  # width changes faster, so here it is H, W, C
  anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
  anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
  length = np.int32(anchors.shape[0])

  return anchors, length

def generate_anchors_pre_tf(height, width, feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
  #返回  特征图的所有点的9个框对应原始坐标的所有坐标
  # 和个数length

  #height, width特征图的尺寸
  shift_x = tf.range(width) * feat_stride # width  shift_x    [-1]  特征图对应原始图片的坐标
  shift_y = tf.range(height) * feat_stride # height shift_y   [-1]
  shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
  sx = tf.reshape(shift_x, shape=(-1,))
  sy = tf.reshape(shift_y, shape=(-1,))

  shifts = tf.transpose(tf.stack([sx, sy, sx, sy]))
  # tf.transpose(input, [dimension_1, dimenaion_2,..,
  # dimension_n]):这个函数主要适用于交换输入张量的不同维度用的，
  # 如果输入张量是二维，就相当是转置。dimension_n是整数，如果张量是三维，
  # 就是用0, 1, 2来表示。
  K = tf.multiply(width, height)
  shifts = tf.transpose(tf.reshape(shifts, shape=[1, K, 4]), perm=(1, 0, 2))

  anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
  """
           9*4矩阵
           直接调用这个函数 就可以 返回 VGG15 conv5_2第一个点 对应再原始图像的
           9个矩形框
           """
  A = anchors.shape[0] #A=9
  anchor_constant = tf.constant(anchors.reshape((1, A, 4)), dtype=tf.int32)

  length = K * A      #(1,9,4)+
  anchors_tf = tf.reshape(tf.add(anchor_constant, shifts), shape=(length, 4))

  #特征图的所有点的9个框对应原始坐标的所有坐标  和个数length
  return tf.cast(anchors_tf, dtype=tf.float32), length
