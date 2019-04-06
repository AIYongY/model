# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

import numpy as np

from snippets import generate_anchors_pre, generate_anchors_pre_tf
from proposal_layer import proposal_layer, proposal_layer_tf
from proposal_top_layer import proposal_top_layer, proposal_top_layer_tf
from anchor_target_layer import anchor_target_layer
from proposal_target_layer import proposal_target_layer
from visualization import draw_bounding_boxes

from config import cfg

print(cfg.USE_E2E_TF)
class Network(object):
  def __init__(self):
    self._predictions = {}
    self._losses = {}
    self._anchor_targets = {}
    self._proposal_targets = {}
    self._layers = {}
    self._gt_image = None
    self._act_summaries = []
    self._score_summaries = {}
    self._train_summaries = []
    self._event_summaries = {}
    self._variables_to_fix = {}

  def _add_gt_image(self):
    # add back mean
    image = self._image + cfg.PIXEL_MEANS
    #cfg.PIXEL_MEANS=[[[102.9801 115.9465 122.7717]]]
    #self._im_info = tf.placeholder(tf.float32, shape=[3])
    # BGR to RGB (opencv uses BGR)

    #resize_bilinear图像操作>调整大小  输入图像可以是不同的类型,但输出图像总是浮点型的.
    resized = tf.image.resize_bilinear(image, tf.to_int32(self._im_info[:2] / self._im_info[2]))
    self._gt_image = tf.reverse(resized, axis=[-1])
    #意思是将图片 镜像 反过来

  def _add_gt_image_summary(self):
    # use a customized visualization function to visualize the boxes
    if self._gt_image is None:
      self._add_gt_image()

    # self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
    ##self._gt_image = tf.reverse(resized, axis=[-1])这个是增加
    #self._im_info = tf.placeholder(tf.float32, shape=[3])
    image = tf.py_func(draw_bounding_boxes, 
                      [self._gt_image, self._gt_boxes, self._im_info],
                      tf.float32, name="gt_boxes")
    #返回的是 画 了边框的图
    
    return tf.summary.image('GROUND_TRUTH', image)

  def _add_act_summary(self, tensor):
    tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
    tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                      tf.nn.zero_fraction(tensor))

  def _add_score_summary(self, key, tensor):
    tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

  def _add_train_summary(self, var):
    tf.summary.histogram('TRAIN/' + var.op.name, var)

  def _reshape_layer(self, bottom, num_dim, name):#num_dim=2
    input_shape = tf.shape(bottom)
    with tf.variable_scope(name) as scope:
      # change the channel to the caffe format
      to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
      # then force it to have channel 2
      reshaped = tf.reshape(to_caffe,     #[1，2，-1，input_shape[2]]
                            tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
      # then swap the channel back
      to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
      return to_tf

  def _softmax_layer(self, bottom, name):
    if name.startswith('rpn_cls_prob_reshape'):
      input_shape = tf.shape(bottom)
      bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
      reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
      return tf.reshape(reshaped_score, input_shape)
    return tf.nn.softmax(bottom, name=name)


  def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
    """
     返回的经过筛选得到的5000个的框 和5000个框进行回归预测得到的 框
    :param rpn_cls_prob:
    :param rpn_bbox_pred:
    :param name:
    :return: rois, rpn_scores
    """
    with tf.variable_scope(name) as scope:
      if cfg.USE_E2E_TF: #True
        # batch_inds 0,1,2,3 blob = tf.concat([batch_inds, proposals], 1)
        # blob, top_scores  5000个  特征图映射到原图
        # 筛选 5000个 惊醒矫正后的框 blob和得分值top_scores
        rois, rpn_scores = proposal_top_layer_tf(
          rpn_cls_prob,
          rpn_bbox_pred,
          self._im_info,
          # self._im_info = tf.placeholder(tf.float32, shape=[3])
          self._feat_stride,#16
          self._anchors, # 特征图的所有点的9个框对应原始坐标的  所有  坐标anchors  anchor_length和个数length
          self._num_anchors#9
        )
      else:
        # #输入的是5000个特征图上映射到原图的框坐标
        # 输入的是5000个特征图上的框坐标

        # 输入时特征  与特征映射到原图    的边框
        # 这两个输入结合在一起 做回归预测
        # 好像与bbox_transform_inv_tf一样
        rois, rpn_scores = tf.py_func(proposal_top_layer,
                              [rpn_cls_prob, rpn_bbox_pred, self._im_info,
                               self._feat_stride, self._anchors, self._num_anchors],
                              [tf.float32, tf.float32], name="proposal_top")
      #cfg.TEST.RPN_TOP_N = 5000
      rois.set_shape([cfg.TEST.RPN_TOP_N, 5])
      rpn_scores.set_shape([cfg.TEST.RPN_TOP_N, 1])

    return rois, rpn_scores

  def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
    """
    # 返回非极大值抑制之后得到的 框 blob shape=(V, 1+4)  1里面是0   有啥用？
    # blob  scores  框值对应的 置信度
    :param rpn_cls_prob:
    :param rpn_bbox_pred:
    :param name:
    :return:
    """
    # 每个位置的9个archors的类别概率和每个位置的9个archors
    # 的回归位置偏移得到post_nms_topN个archors的位置及为1的概率
    # scores  = rpn_cls_prob[:, :, :, num_anchors:]  = [-1,1]

    with tf.variable_scope(name) as scope:
      if cfg.USE_E2E_TF: #cfg.USE_E2E_TF = True

        # 返回非极大值抑制之后得到的 框 blob shape=(V, 1+4)  1里面是0   有啥用？
        # blob  scores  框值对应的 置信度
        rois, rpn_scores = proposal_layer_tf(
          rpn_cls_prob,
          rpn_bbox_pred,
          self._im_info,      # self._im_info = tf.placeholder(tf.float32, shape=[3])
          self._mode,        #self._mode = mode  mode先不知道是啥
          self._feat_stride, #_feat_stride=16
          self._anchors,     #是原图所有的矩形框
          self._num_anchors  # 9
        )

#---------------------------------------未看
      else:
        rois, rpn_scores = tf.py_func(proposal_layer,
                              [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                               self._feat_stride, self._anchors, self._num_anchors],
                              [tf.float32, tf.float32], name="proposal")

      rois.set_shape([None, 5])
      rpn_scores.set_shape([None, 1])
# ---------------------------------------未看

    return rois, rpn_scores

  # Only use it if you have roi_pooling op written in tf.image
  def _roi_pool_layer(self, bootom, rois, name):
    with tf.variable_scope(name) as scope:
      #https://blog.csdn.net/u011436429/article/details/80279536
      #函数解释   类似池化
      return tf.image.roi_pooling(bootom, rois,
                                  pooled_height=cfg.POOLING_SIZE,
                                  pooled_width=cfg.POOLING_SIZE,
                                  spatial_scale=1. / 16.)[0]

  def _crop_pool_layer(self, bottom, rois, name):
    #在原图中  预测到的框  的范围剪切下来  修改尺寸为14*14 再 池化 到7*7
    #输入
    # net_conv = 采样16倍的  特征图
    #rois 是特征图当中的值  他的值反应 原图的矩形框
    # # rois 是框 取128个 前面的fg_rois_per_image是正样本 小于等于32      是非极大值抑制之后 筛选最优的  128
    # # rois 是框 取128个 前面的fg_rois_per_image是正样本 小于等于32      是非极大值抑制之后 筛选最优的  128
    # pool5 = self._crop_pool_layer(net_conv, rois, "pool5")

    with tf.variable_scope(name) as scope:
      #tf.slice的意思时起点到终点的位置截取下来  # 第一列是0到n的值
      batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
      # Get the normalized coordinates of bounding boxes
      bottom_shape = tf.shape(bottom)
      height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
      width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
      x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width  # width height 是输入图像的尺寸
      y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height #
      x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
      y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
      #特征图的值表示什么  是要看自己怎么取计算 他就是什么意思 ？？？？？？？？？
      #在这里  将特征值 的框表示的是原图的框   框表示的是原图的框 /   原图像的尺寸   就是得到    在特征图的尺寸



      bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1)) #停止梯度计算.
      pre_pool_size = cfg.POOLING_SIZE * 2#cfg.POOLING_SIZE = 7
      #tf.image.crop_and_resize  #batch_ids 0到n的值                              14         *        14
      crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")
      #在特征图中   提取crop(裁剪),    并双线调整它们的大小(可能高宽比变化)
      #在特征图中提取crop(裁剪)   减的框 是可以直接映射到原图的
      #输入的  bboxes  第一维度 需要 0-n的索引  而且还是与原图的比例值
    return slim.max_pool2d(crops, [2, 2], padding='SAME')

  def _dropout_layer(self, bottom, name, ratio=0.5):
    return tf.nn.dropout(bottom, ratio, name=name)

  def _anchor_target_layer(self, rpn_cls_score, name):
    """
    #     elf._anchor_targets  label 边框偏移量 边框权重 边框权重的归一化参数
      #     添加到这里  再传递在这里 self._score_summaries = {}
      #    返回 label  浮点数   标签正样本1，负0，不关注-1   (1, 1, A * height, width)



    :param rpn_cls_score:   rpn一条路径得到的  背景前景值
    :param name:
    :return:
    """
    with tf.variable_scope(name) as scope:

      """
      
      rpn_cls_score  rpn一条路径得到的  背景前景值
      self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
      self._feat_stride = 16
      self._im_info = tf.placeholder(tf.float32, shape=[3])
      self._anchors   wgg特征图 对应原始坐标的所有 边框
      self._num_anchors = 9
      """
      # 返回的是     特征图映射到原图的  所有的边框
      # 把超出图像尺寸的 边框 置为-1
      # 在输入的所有的边框与标签中  帅选出  小于等于256/2 个正负样本  总共 就是256 样本  正为1负为0 其他为-1

      # 超出图像尺寸的边框的  label等置为-1  边框偏移量 0 边框权重0  边框权重的归一化参数0
      #    标签正样本1，负0，不关注-1  (1, 1, A * height, width)
      #    边框 偏移量  是偏移量 dx dy dw dh  是中心坐标与  边框长度的偏移量(1, height, width, A * 4)
      #    边框权重1             (1, height, width, A * 4)
      #    边框权重的归一化参数  (1, height, width, A * 4)
      # return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

      #     elf._anchor_targets  label 边框偏移量 边框权重 边框权重的归一化参数
      #     添加到这里  再传递在这里 self._score_summaries = {}
      #    返回 label  浮点数   标签正样本1，负0，不关注-1   (1, 1, A * height, width)#  A=9
      # 限定得到的框在256/2  之内   小于256/2之内则不变    正负样本的和是128  其他为不关注 -1
      # 限定得到的框在256/2  之内    小于256/2之内则不变

      rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
        anchor_target_layer,
        [rpn_cls_score, self._gt_boxes, self._im_info, self._feat_stride, self._anchors, self._num_anchors],
        [tf.float32, tf.float32, tf.float32, tf.float32],
        name="anchor_target")

      rpn_labels.set_shape([1, 1, None, None])
      rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
      rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
      rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

      rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
      self._anchor_targets['rpn_labels'] = rpn_labels
      self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
      # 边框偏移量是偏移量dxdydwdh是中心坐标与边框长度的偏移量(1, height, width, A * 4)
      # 限定得到的框在256 / 2之内小于256 / 2之内则不变正负样本的和是128其他为不关注 0

      self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
      self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

      #self._score_summaries = {}
      # Python 字典(Dictionary) update() 函数把字典dict2的键/值对更新到dict里。
      self._score_summaries.update(self._anchor_targets)

    return rpn_labels

  def _proposal_target_layer(self, rois, roi_scores, name):
    """

      def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
      返回非极大值抑制之后得到的 框 blob  shape=(V, 1+4)  1里面是0   有啥用？
      blob  scores  框值对应的 置信度 交并比  roi_scores  shape [-1,1]

    :param rois:
    :param roi_scores: 非极大值抑制得到对应框的值
    :param name:
    :return:
    """
    with tf.variable_scope(name) as scope:
      rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
        proposal_target_layer,
        [rois, roi_scores, self._gt_boxes, self._num_classes],
        [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
        name="proposal_target")
      # labels  取128个  前面的fg_rois_per_image是正样本 小于等于32   是非极大值抑制之后 筛选最优的  128个
      # rois 是框 取128个 前面的fg_rois_per_image是正样本 小于等于32      是非极大值抑制之后 筛选最优的  128个
      # roi_scores  取128个 前面的fg_rois_per_image是正样本 小于等于32  是非极大值抑制之后 筛选最优的  128个
      # return labels, rois, roi_scores, bbox_targets, bbox_inside_weights

      # bbox_targets：256*(4*21)的矩阵，只有为正样本时，对应类别的坐标才不为0，其他类别的坐标全为0
      # bbox_inside_weights：256*(4*21)的矩阵，正样本时，对应类别四个坐标的权重为1，其他全为0

      # return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

      rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
      roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE])
      labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
      bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
      bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
      bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])

      self._proposal_targets['rois'] = rois
      self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
      self._proposal_targets['bbox_targets'] = bbox_targets
      self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
      self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

      self._score_summaries.update(self._proposal_targets)
      # labels  取128个  前面的fg_rois_per_image是正样本 小于等于32   是非极大值抑制之后 筛选最优的  128个
      # rois 是框 取128个 前面的fg_rois_per_image是正样本 小于等于32      是非极大值抑制之后 筛选最优的  128个
      # roi_scores  取128个 前面的fg_rois_per_image是正样本 小于等于32  是非极大值抑制之后 筛选最优的  128个
      # return labels, rois, roi_scores, bbox_targets, bbox_inside_weights

      # bbox_targets：256*(4*21)的矩阵，只有为正样本时，对应类别的坐标才不为0，其他类别的坐标全为0
      # bbox_inside_weights：256*(4*21)的矩阵，正样本时，对应类别四个坐标的权重为1，其他全为0

      # return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

      return rois, roi_scores

  def _anchor_component(self):
    """
    这个函数的功能就是把参数传进去self._anchors   self._anchor_length

     # 特征图的所有点的9个框对应原始坐标的  所有所有  坐标anchors  anchor_length和个数length

     anchors.set_shape([None, 4])
      anchor_length.set_shape([])
      self._anchors = anchors
      self._anchor_length = anchor_length

    :return:
    """

    with tf.variable_scope('ANCHOR_' + self._tag) as scope:
      # just to get the shape right  原始图片的尺寸除以16 得到特征图的尺寸
      height = tf.to_int32(tf.ceil(self._im_info[0] / np.float32(self._feat_stride[0])))
      width = tf.to_int32(tf.ceil(self._im_info[1] / np.float32(self._feat_stride[0])))
      if cfg.USE_E2E_TF: # cfg.USE_E2E_TF  True
        # 特征图的所有点的9个框对应原始坐标的  所有所有  坐标anchors  anchor_length和个数length
        anchors, anchor_length = generate_anchors_pre_tf(
          height,
          width,
          self._feat_stride,
          self._anchor_scales,
          self._anchor_ratios
        )
      else:
        anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                            [height, width,
                                             self._feat_stride, self._anchor_scales, self._anchor_ratios],
                                            [tf.float32, tf.int32], name="generate_anchors")
      anchors.set_shape([None, 4])
      anchor_length.set_shape([])
      self._anchors = anchors
      self._anchor_length = anchor_length

  def _build_network(self, is_training=True):
    # rois 是框 取128个 前面的fg_rois_per_image是正样本 小于等于32      是非极大值抑制之后 筛选最优的  128
    # rois shape [-1,5]  讲过剪切框得到  rois shape [-1,4]
    #   返回的是 最终  得分值预测   与框的预测
    #   return rois, cls_prob, bbox_pred



    # select initializers
    if cfg.TRAIN.TRUNCATED: #  False   truncated 缩短
      initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
    else:
      initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

    # _build_network 在vgg16类中被重写 定义为VGG16的前向传播
    net_conv = self._image_to_head(is_training)
    # 返回降采样16倍的  特征图

    with tf.variable_scope(self._scope, self._scope):

      self._anchor_component()
      # 这个函数的功能就是把参数传进去self._anchors  self._anchor_length
      # 特征图的所有点的9个框对应原始坐标的  所有所有  坐标anchors  anchor_length和个数length

      # region proposal network    # return rois
      rois = self._region_proposal(net_conv, is_training, initializer)
      # # rois 是框 取128个 前面的fg_rois_per_image是正样本 小于等于32      是非极大值抑制之后 筛选最优的  128

      # self._predictions["rpn_cls_score"] = rpn_cls_score  # rpn路径的特征图
      # self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape  # 1*？*？*18==>1*(?*9)*?*2 是上面的形状改变
      # self._predictions["rpn_cls_prob"] = rpn_cls_prob  # rpn_cls_score_reshape经过softmax 1*？*？*18==>1*(?*9)*?*2
      # self._predictions["rpn_cls_pred"] = rpn_cls_pred
      # # #返回最大值的下标                      1*(?*9)*?*2==> [-1, 2] 得到前景背景 哪个的得分值比较大
      # self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
      # # rpn  的框的特征图  rpn_bbox_pred
      # self._predictions["rois"] = rois


      # region of interest pooling
      if cfg.POOLING_MODE == 'crop':#默认cfg.POOLING_MODE = 'crop'
        #net_conv = 采样16倍的  特征图
        # # rois 是框 取128个 前面的fg_rois_per_image是正样本 小于等于32      是非极大值抑制之后 筛选最优的  128
        # # rois 是框 取128个 前面的fg_rois_per_image是正样本 小于等于32      是非极大值抑制之后 筛选最优的  128
        pool5 = self._crop_pool_layer(net_conv, rois, "pool5")
        #pool5 的个数得到和  r ois 一样
        #rois   作为剪切尺寸的值的一部分
        # 在原特征中  截图框 框的尺寸是 特征图的值除以原图的尺寸（归一化）   修改尺寸为14*14 再 池化 到7*7
      else:
        raise NotImplementedError

    #pool5是
    # 在原特征中  截图框 框的尺寸是 特征图的值除以原图的尺寸（归一化）   修改尺寸为14*14 再 池化 到7*7
    fc7 = self._head_to_tail(pool5, is_training)
    #  fc7 是在   pool5   接了全连接层与 dropout

    with tf.variable_scope(self._scope, self._scope):
      # region classification
      cls_prob, bbox_pred = self._region_classification(fc7, is_training, 
                                                        initializer, initializer_bbox)
      # 在原特征中  截图框 框的尺寸是 特征图的值除以原图的尺寸（归一化）   修改尺寸为14*14 再 池化 到7*7
      # fc7 = self._head_to_tail(pool5, is_training)
      #  fc7 是在   pool5   接了全连接层与 dropout

      #21个类别
      #接了全连接层  =cls_prob               softmax 全连接层  = bbox_pred
      #返回的是 最终  得分值预测   与框的预测

    self._score_summaries.update(self._predictions)
    #rois 是框 取128个 前面的fg_rois_per_image是正样本 小于等于32      是非极大值抑制之后 筛选最优的  128
    # rois shape [-1,5]  讲过剪切框得到  rois shape [-1,4]
    #    #返回的是 最终  得分值预测   与框的预测
    #
    return rois, cls_prob, bbox_pred

  def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    """
     rpn_bbox_pred = self._predictions['rpn_bbox_pred']
      # rpn  的框的特征图  rpn_bbox_pred
      # rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1].........

      rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
      # 边框偏移量是偏移量dxdydwdh是中心坐标与边框长度的偏移量(1, height, width, A * 4)
      # 限定得到的框在256 / 2之内小于256 / 2之内则不变正负样本的和是128其他为不关注 0

      rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
      #   rpn_bbox_inside_weights   边框权重1             (1, height, width, A * 4)  不关注 0
      #   rpn_bbox_outside_weights  边框权重的归一化参数  (1, height, width, A * 4)  不关注 0
      #正样本和负样本一样
      rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']


       rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                          rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])
      #sigma_rpn=3.0
    """
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets#特征图预测的值减去  dx dy dw  dh 就是在 移动


    #   rpn_bbox_inside_weights   边框权重1             (1, height, width, A * 4)  不关注 0
    in_box_diff = bbox_inside_weights * box_diff #不关注的框  *0
    # 这个就是损失函数的   x  in_box_diff

    abs_in_box_diff = tf.abs(in_box_diff)   #为啥要求绝对值  有些框本来就是0  减去dx那就是为负数

    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
    # abs_in_box_diff <  1. / sigma_2   smoothL1_sign =1.0
    #abs_in_box_diff >  1. / sigma_2    smoothL1_sign  = 0.0

    in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    #tf.pow 是幂运算  就是 几次方

    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = tf.reduce_mean(tf.reduce_sum(
      out_loss_box,
      axis=dim  #dim=[1, 2, 3]
    ))
    return loss_box

  def _add_losses(self, sigma_rpn=3.0):

    #返回没有加正则化的 总loss
    with tf.variable_scope('LOSS_' + self._tag) as scope:
      # RPN, class loss
      rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
      # rpn路径的特征图1*？*？*18==>1*(?*9)*?*2 是上面的形状改变
      rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
      # 返回 label  浮点数   标签正样本1，负0，不关注-1   (1, 1, A * height, width)#  A=9
      # 限定得到的框在256/2  之内   小于256/2之内则不变    正负样本的和是128  其他为不关注 -1

      rpn_select = tf.where(tf.not_equal(rpn_label, -1))# (x! = y) 元素的真值.
      rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
      rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])


      rpn_cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

      # RPN, bbox loss
      rpn_bbox_pred = self._predictions['rpn_bbox_pred']
      # rpn  的框的特征图  rpn_bbox_pred
      # rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1].........

      rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
      # 边框偏移量是偏移量dxdydwdh是中心坐标与边框长度的偏移量(1, height, width, A * 4)
      # 限定得到的框在256 / 2之内小于256 / 2之内则不变正负样本的和是128其他为不关注 0

      rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
      #   rpn_bbox_inside_weights   边框权重1             (1, height, width, A * 4)  不关注 0
      #   rpn_bbox_outside_weights  边框权重的归一化参数  (1, height, width, A * 4)  不关注 0
      #正样本和负样本一样
      rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']

      rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                          rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])
      #sigma_rpn=3.0

      # RCNN, class loss
      cls_score = self._predictions["cls_score"]
      label = tf.reshape(self._proposal_targets["labels"], [-1])
      cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

      # RCNN, bbox loss
      bbox_pred = self._predictions['bbox_pred']
      bbox_targets = self._proposal_targets['bbox_targets']
      bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
      bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
      loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)


      self._losses['cross_entropy'] = cross_entropy
      self._losses['loss_box'] = loss_box
      self._losses['rpn_cross_entropy'] = rpn_cross_entropy
      self._losses['rpn_loss_box'] = rpn_loss_box


      loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
      regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
      self._losses['total_loss'] = loss + regularization_loss

      self._event_summaries.update(self._losses)

    return loss

  def _region_proposal(self, net_conv, is_training, initializer):
    """
    self._predictions["rpn_cls_score"] = rpn_cls_score #rpn路径的特征图
    self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape # 1*？*？*18==>1*(?*9)*?*2 是上面的形状改变
    self._predictions["rpn_cls_prob"] = rpn_cls_prob# rpn_cls_score_reshape经过softmax 1*？*？*18==>1*(?*9)*?*2
    self._predictions["rpn_cls_pred"] = rpn_cls_pred
    # #返回最大值的下标                      1*(?*9)*?*2==> [-1, 2] 得到前景背景 哪个的得分值比较大
    self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
    #rpn  的框的特征图  rpn_bbox_pred
    self._predictions["rois"] = rois
    # rois 是框 取128个 前面的fg_rois_per_image是正样本 小于等于32      是非极大值抑制之后 筛选最优的  128

    return rois
    :param net_conv:
    :param is_training:
    :param initializer:
    :return:
    """
    #                          cfg.RPN_CHANNELS=512
    #rpn 网络的开始
    rpn = slim.conv2d(net_conv, cfg.RPN_CHANNELS, [3, 3], trainable=is_training, weights_initializer=initializer,
                        scope="rpn_conv/3x3")
    self._act_summaries.append(rpn)
    # self._act_summaries = []

    #   self._num_anchors =  9
    rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                weights_initializer=initializer,
                                padding='VALID', activation_fn=None, scope='rpn_cls_score')

    # 1*1的conv，得到每个位置的9个archors分类特征1*？*？*(9*2)（二分类），判断当前archors是正样本还是负样本
    rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
    # 1*？*？*18==>1*(?*9)*?*2
    # (1, ?, ?, 2)  就是转化为这种形状

    rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
    #得到前景背景的概率   [-1,2]  => [-1] 得到的值是  [0，1,0,0,1,1....]

    rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")
    #返回最大值的下标                      1*(?*9)*?*2==> [-1, 2] 得到前景背景 哪个的得分值比较大

    # 变换会原始维度1*(?*9)*?*2==>1*?*?*(9*2)
    rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")



    # 这里是另一条路径了[-1,?,?,36]
    rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                weights_initializer=initializer,
                                padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
    if is_training:
      #输入  rpn  是两个路径的 特征图   非极大值抑制是输出的全部特征图进行计算的
      rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
      """
      def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
      返回非极大值抑制之后得到的 框 blob  shape=(V, 1+4)  1里面是0   有啥用？
      blob  scores  框值对应的 置信度 交并比  roi_scores  shape [-1,1]
      """
      rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
      #     elf._anchor_targets  label 边框偏移量 边框权重 边框权重的归一化参数
      #     添加到这里  再传递在这里 self._score_summaries = {}
      #    返回 label  浮点数   标签正样本1，负0，不关注-1   (1, 1, A * height, width)

      with tf.control_dependencies([rpn_labels]):
        #rois     roi_scores
        # 非极大值抑制后得到的 输入的是 特征图所有的值计算非极大值抑制
        rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
      # _proposal_target_layer   return rois, roi_scores
      #self._score_summaries.update(self._proposal_targets)

      #self._proposal_targets['rois'] = rois
      # self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
      # self._proposal_targets['bbox_targets'] = bbox_targets
      # self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
      # self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

      # self._score_summaries.update(self._proposal_targets)
      # # labels  取128个  前面的fg_rois_per_image是正样本 小于等于32   是非极大值抑制之后 筛选最优的  128个
      # rois 是框 取128个 前面的fg_rois_per_image是正样本 小于等于32      是非极大值抑制之后 筛选最优的  128个
      # roi_scores  取128个 前面的fg_rois_per_image是正样本 小于等于32  是非极大值抑制之后 筛选最优的  128个
      # return labels, rois, roi_scores, bbox_targets, bbox_inside_weights

      # bbox_targets：256*(4*21)的矩阵，只有为正样本时，对应类别的坐标才不为0，其他类别的坐标全为0
      # bbox_inside_weights：256*(4*21)的矩阵，正样本时，对应类别四个坐标的权重为1，其他全为0

      # return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights



    else:#   如果不是训练               这是不是在训练的时候的筛选框的方法
      # 两种筛选框的 方法    rpn_cls_prob  rpn_bbox_pred  得到的是 1000，600  /16
      # 60*40* (2*9)  60*40* (4*9) 进行筛选
      #             筛选后的形状是？？？？？？？？？？？？？？？
      if cfg.TEST.MODE == 'nms':
        rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        # 返回非极大值抑制之后得到的 框 blob shape=(V, 1+4)  1里面是0   有啥用？
        # blob  scores  框值对应的 置信度
      elif cfg.TEST.MODE == 'top':
        rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        #返回的经过筛选得到的5000个的框 和5000个框进行回归预测得到的 框？？？
        #5000？？？？？？？？？？？/   是不是错了
      else:
        raise NotImplementedError

    self._predictions["rpn_cls_score"] = rpn_cls_score #rpn路径的特征图
    self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape # rpn路径的特征图1*？*？*18==>1*(?*9)*?*2 是上面的形状改变
    self._predictions["rpn_cls_prob"] = rpn_cls_prob# rpn_cls_score_reshape经过softmax 1*？*？*18==>1*(?*9)*?*2
    self._predictions["rpn_cls_pred"] = rpn_cls_pred
    # #返回最大值的下标                      1*(?*9)*?*2==> [-1, 2] 得到前景背景 哪个的得分值比较大
    self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
    #rpn  的框的特征图  rpn_bbox_pred
    # rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
    #                             weights_initializer=initializer,
    #                             padding='VALID', activation_fn=None, scope='rpn_bbox_pred')


    self._predictions["rois"] = rois
    # rois 是框 取128个 前面的fg_rois_per_image是正样本 小于等于32      是非极大值抑制之后 筛选最优的  128

    return rois

  def _region_classification(self, fc7, is_training, initializer, initializer_bbox):
    cls_score = slim.fully_connected(fc7, self._num_classes, #21
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       activation_fn=None, scope='cls_score')
    cls_prob = self._softmax_layer(cls_score, "cls_prob")

    cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
    bbox_pred = slim.fully_connected(fc7, self._num_classes * 4, 
                                     weights_initializer=initializer_bbox,
                                     trainable=is_training,
                                     activation_fn=None, scope='bbox_pred')
    ##pool5是
    # 在原特征中  截图框 框的尺寸是 特征图的值除以原图的尺寸（归一化）   修改尺寸为14*14 再 池化 到7*7
    # fc7 = self._head_to_tail(pool5, is_training)
    #  fc7 是在   pool5   接了全连接层与 dropout
    #在 fc7的基础上  全连接 _softmax_layer  tf.argmax等 得到
    #
    self._predictions["cls_score"] = cls_score#每个类别的概率
    self._predictions["cls_pred"] = cls_pred#cls_prob  tf.argmax(cls_score, axis=1...求的是21个类别最大的
    self._predictions["cls_prob"] = cls_prob#cls_score 讲过soft_max得到
    self._predictions["bbox_pred"] = bbox_pred#在前向传播  每个边框的预测值   深度self._num_classes * 4

    return cls_prob, bbox_pred

  def _image_to_head(self, is_training, reuse=None):
    """

    这个函数在继承的时候被重写  如果没有被重写 这里就会执行异常


    :param is_training:
    :param reuse:
    :return:
    """

    #当程序出现错误，python会自动引发异常，也可以通过raise显示地引发异常。
    # 一旦执行了raise语句，raise后面的语句将不能执行。
    raise NotImplementedError

  def _head_to_tail(self, pool5, is_training, reuse=None):
    """

    这个函数在继承的时候被重写  如果没有被重写 这里就会执行异常

    :param pool5:
    :param is_training:
    :param reuse:
    :return:
    """

    raise NotImplementedError

  # net.create_architecture("TEST", 21,
  #                         tag='default', anchor_scales=[8, 16, 32])
  def create_architecture(self, mode, num_classes, tag=None,
                          anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    """
    这个函数时  test  faster r-cnn 的入口


    :param mode:
    :param num_classes:
    :param tag:
    :param anchor_scales:
    :param anchor_ratios:
    :return:
    """
    self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    self._im_info = tf.placeholder(tf.float32, shape=[3])
    self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
    self._tag = tag

    self._num_classes = num_classes #  21
    self._mode = mode # TEST
    self._anchor_scales = anchor_scales  # (8, 16, 32)
    # anchor_scales = (8, 16, 32), anchor_ratios = (0.5, 1, 2)
    self._num_scales = len(anchor_scales)  #3

    self._anchor_ratios = anchor_ratios  # (0.5, 1, 2)
    # anchor_scales = (8, 16, 32), anchor_ratios = (0.5, 1, 2)
    self._num_ratios = len(anchor_ratios) # 3
    #_num_scales = 3
    self._num_anchors = self._num_scales * self._num_ratios # 9

    training = mode == 'TRAIN'
    testing = mode == 'TEST'

    assert tag != None  #default    默认是None 传进来 default

    # handle most of the regularizers here

    #L2正则             #0.0001 cfg.TRAIN.WEIGHT_DECAY
    weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
    if cfg.TRAIN.BIAS_DECAY:  # False
      biases_regularizer = weights_regularizer
    else:
      biases_regularizer = tf.no_regularizer

    # list as many types of layers as possible, even if they are not used now
    with arg_scope([slim.conv2d, slim.conv2d_in_plane,\
                    slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected], 
                    weights_regularizer=weights_regularizer,
                    biases_regularizer=biases_regularizer, 
                    biases_initializer=tf.constant_initializer(0.0)):

      rois, cls_prob, bbox_pred = self._build_network(training)
      # rois 是框 取128个 前面的fg_rois_per_image是正样本 小于等于32      是非极大值抑制之后 筛选最优的  128
      # rois shape [-1,5]  讲过剪切框得到  rois shape [-1,4]
      #   返回的是 最终  得分值预测   与框的预测
      #   return rois, cls_prob, bbox_pred

    layers_to_output = {'rois': rois}

    for var in tf.trainable_variables():
      self._train_summaries.append(var)

     # tf.trainable_variables返回的是需要训练的变量列表
     # tf.all_variables返回的是所有变量的列表

    if testing:
      #cfg.TRAIN.BBOX_NORMALIZE_STDS = [0.1, 0.1, 0.2, 0.2]         21
      stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
      #tile函数的主要功能就是将一个数组重复一定次数形成一个新的数组,但是无论如何,最后形成的一定还是一个数组

      # cfg.TRAIN.BBOX_NORMALIZE_MEANS   [0.0, 0.0, 0.0, 0.0]
      means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
      self._predictions["bbox_pred"] *= stds
      self._predictions["bbox_pred"] += means
    else:
      self._add_losses()
      layers_to_output.update(self._losses)
      #layers_to_output = {}

      val_summaries = []
      with tf.device("/cpu:0"):
        val_summaries.append(self._add_gt_image_summary())
        for key, var in self._event_summaries.items():
          val_summaries.append(tf.summary.scalar(key, var))
        for key, var in self._score_summaries.items():
          self._add_score_summary(key, var)
        for var in self._act_summaries:
          self._add_act_summary(var)
        for var in self._train_summaries:
          self._add_train_summary(var)

      self._summary_op = tf.summary.merge_all()
      self._summary_op_val = tf.summary.merge(val_summaries)

    layers_to_output.update(self._predictions)
    # layers_to_output = {}

    return layers_to_output

  def get_variables_to_restore(self, variables, var_keep_dic):
    raise NotImplementedError

  def fix_variables(self, sess, pretrained_model):
    raise NotImplementedError

  # Extract the head feature maps, for example for vgg16 it is conv5_3
  # only useful during testing mode
  def extract_head(self, sess, image):
    feed_dict = {self._image: image}
    feat = sess.run(self._layers["head"], feed_dict=feed_dict)
    return feat

  # only useful during testing mode
  def test_image(self, sess, image, im_info):
    """
     ##pool5是
    # #在原特征中  截图框 框的尺寸是 特征图的值除以原图的尺寸（归一化）   修改尺寸为14*14 再 池化 到7*7
    # fc7 = self._head_to_tail(pool5, is_training)
    #  fc7 是在   pool5   接了全连接层与 dropout
    #在 fc7的基础上  全连接 _softmax_layer  tf.argmax等 得到
    #
    self._predictions["cls_score"] = cls_score#每个类别的概率
    self._predictions["cls_pred"] = cls_pred#cls_prob  tf.argmax(cls_score, axis=1...求的是21个类别最大的
    self._predictions["cls_prob"] = cls_prob#cls_score 讲过soft_max得到
    self._predictions["bbox_pred"] = bbox_pred#在前向传播  每个边框的预测值   深度self._num_classes * 4

    #rpn_cls_prob,  rpn_bbox_pred  输入的是  rpn的特征图
    # post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N#__C.TEST.RPN_POST_NMS_TOP_N = 300    非极大值抑制输出的 最大个数
    # nms_thresh = cfg[cfg_key].RPN_NMS_THRESH#__C.TEST.RPN_NMS_THRESH = 0.7

    indices = tf.image.non_max_suppression(proposals, scores, max_output_size=post_nms_topN, iou_threshold=nms_thresh)
    #非极大值抑制  输入的预测边框  和预测的置信度  都是预测的输入
    #rois  非极大值抑制 之后 得到小于300个 而且  iou大于 0.7的
    rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
    # 返回非极大值抑制之后得到的 框 blob shape=(V, 1+4)  1里面是0   有啥用？
    # blob  scores  框值对应的 置信度

    """
    feed_dict = {self._image: image,
                 self._im_info: im_info}


    cls_score, cls_prob, bbox_pred, rois = sess.run([self._predictions["cls_score"],
                                                     self._predictions['cls_prob'],
                                                     self._predictions['bbox_pred'],
                                                     self._predictions['rois']],
                                                    feed_dict=feed_dict)

    return cls_score, cls_prob, bbox_pred, rois

  def get_summary(self, sess, blobs):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes']}
    summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

    return summary

  def train_step(self, sess, blobs, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes']}
    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                        self._losses['rpn_loss_box'],
                                                                        self._losses['cross_entropy'],
                                                                        self._losses['loss_box'],
                                                                        self._losses['total_loss'],
                                                                        train_op],
                                                                       feed_dict=feed_dict)
    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss

  def train_step_with_summary(self, sess, blobs, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes']}
    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                                 self._losses['rpn_loss_box'],
                                                                                 self._losses['cross_entropy'],
                                                                                 self._losses['loss_box'],
                                                                                 self._losses['total_loss'],
                                                                                 self._summary_op,
                                                                                 train_op],
                                                                                feed_dict=feed_dict)
    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary

  def train_step_no_return(self, sess, blobs, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes']}
    sess.run([train_op], feed_dict=feed_dict)

