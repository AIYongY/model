# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
try:
  import cPickle as pickle
except ImportError:
  import pickle
import os
import math

from timer import Timer
from blob import im_list_to_blob

from config import cfg, get_output_dir
from bbox_transform import clip_boxes, bbox_transform_inv
from nms_wrapper import nms

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS  #[[[102.9801 115.9465 122.7717]]]

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:# cfg.TEST.SCALES = [600]
    im_scale = float(target_size) / float(im_size_min)#600  /  输入图像比较近小的尺寸
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE: #1000
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    #fx fy是缩放的比例因子
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  #以上就是  将 输入的图片  修改尺寸到   600  1000   限定在这个范围内

  # processed_ims = [ im ]里面是修改内容图片尺寸在  600  1000   限定在这个范围内
  blob = im_list_to_blob(processed_ims)
  #  im_scale_factors.append(im_scale)
  #processed_ims.append(im)
  # for i in range(num_images):
  #   im = ims[i]
  #   blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

  return blob, np.array(im_scale_factors)

def _get_blobs(im):
  """Convert an image and RoIs within that image into network inputs."""
  blobs = {}
  blobs['data'], im_scale_factors = _get_image_blob(im)

  # processed_ims = [ im ]里面是修改内容图片尺寸在  600  1000   限定在这个范围内
  # blob = im_list_to_blob(processed_ims)
  #  im_scale_factors.append(im_scale)
  # processed_ims.append(im)
  # for i in range(num_images):
  #   im = ims[i]
  #   blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
  # return blob, np.array(im_scale_factors)

  return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes

def _rescale_boxes(boxes, inds, scales):
  """Rescale boxes according to image rescaling."""
  for i in range(boxes.shape[0]):
    boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

  return boxes

def im_detect(sess, net, im):
  """
  #scores 是rpn scores = self._predictions['cls_prob']  =  每个类别的概率cls_score 讲过soft_max得到
    #pred_boxes
    # pred_boxes = bbox_transform_inv(boxes, box_deltas)
    # 做回归预测   两条路劲rpn得到的的 box_deltas 作为 dx dy dw dh 与 筛选出来的框做回归预测
    # pred_boxes  anchors回归预测后的值
  return scores, pred_boxes

  :param sess:
  :param net:
  :param im:
  :return:
  """
  # 输入 net = vgg16()   im一张图片
  blobs, im_scales = _get_blobs(im)

  # processed_ims = [ im ]里面是修改内容图片尺寸在  600  1000   限定在这个范围内
  # blob = im_list_to_blob(processed_ims)
  #  im_scale_factors.append(im_scale)
  # processed_ims.append(im)

  # for i in range(num_images):
  #   im = ims[i]
  #   blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

  # return blob, np.array(im_scale_factors)
  # return blobs, im_scale_factors


  assert len(im_scales) == 1, "Only single-image batch implemented"
  # blobs, im_scales  就是一个图像   blobs也是一张图片

  im_blob = blobs['data']
  # # blobs, im_scales  就是一个图像   blobs也是一张图片

  blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)

  _, scores, bbox_pred, rois = net.test_image(sess, blobs['data'], blobs['im_info'])
  # scores = self._predictions['cls_prob']  =  每个类别的概率cls_score 讲过soft_max得到

  ##pool5是
  # #在原特征中  截图框 框的尺寸是 特征图的值除以原图的尺寸（归一化）   修改尺寸为14*14 再 池化 到7*7
  # fc7 = self._head_to_tail(pool5, is_training)
  #  fc7 是在   pool5   接了全连接层与 dropout
  # 在 fc7的基础上  全连接 _softmax_layer  tf.argmax等 得到
  #
      # self._predictions["cls_score"] = cls_score  # 每个类别的概率
      # self._predictions["cls_pred"] = cls_pred  # cls_prob  tf.argmax(cls_score, axis=1...求的是21个类别最大的
      # self._predictions["cls_prob"] = cls_prob  # cls_score 讲过soft_max得到
      # self._predictions["bbox_pred"] = bbox_pred  # 在前向传播  每个边框的预测值   深度self._num_classes * 4
      # rois  非极大值抑制 之后 得到小于300个 而且  iou大于 0.7的   shape=(V, 1+4)

  # rpn_cls_prob,  rpn_bbox_pred  输入的是  rpn的特征图
  # post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N#__C.TEST.RPN_POST_NMS_TOP_N = 300    非极大值抑制输出的 最大个数
  # nms_thresh = cfg[cfg_key].RPN_NMS_THRESH#__C.TEST.RPN_NMS_THRESH = 0.7

      #indices = tf.image.non_max_suppression(proposals, scores, max_output_size=post_nms_topN, iou_threshold=nms_thresh)
  # 非极大值抑制  输入的预测边框  和预测的置信度  都是预测的输入

      #rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
  # 返回非极大值抑制之后得到的 框 blob shape=(V, 1+4)
  #   # blob  scores  框值对应的 置信度


  # rois  非极大值抑制 之后 得到小于300个 而且  iou大于 0.7的   shape=(V, 1+4)
  boxes = rois[:, 1:5] / im_scales[0]  # 是将boxes 转换到修改后的尺寸图里
  # im_scales 是输入的图片  修改尺寸600  1000 限定在这个范围内 的
  #            比例  比例   原图最大边 / 600或者1000

  scores = np.reshape(scores, [scores.shape[0], -1])
  bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
  if cfg.TEST.BBOX_REG: #True
    # Apply bounding-box regression deltas
    # bbox_pred  是前向传播全连接后 预测的 边框
    box_deltas = bbox_pred#  [bbox_pred.shape[0], -1]

    # boxes  是 rois 非极大值抑制 之后 得到小于300个 而且  iou大于 0.7的   shape=(V, 1+4)
    #并且转换 到 修改后的 图片尺寸里面


    # #pool5 的个数得到和  r ois 个数 一样  所以最终得到的 boxes 个数也一样
    #boxes
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    #做回归预测   两条路劲rpn得到的的 box_deltas 作为 dx dy dw dh 与 筛选出来的框做回归预测
    #pred_boxes  anchors回归预测后的值


    #输入   #pred_boxes    anchors回归预测后的值
    # im.shape修改图片尺寸之后的形状  修改尺寸600  1000 限定在这个范围内
    pred_boxes = _clip_boxes(pred_boxes, im.shape)#限定预测的边框  不可超出图像的尺寸
  else:
    # Simply repeat the boxes, once for each class
    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    #scores 是rpn scores = self._predictions['cls_prob']  =  每个类别的概率cls_score 讲过soft_max得到
    #pred_boxes
    # pred_boxes = bbox_transform_inv(boxes, box_deltas)
    # 做回归预测   两条路劲rpn得到的的 box_deltas 作为 dx dy dw dh 与 筛选出来的框做回归预测
    # pred_boxes  anchors回归预测后的值
  return scores, pred_boxes

def apply_nms(all_boxes, thresh):
  """Apply non-maximum suppression to all predicted boxes output by the
  test_net method.
  """
  num_classes = len(all_boxes)
  num_images = len(all_boxes[0])
  nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
  for cls_ind in range(num_classes):
    for im_ind in range(num_images):
      dets = all_boxes[cls_ind][im_ind]
      if dets == []:
        continue

      x1 = dets[:, 0]
      y1 = dets[:, 1]
      x2 = dets[:, 2]
      y2 = dets[:, 3]
      scores = dets[:, 4]
      inds = np.where((x2 > x1) & (y2 > y1))[0]
      dets = dets[inds,:]
      if dets == []:
        continue

      keep = nms(dets, thresh)
      if len(keep) == 0:
        continue
      nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
  return nms_boxes

def test_net(sess, net, imdb, weights_filename, max_per_image=100, thresh=0.):
  np.random.seed(cfg.RNG_SEED)  # 3
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]

  output_dir = get_output_dir(imdb, weights_filename)
  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}

  for i in range(num_images):
    im = cv2.imread(imdb.image_path_at(i))

    _t['im_detect'].tic()
    #得到的就是 是前景的概率scores  与    预测到的boxes边框
    scores, boxes = im_detect(sess, net, im)
    # scores 是rpn scores = self._predictions['cls_prob']  =  每个类别的概率cls_score 讲过soft_max得到
    # pred_boxes
    # pred_boxes = bbox_transform_inv(boxes, box_deltas)
    # 做回归预测   两条路劲rpn得到的的 box_deltas 作为 dx dy dw dh 与 筛选出来的框做回归预测
    # pred_boxes  anchors回归预测后的值
    #return scores, pred_boxes

    _t['im_detect'].toc()

    _t['misc'].tic()

    # skip j = 0, because it's the background class
    for j in range(1, imdb.num_classes):
      inds = np.where(scores[:, j] > thresh)[0]
      cls_scores = scores[inds, j]
      cls_boxes = boxes[inds, j*4:(j+1)*4]
      cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
        .astype(np.float32, copy=False)
      keep = nms(cls_dets, cfg.TEST.NMS)
      cls_dets = cls_dets[keep, :]
      all_boxes[j][i] = cls_dets

    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
      image_scores = np.hstack([all_boxes[j][i][:, -1]
                    for j in range(1, imdb.num_classes)])
      if len(image_scores) > max_per_image:
        image_thresh = np.sort(image_scores)[-max_per_image]
        for j in range(1, imdb.num_classes):
          keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
          all_boxes[j][i] = all_boxes[j][i][keep, :]
    _t['misc'].toc()

    print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
        .format(i + 1, num_images, _t['im_detect'].average_time,
            _t['misc'].average_time))

  det_file = os.path.join(output_dir, 'detections.pkl')
  with open(det_file, 'wb') as f:
    pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)

