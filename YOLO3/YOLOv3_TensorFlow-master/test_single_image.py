# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box

from model import yolov3

parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
#https://github.com/wizyoung/YOLOv3_TensorFlow
# parser.add_argument("input_image", type=str,#default="C:/picture/xing2.png",
#                     help="The path of the input image.")

parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")

parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--class_name_path", type=str, default="./data/coco.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3.ckpt",
                    help="The path of the weights to restore.")
args = parser.parse_args()
#这个函数是打开anchors  text文本  9个值
args.anchors = parse_anchors(args.anchor_path)

#函数是把 txt 的文本标签转化为文本   比如：flow :1
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)
#生成80 个颜色
# img_ori = cv2.imread(args.input_image)
img_ori = cv2.imread("C:/picture/zjz.png")

height_ori, width_ori = img_ori.shape[:2] #输入图像的尺寸
img = cv2.resize(img_ori, tuple(args.new_size))
cv2.imshow("aa",img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.asarray(img, np.float32)
#asarray不会构建新的变量 占内存 array会

img = img[np.newaxis, :] / 255.
#np.newaxis增加一个维度
print("aaaaaaaaaaaaaaaaaaaa",img.shape)

with tf.Session() as sess:#416 416   1  416 416 3
    input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
    yolo_model = yolov3(args.num_class, args.anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)
        #返回三个feature map

        # 传进去三个feature map
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
    """
    # 传进来时前向传播的三个feature map
    返回的是具体的
    坐标值
    置信度
    哪个类别的概率
    boxes是矩形的四个坐标值
    # boxes shape[N, (13*13+26*26+52*52)*3, 4]
    # confs shape: [N, (13*13+26*26+52*52)*3, 1]
    # probs shape: [N, (13 * 13 + 26 * 26 + 52 * 52) * 3, class_num]
    """
    pred_scores = pred_confs * pred_probs

    #返回[N, (13 * 13 + 26 * 26 + 52 * 52) * 3, class_num]   就是行的一个值乘以所有列
    #boxes shape[N, (13*13+26*26+52*52)*3, 4]
    #pred_scores[N, (13 * 13 + 26 * 26 + 52 * 52) * 3, class_num]
    # args.num_class=80
    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=30, score_thresh=0.4, iou_thresh=0.5)
    """
        这里只能输入一张图片  输入一张图片 检测到的 所有的值  然后返回非极大值抑制nms帅选出来的值
    返回 return boxes, score, label  帅选出来的的值   边框是具体的坐标值
    """
    saver = tf.train.Saver()
    saver.restore(sess, args.restore_path)#
    print("aaaaaaaaaaaaaaaaaaaa",img.shape)
    boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
    #这里并没有将图片缩放，但为啥可以输入  ？它里面自己实现缩放？？
#width_ori height_ori是输入图像的尺寸   args.new_size是网络要求的尺寸 416*416
    # rescale the coordinates to the original image  比例转换  缩放到原来的尺寸
    boxes_[:, 0] *= (width_ori/float(args.new_size[0]))
    boxes_[:, 2] *= (width_ori/float(args.new_size[0]))
    boxes_[:, 1] *= (height_ori/float(args.new_size[1]))
    boxes_[:, 3] *= (height_ori/float(args.new_size[1]))

    print("box coords:")
    print(boxes_)
    print('*' * 30)
    print("scores:")
    print(scores_)
    print('*' * 30)
    print("labels:")
    print(labels_)

    for i in range(len(boxes_)):#遍历检测到的边框的个数
        x0, y0, x1, y1 = boxes_[i]
        #boxes_是sess后检测不到的边框
        #img_ori原图   [x0, y0, x1, y1]原图中的矩形框   classes80个标签的名字 labels_下标
        plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]], color=color_table[labels_[i]])
    cv2.imshow('Detection result', img_ori)
    cv2.imwrite('detection_result.jpg', img_ori)
    cv2.waitKey(0)
