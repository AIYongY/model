# coding: utf-8

from __future__ import division, print_function

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

def conv2d(inputs, filters, kernel_size, strides=1):
    """
    这个函数作用
    if（strides=1) kernel_size=7  假设输入2，24，24，3   总填充6   2，30，30，3  VALID卷积
       ( 30-7）/1+1=23+1=24  最终尺寸不变  为啥要这样写，因为这样写，运行的速度较快
   if（strides>1)  那么这个_fixed_padding（）填充函数就是不执行，就是一个slim.conv2d（）函数进行
        卷积降维  操作

    :param inputs: 输入的图片数据
    :param filters: 输入卷积需要的深度
    :param kernel_size: 卷积核的尺寸
    :param strides:卷积的步长
    :return:
    """
    def _fixed_padding(inputs, kernel_size):
        #填充padding  比如输入 卷积核尺寸  7   7-1=6  6/2=3   就是在输入的图像的宽和高上下左右填充 3
        #填充的目的是为了：
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
        return padded_inputs
    if strides > 1: 
        inputs = _fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides,
                         padding=('SAME' if strides == 1 else 'VALID'))
    return inputs

def darknet53_body(inputs):
    """
    全程不用加激活函数和BN  在前面arg_scope已经定义好了，
    :param i nputs:
    :return:  (-1, 52, 52, 256) (-1, 26, 26, 512) (-1, 13, 13, 1024)

    """
    def res_block(inputs, filters):
        """
        这里stridesz只能等于1
        比如输入 (-1, 207， 207,64)  strides=1
        输出  (-1, 207， 207,64)    尺寸不变
        :param inputs:输入的batch  (-1,413，413,3)图像
        :param filters:输入的图像深度   32
        :return:
        """
        shortcut = inputs #(-1,416,416,3)
        net = conv2d(inputs, filters * 1, 1)
        net = conv2d(net, filters * 2, 3)

        net = net + shortcut
        return net
    
    # first two conv2d layers  darknet53_body最开始的两个卷积 没有残差项
    net = conv2d(inputs, 32,  3, strides=1)
    net = conv2d(net, 64,  3, strides=2)

    # res_block * 1 输入  残差块 有1*1  与  3*3   和  +
    net = res_block(net, 32)

    net = conv2d(net, 128, 3, strides=2)#作用降维

    # res_block * 2
    for i in range(2):
        net = res_block(net, 64)

    net = conv2d(net, 256, 3, strides=2)

    # res_block * 8
    for i in range(8):
        net = res_block(net, 128)

    route_1 = net
    net = conv2d(net, 512, 3, strides=2)

    # res_block * 8
    for i in range(8):
        net = res_block(net, 256)

    route_2 = net
    net = conv2d(net, 1024, 3, strides=2)

    # res_block * 4
    for i in range(4):
        net = res_block(net, 512)
    route_3 = net

    return route_1, route_2, route_3


def yolo_block(inputs, filters):
    """
    :param inputs:  (-1, 13, 13, 1024)
    :param filters: 512    其中strides默认为  1  就是说尺寸是不变的  变化的是深度
    :return:
    """
    net = conv2d(inputs, filters * 1, 1)    #512
    net = conv2d(net, filters * 2, 3)       #1024
    net = conv2d(net, filters * 1, 1)       #512
    net = conv2d(net, filters * 2, 3)       #1024
    net = conv2d(net, filters * 1, 1)       #512
    route = net                             #512
    net = conv2d(net, filters * 2, 3)       #1024
    return route, net                       #(-1, 13, 13, 512)   (-1, 13, 13, 1024)


def upsample_layer(inputs, out_shape):
    """
    函数的作用就是  上采样  改变图片的尺寸  使图片的尺寸增大
    :param inputs: 输入的图片
    :param out_shape: 需要将图片上采样到什么尺寸
    :return:
    """
    new_height, new_width = out_shape[1], out_shape[2]
    # NOTE: here height is the first
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), align_corners=True, name='upsampled')
    return inputs

#自己加的测试尺寸代码
# inputs = tf.random_normal(shape=((1,416,416,3)))
#
# route_1, route_2, route_3 = darknet53_body(inputs)
# print(route_1.get_shape(),route_2.get_shape(),route_3.get_shape())
