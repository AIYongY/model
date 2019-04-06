# coding: utf-8

import numpy as np
import tensorflow as tf
import random

from tensorflow.core.framework import summary_pb2


def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])


def parse_anchors(anchor_path):
    '''
    #这个函数是打开anchors  text文本  9个值
    parse anchors.
    returned data: shape [N, 2], dtype float32
    '''
    anchors = np.reshape(np.asarray(open(anchor_path, 'r').read().split(','), np.float32), [-1, 2])
    return anchors


def read_class_names(class_name_path):
    """
    函数是把 txt 的文本标签转化为文本   比如：flow :1
    :param class_name_path:
    :return:
    """
    #enumerate在字典上是枚举、列举的意思
    names = {}
    with open(class_name_path, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
            #str.strip([chars]);chars -- 移除字符串头尾指定的字符序列。
    return names


def shuffle_and_overwrite(file_name):#file_name 图片名字文件txt
    #这个函数 就是为了打乱顺序
    content = open(file_name, 'r').readlines()
    #readlines() 方法用于读取所有行  图片名字
    random.shuffle(content)
    #shuffle() 方法将序列的所有元素随机排序。
    #shuffle()是不能直接访问的，需要导入 random 模块，然后通过 random 静态对象调用该方法
    with open(file_name, 'w') as f:
        for line in content:
            f.write(line)


def update_dict(ori_dict, new_dict):
    if not ori_dict:
        return new_dict
    for key in ori_dict:
        ori_dict[key] += new_dict[key]
    return ori_dict


def list_add(ori_list, new_list):
    for i in range(len(ori_list)):
        ori_list[i] += new_list[i]
    return ori_list


def load_weights(var_list, weights_file):
    """
    Loads and converts pre-trained weights.
    param:
        var_list: list of network variables.
        weights_file: name of the binary file.
    """
    with open(weights_file, "rb") as fp:
        np.fromfile(fp, dtype=np.int32, count=5)
        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        # do something only if we process conv layer
        if 'Conv' in var1.name.split('/')[-2]:
            # check type of next layer
            if 'BatchNorm' in var2.name.split('/')[-2]:
                # load batch norm params
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))
                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            elif 'Conv' in var2.name.split('/')[-2]:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr +
                                       bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
                # we loaded 1 variable
                i += 1
            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(
                tf.assign(var1, var_weights, validate_shape=True))
            i += 1

    return assign_ops


def config_learning_rate(args, global_step):
    # parser.add_argument("--lr_type", type=str, default='fixed',
    #                     help="The learning rate type. Chosen from [fixed, exponential]")
    # parser.add_argument("--learning_rate_init", type=float, default=1e-3,
    #                     help="The initial learning rate.")
    # parser.add_argument("--lr_decay_freq", type=int, default=1000,
    #                     help="The learning rate decay frequency. Used when chosen exponential lr_type.")
    # parser.add_argument("--lr_decay_factor", type=float, default=0.96,
    #                     help="The learning rate decay factor. Used when chosen exponential lr_type.")
    #parser.add_argument("--lr_lower_bound", type=float, default=1e-6,
                    #help="The minimum learning rate. Used when chosen exponential lr type.")
    if args.lr_type == 'exponential':#global_step是为了纪律当前训练的几步
        lr_tmp = tf.train.exponential_decay(args.learning_rate_init, global_step, args.lr_decay_freq,
                                            args.lr_decay_factor, staircase=True, name='exponential_learning_rate')
        return tf.maximum(lr_tmp, args.lr_lower_bound)#lr_lower_bound为要求的最小的学习率
    elif args.lr_type == 'fixed':
        return tf.convert_to_tensor(args.learning_rate_init, name='fixed_learning_rate')
    #tf.convert_to_tensor用于将不同数据变成张量：比如可以让数组变成张量、也可以让列表变成张量。
    else:
        raise ValueError('Unsupported learning rate type!')


def config_optimizer(optimizer_name, learning_rate, decay=0.9, momentum=0.9):
    #"--optimizer_name", type=str, default='adam',
    if optimizer_name == 'momentum':
        return tf.train.MomentumOptimizer(learning_rate, momentum=momentum)
    elif optimizer_name == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=momentum)
    elif optimizer_name == 'adam':
        return tf.train.AdamOptimizer(learning_rate)
    elif optimizer_name == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Unsupported optimizer type!')