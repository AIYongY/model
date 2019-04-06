# coding=utf-8
# for better understanding about yolov3 architecture, refer to this website (in Chinese):
# https://blog.csdn.net/leviopku/article/details/82660381

from __future__ import division, print_function

import tensorflow as tf
slim = tf.contrib.slim

from utils.layer_utils import conv2d, darknet53_body, yolo_block, upsample_layer

class yolov3(object):

    def __init__(self, class_num, anchors, batch_norm_decay=0.9):

        # self.anchors = [[10, 13], [16, 30], [33, 23],
                         # [30, 61], [62, 45], [59,  119],
                         # [116, 90], [156, 198], [373,326]]
        self.class_num = class_num
        self.anchors = anchors
        self.batch_norm_decay = batch_norm_decay

    def forward(self, inputs, is_training=False, reuse=False):
        """
        得到的feature_map就是映射到原图像的网格的尺寸
        :param inputs:
        :param is_training:
        :param reuse:
        :return:
        """
        # the input img_size, form: [height, weight]
        # it will be used later
        self.img_size = tf.shape(inputs)[1:3]#[416,416]
        # set batch norm params
        batch_norm_params = {
            'decay': self.batch_norm_decay,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }
        #这些参数是什么?具体含义

        with slim.arg_scope([slim.conv2d, slim.batch_norm],reuse=reuse):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1)):
                #参数是x  函数里面的内容是   ：后面
                with tf.variable_scope('darknet53_body'):
                    route_1, route_2, route_3 = darknet53_body(inputs)
                # (-1, 52, 52, 256) (-1, 26, 26, 512) (-1, 13, 13, 1024)
                with tf.variable_scope('yolov3_head'):
                    #返回(-1, 13, 13, 512)   (-1, 13, 13, 1024)
                    #里面做了5个 和6个DBL   inter1 5个DBL的输出   net 是6个
                    inter1, net = yolo_block(route_3, 512)
                    #-------------------------------------------------------------------------------------
                    #第一个尺度预测的输出，这里要把normalizer_fn，activation_fn设置为None
                    #输入(-1, 13, 13, 1024)   输出 (-1，13，13，255)  3 * (5 + self.class_num)=255
                    #3 * (5 + self.class_num)的意思是  一个像素点预测三个框，一个框有5个值  需要预测几个类别
                    feature_map_1 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')


                    #-------------------------------------------------------------------------------------
                    #第二个尺度的预测输出代码
                    #inter1输入 (-1, 13, 13, 512)==>256
                    inter1 = conv2d(inter1, 256, 1)
                    #将route_3那条路线输出的图片进行  上采样 使与  route_2尺寸相同 深度不变
                    inter1 = upsample_layer(inter1, route_2.get_shape().as_list())
                    #route_2 = (-1, 26, 26, 512)   (-1, 26, 26, 256) ==> (-1, 26, 26, 768)
                    concat1 = tf.concat([inter1, route_2], axis=3)

                    # 里面做了5个 和6个DBL   inter2 5个DBL的输出   net 是6个
                    inter2, net = yolo_block(concat1, 256)
                    #-------------------------------------------------------------------------------------
                    #第二个尺度预测的输出，这里要把normalizer_fn，activation_fn设置为None
                    #输出(-1, 26, 26, 255)
                    feature_map_2 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')


                    #-------------------------------------------------------------------------------------
                    #第三个尺度
                    inter2 = conv2d(inter2, 128, 1)
                    inter2 = upsample_layer(inter2, route_1.get_shape().as_list())
                    concat2 = tf.concat([inter2, route_1], axis=3)
                    _, feature_map_3 = yolo_block(concat2, 128)
                    #输出尺寸(-1, 52, 52, 256)
                    feature_map_3 = slim.conv2d(feature_map_3, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')
                    #-------------------------------------------------------------------------------------

            return feature_map_1, feature_map_2, feature_map_3

    def reorg_layer(self, feature_map, anchors):
        '''
        这是一个feature_map anchors的输入，每个尺度的输入需要分开输入与分开返回
        # 返回  shape:
        # x_y_offset: [13, 13, 1, 2]
        # boxes: [N, 13, 13, 3, 4], rescaled to the original image scale
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        #boxes 是包含着特征图映射到原图 的中心值+宽度值

        一个尺度预测三个坐标   三个尺度就是9个坐标   缩放做大的 预测框较大，因为感受野较大
        feature_map: a feature_map from [feature_map_1, feature_map_2, feature_map_3] returned
            from `forward` function
        anchors: shape: [3, 2]
        self.anchors = [[10, 13], [16, 30], [33, 23]  在原图中，一个点对应的最小的三个框 需要进行放大 算出9个
        原图中最小的三个框， 除以原图与特征图的比例
        '''
        # NOTE: size in [h, w] format! don't get messed up!
        grid_size = feature_map.shape.as_list()[1:3]  # [13, 13] 输入的尺寸是(-1，13，13，255)
        # the downscale ratio in height and weight  self.img_size = tf.shape(inputs)[1:3]是输入的宽和高
        ratio = tf.cast(self.img_size / grid_size, tf.float32)#得到原图与第一个特征图的尺寸比例
        #[32. 32.]
        # rescale the anchors to the feature_map
        # NOTE: the anchor is in [w, h] format!  anchor[0]是宽 ratio[1]是宽 anchor[1] / ratio[0] 高
        rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]
        #rescaled_anchors = [(0.3125, 0.40625), (0.5, 0.9375), (1.03125, 0.71875)]

        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + self.class_num])

        # split the feature_map along the last dimension
        # shape info: take 416x416 input image and the 13*13 feature_map for example:
        # box_centers: [N, 13, 13, 3, 2] last_dimension: [center_x, center_y]
        # box_sizes: [N, 13, 13, 3, 2] last_dimension: [width, height]
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        box_centers, box_sizes, conf_logits, prob_logits = tf.split(feature_map, [2, 2, 1, self.class_num], axis=-1)
        #虽然说  图片的定义是  heigh width depth  但是里面的值是看  depth  里面的额排列是先  X，后Y
        #(-1，13，13，255)   box_centers, box_sizes里面的具体的值 都是最终特征图的值  需要映射到原图
        #axis=-1意思是最后一个维度 就是深度  [2, 2, 1, self.class_num]是说要怎么分 比如分成 先分2个，2个，1个，最后80个
        #(?, 13, 13, 3, 2) (?, 13, 13, 3, 2) (?, 13, 13, 3, 1) (?, 13, 13, 3, 80)
        #  中心坐标(一个框预测三个坐标，中心点一般是不同的)           尺寸             置信度               分类个数
        box_centers = tf.nn.sigmoid(box_centers)

        # use some broadcast tricks to get the mesh coordinates
        grid_x = tf.range(grid_size[1], dtype=tf.int32)#13 [ 0  1  2  3  4  5  6  7  8  9 10 11 12]
        grid_y = tf.range(grid_size[0], dtype=tf.int32)#13 [ 0  1  2  3  4  5  6  7  8  9 10 11 12]
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        #grid_x[[ 0  1  2  3  4  5  6  7  8  9 10 11 12][ 0  1  2  3  4  5  6  7  8  9 10 11 12]...13个]]
        #grid_y[[ 0  0  0  0  0  0  0  0  0  0  0  0  0] [ 1  1  1  1  1  1  1  1  1  1  1  1  1]...13个]]

        x_offset = tf.reshape(grid_x, (-1, 1))#[[ 0] [ 1][ 2] [ 3]
        y_offset = tf.reshape(grid_y, (-1, 1))#[[ 0] [ 0][ 0] [ 0]
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1) #[[ 0,0] [ 1,0][ 2,0] [ 3,0]
        #x_y_offset的shape (169, 2)
        #意思是最左上角的坐标 加上  x_y_offset 就能得到整个图像的 框框坐标
        # shape: [13, 13, 1, 2]  意思是
        #有13个    [[[[0.  0.]][[1.  0.]] [[2.  0.]][[3.  0.]] [[4.  0.]] ...[[12.  0.]]  13个 shape【1，2】
        x_y_offset = tf.cast(tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)

        # get the absolute box coordinates on the feature_map
        #得到的就是所有中心点的坐标，是基于 网格的坐标
        box_centers = box_centers + x_y_offset
        # rescale to the original image scale
        box_centers = box_centers * ratio[::-1]# 取从后向前（相反）的元素
        #得到的x_y_offset是0123的值，其实就是特征图输出的坐标值  乘以缩放的倍数 就是得到  原始图片的坐标值
        #box_centers这个中心值是经过   sigmoid函数的值也可以看作是特征图中一个亚像素点的特征 *倍数映射到原图


        #----------现在得到了所有网格的中心点的坐标了-------------


        # avoid getting possible nan value with tf.clip_by_value
        #tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。
        # 小于min的让它等于min，大于max的元素的值等于max。  e=2.71828. -6.3
        box_sizes = tf.clip_by_value(tf.exp(box_sizes), 1e-9, 50) * rescaled_anchors
        # rescale to the original image scale
        box_sizes = box_sizes * ratio[::-1]

        # shape: [N, 13, 13, 3, 4]
        # last dimension: (center_x, center_y, w, h)
        boxes = tf.concat([box_centers, box_sizes], axis=-1)

        # shape:
        # x_y_offset: [13, 13, 1, 2]
        # boxes: [N, 13, 13, 3, 4], rescaled to the original image scale
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        #boxes 是包含着特征图映射到原图 的中心值+宽度值
        return x_y_offset, boxes, conf_logits, prob_logits


    def predict(self, feature_maps):
        '''
        返回的是具体的 坐标值 置信度  哪个类别的概率
        boxes是矩形的四个坐标值
        #boxes shape[N, (13*13+26*26+52*52)*3, 4]
        #confs shape: [N, (13*13+26*26+52*52)*3, 1]
        #probs shape: [N, (13 * 13 + 26 * 26 + 52 * 52) * 3, class_num]


        Receive the returned feature_maps from `forward` function,
        the produce the output predictions at the test stage.
        # self.anchors = [[10, 13], [16, 30], [33, 23],
                         # [30, 61], [62, 45], [59,  119],
                         # [116, 90], [156, 198], [373,326]]
        '''
        feature_map_1, feature_map_2, feature_map_3 = feature_maps

        feature_map_anchors = [(feature_map_1, self.anchors[6:9]),
                               (feature_map_2, self.anchors[3:6]),
                               (feature_map_3, self.anchors[0:3])]
        #return x_y_offset, boxes, conf_logits, prob_logits  一个函数reorg_layer   =A
        #三个加在一起[A,B,C]  里面有三个 块矩阵  需要  用for去遍历  一个个的计算
        reorg_results = [self.reorg_layer(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]

        def _reshape(result):
            """
            返回三个尺度的组合concat boxes是
            return boxes, confs, probs
            :param result:
            :return:
            """

            #result[A,B,C]
            # x_y_offset: [13, 13, 1, 2]
            # boxes: [N, 13, 13, 3, 4], rescaled to the original image scale
            # conf_logits: [N, 13, 13, 3, 1]
            # prob_logits: [N, 13, 13, 3, class_num]
            # boxes 是包含着特征图映射到原图 的中心值+宽度值

            x_y_offset, boxes, conf_logits, prob_logits = result
            grid_size = x_y_offset.shape.as_list()[:2]#[:2]是从1开始的  1，2   返回[13,13]
            # boxes[N, 13, 13, 3, 4]
            boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
            conf_logits = tf.reshape(conf_logits, [-1, grid_size[0] * grid_size[1] * 3, 1])
            prob_logits = tf.reshape(prob_logits, [-1, grid_size[0] * grid_size[1] * 3, self.class_num])
            # shape: (take 416*416 input image and feature_map_1 for example)
            # boxes: [N, 13*13*3, 4]
            # conf_logits: [N, 13*13*3, 1]
            # prob_logits: [N, 13*13*3, class_num]
            return boxes, conf_logits, prob_logits

        boxes_list, confs_list, probs_list = [], [], []
        for result in reorg_results:
            #三个加在一起[A,B,C]  里面有三个块矩阵
            boxes, conf_logits, prob_logits = _reshape(result)
            #返回
            # boxes: [N, 13*13*3, 4]
            # conf_logits: [N, 13*13*3, 1]
            # prob_logits: [N, 13*13*3, class_num]
            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)
            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)
        
        # collect results on three scales
        # take 416*416 input image for example:
        # shape: [N, (13*13+26*26+52*52)*3, 4]  有三个 尺度空间 让他们拼接
        boxes = tf.concat(boxes_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, 1]
        confs = tf.concat(confs_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, class_num]
        probs = tf.concat(probs_list, axis=1)

        center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = center_x + width / 2
        y_max = center_y + height / 2

        boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)
        #boxes shape[N, (13*13+26*26+52*52)*3, 4]
        #confs shape: [N, (13*13+26*26+52*52)*3, 1]
        #probs shape: [N, (13 * 13 + 26 * 26 + 52 * 52) * 3, class_num]
        return boxes, confs, probs
    
    def loss_layer(self, feature_map_i, y_true, anchors):
        """
        这里是包含batch的输入
        :param feature_map_i: 输入某个特征图
        :param y_true: 输入真实的标签  不知道他是什么形状的？？？？？？？？？？？？？？？？？？？？？？
        :param anchors: 输入某个特征图的 anchors 三个宽高度值
        :return:
        """

        #feature_map_1 = (-1，13，13，255)
        # size in [h, w] format! don't get messed up!
        grid_size = tf.shape(feature_map_i)[1:3]#[13,13]
        # the downscale ratio in height and weight
        #img_size = [416,416]/[13,13]
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # N: batch_size
        N = tf.cast(tf.shape(feature_map_i)[0], tf.float32)

        x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = self.reorg_layer(feature_map_i, anchors)

        ###########
        # get mask
        ###########
        # shape: take 416x416 input image and 13*13 feature_map for example:
        # object_mask  shape =[N, 13, 13, 3, 1]  最后一个维度应该是  实际分到哪一类的标签值
        object_mask = y_true[..., 4:5]# 4:5表示最后一个维度，  ...的意思取所有的可能组合    4：5就是4
        #相当与三维盒子，俯视往下看的面  ，的每一列
        #object_mask 置信度掩模
        # shape: [N, 13, 13, 3, 4] & [N, 13, 13, 3] ==> [V, 4]
        # V: num of true gt box      0：4 就是0，1，2，3    0:4是中心坐标和，宽高度值
        valid_true_boxes = tf.boolean_mask(y_true[..., 0:4], tf.cast(object_mask[..., 0], 'bool'))
        #返回只有一个矩形框   掩模里面只有一个1
        # shape: [V, 2]
        valid_true_box_xy = valid_true_boxes[:, 0:2]#一个矩形框这个是中心点
        valid_true_box_wh = valid_true_boxes[:, 2:4]#一个矩形框宽高度
        # shape: [N, 13, 13, 3, 2]
        pred_box_xy = pred_boxes[..., 0:2]#所有矩形框的中心
        pred_box_wh = pred_boxes[..., 2:4]#所有矩形框的宽高度

        # calc iou
        # shape: [N, 13, 13, 3, V]  返回所有的置信度
        iou = self.broadcast_iou(valid_true_box_xy, valid_true_box_wh, pred_box_xy, pred_box_wh)

        #返回 shape: [N, 13, 13, 3]   输入 [N, 13, 13, 3, V]
        best_iou = tf.reduce_max(iou, axis=-1) #输出每一个框只有一个最大的置信度  ，有几个框就有几个置信度

        # get_ignore_mask
        ignore_mask = tf.cast(best_iou < 0.5, tf.float32)#意思是判断是否小于0.5，小于0.5的地方返回1，大于返回0
        # shape: [N, 13, 13, 3, 1] 是预测的置信度
        ignore_mask = tf.expand_dims(ignore_mask, -1)

        # get xy coordinates in one cell from the feature_map
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_xy = y_true[..., 0:2] / ratio[::-1] - x_y_offset#在特征图上的中心
        pred_xy = pred_box_xy / ratio[::-1] - x_y_offset#在特征图上的中心

        # get_tw_th
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_tw_th = y_true[..., 2:4] / anchors#在特征图上的宽高度
        pred_tw_th = pred_box_wh / anchors#在特征图上的宽高度
        # for numerical stability
        #函数的目的 ：true_tw_th为0的地方变为1，不为0的地方等于他的原值
        true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0),#等于0的地方不变 不等于0的地方替换
                              x=tf.ones_like(true_tw_th), y=true_tw_th)
        #tf.ones_like(true_tw_th)返回与true_tw_th相同形状为1的矩阵

        pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0),
                              x=tf.ones_like(pred_tw_th), y=pred_tw_th)
        true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
        #clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间
        pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

        # box size punishment: 
        # box with smaller area has bigger weight. This is taken from the yolo darknet C source code.
        # shape: [N, 13, 13, 3, 1]
        box_loss_scale = 2. - (y_true[..., 2:3] / tf.cast(self.img_size[1],
                         tf.float32)) * (y_true[..., 3:4] / tf.cast(self.img_size[0], tf.float32))
        #意思是2减去  标签矩形框的X/图片宽  *   标签矩形框的Y/图片高  为啥？？？？？？？？？？？？？？？？

        ############
        # loss_part
        ############
        # shape: [N, 13, 13, 3, 1]   object_mask是标签的置信度 就是只有一个网格框才有loss 预测3个
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale) / N
        wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale) / N

        # shape: [N, 13, 13, 3, 1]
        conf_pos_mask = object_mask
        #ignore_mask01矩阵     object_mask01矩阵
        #1-标签的置信度就是相当于取反 *  ignore_mask  行*行 就是对应位置相乘
        conf_neg_mask = (1 - object_mask) * ignore_mask  #相当与交集

        #交叉熵是 把所有的 网格 都计算了
        #只计算检测物体的网格
        conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
        #conf_pos_mask是标签的置信度，打标签的矩形框的中心落在网格框  置信度才是1 可以理解为就只有一个1 但是有三个矩形
        #这个交叉熵损失求得的值，并没有相加，计算公式是 ylogx+(1-y)log(1-x)*掩模  掩模就是 有真实标签的网格预测得值才有损失

        #计算没有物体的网格
        conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
        conf_loss = tf.reduce_sum(conf_loss_pos + conf_loss_neg) / N

        # shape: [N, 13, 13, 3, 1]
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[..., 5:], logits=pred_prob_logits)
        class_loss = tf.reduce_sum(class_loss) / N

        return xy_loss, wh_loss, conf_loss, class_loss

    
    def compute_loss(self, y_pred, y_true):
        '''

        这个函数就是计算了所有的损失和
        param:
            y_pred: returned feature_map list by `forward` function: [feature_map_1, feature_map_2, feature_map_3]
            y_true: input y_true by the tf.data pipeline
        '''
        loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
        anchor_group = [self.anchors[6:9], self.anchors[3:6], self.anchors[0:3]]

        # calc loss in 3 scales
        for i in range(len(y_pred)):
            result = self.loss_layer(y_pred[i], y_true[i], anchor_group[i])
            loss_xy += result[0]
            loss_wh += result[1]
            loss_conf += result[2]
            loss_class += result[3]
        total_loss = loss_xy + loss_wh + loss_conf + loss_class
        return [total_loss, loss_xy, loss_wh, loss_conf, loss_class]



    def broadcast_iou(self, true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):
        '''

                     自己的理解：            这里的输入不包括batch   就是计算一张图内的置信度
        返回的是 预测的框框与自己打标签组合的置信值  真实边框的面积/真实边框与预测边框所占的面积
        maintain an efficient way to calculate the ios matrix between ground truth true boxes and the predicted boxes
        note: here we only care about the size match
        '''
        # shape:
        # true_box_??: [V, 2] V = N*13*13*3  这是一个特征图
        # pred_box_??: [N, 13, 13, 3, 2]

        # shape: [N, 13, 13, 3, 1, 2]
        pred_box_xy = tf.expand_dims(pred_box_xy, -2)#在倒数第二个维度增加一个维度
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)

        # shape: [1, V, 2]
        true_box_xy = tf.expand_dims(true_box_xy, 0)#第一个维度增加
        true_box_wh = tf.expand_dims(true_box_wh, 0)

        # [N, 13, 13, 3, 1, 2] & [1, V, 2] ==> [N, 13, 13, 3, V, 2]  两个矩形求他的坐标得最大值，就是求他的交集坐标
        intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2.,
                                    true_box_xy - true_box_wh / 2.)
        #作用就是用一个边框 去遍历所有的边框的 进行计算
        #tf.maximum这个函数是求最大值，但是不是单单的相对应的形状去比较 ，而是用a的元素去遍历b的所有元素求得最大值
        #所以最终得到的维度数是V  输入的矩阵是有要求的  比如 [N, 13, 13, 3, 1, 2]倒数第二个的维度必须是1，
        #[1, V, 2]  两个矩阵的最后一个维度必须相同
        intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,  #矩形框的右下角
                                    true_box_xy + true_box_wh / 2.)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.) #求出交集得 宽高度

        # shape: [N, 13, 13, 3, V]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]  #交叉的面积
        # shape: [N, 13, 13, 3, 1]
        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
        # shape: [1, V]
        true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]

        # [N, 13, 13, 3, V] N个图片 13行 13列  的每一个点 预测 的3个框 每一个边框与图片所有边框的置信度
        iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)
        #   [N, 13, 13, 3, V] /  [N, 13, 13, 3, 1] + shape: [1, V] - [N, 13, 13, 3, V]
        return iou
