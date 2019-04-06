import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

import numpy as np


class  AA():
    # 没看
    def create_architecture(self, mode, num_classes, tag=None, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
        self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])  # 由于图像宽高不定，因而第二维和第三维都是None
        self._im_info = tf.placeholder(tf.float32, shape=[3])  # 图像信息，高、宽、缩放到宽为600或者高为1000的最小比例
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None,
                                                           5])  # ground truth框的信息。前四个为位置信息，最后一个为该框对应的类别（见roi_data_layer/minibatch.py/get_minibatch）
        self._tag = tag

        self._num_classes = num_classes
        self._mode = mode
        self._anchor_scales = anchor_scales
        self._num_scales = len(anchor_scales)

        self._anchor_ratios = anchor_ratios
        self._num_ratios = len(anchor_ratios)

        self._num_anchors = self._num_scales * self._num_ratios  # self._num_anchors=9

        training = mode == 'TRAIN'
        testing = mode == 'TEST'

        weights_regularizer = tf.contrib.layers.l2_regularizer(
            cfg.TRAIN.WEIGHT_DECAY)  # handle most of the regularizers here
        if cfg.TRAIN.BIAS_DECAY:
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        # list as many types of layers as possible, even if they are not used now
        with arg_scope(
                [slim.conv2d, slim.conv2d_in_plane, slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                weights_regularizer=weights_regularizer, biases_regularizer=biases_regularizer,
                biases_initializer=tf.constant_initializer(0.0)):
            # rois：256个archors的类别（训练时为每个archors的类别，测试时全0）
            # cls_prob：256个archors每一类别的概率
            # bbox_pred：预测位置信息的偏移
            rois, cls_prob, bbox_pred = self._build_network(training)

        layers_to_output = {'rois': rois}

        for var in tf.trainable_variables():
            self._train_summaries.append(var)

        if testing:
            stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
            means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
            self._predictions["bbox_pred"] *= stds  # 训练时_region_proposal中预测的位置偏移减均值除标准差，因而测试时需要反过来。
            self._predictions["bbox_pred"] += means
        else:
            self._add_losses()
            layers_to_output.update(self._losses)

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

        return layers_to_output

    def _build_network(self, is_training=True):
        """
        没看完


        _build_network用于创建网络
    _build_network = _image_to_head + //得到输入图像的特征
    _anchor_component + //得到所有可能的archors在原始图像中的坐标（可能超出图像边界）及archors的数量
    _region_proposal + //对输入特征进行处理，最终得到2000个archors（训练）或300个archors（测试）
    _crop_pool_layer + //将256个archors裁剪出来，并缩放到7*7的固定大小，得到特征
    _head_to_tail + //将256个archors的特征增加fc及dropout，得到4096维的特征
    _region_classification // 增加fc层及dropout层，用于rcnn的分类及回归
    总体流程：网络通过vgg1-5得到特征net_conv后，送入rpn网络得到候选区域archors，去除超出图像边界的archors并选出2000个archors用于训练rpn网络（300个用于测试）。并进一步选择256个archors（用于rcnn分类）。之后将这256个archors的特征根据rois进行裁剪缩放及pooling，得到相同大小7*7的特征pool5，pool5通过两个fc层得到4096维特征fc7，fc7送入_region_classification（2个并列的fc层），得到21维的cls_score和21*4维的bbox_pred。
        :param self:
        :param is_training:
        :return:
        """
        # train truncated
        # tf.truncated_normal(shape, mean,stddev):
        # shape表示生成张量的维度，mean是均值，stddev是标准差。这个函数产生正太分布，均值和标准差自己设定。
        # 这是一个截断的产生正太分布的函数，就是说产生正太分布的值如果与均值的差值大于两倍的标准差，
        # 那就重新生成。和一般的正太分布的产生随机数据比起来，这个函数产生的随机数与均值的差距不会超过两倍的标准差，
        # 但是一般的别的函数是可能的

        # 判断用哪个标准差初始化
        if cfg.TRAIN.TRUNCATED:  # select initializers
            initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
        else:
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

        # 这是个函数 是VGG16的conv5_3输出conv5_3
        net_conv = self._image_to_head(is_training)  # 得到vgg16的conv5_3
        with tf.variable_scope(self._scope, self._scope):
            self._anchor_component()  # 通过特征图及相对原始图像的缩放倍数_feat_stride得到所有archors的起点及终点坐标

 #——————————————————————————————————————-

            rois = self._region_proposal(net_conv, is_training,
                                         initializer)  # 通过rpn网络，得到256个archors的类别（训练时为每个archors的类别，测试时全0）及位置（后四维）
            pool5 = self._crop_pool_layer(net_conv, rois,
                                          "pool5")  # 对特征图通过rois得到候选区域，并对候选区域进行缩放，得到14*14的固定大小，进一步pooling成7*7大小

        fc7 = self._head_to_tail(pool5, is_training)  # 对固定大小的rois增加fc及dropout，得到4096维的特征，用于分类及回归
        with tf.variable_scope(self._scope, self._scope):
            cls_prob, bbox_pred = self._region_classification(fc7, is_training, initializer,
                                                              initializer_bbox)  # 对rois进行分类，完成目标检测；进行回归，得到预测坐标

        self._score_summaries.update(self._predictions)

        # rois：256个archors的类别（训练时为每个archors的类别，测试时全0）
        # cls_prob：256个archors每一类别的概率
        # bbox_pred：预测位置信息的偏移
        return rois, cls_prob, bbox_pred

    def _image_to_head(self, is_training, reuse=None):
        """
        已看
        构建VGG16网路
        :param self:
        :param is_training:
        :param reuse:
        :return:
        """
        with tf.variable_scope(self._scope, self._scope, reuse=reuse):
            # 自己的理解slim.repeat就是用来多次重复某个函数
            # repeat(inputs输入, repetitions重复次数, layer什么, *args, **kwargs):
            net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3], trainable=False, scope='conv1')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=False, scope='conv2')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], trainable=is_training, scope='conv3')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=is_training, scope='conv4')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=is_training, scope='conv5')

        self._act_summaries.append(net)
        self._layers['head'] = net

        return net

    def _anchor_component(self):
        """
         self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
          # 由于图像宽高不定，因而第二维和第三维都是None
        self._im_info = tf.placeholder(tf.float32, shape=[3])
         # 图像信息，高、宽、缩放到宽为600或者高为1000的最小比例
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None,

        _anchor_component：用于得到所有可能的archors在原始图像中的坐标（可能超出图像边界）
        及archors的数量（特征图宽*特征图高*9
        """
        with tf.variable_scope('ANCHOR_' + self._tag) as scope:
            # tf.ceil(x, name=None)          # 向上取整
            height = tf.to_int32(tf.ceil(self._im_info[0] / np.float32(self._feat_stride[0])))
            # 图像经过vgg16得到特征图的宽高
            width = tf.to_int32(tf.ceil(self._im_info[1] / np.float32(self._feat_stride[0])))
            if cfg.USE_E2E_TF:
                # 通过特征图宽高、_feat_stride（特征图相对原始图缩小的比例）及_anchor_scales、_anchor_ratios得到原始图像上
                # 所有可能的archors（坐标可能超出原始图像边界）和archor的数量
                anchors, anchor_length = generate_anchors_pre_tf(height, width, self._feat_stride, self._anchor_scales,
                                                                 self._anchor_ratios)
            else:
                anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                                    [height, width, self._feat_stride, self._anchor_scales,
                                                     self._anchor_ratios], [tf.float32, tf.int32],
                                                    name="generate_anchors")
            anchors.set_shape([None, 4])  # 起点坐标，终点坐标，共4个值
            anchor_length.set_shape([])
            self._anchors = anchors
            self._anchor_length = anchor_length

     def generate_anchors_pre_tf(self,height, width, feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
            """
            注意：返回的是 VGG16 con2d5_2输出的特征点，对应在原图中 所有的矩形框的坐标值
            形状是  (width*height)*9*4
            :param width: VGG16输出的宽度
            :param feat_stride: 这个是V166网络输入如输出，图像尺寸变换的  比
            :param anchor_scales: 锚点比例
            :param anchor_ratios:锚点比率  一块的锚点是三个  三个的比率
            :return: return tf.cast(anchors_tf, dtype=tf.float32), length
            """
            #VGG16特征图的对应   0对应0  1对应16
            shift_x = tf.range(width) * feat_stride
            # 得到  所有  archors在原始图像的起始x坐标：(0,feat_stride,2*feat_stride...)
            shift_y = tf.range(height) * feat_stride
            # 得到  所有  archors在原始图像的起始y坐标：(0,feat_stride,2*feat_stride...)

            #输入两个一维的矩阵   meshgrid复制N行，N行由另外一个参数的行数决定
            shift_x, shift_y = tf.meshgrid(shift_x,#列数width
                                           shift_y)#行数height
            #得到在原始图相所有的原始X Y坐标   Y是要 转置的
            # shift_x：height个(0,feat_stride,2*feat_stride...);shift_x列  shift_y行  shift_y行个0
            # shift_y：width个(0,feat_stride,2*feat_stride...)'  shift_x列 shift_y行  shift_x列个0

            sx = tf.reshape(shift_x, shape=(-1,)) #意思是变成1维度
            # 0,feat_stride,2*feat_stride...0,feat_stride,2*feat_stride...0,feat_stride,2*feat_stride...
            #0,feat_stride,2*feat_stride... 有shift_y个
            sy = tf.reshape(shift_y, shape=(-1,))
            # 0,0,0...feat_stride,feat_stride,feat_stride...2*feat_stride,2*feat_stride,2*feat_stride..
            #shift_x列个0
                        #总共有   shift_y*shift_x   个值


            # tf.transpose是转置   tf.stack 是矩阵拼接 ，并不是连在一起  ，多个矩阵用一个矩阵表示，里面由若干矩阵
            #四个1维度拼接  axis=0 按照行拼接 就是添加行   shape = [4, shift_y*shift_x ]
            #转置  shape = [shift_y*shift_x ，4]
            #例子 [0，0，0，0]  [feat_stride，0，feat_stride，0] [2*feat_stride，0，,2*feat_stride，0]
            #shifts就是用来偏移的 ，  例如：在同一行feat_stride，0左上角的偏移  右上角的偏移一样  就是往下移动
            #第一个代表高度  第二个是宽度 ，
            shifts = tf.transpose(tf.stack([sx, sy, sx, sy]))  # width*height个四位矩阵
            K = tf.multiply(width, height)  # 特征图总共像素数

            #perm=(1, 0, 2) 这个参数是维度的调换 排列
            #1 * (width * height) * 4
            shifts = tf.transpose(tf.reshape(shifts, shape=[1, K, 4]),
                                  perm=(1, 0, 2))  # 增加一维，变成1*(width*height)*4矩阵，而后变换维度为(width*height)*1*4矩阵


            """
             直接调用这个函数generate_anchors 就可以 返回 VGG15 conv5_2第一个点 对应再原始图像的
            9个矩形框   (9,4)  anchors.shape
                     """
            # anchor_scales=(8, 16, 32) anchor_ratios=[0,0,15,15]
            anchors = generate_anchors(ratios=np.array(anchor_ratios),
                                       scales=np.array(anchor_scales))
            # 9*4矩阵    得到9个archors的在原始图像中的四个坐标（放大比例默认为16）

            A = anchors.shape[0]  # A=9
            # anchors增加维度为1*9*4
            anchor_constant = tf.constant(anchors.reshape((1, A, 4)), dtype=tf.int32)
            length = K * A  # 总共的archors的个数（每个点对应A=9个archor，共K=height*width个点）
            # 1*9*4的base archors和(width*height)*1*4的偏移矩阵进行broadcast相加，得到(width*height)*9*4，
            # 并改变形状为(width*height*9)*4，得到所有的archors的四个坐标
            anchors_tf = tf.reshape(tf.add(anchor_constant, shifts), shape=(length, 4))

            return tf.cast(anchors_tf, dtype=tf.float32), length



            # np.arange(3, 6))  arange是np的函数 range是for的函数
        #scales=2 ** np.arange(3, 6))         scales=2 ** np.arange(3, 6) =  [ 8 16 32]

def generate_anchors(base_size=16,ratios=[0.5,1,2],scales=2 ** np.arange(3, 6)):
        """
         9*4矩阵
         直接调用这个函数 就可以 返回 VGG15 conv5_2第一个点 对应再原始图像的
         9个矩形框
         """
        #base_anchor = (0, 0, 15, 15)
        base_anchor = np.array([1, 1, base_size, base_size]) - 1  # base archor的四个坐标
        ratio_anchors = _ratio_enum(base_anchor, ratios)
        # 三个基准锚点的坐标4   3 *4
        # 通过ratio得到3个archors的坐标（3*4矩阵）
        anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales) for i in
                             range(ratio_anchors.shape[0])])  # 3遍历行数
        # _scale_enum  3*4矩阵变成9*4矩阵，得到9个archors的坐标
        """arr1=np.array([1,2,3])
            arr2=np.array([4,5,6])
            print np.vstack((arr1,arr2))     
            output [[1 2 3]
                    [4 5 6]]
        """
        return anchors

    #base_anchor=anchor=[0, 0, 15, 15])
def _whctrs(anchor):
        """
        得到中心坐标，与宽高   ，可以输入任何的   点坐标
        返回base_anchor[0, 0, 15, 15]的宽16，宽16，中心x7.5 中心y7.5  是一个宽度值，中心是坐标
         #base_anchor=anchor=[0, 0, 15, 15])
        Return width, height, x center, and y center for an anchor (window). """
        w = anchor[2] - anchor[0] + 1  # 宽
        h = anchor[3] - anchor[1] + 1  # 宽
        x_ctr = anchor[0] + 0.5 * (w - 1)  # 中心x
        y_ctr = anchor[1] + 0.5 * (h - 1)  # 中心y
        return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
        """
        该函数原理
        输入 3个基准的锚，宽度  高度  中心点坐标 X,Y  X+-宽高的一半就是坐标点
        返回  3*4的坐标点  就是三个基准锚 的坐标点
        [[-3.5  2.  18.5 13. ]
        [ 0.   0.  15.  15. ]
         [ 2.5 -3.  12.5 18. ]]
array([0, 1, 2])
>> x.shape
(3,)
>> x[:, np.newaxis]
array([[0],
       [1],
       [2]])
        """
        # ws, hs, x_ctr, y_ctr =[23. 16. 11.] [12. 16. 22.] 7.5 7.5
        ws = ws[:, np.newaxis]  # 3维向量变成3*1矩阵  相当于reshape(:,1) 一维度变成2维
        hs = hs[:, np.newaxis]  # 3维向量变成3*1矩阵   np.newaxis就是增加一个维度的意思
        anchors = np.hstack(
            (x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1), x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)))  # 3*4矩阵
        """anchors = 
        [[-3.5  2.  18.5 13. ]
        [ 0.   0.  15.  15. ]
         [ 2.5 -3.  12.5 18. ]]"""
        return anchors

def _ratio_enum(anchor, ratios):  # 缩放比例为像素总数的比例，而非单独宽或者高的比例
        """
        anchor = [0,0,15,15]
        返回
        三个基准锚点的坐标4   3 *4
        anchors =
        [[-3.5  2.  18.5 13.]
        [0.   0.  15.  15.]
        [2.5 - 3.  12.5  18.]]
         """
        w, h, x_ctr, y_ctr = _whctrs(anchor)  # 得到bace_anchor(0，0，15，15)中心位置和宽高
        #16,16,7.5,7.5
        size = w * h  # 总共像素数
        size_ratios = []

        #自己修改的写法
        for i in range(len(ratios)):
            size_ratio = size / ratios[i]
            size_ratios.append(size_ratio)
        #ratios=[0.5,1, 2]
        #-----------------------------------------------
        # size_ratios = size / ratios#是不是错误的写法 int/list
        #-----------------------------------------------

        # 缩放比例  anchor=(0,0,15,15) 的总像素个数  进行缩放
        #round() 方法返回浮点数x的四舍五入值。np.sqrt平方根 根号x
        ws = np.round(np.sqrt(size_ratios))  # 缩放后的宽,3维向量(值由大到小)
        hs = np.round(ws * ratios)  # 缩放后的高，两个3维向量对应元素相乘，为3维向量（值由小到大）
        #ws, hs, x_ctr, y_ctr =[23. 16. 11.] [12. 16. 22.] 7.5 7.5
        anchors = _mkanchors(ws, hs, x_ctr, y_ctr)  # 根据中心及宽高得到3个archors的四个坐标
        """
        三个基准锚点的坐标4   3 *4
        anchors =
        [[-3.5  2.  18.5 13.]
        [0.   0.  15.  15.]
        [2.5 - 3.  12.5  18.]]
         """
        return anchors

def _scale_enum(anchor, scales):
        """
        一个个的输入  也就是 一个个的返回
        一个个返回放大后[ 8 16 32]   的坐标值   返回去  需要组合拼接在一块

        坐标值   输入的是 3*4的坐标值  矩形框的坐标值  1*4一个个的输入
        宽度值   转化为  宽度高度值  中心点 值
        宽度值   宽度 高度 放大倍数
        坐标值   然后再  转化  为坐标值

        作用就是  放大 宽和高的倍数  然后调用_mkanchors
        scales = [8 16 32]
        Enumerate a set of anchors for each scale wrt an anchor. """
        w, h, x_ctr, y_ctr = _whctrs(anchor)  # 得到中心位置和宽高 这是
        ws = w * scales  # 得到宽的放大倍数
        hs = h * scales  # 得到宽的放大倍数
        anchors = _mkanchors(ws, hs, x_ctr, y_ctr)  # 根据中心及宽高得到3个archors的四个坐标
        return anchors

class BB():
        #cfg.RPN_CHANNELS=512
    def _region_proposal(self, net_conv, is_training, initializer):  # 对输入特征图进行处理
        rpn = slim.conv2d(net_conv, cfg.RPN_CHANNELS, [3, 3], trainable=is_training, weights_initializer=initializer,
                          scope="rpn_conv/3x3")  # 3*3的conv，作为rpn网络
        self._act_summaries.append(rpn)
        #_act_summaries类初始化时定义的list
        #_num_anchors=9
        rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                    weights_initializer=initializer,  # _num_anchors为9
                                    padding='VALID', activation_fn=None,
                                    scope='rpn_cls_score')

        # 1*1的conv，得到每个位置的9个archors分类特征-1*？*？*(9*2)（二分类），判断当前archors是正样本还是负样本
        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
        # 1*？*？*18==>1*(?*9)*?*2
        # (1, ?, ?, 2)  就是转化为这种形状
        rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape,
                                                   "rpn_cls_prob_reshape")
        # 以最后一维为特征长度，得到所有特征的概率1*(?*9)*?*2


        #————————————————————————————————————————————

        rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1,
                                 name="rpn_cls_pred")
        # 得到每个位置的9个archors预测的类别，(1*?*9*?)的列向量
        #————————————————————————————————————————————


        # 变换会原始维度1*(?*9)*?*2==>1*?*?*(9*2)
        rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2,
                                           "rpn_cls_prob")

        #这里是另一条路径了
        rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                    weights_initializer=initializer,
                                    padding='VALID', activation_fn=None,
                                    scope='rpn_bbox_pred')  # 1*1的conv，每个位置的9个archors回归位置偏移1*？*？*(9*4)
        if is_training:
            # 每个位置的9个archors的类别概率和每个位置的9个archors的回归位置偏移得到post_nms_topN=2000个archors的位置（包括全0的batch_inds）及为1的概率
            rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")  # rpn_labels：特征图中每个位置对应的是正样本、负样本还是不关注
            with tf.control_dependencies(
                    [rpn_labels]):  # Try to have a deterministic order for the computing graph, for reproducibility
                rois, _ = self._proposal_target_layer(rois, roi_scores,
                                                      "rpn_rois")  # 通过post_nms_topN个archors的位置及为1（正样本）的概率得到256个rois（第一列的全0更新为每个archors对应的类别）及对应信息
        else:
            if cfg.TEST.MODE == 'nms':
                # 每个位置的9个archors的类别概率和每个位置的9个archors的回归位置偏移得到post_nms_topN=300个archors的位置（包括全0的batch_inds）及为1的概率
                rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            elif cfg.TEST.MODE == 'top':
                rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            else:
                raise NotImplementedError

        self._predictions["rpn_cls_score"] = rpn_cls_score  # 每个位置的9个archors是正样本还是负样本
        self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape  # 每个archors是正样本还是负样本
        self._predictions["rpn_cls_prob"] = rpn_cls_prob  # 每个位置的9个archors是正样本和负样本的概率
        self._predictions["rpn_cls_pred"] = rpn_cls_pred  # 每个位置的9个archors预测的类别，(1*?*9*?)的列向量
        self._predictions["rpn_bbox_pred"] = rpn_bbox_pred  # 每个位置的9个archors回归位置偏移
        self._predictions["rois"] = rois  # 256个archors的类别（第一维）及位置（后四维）

        return rois  # 返回256个archors的类别（第一维，训练时为每个archors的类别，测试时全0）及位置（后四维）

    def _reshape_layer(self, bottom, num_dim, name):
        #(1, ?, 24, 2)  就是转化为这种形状
        input_shape = tf.shape(bottom)
        with tf.variable_scope(name) as scope:
            to_caffe = tf.transpose(bottom, [0, 3, 1, 2])  # NHWC（TF数据格式）变成NCHW（caffe格式）
            reshaped = tf.reshape(to_caffe, tf.concat(axis=0, values=[[1, num_dim, -1], [
                input_shape[2]]]))  #1,2,-1,行数
            # 1*(num_dim*9)*?*?==>1*num_dim*(9*?)*?  或 1*num_dim*(9*?)*?==>1*(num_dim*9)*?*?
            to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
            return to_tf

    def _softmax_layer(self, bottom, name):
        if name.startswith('rpn_cls_prob_reshape'):  # bottom：1*(?*9)*?*2
            input_shape = tf.shape(bottom)
            bottom_reshaped = tf.reshape(bottom, [-1, input_shape[
                -1]])  # 只保留最后一维，用于计算softmax的概率，其他的全合并：1*(?*9)*?*2==>(1*?*9*?)*2
            reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)  # 得到所有特征的概率
            return tf.reshape(reshaped_score, input_shape)  # (1*?*9*?)*2==>1*(?*9)*?*2
        return tf.nn.softmax(bottom, name=name)

    def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred,
                        name):
    # 每个位置的9个archors的类别概率和每个位置的9个archors的回归位置偏移得到post_nms_topN个archors的位置及为1的概率
        with tf.variable_scope(name) as scope:
            if cfg.USE_E2E_TF:  # post_nms_topN*5的rois（第一列为全0的batch_inds，后4列为坐标）；rpn_scores：post_nms_topN*1个对应的为1的概率
                rois, rpn_scores = proposal_layer_tf(rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                                     self._feat_stride, self._anchors, self._num_anchors)
            else:
                rois, rpn_scores = tf.py_func(proposal_layer, [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                                               self._feat_stride, self._anchors, self._num_anchors],
                                              [tf.float32, tf.float32], name="proposal")

            rois.set_shape([None, 5])
            rpn_scores.set_shape([None, 1])

        return rois, rpn_scores

    def proposal_layer_tf(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors,
                          num_anchors):
        # 每个位置的9个archors的类别概率和每个位置的9个archors的回归位置偏移
        if type(cfg_key) == bytes:
            cfg_key = cfg_key.decode('utf-8')
        pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N  # 训练时为2000，测试时为300
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH  # nms的阈值，为0.7

        scores = rpn_cls_prob[:, :, :,
                 num_anchors:]  # 1*?*?*(9*2)取后9个：1*?*?*9。应该是前9个代表9个archors为背景景的概率，后9个代表9个archors为前景的概率（二分类，只有背景和前景）
        scores = tf.reshape(scores, shape=(-1,))  # 所有的archors为1的概率
        rpn_bbox_pred = tf.reshape(rpn_bbox_pred, shape=(-1, 4))  # 所有的archors的四个坐标

        proposals = bbox_transform_inv_tf(anchors, rpn_bbox_pred)  # 已知archor和偏移求预测的坐标
        proposals = clip_boxes_tf(proposals, im_info[:2])  # 限制预测坐标在原始图像上

        indices = tf.image.non_max_suppression(proposals, scores, max_output_size=post_nms_topN,
                                               iou_threshold=nms_thresh)  # 通过nms得到分值最大的post_nms_topN个坐标的索引

        boxes = tf.gather(proposals, indices)  # 得到post_nms_topN个对应的坐标
        boxes = tf.to_float(boxes)
        scores = tf.gather(scores, indices)  # 得到post_nms_topN个对应的为1的概率
        scores = tf.reshape(scores, shape=(-1, 1))

        batch_inds = tf.zeros((tf.shape(indices)[0], 1), dtype=tf.float32)  # Only support single image as input
        blob = tf.concat([batch_inds, boxes],
                         1)
        # post_nms_topN*1个batch_inds和post_nms_topN*4个坐标concat，得到post_nms_topN*5的blob

        return blob, scores

    def bbox_transform_inv_tf(boxes, deltas):  # 已知archor和偏移求预测的坐标
        boxes = tf.cast(boxes, deltas.dtype)
        widths = tf.subtract(boxes[:, 2], boxes[:, 0]) + 1.0  # 宽
        heights = tf.subtract(boxes[:, 3], boxes[:, 1]) + 1.0  # 高
        ctr_x = tf.add(boxes[:, 0], widths * 0.5)  # 中心x
        ctr_y = tf.add(boxes[:, 1], heights * 0.5)  # 中心y

        dx = deltas[:, 0]  # 预测的dx
        dy = deltas[:, 1]  # 预测的dy
        dw = deltas[:, 2]  # 预测的dw
        dh = deltas[:, 3]  # 预测的dh

        pred_ctr_x = tf.add(tf.multiply(dx, widths), ctr_x)  # 公式2已知xa，wa，tx反过来求预测的x中心坐标
        pred_ctr_y = tf.add(tf.multiply(dy, heights), ctr_y)  # 公式2已知ya，ha，ty反过来求预测的y中心坐标
        pred_w = tf.multiply(tf.exp(dw), widths)  # 公式2已知wa，tw反过来求预测的w
        pred_h = tf.multiply(tf.exp(dh), heights)  # 公式2已知ha，th反过来求预测的h

        pred_boxes0 = tf.subtract(pred_ctr_x, pred_w * 0.5)  # 预测的框的起始和终点四个坐标
        pred_boxes1 = tf.subtract(pred_ctr_y, pred_h * 0.5)
        pred_boxes2 = tf.add(pred_ctr_x, pred_w * 0.5)
        pred_boxes3 = tf.add(pred_ctr_y, pred_h * 0.5)

        return tf.stack([pred_boxes0, pred_boxes1, pred_boxes2, pred_boxes3], axis=1)

    def clip_boxes_tf(boxes, im_info):  # 限制预测坐标在原始图像上
        b0 = tf.maximum(tf.minimum(boxes[:, 0], im_info[1] - 1), 0)
        b1 = tf.maximum(tf.minimum(boxes[:, 1], im_info[0] - 1), 0)
        b2 = tf.maximum(tf.minimum(boxes[:, 2], im_info[1] - 1), 0)
        b3 = tf.maximum(tf.minimum(boxes[:, 3], im_info[0] - 1), 0)
        return tf.stack([b0, b1, b2, b3], axis=1)

    def _anchor_target_layer(self, rpn_cls_score, name):  # rpn_cls_score:每个位置的9个archors分类特征1*？*？*(9*2)
        with tf.variable_scope(name) as scope:
            # rpn_labels; 特征图中每个位置对应的是正样本、负样本还是不关注（去除了边界在图像外面的archors）
            # rpn_bbox_targets:# 特征图中每个位置和对应的正样本的坐标偏移（很多为0）
            # rpn_bbox_inside_weights:  正样本的权重为1（去除负样本和不关注的样本，均为0）
            # rpn_bbox_outside_weights:  正样本和负样本（不包括不关注的样本）归一化的权重
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
                anchor_target_layer,
                [rpn_cls_score, self._gt_boxes, self._im_info, self._feat_stride, self._anchors, self._num_anchors],
                [tf.float32, tf.float32, tf.float32, tf.float32], name="anchor_target")

            rpn_labels.set_shape([1, 1, None, None])
            rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

            rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
            self._anchor_targets['rpn_labels'] = rpn_labels  # 特征图中每个位置对应的是正样本、负样本还是不关注（去除了边界在图像外面的archors）
            self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets  # 特征图中每个位置和对应的正样本的坐标偏移（很多为0）
            self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights  # 正样本的权重为1（去除负样本和不关注的样本，均为0）
            self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights  # 正样本和负样本（不包括不关注的样本）归一化的权重

            self._score_summaries.update(self._anchor_targets)

        return rpn_labels

    def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors,
                            num_anchors):  # 1*？*？*(9*2); ?*5; 3; [16], ?*4; [9]
        """Same as the anchor target layer in original Fast/er RCNN """
        A = num_anchors  # [9]
        total_anchors = all_anchors.shape[0]  # 所有archors的个数，9*特征图宽*特征图高 个
        K = total_anchors / num_anchors

        _allowed_border = 0  # allow boxes to sit over the edge by a small amount
        height, width = rpn_cls_score.shape[1:3]  # rpn网络得到的特征的高宽

        inds_inside = np.where(  # 所有archors边界可能超出图像，取在图像内部的archors的索引
            (all_anchors[:, 0] >= -_allowed_border) & (all_anchors[:, 1] >= -_allowed_border) &
            (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
            (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
        )[0]

        anchors = all_anchors[inds_inside, :]  # 得到在图像内部archors的坐标

        labels = np.empty((len(inds_inside),), dtype=np.float32)  # label: 1 正样本, 0 负样本, -1 不关注
        labels.fill(-1)

        # 计算  每个anchors:n*4和每个真实位置gt_boxes:m*4的重叠区域  的比  的矩阵:n*m
        overlaps = bbox_overlaps(np.ascontiguousarray(anchors, dtype=np.float),
                                 np.ascontiguousarray(gt_boxes, dtype=np.float))


        argmax_overlaps = overlaps.argmax(axis=1)
        # 找到每行最大值的位置，即每个archors对应的正样本的位置，得到n维的行向量

        # overlaps是
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        # 取出每个archors对应的正样本的重叠区域，n维向量

        gt_argmax_overlaps = overlaps.argmax(axis=0)
        # 找到每列最大值的位置，即每个真实位置对应的archors的位置，得到m维的行向量
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        # 取出每个真实位置对应的archors的重叠区域，m维向量
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
        # 得到从小到大顺序的位置  是锚点从小到大

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:  # assign bg labels first so that positive labels can clobber them first set the negatives
            #label 是和 置信度 具有一样长度 值全为-1
            # labels = np.empty((len(inds_inside),), dtype=np.float32)
            # label: 1 正样本, 0 负样本, -1 不关注
            # labels.fill(-1)
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
            # 将archors对应的正样本的重叠区域中小于阈值的置0

        labels[gt_argmax_overlaps] = 1  # fg label: for each gt, anchor with highest overlap 每个真实位置对应的archors置1
        labels[
            max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1  # fg label: above threshold IOU 将archors对应的正样本的重叠区域中大于阈值的置1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:  # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # 如果有过多的正样本，则只随机选择num_fg=0.5*256=128个正样本
        num_fg = int(
            cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)  # subsample positive labels if we have too many
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1  # 将多于的正样本设置为不关注

        # 如果有过多的负样本，则只随机选择 num_bg=256-正样本个数 个负样本
        num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)  # subsample negative labels if we have too many
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1  # 将多于的负样本设置为不关注

        bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])  # 通过archors和archors对应的正样本计算坐标的偏移

        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)  # 正样本的四个坐标的权重均设置为1

        bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:  # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)  # 正样本和负样本的总数（去除不关注的样本）
            positive_weights = np.ones((1, 4)) * 1.0 / num_examples  # 归一化的权重
            negative_weights = np.ones((1, 4)) * 1.0 / num_examples  # 归一化的权重
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) & (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT / np.sum(labels == 1))
            negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) / np.sum(labels == 0))
        bbox_outside_weights[labels == 1, :] = positive_weights  # 归一化的权重
        bbox_outside_weights[labels == 0, :] = negative_weights  # 归一化的权重

        # 由于上面使用了inds_inside，此处将labels，bbox_targets，bbox_inside_weights，bbox_outside_weights映射到原始的archors（包含未知
        # 参数超出图像边界的archors）对应的labels，bbox_targets，bbox_inside_weights，bbox_outside_weights，同时将不需要的填充fill的值
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside,
                                     fill=0)  # 所有archors中正样本的四个坐标的权重均设置为1，其他为0
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)  # (1*？*？)*9==>1*？*？*9==>1*9*？*？
        labels = labels.reshape((1, 1, A * height, width))  # 1*9*？*？==>1*1*(9*？)*？
        rpn_labels = labels  # 特征图中每个位置对应的是正样本、负样本还是不关注（去除了边界在图像外面的archors）

        bbox_targets = bbox_targets.reshape((1, height, width, A * 4))  # 1*(9*？)*？*4==>1*？*？*(9*4)

        rpn_bbox_targets = bbox_targets  # 特征图中每个位置和对应的正样本的坐标偏移（很多为0）
        bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A * 4))  # 1*(9*？)*？*4==>1*？*？*(9*4)
        rpn_bbox_inside_weights = bbox_inside_weights
        bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, A * 4))  # 1*(9*？)*？*4==>1*？*？*(9*4)
        rpn_bbox_outside_weights = bbox_outside_weights  # 归一化的权重
        return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

    def _unmap(data, count, inds, fill=0):
        """ Unmap a subset of item (data) back to the original set of items (of size count) """
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=np.float32)  # 得到1维矩阵
            ret.fill(fill)  # 默认填充fill的值
            ret[inds] = data  # 有效位置填充具体数据
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=np.float32)  # 得到对应维数的矩阵
            ret.fill(fill)  # 默认填充fill的值
            ret[inds, :] = data  # 有效位置填 充具体数据
        return ret

    def _compute_targets(ex_rois, gt_rois):
        """Compute bounding-box regression targets for an image."""
        assert ex_rois.shape[0] == gt_rois.shape[0]
        assert ex_rois.shape[1] == 4
        assert gt_rois.shape[1] == 5

        # 通过公式2后四个，结合archor和对应的正样本的坐标计算坐标的偏移
        return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
        # 由于gt_rois是5列，去掉第一列的batch_inds

    def bbox_transform(ex_rois, gt_rois):
        ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0  # archor的宽
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0  # archor的高
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths  # archor的中心x
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights  # archor的中心y

        gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0  # 真实正样本w
        gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0  # 真实正样本h
        gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths  # 真实正样本中心x
        gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights  # 真实正样本中心y

        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths  # 通过公式2后四个的x*，xa，wa得到dx
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights  # 通过公式2后四个的y*，ya，ha得到dy
        targets_dw = np.log(gt_widths / ex_widths)  # 通过公式2后四个的w*，wa得到dw
        targets_dh = np.log(gt_heights / ex_heights)  # 通过公式2后四个的h*，ha得到dh

        targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
        return targets

    def bbox_overlaps(
            np.

        ndarray[DTYPE_t, ndim = 2] boxes,
                                   np.ndarray[DTYPE_t, ndim = 2] query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
        cdef
        unsigned
        int
        N = boxes.shape[0]
        cdef
        unsigned
        int
        K = query_boxes.shape[0]
        cdef
        np.ndarray[DTYPE_t, ndim = 2] overlaps = np.zeros((N, K), dtype=DTYPE)
        cdef
        DTYPE_t
        iw, ih, box_area
        cdef
        DTYPE_t
        ua
        cdef
        unsigned
        int
        k, n
        for k in range(K):
            box_area = (
                    (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
                    (query_boxes[k, 3] - query_boxes[k, 1] + 1)
            )
            for n in range(N):
                iw = (
                        min(boxes[n, 2], query_boxes[k, 2]) -
                        max(boxes[n, 0], query_boxes[k, 0]) + 1
                )
                if iw > 0:
                    ih = (
                            min(boxes[n, 3], query_boxes[k, 3]) -
                            max(boxes[n, 1], query_boxes[k, 1]) + 1
                    )
                    if ih > 0:
                        ua = float(
                            (boxes[n, 2] - boxes[n, 0] + 1) *
                            (boxes[n, 3] - boxes[n, 1] + 1) +
                            box_area - iw * ih
                        )
                        overlaps[n, k] = iw * ih / ua
        return overlaps


    def _proposal_target_layer(self, rois, roi_scores, name):  # post_nms_topN个archors的位置及为1（正样本）的概率
        # 只在训练时使用该层，从post_nms_topN个archors中选择256个archors
        with tf.variable_scope(name) as scope:
            # labels：正样本和负样本对应的真实的类别
            # rois：从post_nms_topN个archors中选择256个archors（第一列的全0更新为每个archors对应的类别）
            # roi_scores：256个archors对应的为正样本的概率
            # bbox_targets：256*(4*21)的矩阵，只有为正样本时，对应类别的坐标才不为0，其他类别的坐标全为0
            # bbox_inside_weights：256*(4*21)的矩阵，正样本时，对应类别四个坐标的权重为1，其他全为0
            # bbox_outside_weights：256*(4*21)的矩阵，正样本时，对应类别四个坐标的权重为1，其他全为0
            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                proposal_target_layer, [rois, roi_scores, self._gt_boxes, self._num_classes],
                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32], name="proposal_target")

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

            return rois, roi_scores


    def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, _num_classes):
        """Assign object detection proposals to ground-truth targets. Produces proposal classification labels and bounding-box regression targets."""
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = rpn_rois  # rpn_rois为post_nms_topN*5的矩阵
        all_scores = rpn_scores  # rpn_scores为post_nms_topN的矩阵，代表对应的archors为正样本的概率

        if cfg.TRAIN.USE_GT:  # Include ground-truth boxes in the set of candidate rois;  USE_GT=False，未使用这段代码
            zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
            all_rois = np.vstack((all_rois, np.hstack((zeros, gt_boxes[:, :-1]))))
            all_scores = np.vstack((all_scores, zeros))  # not sure if it a wise appending, but anyway i am not using it

        num_images = 1  # 该程序只能一次处理一张图片
        rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images  # 每张图片中最终选择的rois
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)  # 正样本的个数：0.25*rois_per_image

        # Sample rois with classification labels and bounding box regression targets
        # labels：正样本和负样本对应的真实的类别
        # rois：从post_nms_topN个archors中选择256个archors（第一列的全0更新为每个archors对应的类别）
        # roi_scores：256个archors对应的为正样本的概率
        # bbox_targets：256*(4*21)的矩阵，只有为正样本时，对应类别的坐标才不为0，其他类别的坐标全为0
        # bbox_inside_weights：256*(4*21)的矩阵，正样本时，对应类别四个坐标的权重为1，其他全为0
        labels, rois, roi_scores, bbox_targets, bbox_inside_weights = _sample_rois(all_rois, all_scores, gt_boxes,
                                                                                   fg_rois_per_image, rois_per_image,
                                                                                   _num_classes)  # 选择256个archors

        rois = rois.reshape(-1, 5)
        roi_scores = roi_scores.reshape(-1)
        labels = labels.reshape(-1, 1)
        bbox_targets = bbox_targets.reshape(-1, _num_classes * 4)
        bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)
        bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(
            np.float32)  # 256*(4*21)的矩阵，正样本时，对应类别四个坐标的权重为1，其他全为0

        return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


    def _get_bbox_regression_labels(bbox_target_data, num_classes):
        """Bounding-box regression targets (bbox_target_data) are stored in a compact form N x (class, tx, ty, tw, th)
        This function expands those targets into the 4-of-4*K representation used by the network (i.e. only one class has non-zero targets).
        Returns:
            bbox_target (ndarray): N x 4K blob of regression targets
            bbox_inside_weights (ndarray): N x 4K blob of loss weights
        """
        clss = bbox_target_data[:, 0]  # 第1列，为类别
        bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)  # 256*(4*21)的矩阵
        bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
        inds = np.where(clss > 0)[0]  # 正样本的索引
        for ind in inds:
            cls = clss[ind]  # 正样本的类别
            start = int(4 * cls)  # 每个正样本的起始坐标
            end = start + 4  # 每个正样本的终止坐标（由于坐标为4）
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]  # 对应的坐标偏移赋值给对应的类别
            bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS  # 对应的权重(1.0, 1.0, 1.0, 1.0)赋值给对应的类别

        # bbox_targets：256*(4*21)的矩阵，只有为正样本时，对应类别的坐标才不为0，其他类别的坐标全为0
        # bbox_inside_weights：256*(4*21)的矩阵，正样本时，对应类别四个坐标的权重为1，其他全为0
        return bbox_targets, bbox_inside_weights


    def _compute_targets(ex_rois, gt_rois, labels):
        """Compute bounding-box regression targets for an image."""
        assert ex_rois.shape[0] == gt_rois.shape[0]
        assert ex_rois.shape[1] == 4
        assert gt_rois.shape[1] == 4

        targets = bbox_transform(ex_rois, gt_rois)  # 通过公式2后四个，结合256个archor和对应的正样本的坐标计算坐标的偏移
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:  # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS)) / np.array(
                cfg.TRAIN.BBOX_NORMALIZE_STDS))  # 坐标减去均值除以标准差，进行归一化
        return np.hstack((labels[:, np.newaxis], targets)).astype(np.float32, copy=False)  # 之前的bbox第一列为全0，此处第一列为对应的类别


    def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image,
                     num_classes):  # all_rois第一列全0，后4列为坐标；gt_boxes前4列为坐标，最后一列为类别
        """Generate a random sample of RoIs comprising foreground and background examples."""
        # 计算archors和gt_boxes重叠区域面积的比值
        overlaps = bbox_overlaps(np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
                                 np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))  # overlaps: (rois x gt_boxes)
        gt_assignment = overlaps.argmax(axis=1)  # 得到每个archors对应的gt_boxes的索引
        max_overlaps = overlaps.max(axis=1)  # 得到每个archors对应的gt_boxes的重叠区域的值
        labels = gt_boxes[gt_assignment, 4]  # 得到每个archors对应的gt_boxes的类别

        # 每个archors对应的gt_boxes的重叠区域的值大于阈值的作为正样本，得到正样本的索引
        fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[
            0]  # Select foreground RoIs as those with >= FG_THRESH overlap
        # Guard against the case when an image has fewer than fg_rois_per_image. Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        # 每个archors对应的gt_boxes的重叠区域的值在给定阈值内的作为负样本，得到负样本的索引
        bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) & (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]

        # Small modification to the original version where we ensure a fixed number of regions are sampled
        # 最终选择256个archors
        if fg_inds.size > 0 and bg_inds.size > 0:  # 正负样本均存在，则选择最多fg_rois_per_image个正样本，不够的话，补充负样本
            fg_rois_per_image = min(fg_rois_per_image, fg_inds.size)
            fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_image), replace=False)
            bg_rois_per_image = rois_per_image - fg_rois_per_image
            to_replace = bg_inds.size < bg_rois_per_image
            bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_image), replace=to_replace)
        elif fg_inds.size > 0:  # 只有正样本，选择rois_per_image个正样本
            to_replace = fg_inds.size < rois_per_image
            fg_inds = npr.choice(fg_inds, size=int(rois_per_image), replace=to_replace)
            fg_rois_per_image = rois_per_image
        elif bg_inds.size > 0:  # 只有负样本，选择rois_per_image个负样本
            to_replace = bg_inds.size < rois_per_image
            bg_inds = npr.choice(bg_inds, size=int(rois_per_image), replace=to_replace)
            fg_rois_per_image = 0
        else:
            import pdb
            pdb.set_trace()

        keep_inds = np.append(fg_inds, bg_inds)  # 正样本和负样本的索引
        labels = labels[keep_inds]  # 正样本和负样本对应的真实的类别
        labels[int(fg_rois_per_image):] = 0  # 负样本对应的类别设置为0
        rois = all_rois[keep_inds]  # 从post_nms_topN个archors中选择256个archors
        roi_scores = all_scores[keep_inds]  # 256个archors对应的为正样本的概率

        # 通过256个archors的坐标和每个archors对应的gt_boxes的坐标及这些archors的真实类别得到坐标偏移（将rois第一列的全0更新为每个archors对应的类别）
        bbox_target_data = _compute_targets(rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)
        # bbox_targets：256*(4*21)的矩阵，只有为正样本时，对应类别的坐标才不为0，其他类别的坐标全为0
        # bbox_inside_weights：256*(4*21)的矩阵，正样本时，对应类别四个坐标的权重为1，其他全为0
        bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(bbox_target_data, num_classes)

        # labels：正样本和负样本对应的真实的类别
        # rois：从post_nms_topN个archors中选择256个archors（第一列的全0更新为每个archors对应的类别）
        # roi_scores：256个archors对应的为正样本的概率
        # bbox_targets：256*(4*21)的矩阵，只有为正样本时，对应类别的坐标才不为0，其他类别的坐标全为0
        # bbox_inside_weights：256*(4*21)的矩阵，正样本时，对应类别四个坐标的权重为1，其他全为0
        return labels, rois, roi_scores, bbox_targets, bbox_inside_weights


    def _proposal_target_layer(self, rois, roi_scores, name):  # post_nms_topN个archors的位置及为1（正样本）的概率
        # 只在训练时使用该层，从post_nms_topN个archors中选择256个archors
        with tf.variable_scope(name) as scope:
            # labels：正样本和负样本对应的真实的类别
            # rois：从post_nms_topN个archors中选择256个archors（第一列的全0更新为每个archors对应的类别）
            # roi_scores：256个archors对应的为正样本的概率
            # bbox_targets：256*(4*21)的矩阵，只有为正样本时，对应类别的坐标才不为0，其他类别的坐标全为0
            # bbox_inside_weights：256*(4*21)的矩阵，正样本时，对应类别四个坐标的权重为1，其他全为0
            # bbox_outside_weights：256*(4*21)的矩阵，正样本时，对应类别四个坐标的权重为1，其他全为0
            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                proposal_target_layer, [rois, roi_scores, self._gt_boxes, self._num_classes],
                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32], name="proposal_target")

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

            return rois, roi_scores


    def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, _num_classes):
        """Assign object detection proposals to ground-truth targets. Produces proposal classification labels and bounding-box regression targets."""
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = rpn_rois  # rpn_rois为post_nms_topN*5的矩阵
        all_scores = rpn_scores  # rpn_scores为post_nms_topN的矩阵，代表对应的archors为正样本的概率

        if cfg.TRAIN.USE_GT:  # Include ground-truth boxes in the set of candidate rois;  USE_GT=False，未使用这段代码
            zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
            all_rois = np.vstack((all_rois, np.hstack((zeros, gt_boxes[:, :-1]))))
            all_scores = np.vstack((all_scores, zeros))  # not sure if it a wise appending, but anyway i am not using it

        num_images = 1  # 该程序只能一次处理一张图片
        rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images  # 每张图片中最终选择的rois
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)  # 正样本的个数：0.25*rois_per_image

        # Sample rois with classification labels and bounding box regression targets
        # labels：正样本和负样本对应的真实的类别
        # rois：从post_nms_topN个archors中选择256个archors（第一列的全0更新为每个archors对应的类别）
        # roi_scores：256个archors对应的为正样本的概率
        # bbox_targets：256*(4*21)的矩阵，只有为正样本时，对应类别的坐标才不为0，其他类别的坐标全为0
        # bbox_inside_weights：256*(4*21)的矩阵，正样本时，对应类别四个坐标的权重为1，其他全为0
        labels, rois, roi_scores, bbox_targets, bbox_inside_weights = _sample_rois(all_rois, all_scores, gt_boxes,
                                                                                   fg_rois_per_image, rois_per_image,
                                                                                   _num_classes)  # 选择256个archors

        rois = rois.reshape(-1, 5)
        roi_scores = roi_scores.reshape(-1)
        labels = labels.reshape(-1, 1)
        bbox_targets = bbox_targets.reshape(-1, _num_classes * 4)
        bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)
        bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(
            np.float32)  # 256*(4*21)的矩阵，正样本时，对应类别四个坐标的权重为1，其他全为0

        return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


    def _get_bbox_regression_labels(bbox_target_data, num_classes):
        """Bounding-box regression targets (bbox_target_data) are stored in a compact form N x (class, tx, ty, tw, th)
        This function expands those targets into the 4-of-4*K representation used by the network (i.e. only one class has non-zero targets).
        Returns:
            bbox_target (ndarray): N x 4K blob of regression targets
            bbox_inside_weights (ndarray): N x 4K blob of loss weights
        """
        clss = bbox_target_data[:, 0]  # 第1列，为类别
        bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)  # 256*(4*21)的矩阵
        bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
        inds = np.where(clss > 0)[0]  # 正样本的索引
        for ind in inds:
            cls = clss[ind]  # 正样本的类别
            start = int(4 * cls)  # 每个正样本的起始坐标
            end = start + 4  # 每个正样本的终止坐标（由于坐标为4）
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]  # 对应的坐标偏移赋值给对应的类别
            bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS  # 对应的权重(1.0, 1.0, 1.0, 1.0)赋值给对应的类别

        # bbox_targets：256*(4*21)的矩阵，只有为正样本时，对应类别的坐标才不为0，其他类别的坐标全为0
        # bbox_inside_weights：256*(4*21)的矩阵，正样本时，对应类别四个坐标的权重为1，其他全为0
        return bbox_targets, bbox_inside_weights


    def _compute_targets(ex_rois, gt_rois, labels):
        """Compute bounding-box regression targets for an image."""
        assert ex_rois.shape[0] == gt_rois.shape[0]
        assert ex_rois.shape[1] == 4
        assert gt_rois.shape[1] == 4

        targets = bbox_transform(ex_rois, gt_rois)  # 通过公式2后四个，结合256个archor和对应的正样本的坐标计算坐标的偏移
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:  # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS)) / np.array(
                cfg.TRAIN.BBOX_NORMALIZE_STDS))  # 坐标减去均值除以标准差，进行归一化
        return np.hstack((labels[:, np.newaxis], targets)).astype(np.float32, copy=False)  # 之前的bbox第一列为全0，此处第一列为对应的类别


    def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image,
                     num_classes):  # all_rois第一列全0，后4列为坐标；gt_boxes前4列为坐标，最后一列为类别
        """Generate a random sample of RoIs comprising foreground and background examples."""
        # 计算archors和gt_boxes重叠区域面积的比值
        overlaps = bbox_overlaps(np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
                                 np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))  # overlaps: (rois x gt_boxes)
        gt_assignment = overlaps.argmax(axis=1)  # 得到每个archors对应的gt_boxes的索引
        max_overlaps = overlaps.max(axis=1)  # 得到每个archors对应的gt_boxes的重叠区域的值
        labels = gt_boxes[gt_assignment, 4]  # 得到每个archors对应的gt_boxes的类别

        # 每个archors对应的gt_boxes的重叠区域的值大于阈值的作为正样本，得到正样本的索引
        fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[
            0]  # Select foreground RoIs as those with >= FG_THRESH overlap
        # Guard against the case when an image has fewer than fg_rois_per_image. Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        # 每个archors对应的gt_boxes的重叠区域的值在给定阈值内的作为负样本，得到负样本的索引
        bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) & (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]

        # Small modification to the original version where we ensure a fixed number of regions are sampled
        # 最终选择256个archors
        if fg_inds.size > 0 and bg_inds.size > 0:  # 正负样本均存在，则选择最多fg_rois_per_image个正样本，不够的话，补充负样本
            fg_rois_per_image = min(fg_rois_per_image, fg_inds.size)
            fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_image), replace=False)
            bg_rois_per_image = rois_per_image - fg_rois_per_image
            to_replace = bg_inds.size < bg_rois_per_image
            bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_image), replace=to_replace)
        elif fg_inds.size > 0:  # 只有正样本，选择rois_per_image个正样本
            to_replace = fg_inds.size < rois_per_image
            fg_inds = npr.choice(fg_inds, size=int(rois_per_image), replace=to_replace)
            fg_rois_per_image = rois_per_image
        elif bg_inds.size > 0:  # 只有负样本，选择rois_per_image个负样本
            to_replace = bg_inds.size < rois_per_image
            bg_inds = npr.choice(bg_inds, size=int(rois_per_image), replace=to_replace)
            fg_rois_per_image = 0
        else:
            import pdb
            pdb.set_trace()

        keep_inds = np.append(fg_inds, bg_inds)  # 正样本和负样本的索引
        labels = labels[keep_inds]  # 正样本和负样本对应的真实的类别
        labels[int(fg_rois_per_image):] = 0  # 负样本对应的类别设置为0
        rois = all_rois[keep_inds]  # 从post_nms_topN个archors中选择256个archors
        roi_scores = all_scores[keep_inds]  # 256个archors对应的为正样本的概率

        # 通过256个archors的坐标和每个archors对应的gt_boxes的坐标及这些archors的真实类别得到坐标偏移（将rois第一列的全0更新为每个archors对应的类别）
        bbox_target_data = _compute_targets(rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)
        # bbox_targets：256*(4*21)的矩阵，只有为正样本时，对应类别的坐标才不为0，其他类别的坐标全为0
        # bbox_inside_weights：256*(4*21)的矩阵，正样本时，对应类别四个坐标的权重为1，其他全为0
        bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(bbox_target_data, num_classes)

        # labels：正样本和负样本对应的真实的类别
        # rois：从post_nms_topN个archors中选择256个archors（第一列的全0更新为每个archors对应的类别）
        # roi_scores：256个archors对应的为正样本的概率
        # bbox_targets：256*(4*21)的矩阵，只有为正样本时，对应类别的坐标才不为0，其他类别的坐标全为0
        # bbox_inside_weights：256*(4*21)的矩阵，正样本时，对应类别四个坐标的权重为1，其他全为0
        return labels, rois, roi_scores, bbox_targets, bbox_inside_weights


    def _head_to_tail(self, pool5, is_training, reuse=None):
        with tf.variable_scope(self._scope, self._scope, reuse=reuse):
            pool5_flat = slim.flatten(pool5, scope='flatten')
            fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
            if is_training:
                fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, scope='dropout6')
            fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
            if is_training:
                fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True, scope='dropout7')

        return fc7


    def _region_classification(self, fc7, is_training, initializer, initializer_bbox):
        # 增加fc层，输出为总共类别的个数，进行分类
        cls_score = slim.fully_connected(fc7, self._num_classes, weights_initializer=initializer, trainable=is_training,
                                         activation_fn=None, scope='cls_score')
        cls_prob = self._softmax_layer(cls_score, "cls_prob")  # 得到每一类别的概率
        cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")  # 得到预测的类别
        # 增加fc层，预测位置信息的偏移
        bbox_pred = slim.fully_connected(fc7, self._num_classes * 4, weights_initializer=initializer_bbox,
                                         trainable=is_training, activation_fn=None, scope='bbox_pred')

        self._predictions["cls_score"] = cls_score  # 用于rcnn分类的256个archors的特征
        self._predictions["cls_pred"] = cls_pred
        self._predictions["cls_prob"] = cls_prob
        self._predictions["bbox_pred"] = bbox_pred

        return cls_prob, bbox_pred


    def _add_losses(self, sigma_rpn=3.0):
        with tf.variable_scope('LOSS_' + self._tag) as scope:
            rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])  # 每个archors是正样本还是负样本
            rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])  # 特征图中每个位置对应的是正样本、负样本还是不关注（去除了边界在图像外面的archors）
            rpn_select = tf.where(tf.not_equal(rpn_label, -1))  # 不关注的archor到的索引
            rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])  # 去除不关注的archor
            rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])  # 去除不关注的label
            rpn_cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))  # rpn二分类的损失

            rpn_bbox_pred = self._predictions['rpn_bbox_pred']  # 每个位置的9个archors回归位置偏移
            rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']  # 特征图中每个位置和对应的正样本的坐标偏移（很多为0）
            rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']  # 正样本的权重为1（去除负样本和不关注的样本，均为0）
            rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']  # 正样本和负样本（不包括不关注的样本）归一化的权重
            rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

            cls_score = self._predictions["cls_score"]  # 用于rcnn分类的256个archors的特征
            label = tf.reshape(self._proposal_targets["labels"], [-1])  # 正样本和负样本对应的真实的类别
            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))  # rcnn分类的损失

            bbox_pred = self._predictions['bbox_pred']  # RCNN, bbox loss
            bbox_targets = self._proposal_targets['bbox_targets']  # 256*(4*21)的矩阵，只有为正样本时，对应类别的坐标才不为0，其他类别的坐标全为0
            bbox_inside_weights = self._proposal_targets['bbox_inside_weights']  # 256*(4*21)的矩阵，正样本时，对应类别四个坐标的权重为1，其他全为0
            bbox_outside_weights = self._proposal_targets['bbox_outside_weights']  # 256*(4*21)的矩阵，正样本时，对应类别四个坐标的权重为1，其他全为0
            loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

            self._losses['cross_entropy'] = cross_entropy
            self._losses['loss_box'] = loss_box
            self._losses['rpn_cross_entropy'] = rpn_cross_entropy
            self._losses['rpn_loss_box'] = rpn_loss_box

            loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box  # 总共的损失
            regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
            self._losses['total_loss'] = loss + regularization_loss

            self._event_summaries.update(self._losses)

        return loss


    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets  # 预测的和真实的相减
        in_box_diff = bbox_inside_weights * box_diff  # 乘以正样本的权重1（rpn：去除负样本和不关注的样本，rcnn：去除负样本）
        abs_in_box_diff = tf.abs(in_box_diff)  # 绝对值
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))  # 小于阈值的截断的标志位
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (
                    1. - smoothL1_sign)  # smooth l1 loss
        out_loss_box = bbox_outside_weights * in_loss_box  # rpn：除以有效样本总数（不考虑不关注的样本），进行归一化；rcnn：正样本四个坐标权重为1，负样本为0
        loss_box = tf.reduce_mean(tf.reduce_sum(out_loss_box, axis=dim))
        return loss_box