# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import logging

from utils.data_utils import parse_data
from utils.misc_utils import parse_anchors, read_class_names, shuffle_and_overwrite, update_dict, make_summary, config_learning_rate, config_optimizer, list_add
from utils.eval_utils import evaluate_on_cpu, evaluate_on_gpu
from utils.nms_utils import gpu_nms

from model import yolov3
#                    输入的y_true    （ 图片名字  +  标签 +  框  )  n个
#                    输入进去之后  转换为   和feature一样形状的 标签

#    转换后是  如下

# shape: [13, 13, 3, 80]   [..., 5:]里面的维度是 [13, 13, 3, 80]
# true_probs_temp = y_true[j][i][..., 5:]
# object_mask = y_true[..., 4:5]# 4:5  4：5就是4  是背景值前景值  这里这是前景值1和0
                    #应该全都是1？？？？？？？？？？？？？？？？？？？？没有负样本？？？？？？？？？？
# shape: [13, 13, 3, 4] (x_center, y_center, w, h)
# true_boxes_temp = y_true[j][i][..., 0:4]


#预测的所有 中心值边框等 在计算损失的时候并没有帅选     所有y_true 需要和 y_predict一样的维度

# y_true[feature_map_group][y, x, k, :2] = box_centers[i]
#         y_true[feature_map_group][y, x, k, 2:4] = box_sizes[i]
#         y_true[feature_map_group][y, x, k, 4] = 1.#这里的1表示是否含有，待检测的物体在里面，表示背景还是前景
#         y_true[feature_map_group][y, x, k, 5+c] = 1.
#         # k=0,1,2  [3, 4, 5]  [6, 7, 8] 的其中一个  一次只能是一个
#         # box_centers[i]  box_sizes[i] 都是在原图上的中心 和框的尺寸宽度值
#         #自己打的标签 与框 ，通过与   9个锚点   进行计算IOU  筛选最大的填入对应的值
#     return y_true_13, y_true_26, y_true_52   #process_box函数

# y_pred[i]就是三个feature_map   =   feature_map_i    9个anchors,输入某个特征图的  anchors  3个
# feature_map_i 里面的 中心值是一个网格的中心值  加上每个网格的 偏移量得到 x_y_offset
# 这里所有的个数  并没有进行帅选
# x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = self.reorg_layer(feature_map_i, anchors)

#################
# ArgumentParser
#################
parser = argparse.ArgumentParser(description="YOLO-V3 training procedure.")
# some paths
parser.add_argument("--train_file", type=str, default="./data/my_data/train.txt",
                    help="The path of the training txt file.")

parser.add_argument("--val_file", type=str, default="./data/my_data/val.txt",
                    help="The path of the validation txt file.")

parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3.ckpt",
                    help="The path of the weights to restore.")

parser.add_argument("--save_dir", type=str, default="./checkpoint/",
                    help="The directory of the weights to save.")

parser.add_argument("--log_dir", type=str, default="./data/logs/",
                    help="The directory to store the tensorboard log files.")

parser.add_argument("--progress_log_path", type=str, default="./data/progress.log",
                    help="The path to record the training progress.")

parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")

parser.add_argument("--class_name_path", type=str, default="./data/coco.names",
                    help="The path of the class names.")

# some numbers
parser.add_argument("--batch_size", type=int, default=20,
                    help="The batch size for training.")

parser.add_argument("--img_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image to `img_size`, size format: [width, height]")

parser.add_argument("--total_epoches", type=int, default=10000,
                    help="Total epoches to train.")

parser.add_argument("--train_evaluation_freq", type=int, default=100,
                    help="Evaluate on the training batch after some steps.")

parser.add_argument("--val_evaluation_freq", type=int, default=100,
                    help="Evaluate on the whole validation dataset after some steps.")

parser.add_argument("--save_freq", type=int, default=500,
                    help="Save the model after some steps.")

parser.add_argument("--num_threads", type=int, default=10,
                    help="Number of threads for image processing used in tf.data pipeline.")

parser.add_argument("--prefetech_buffer", type=int, default=3,
                    help="Prefetech_buffer used in tf.data pipeline.")

# learning rate and optimizer
parser.add_argument("--optimizer_name", type=str, default='adam',
                    help="The optimizer name. Chosen from [sgd, momentum, adam, rmsprop]")

parser.add_argument("--save_optimizer", type=lambda x: (str(x).lower() == 'true'), default=False,
                    help="Whether to save the optimizer parameters into the checkpoint file.")

parser.add_argument("--learning_rate_init", type=float, default=1e-3,
                    help="The initial learning rate.")

parser.add_argument("--lr_type", type=str, default='fixed',
                    help="The learning rate type. Chosen from [fixed, exponential]")

parser.add_argument("--lr_decay_freq", type=int, default=1000,
                    help="The learning rate decay frequency. Used when chosen exponential lr_type.")

parser.add_argument("--lr_decay_factor", type=float, default=0.96,
                    help="The learning rate decay factor. Used when chosen exponential lr_type.")

parser.add_argument("--lr_lower_bound", type=float, default=1e-6,
                    help="The minimum learning rate. Used when chosen exponential lr type.")

# finetune
parser.add_argument("--restore_part", nargs='*', type=str, default=['yolov3/darknet53_body'],
                    help="Partially restore part of the model for finetuning. Set [None] to restore the whole model.")

parser.add_argument("--update_part", nargs='*', type=str, default=['yolov3/yolov3_head'],
                    help="Partially restore part of the model for finetuning. Set [None] to train the whole model.")

# warm up strategy
parser.add_argument("--use_warm_up", type=lambda x: (str(x).lower() == 'true'), default=False,
                    help="Whether to use warm up strategy.")
#Python lower() 方法转换字符串中所有大写字符为小写

parser.add_argument("--warm_up_lr", type=float, default=5e-5,
                    help="Warm up learning rate.")

parser.add_argument("--warm_up_epoch", type=int, default=5,
                    help="Warm up training epoches.")
args = parser.parse_args()
"""

from utils.data_utils import parse_data
from utils.misc_utils import parse_anchors, read_class_names, shuffle_and_overwrite, update_dict, make_summary, config_learning_rate, config_optimizer, list_add
from utils.eval_utils import evaluate_on_cpu, evaluate_on_gpu
from utils.nms_utils import gpu_nms
"""
# args params
args.anchors = parse_anchors(args.anchor_path)
#这个函数是打开anchors  text文本  9个值
args.classes = read_class_names(args.class_name_path)
#enumerate在字典上是枚举、列举的意思
args.class_num = len(args.classes)#80



#——————————————————————————————
#                 读取数据batch
args.train_img_cnt = len(open(args.train_file, 'r').readlines())
#--train_file", type=str, default="./data/my_data/train.txt"  训练文件

args.val_img_cnt = len(open(args.val_file, 'r').readlines())#测试的文件

args.train_batch_num = int(np.ceil(float(args.train_img_cnt) / args.batch_size))
args.val_batch_num = int(np.ceil(float(args.val_img_cnt) / args.batch_size))

# setting loggers
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S', filename=args.progress_log_path, filemode='w')
#??????????????????????///  可以先不管
# setting placeholders
is_training = tf.placeholder(dtype=tf.bool, name="phase_train")
handle_flag = tf.placeholder(tf.string, [], name='iterator_handle_flag')
#训练的数据       这个占位符  是说只有有值的时候  才能执行  也就是传进去处理器
#iterator_handle_flag 迭代处理器

##################
# tf.data pipeline
##################
# Selecting `feedable iterator` to switch between training dataset and validation dataset

# manually shuffle the train txt file because tf.data.shuffle is soooo slow!!
# you can google it for more details.
shuffle_and_overwrite(args.train_file)
#打乱顺序
# parser.add_argument("--num_threads", type=int, default=10,
#                     help="Number of threads for image processing used in tf.data pipeline.")

train_dataset = tf.data.TextLineDataset(args.train_file)
#每一行是  图片名字  +  标签
#生成一个dataset，dataset中的每一个元素就对应了文件中的一行   args.img_size = [416，416]
#                                                       anchors  text文本  9个值



#  自己猜什么意思：   把train_dataset  的每一行的值  传入  x
#自己的理解    tf.contrib.data.map_and_batch 就是为了构建batch
# 当然需要输入batch的数量

train_dataset = train_dataset.apply(tf.contrib.data.map_and_batch(
    lambda x: tf.py_func(parse_data, [x, args.class_num, args.img_size, args.anchors, 'train'], [tf.float32, tf.float32, tf.float32, tf.float32]),
    num_parallel_calls=args.num_threads, batch_size=args.batch_size))
# batch_size表示要在此数据集合并的单个batch中的连续元素数。
#num_parallel_calls表示要并行处理的元素数


train_dataset = train_dataset.prefetch(args.prefetech_buffer)
#prefetch 预取 =3  ？？？？？？？？？？？？？？？？？？？？？是啥  缓存？？？？？？//
# parser.add_argument("--prefetech_buffer", type=int, default=3,
#                     help="Prefetech_buffer used in tf.data pipeline.")


val_dataset = tf.data.TextLineDataset(args.val_file)


#----------------------------------------
#生成一个dataset，dataset中的每一个元素就对应了文件中的一行
#只是 在0矩阵里面 进行  填充
#feature_map_group  是表示哪个特征图   y,x是转化尺度后的中心  k是


# val_dataset
# y_true[feature_map_group][y, x, k, :2] = box_centers[i]
# y_true[feature_map_group][y, x, k, 2:4] = box_sizes[i]
# y_true[feature_map_group][y, x, k, 4] = 1.#这里的1表示是否含有，待检测的物体在里面，表示背景还是前景
# y_true[feature_map_group][y, x, k, 5+c] = 1.
#隐藏了一个含义，在遍历的时候  5+0 5+1  只能是一个0一个为1   5+0的值为1意思是这个框

#总共的标签值  box_centers0，1， box_sizes 2，3   置信度4   标签5，6
# return y_true_13, y_true_26, y_true_52

val_dataset = val_dataset.apply(tf.contrib.data.map_and_batch(
    lambda x: tf.py_func(parse_data, [x, args.class0_num,
            args.img_size, args.anchors, 'val'], [tf.float32, tf.float32, tf.float32, tf.float32]),
    num_parallel_calls=args.num_threads, batch_size=args.batch_size))#batch_size=20
val_dataset.prefetch(args.prefetech_buffer)



#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
                                                #定义管道的开始


# creating two dataset iterators
#？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
train_iterator = train_dataset.make_initializable_iterator()
# make_initializable_iterator 迭代器
# 可初始化迭代器允许Dataset中存在占位符，
# 这样可以在数据需要输出的时候，再进行feed操作
val_iterator = val_dataset.make_initializable_iterator()

                    #定义可初始化迭代器

# creating two dataset handles
#？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
train_handle = train_iterator.string_handle()
val_handle = val_iterator.string_handle()

                   #定义处理器

# select a specific iterator based on the passed handle
#         handle_flag  在feed_dict传进来的是 train_handle  可初始化迭代器的处理器
dataset_iterator = tf.data.Iterator.from_string_handle(handle_flag, train_dataset.output_types,
                                                       train_dataset.output_shapes)

                    # 通过一个占位符和数据集的结构来定义   可馈送迭代器

# get an element from the choosed dataset iterator
# 取出张量对象
image, y_true_13, y_true_26, y_true_52 = dataset_iterator.get_next()



y_true = [y_true_13, y_true_26, y_true_52]
# https://blog.csdn.net/weixin_39506322/article/details/82455860
# tf.data pipeline will lose the data shape, so we need to set it manually
image.set_shape([None, args.img_size[1], args.img_size[0], 3])
for y in y_true:
    y.set_shape([None, None, None, None, None])
#——————————————————————————————batch
##################
# Model definition
##################

# define yolo-v3 model here
yolo_model = yolov3(args.class_num, args.anchors)
with tf.variable_scope('yolov3'):
    pred_feature_maps = yolo_model.forward(image, is_training=is_training)
loss = yolo_model.compute_loss(pred_feature_maps, y_true)
#y_true和三个特征图一样的维度标签                      ？？？？？自己做标签要怎么做啊？？？？？

#输入(self, y_pred, y_true):y_pred就是三个特征图
# 这个函数就是计算了所有的损失和total_loss
# return [total_loss, loss_xy, loss_wh, loss_conf, loss_class]

y_pred = yolo_model.predict(pred_feature_maps)#可以传进去batch
# 传进来时前向传播的三个feature map
"""
返回的是具体的
坐标值
置信度
哪个类别的概率
boxes是矩形的四个坐标值
# boxes shape[N, (13*13+26*26+52*52)*3, 4]
# confs shape: [N, (13*13+26*26+52*52)*3, 1]
# probs shape: [N, (13 * 13 + 26 * 26 + 52 * 52) * 3, class_num]
"""
################
# register the gpu nms operation here for the following evaluation scheme
pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])#输入只能是1个图片
pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])
gpu_nms_op = gpu_nms(pred_boxes_flag, pred_scores_flag, args.class_num)
# 这里只能输入一张图片 输入一张图片 检测到的所有的值
# 然后返回非极大值抑制nms帅选出来的值 返回
# return boxes, score, label 帅选出来的的值  边框是具体的坐标值
# 返回[N, (13 * 13 + 26 * 26 + 52 * 52) * 3, class_num]   就是行的一个值乘以所有列
# boxes shape[N, (13*13+26*26+52*52)*3, 4]
# pred_scores[N, (13 * 13 + 26 * 26 + 52 * 52) * 3, class_num]
# args.num_class=80
################

if args.restore_part == ['None']:
    args.restore_part = [None]
if args.update_part == ['None']:
    args.update_part = [None]
"""
"--restore_part", nargs='*', type=str, default=['yolov3/darknet53_body'],
"--update_part", nargs='*', type=str, default=['yolov3/yolov3_head'],
"""
saver_to_restore = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(include=args.restore_part))
# exclude=['resnet50/fc']表示加载预训练参数中除了resnet50/fc这一层之外的其他所有参数。
# include=["inceptionv3"]表示只加载inceptionv3这一层的所有参数。
# var_list指定将保存和恢复的变量。它可以作为一个dict或一个列表传递：
# 一个dict名字变量：键是将被用来在检查点文件保存或恢复变量名。
# 变量列表：变量将在检查点文件中使用其op名称进行键控
update_vars = tf.contrib.framework.get_variables_to_restore(include=args.update_part)

tf.summary.scalar('train_batch_statistics/total_loss', loss[0])
tf.summary.scalar('train_batch_statistics/loss_xy', loss[1])
tf.summary.scalar('train_batch_statistics/loss_wh', loss[2])
tf.summary.scalar('train_batch_statistics/loss_conf', loss[3])
tf.summary.scalar('train_batch_statistics/loss_class', loss[4])

global_step = tf.Variable(0, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
#global_step  步数指的是 batch执行了几个
#  collections=[tf.GraphKeys.LOCAL_VARIABLES]  这个有啥用？？？？？？？？？？

# "--use_warm_up", type=lambda x: (str(x).lower() == 'true'), default=False
# parser.add_argument("--warm_up_epoch", type=int, default=5,
#                     help="Warm up training epoches.")热身训练阶段
# parser.add_argument("--warm_up_lr", type=float, default=5e-5,
#                     help="Warm up learning rate.")

#args.use_warm_up这个参数 就是为了是否用  args.warm_up_epoch=5个人batch做热身训练
if args.use_warm_up:#global_step这是个变量 初始值为0是为了记录当前训练了几步

    #args.lr_type  这个参数决定 学习率是 固定的 还是  下降的
    learning_rate = tf.cond(tf.less(global_step, args.train_batch_num * args.warm_up_epoch), 
        lambda: args.warm_up_lr, lambda: config_learning_rate(args, global_step - args.train_batch_num * args.warm_up_epoch))
else:
    learning_rate = config_learning_rate(args, global_step)
tf.summary.scalar('learning_rate', learning_rate)


# parser.add_argument("--save_optimizer", type=lambda x: (str(x).lower() == 'true'), default=False,

#--save_optimizer", type=lambda x: (str(x).lower() == 'true'), default=False
if not args.save_optimizer:
    saver_to_save = tf.train.Saver()

optimizer = config_optimizer(args.optimizer_name, learning_rate)

# "--optimizer_name", type=str, default='adam',
# return tf.train.AdamOptimizer(learning_rate)
if args.save_optimizer:
    saver_to_save = tf.train.Saver()

# set dependencies for BN ops
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#这是一个tensorflow的计算图中内置的一个集合，
# 其中会保存一些需要在训练操作之前完成的操作，并配合tf.control_dependencies函数使用
with tf.control_dependencies(update_ops):#意思是说  要把前面的op都执行之后才能执行  with的函数
    train_op = optimizer.minimize(loss[0], var_list=update_vars, global_step=global_step)



with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), train_iterator.initializer])
    train_handle_value, val_handle_value = sess.run([train_handle, val_handle])

    #parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3.ckpt",
    #               help="The path of the weights to restore.")
    saver_to_restore.restore(sess, args.restore_path)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(args.log_dir, sess.graph)

    print('\n----------- start to train -----------\n')

    # parser.add_argument("--total_epoches", type=int, default=10000,
    #                     help="Total epoches to train.")
    for epoch in range(args.total_epoches):#执行几次   所有的数据集 训练N遍

        for i in range(args.train_batch_num):#batch的数量  遍历这里 就是把 所有的数据训练一遍

            #merged是可视化的合并操作
            #y_pred = yolo_model.predict(pred_feature_maps)#可以传进去batch
            #loss = yolo_model.compute_loss(pred_feature_maps, y_true)
            #global_step = tf.Variable(0, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]) 记录运行哪个步骤
            #learning_rate学习率
            #global_step  步数指的是 batch执行了几个
            _, summary, y_pred_, y_true_, loss_, global_step_, lr = sess.run([train_op, merged, y_pred, y_true, loss, global_step, learning_rate],
                                                                             feed_dict={is_training: True, handle_flag: train_handle_value})

            writer.add_summary(summary, global_step=global_step_)
            #??????????????????????????????

            info = "Epoch: {}, global_step: {}, total_loss: {:.3f}, loss_xy: {:.3f}, loss_wh: {:.3f}, loss_conf: {:.3f}, loss_class: {:.3f}".format(
                epoch, global_step_, loss_[0], loss_[1], loss_[2], loss_[3], loss_[4])
            print(info)

            logging.info(info)   #             ??????????????????先不管




            # --train_evaluation_freq, type=int, default=100,
            if global_step_ % args.train_evaluation_freq == 0 and global_step_ > 0:

                #gpu_nms_op是非极大值抑制  输入进入很多框 然后帅选一些  最终的结果  返回的是
                # 返回[N, (13 * 13 + 26 * 26 + 52 * 52) * 3, class_num]   只能输入一张图片
                # 就是行的一个值乘以所有列
                # # boxes shape[N, (13*13+26*26+52*52)*3, 4]
                # # pred_scores[N, (13 * 13 + 26 * 26 + 52 * 52) * 3, class_num]
                # # args.num_class=80
                        #pred_boxes_flag  这两个是非极大值抑制的输入 的一部分 的占位符
                        #pred_scores_flag
                # recall, precision = evaluate_on_cpu(y_pred_, y_true_, args.class_num, calc_now=True)
                recall, precision = evaluate_on_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, y_pred_, y_true_, args.class_num, calc_now=True)

                info = "===> batch recall: {:.3f}, batch precision: {:.3f} <===".format(recall, precision)
                print(info)
                logging.info(info)

                writer.add_summary(make_summary('evaluation/train_batch_recall', recall), global_step=global_step_)
                writer.add_summary(make_summary('evaluation/train_batch_precision', precision), global_step=global_step_)



            # start to save
            # NOTE: this is just demo. You can set the conditions when to save the weights.


            # --save_freq", type=int, default=500
            if global_step_ % args.save_freq == 0 and global_step_ > 0:
                if loss_[0] <= 2.:
                    saver_to_save.save(sess, args.save_dir + 'model-step_{}_loss_{:4f}_lr_{:.7g}'.format(global_step_, loss_[0], lr))



            # --val_evaluation_freq, type=int, default=100,
            if global_step_ % args.val_evaluation_freq == 0 and global_step_ > 0:
                sess.run(val_iterator.initializer)

                true_positive_dict, true_labels_dict, pred_labels_dict = {}, {}, {}

                val_loss = [0., 0., 0., 0., 0.]


                #这里是每个batch叠加得 batch     的精确率 与召回率  和各种损失
                #这里是每个batch叠加的 loss
                for j in range(args.val_batch_num):

                    y_pred_, y_true_, loss_ = sess.run([y_pred, y_true, loss],
                                                        feed_dict={is_training: False, handle_flag: val_handle_value})


                    true_positive_dict_tmp, true_labels_dict_tmp, pred_labels_dict_tmp = \
                        evaluate_on_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag,
                                        y_pred_, y_true_, args.class_num, calc_now=False)

                    # true_positive_dict, true_labels_dict, pred_labels_dict = {}, {}, {}
                    true_positive_dict = update_dict(true_positive_dict, true_positive_dict_tmp)
                    #update_dict是当true_positive_dict为  空  ，则true_positive_dict=true_positive_dict_tmp
                    #不为空则把   true_positive_dict_tmp增加到里面

                    true_labels_dict = update_dict(true_labels_dict, true_labels_dict_tmp)
                    pred_labels_dict = update_dict(pred_labels_dict, pred_labels_dict_tmp)

                    val_loss = list_add(val_loss, loss_)
                    #val_loss 所有batch的损失值之和



                # make sure there is at least one ground truth object in each image
                # avoid divided by 0
                recall = float(sum(true_positive_dict.values())) / (sum(true_labels_dict.values()) + 1e-6)
                precision = float(sum(true_positive_dict.values())) / (sum(pred_labels_dict.values()) + 1e-6)




                info = "===> Epoch: {}, global_step: {}, recall: {:.3f}, precision: {:.3f}, total_loss: {:.3f}, loss_xy: {:.3f}, loss_wh: {:.3f}, loss_conf: {:.3f}, loss_class: {:.3f}".format(
                    epoch, global_step_, recall, precision, val_loss[0] / args.val_img_cnt, val_loss[1] / args.val_img_cnt, val_loss[2] / args.val_img_cnt, val_loss[3] / args.val_img_cnt, val_loss[4] / args.val_img_cnt)
                print(info)
                logging.info(info)
                writer.add_summary(make_summary('evaluation/val_recall', recall), global_step=epoch)
                writer.add_summary(make_summary('evaluation/val_precision', precision), global_step=epoch)

                writer.add_summary(make_summary('validation_statistics/total_loss', val_loss[0] / args.val_img_cnt), global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/loss_xy', val_loss[1] / args.val_img_cnt), global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/loss_wh', val_loss[2] / args.val_img_cnt), global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/loss_conf', val_loss[3] / args.val_img_cnt), global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/loss_class', val_loss[4] / args.val_img_cnt), global_step=epoch)

        # manually shuffle the training data in a new epoch
        shuffle_and_overwrite(args.train_file)
        sess.run(train_iterator.initializer)#这里进行初始化  就是将数据重新开始
