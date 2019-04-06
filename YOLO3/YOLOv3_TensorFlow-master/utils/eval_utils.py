# coding: utf-8

from __future__ import division, print_function

import numpy as np
from collections import Counter

from utils.nms_utils import cpu_nms, gpu_nms


def calc_iou(pred_boxes, true_boxes):
    '''
    Maintain an efficient way to calculate the ios matrix using the numpy broadcast tricks.
    shape_info: pred_boxes: [N, 4]
                true_boxes: [V, 4]
    '''

    # [N, 1, 4]
    pred_boxes = np.expand_dims(pred_boxes, -2)
    # [1, V, 4]
    true_boxes = np.expand_dims(true_boxes, 0)

    # [N, 1, 2] & [1, V, 2] ==> [N, V, 2]
    intersect_mins = np.maximum(pred_boxes[..., :2], true_boxes[..., :2])
    intersect_maxs = np.minimum(pred_boxes[..., 2:], true_boxes[..., 2:])
    intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)

    # shape: [N, V]
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    # shape: [N, 1, 2]
    pred_box_wh = pred_boxes[..., 2:] - pred_boxes[..., :2]
    # shape: [N, 1]
    pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    # [1, V, 2]
    true_boxes_wh = true_boxes[..., 2:] - true_boxes[..., :2]
    # [1, V]
    true_boxes_area = true_boxes_wh[..., 0] * true_boxes_wh[..., 1]

    # shape: [N, V]
    iou = intersect_area / (pred_box_area + true_boxes_area - intersect_area + 1e-10)

    return iou


def evaluate_on_cpu(y_pred, y_true, num_classes, calc_now=True, score_thresh=0.5, iou_thresh=0.5):
    # y_pred -> [None, 13, 13, 255],
    #           [None, 26, 26, 255],
    #           [None, 52, 52, 255],

    num_images = y_true[0].shape[0]
    true_labels_dict = {i: 0 for i in range(num_classes)}  # {class: count}
    pred_labels_dict = {i: 0 for i in range(num_classes)}
    true_positive_dict = {i: 0 for i in range(num_classes)}
    #{0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    for i in range(num_images):
        true_labels_list, true_boxes_list = [], []
        for j in range(3):  # three feature maps
            # shape: [13, 13, 3, 80]
            true_probs_temp = y_true[j][i][..., 5:]
            # shape: [13, 13, 3, 4] (x_center, y_center, w, h)
            true_boxes_temp = y_true[j][i][..., 0:4]

            # [13, 13, 3]
            object_mask = true_probs_temp.sum(axis=-1) > 0

            # [V, 3] V: Ground truth number of the current image
            true_probs_temp = true_probs_temp[object_mask]
            # [V, 4]
            true_boxes_temp = true_boxes_temp[object_mask]

            # [V], labels
            true_labels_list += np.argmax(true_probs_temp, axis=-1).tolist()
            # [V, 4] (x_center, y_center, w, h)
            true_boxes_list += true_boxes_temp.tolist()

        if len(true_labels_list) != 0:
            for cls, count in Counter(true_labels_list).items():
                #aa [0, 1, 1]
                # bb Counter({1: 2, 0: 1})    Counter是统计相同元素的个数
                # 0 1
                # 1 2
                true_labels_dict[cls] += count

        # [V, 4] (xmin, ymin, xmax, ymax)
        true_boxes = np.array(true_boxes_list)
        box_centers, box_sizes = true_boxes[:, 0:2], true_boxes[:, 2:4]
        true_boxes[:, 0:2] = box_centers - box_sizes / 2.
        true_boxes[:, 2:4] = true_boxes[:, 0:2] + box_sizes

        # [1, xxx, 4]
        pred_boxes = y_pred[0][i:i + 1]
        pred_confs = y_pred[1][i:i + 1]
        pred_probs = y_pred[2][i:i + 1]#pred_probs预测可能性

        # pred_boxes: [N, 4]
        # pred_confs: [N]
        # pred_labels: [N]
        # N: Detected box number of the current image
        pred_boxes, pred_confs, pred_labels = cpu_nms(pred_boxes, pred_confs * pred_probs, num_classes,
                                                      score_thresh=score_thresh, iou_thresh=iou_thresh)

        # len: N
        pred_labels_list = [] if pred_labels is None else pred_labels.tolist()
        if pred_labels_list == []:
            continue

        # calc iou
        # [N, V]
        iou_matrix = calc_iou(pred_boxes, true_boxes)
        # [N]
        max_iou_idx = np.argmax(iou_matrix, axis=-1)

        correct_idx = []
        correct_conf = []
        for k in range(max_iou_idx.shape[0]):
            pred_labels_dict[pred_labels_list[k]] += 1
            match_idx = max_iou_idx[k]  # V level
            if iou_matrix[k, match_idx] > iou_thresh and true_labels_list[match_idx] == pred_labels_list[k]:
                if not match_idx in correct_idx:
                    correct_idx.append(match_idx)
                    correct_conf.append(pred_confs[k])
                else:
                    same_idx = correct_idx.index(match_idx)
                    if pred_confs[k] > correct_conf[same_idx]:
                        correct_idx.pop(same_idx)
                        correct_conf.pop(same_idx)
                        correct_idx.append(match_idx)
                        correct_conf.append(pred_confs[k])

        for t in correct_idx:
            true_positive_dict[true_labels_list[t]] += 1

    if calc_now:
        # avoid divided by 0
        recall = sum(true_positive_dict.values()) / (sum(true_labels_dict.values()) + 1e-6)
        precision = sum(true_positive_dict.values()) / (sum(pred_labels_dict.values()) + 1e-6)

        return recall, precision
    else:
        return true_positive_dict, true_labels_dict, pred_labels_dict

        # gpu_nms_op是非极大值抑制  输入进入很多框 然后帅选一些  最终的结果  返回的是
        # 返回[N, (13 * 13 + 26 * 26 + 52 * 52) * 3, class_num]   只能输入一张图片
        # 就是行的一个值乘以所有列
        # # boxes shape[N, (13*13+26*26+52*52)*3, 4]
        # # pred_scores[N, (13 * 13 + 26 * 26 + 52 * 52) * 3, class_num]
        # # args.num_class=80
        # pred_boxes_flag  这两个是非极大值抑制的输入 的一部分 的占位符
        # pred_scores_flag
        # recall, precision = evaluate_on_cpu(y_pred_, y_true_, args.class_num, calc_now=True)
        # recall, precision = evaluate_on_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, y_pred_, y_true_,
        #                                     args.class_num, calc_now=True)
def evaluate_on_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, y_pred, y_true, num_classes, calc_now=True, score_thresh=0.5, iou_thresh=0.5):
    # y_pred -> [None, 13, 13, 255],
    #           [None, 26, 26, 255],
    #           [None, 52, 52, 255],


    # y_pred
    # 返回的是具体的
    # 坐标值
    # 置信度
    # 哪个类别的概率
    # boxes是矩形的四个坐标值
    # boxes shape: [N, (13*13+26*26+52*52)*3, 4]
    # confs shape: [N, (13*13+26*26+52*52)*3, 1]
    # probs shape: [N, (13 * 13 + 26 * 26 + 52 * 52) * 3, class_num]

    num_images = y_true[0].shape[0]#图片batch
    true_labels_dict = {i: 0 for i in range(num_classes)}  # {class: count}80个0
    pred_labels_dict = {i: 0 for i in range(num_classes)}# {class: count}80个0
    true_positive_dict = {i: 0 for i in range(num_classes)}# {class: count}80个0



    for i in range(num_images):

        #true_probs_temp 真实的概率值 里面  因为是每个网格  都需要一个预测属于哪一类[80]的概率
        # 就需要排除 没有需要检测到物体的  标签  这个在   精确率召回率 里面用不到

        true_labels_list, true_boxes_list = [], []
        for j in range(3):  # three feature maps
            # shape: [13, 13, 3, 80]   [..., 5:]里面的维度是 [13, 13, 3, 80]
            true_probs_temp = y_true[j][i][..., 5:]

            # shape: [13, 13, 3, 4] (x_center, y_center, w, h)
            true_boxes_temp = y_true[j][i][..., 0:4]

            # [13, 13, 3]
            object_mask = true_probs_temp.sum(axis=-1) > 0
            #object_mask  是Fauls  和 True 值
            #因为 [13, 13, 3, 80] 的80是one_hot编码加起来，判断是否大于0

            # 来判断是否有物体

            #返回有到物体的概率  真实

            # object_mask表示在[13, 13, 3]这里哪里有检测到  物体
            #shape: [13, 13, 3, 80] 与 [13, 13, 3]个turehe false  返回  v*3, 80
            # [V, 3] V: Ground truth number of the current image
            true_probs_temp = true_probs_temp[object_mask]
            #true_probs_temp是得到在 [13* 13]里面有检测到物体的哪个网格   假如有2个网格检测到  则返回[2，2，3] 4，80
            """
aa = np.array([[1,2],[3,4]])
bb = np.array([[True,False],[True,True]])
cc = aa[bb]
print(cc)     [1 3 4]      
            """
            # [V, 4]  输入true_boxes_temp  为shape: [13, 13, 3, 4]
            # object_mask表示在[13, 13, 3]这里哪里有检测到  物体  [13, 13, 3]个turehe false  返回
            true_boxes_temp = true_boxes_temp[object_mask]
            #假如返回   2，2，4   意思是  13，13的网格里面   有两个网格 检测到的框   6，4

            #true_boxes_temp  是有物体的  的边框  和true_probs_temp  使用相同的   object_mask
            #也就是说   概率  和边框   还是  一一对应



            # [V], labels
            # #true_probs_temp  假如是 6，80
            # 意思是  13，13的网格里面   有两个网格 检测到的框   6个，80
            true_labels_list += np.argmax(true_probs_temp, axis=-1).tolist()#为1的下标
            #[6]   里面的值是80 个概率值的最大下标
            #true_labels_list 只是下标值  不是one_hot编码

            #true_probs_temp 是 one_hot编码  转化为  [N]

            # [V, 4] (x_center, y_center, w, h)
            true_boxes_list += true_boxes_temp.tolist()#这个+是把元素添加到里面
            #true_boxes_list就是打标签的矩形框值  和ceil网格没有了对应的关系
            # 6，4
            #6  是假设

        if len(true_labels_list) != 0:
            for cls, count in Counter(true_labels_list).items():
                true_labels_dict[cls] += count
                #把列表  转化为  计数的字典   标签字典  键 是80个概率的下标  值是个数

        # [V, 4] (xmin, ymin, xmax, ymax)
        true_boxes = np.array(true_boxes_list)
        box_centers, box_sizes = true_boxes[:, 0:2], true_boxes[:, 2:4]
        true_boxes[:, 0:2] = box_centers - box_sizes / 2.
        true_boxes[:, 2:4] = true_boxes[:, 0:2] + box_sizes

        # [1, xxx, 4]
        # y_pred
        # 返回的是具体的
        # 坐标值
        # 置信度
        # 哪个类别的概率
        # boxes是矩形的四个坐标值
        # boxes shape[N, (13*13+26*26+52*52)*3, 4]   0
        # confs shape: [N, (13*13+26*26+52*52)*3, 1]  1
        # probs shape: [N, (13 * 13 + 26 * 26 + 52 * 52) * 3, class_num]   2
        pred_boxes = y_pred[0][i:i + 1] #这里就是遍历 不同的尺度的每一个图片    [i:i + 1]表示每一个图片
        pred_confs = y_pred[1][i:i + 1]    # i是图片
        pred_probs = y_pred[2][i:i + 1]

        # pred_boxes: [N, 4]
        # pred_confs: [N]
        # pred_labels: [N]
        # N: Detected box number of the current image
        pred_boxes, pred_confs, pred_labels = sess.run(gpu_nms_op,
                                                       feed_dict={pred_boxes_flag: pred_boxes,
                                                                  pred_scores_flag: pred_confs * pred_probs})
        # len: N
        pred_labels_list = [] if pred_labels is None else pred_labels.tolist()
        if pred_labels_list == []:
            continue

        # calc iou
        # [N, V]  pred_boxes, true_boxes只有 坐标  没有与网格有对应的关系
        iou_matrix = calc_iou(pred_boxes, true_boxes)  #返回置信度
        #pred_boxes 的一个值 与真实边框 所有的值  的置信度
        # [N]
        max_iou_idx = np.argmax(iou_matrix, axis=-1) #最大的置信度的下标

        correct_idx = []
        correct_conf = []



        #------------------------
        for k in range(max_iou_idx.shape[0]):

#得到  预测的  ono_hot编码  字典的方式

            #K  0,1,2,3     pred_labels_list   len [N]
            pred_labels_dict[pred_labels_list[k]] += 1 #pred_labels_dict是0到79的键，值为0 += 1  是把键为K的值为1
            #实际就是将[N]转换为为one_hot编码

            #预测边框与真实边框的置信度
            match_idx = max_iou_idx[k]  # V level   [N, V] 的V中 iou最大值的索引



            #iou_matrix 是预测的边框  与真实边框的  置信度      [N, V] N个预测的边框 与V个真实边框的  置信度

            #iou_matrix[k, match_idx]最大置信度的值    true_labels_list[match_idx]

            #  match_idx  是最大置信度的下标索引值       True   _labels_list 只是下标值[N] 不是one_hot编码
                                                    #    pred   _labels_list     len [N]
            # 6个，80                                            是0和1值得比较
            #match_idx  本质是   预测的一个边框 与所有 真实 存在物体的 边框的  置信度  最大的那个
                                                                   #match_idx = max_iou_idx[k]
            #      假如 预测的边框有6个  与所有真实边框V 求置信度   得到   6，V  max_iou_idx[k]则是len=[6]
                                            #对应的是V的位置  意思是 预测的第一个索引  对应真实的哪一个
            #                                 match_idx则是 [6]里面操你个0-5的max_iou_idx的值  值则是对应one_hot80的位置
            #                                          rue_labels_list[match_idx] 则是从 [N]


            # 下面 是 把  match_idx 预测的边框  与所有真实边框的置信度 的最大值的索引
            # 加入到  correct_idx  必须按照顺序
            #   match_idx的个数与 k是相同的     pred_confs 加入到correct_conf
            if iou_matrix[k, match_idx] > iou_thresh and true_labels_list[match_idx] == pred_labels_list[k]:

                if not match_idx in correct_idx:    #correct_idx=[]
                    #如果a不在列表b中，那么就执行冒号后面的语句

                    correct_idx.append(match_idx)   # correct_conf = []
                    correct_conf.append(pred_confs[k])
                else:
                    same_idx = correct_idx.index(match_idx)
                    if pred_confs[k] > correct_conf[same_idx]:
                        #pop()函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。
                        correct_idx.pop(same_idx)
                        correct_conf.pop(same_idx)
                        correct_idx.append(match_idx)
                        correct_conf.append(pred_confs[k])

# 得到  真实的  ono_hot编码  字典的方式
        for t in correct_idx:
            # true_positive_dict = {i: 0 for i in range(num_classes)}  # {class: count}80个0
            #true_labels_list   len [N]
            true_positive_dict[true_labels_list[t]] += 1
            # 实际就是将[N]转换为为one_hot编码


    if calc_now:
        # avoid divided by 0
        recall = sum(true_positive_dict.values()) / (sum(true_labels_dict.values()) + 1e-6)
        precision = sum(true_positive_dict.values()) / (sum(pred_labels_dict.values()) + 1e-6)

        return recall, precision
    else:
        return true_positive_dict, true_labels_dict, pred_labels_dict