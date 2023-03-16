#!/usr/bin/env python
# -*- coding  : utf-8 -*-
# @Time       : 2022/2/27 21:37
# @Author     : wanghaoran
# @Site       : SCNU
# @File       : video_tools.py
# @Description: 关于视频初步处理的函数工具集
# @Software   : PyCharm
import dlib
import cv2
import scipy
import numpy as np
from PIL import Image, ImageDraw
# import SkinDetector

"""
1、得到多个roi的值
2、得到多尺度ROI和背景区域信号各自的平均值
3、返回额头三维ROI数组
4、人脸皮肤分割
"""



# IOU计算函数
def cal_iou(box1, box2):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return:
    """
    [[xmin1, ymin1], [xmax1, ymin1], [xmin1, ymax1], [xmax1, ymax1]] = box1[0, :, :]
    [[xmin2, ymin2], [xmax2, ymin2], [xmin2, ymax2], [xmax2, ymax2]] = box2[0, :, :]
    # xmin2, ymin2, xmax2, ymax2 = box2[0,:,:]
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # b1的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # b2的面积

    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    a1 = w * h  # C∩G的面积
    a2 = s1 + s2 - a1
    iou = a1 / a2  # iou = a1/ (s1 + s2 - a1)
    return iou



def gridding_value(xmin, ymin, boxw, boxh, frame):
    """
    得到多个网格的均值
    """
    frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    left = xmin
    right = xmin + boxw
    top = ymin
    bottom = ymin + boxh
    # 得到扩大后的人脸rectangle，l-1 level
    left_e = int(left - (right - left) * 0.1)
    right_e = int(right + (right - left) * 0.1)
    top_e = int(top - (bottom - top) * 0.2)
    bottom_e = int(bottom + (bottom - top) * 0.2)

    grid_means = [] # 用来放每个grid的值
    show_frame1 = frame.copy()

    row = np.linspace(top_e, bottom_e, 6)
    lin = np.linspace(left_e, right_e, 6)
    for i ,element_row in enumerate(row[:-1]):  # 删除最后一个数
        for j , element_lin in enumerate(lin[:-1]):
            grid_mean = np.mean(frame1[int(row[i]):int(row[i+1]), int(lin[j]):int(lin[j+1])])
            grid_means.append(grid_mean)
            cv2.rectangle(show_frame1, (int(lin[j]), int(row[i])), (int(lin[j+1]), int(row[i+1])), (255, 0, 0), 1)
            # cv2.rectangle(show_frame1, (int(left_e), int(top_e)), (int(right_e), int(bottom_e)), (0, 0, 255), 1)

    return show_frame1, grid_means


# 得到额头rectangle边界
def get_forehead_boundary_gray(frame1, detector1, predictor1):
    """
    获取灰度帧额头矩形ROI边界值
    :param frame1:
    :param detector1:
    :param predictor1:
    :return:
    """
    rects = detector1(frame1, 0)  # 人脸数rects（耗时），必须是灰度化的图像呢
    if len(rects) == 1:
        face_points = predictor1(frame1, rects[0])  #
        points = np.zeros((len(face_points.parts()), 2))  # 将这些点存储在Numpy数组array中，这样我们就可以很容易地通过切片得到x和y的最小值和最大值68行2列
        for j, part in enumerate(face_points.parts()):  # numerate用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
            points[j] = (part.x, part.y)  # 把特征点的X坐标和Y坐标存进points数组

        # 得到额头矩形的左右上下边界值
        min_x = int(points[21, 0])
        min_y = int(min(points[21, 1], points[22, 1]))
        max_x = int(points[22, 0])
        max_y = int(max(points[21, 1], points[22, 1]))
        left1 = min_x
        right1 = max_x
        top1 = min_y - (max_x - min_x)
        bottom1 = max_y * 0.98
        # boundary1 = [left, right, top, bottom]
        # # 为什么要这样做复制一下？
        show_frame1 = frame1.copy()
        # # 用于在任何图像上绘制矩形，返回带矩形的原图
        cv2.rectangle(show_frame1, (int(left1), int(top1)), (int(right1), int(bottom1)), (255, 255, 0), 3)
        # fh_roi = frame[int(top):int(bottom), int(left):int(right)]  # 得到额头ROI
        # fh_mean = np.mean(fh_roi)

        return show_frame1, left1,  right1, top1, bottom1  # 返回值是额头ROI的平均值和画出矩形框的图


# 得到人脸rectangle边界
def get_face_rectangle(frame1, detector1):
    """
    得到人脸rectangle边界
    :param frame1: 输入的二维灰度图片
    :param detector1: dlib函数
    :return:
    """
    rects = detector1(frame1, 0)  # 人脸数rects（耗时），必须是灰度化的图像呢
    rect = rects[0]  # 得到第一个人脸
    # 得到检测到的人脸，l0 level 的rectangle
    left1, right1, top1, bottom1 = rect.left(), rect.right(), rect.top(), rect.bottom()
    return left1, right1, top1, bottom1


# 得到整张人脸rectangle的均值
def get_face_rectangle_average(frame1, detector1):
    """
    得到人脸rectangle边界
    :param frame1: 输入的二维灰度图片
    :param detector1: dlib函数
    :return:
    """
    rects = detector1(frame1, 0)  # 人脸数rects（耗时），必须是灰度化的图像呢
    if len(rects) == 1:  # 确保只有一个人脸被检测到
        rect = rects[0]

        # 得到检测到的人脸，l0 level 的rectangle
        left, right, top, bottom = rect.left(), rect.right(), rect.top(), rect.bottom()
        face_mean = np.mean(frame1[top:bottom, left:right])
        return face_mean
    else:
        return 0


# 1对一帧frame进行处理得到mask与对应的各个region
def extract_mask_and_regions(frame1, detector, predictor):
    """
    杜克大学2019年那篇近红外CVPR修改：https://ieeexplore.ieee.org/abstract/document/8803109
    输入一帧frame
    得到一帧图片的mask与各个region的均值、标注plot的一帧图
    效果不是很好
    :param frame1:灰度化后的图片,单通道
    :param detector:dlib函数
    :param predictor:dlib函数
    :return:
    mask:
    region_values: array，25行1列
    show_frame1:
    regions_keys1：
    """
    rects = detector(frame1, 0)
    if len(rects) == 1:
        rect = rects[0]
        face_points = predictor(frame1, rect)
        points = np.zeros((len(face_points.parts()), 2))  # 新建81行2列array，0填充
        for m, part in enumerate(face_points.parts()):
            points[m] = (part.x, part.y)

        # add forehead添加上额头部分
        face_length = np.max(points[:, 1]) - np.min(points[:, 1])  # 计算最高点与最低点的距离
        idx_forehead = np.arange(18, 27)  # 返回一维array：18 19 ...26
        aux_fh = np.array(points[idx_forehead, :])  # 18到26行array化，赋予aux_fh，9个
        aux_fh[:, 1] = aux_fh[:, 1] - face_length * 0.25  # aux_fh全部减去四分之一张脸高度？用作参考
        # 新增六个
        new_points = np.zeros([6, 2])  # 六行两列0array
        # between eyebrows
        new_points[0, :] = (points[21, :] + points[22, :]) / 2
        new_points[1, :] = (aux_fh[3, :] + aux_fh[4, :]) / 2
        # middle left cheek
        new_points[2, :] = (points[48, :] + points[3, :]) / 2
        new_points[3, :] = (points[31, :] + points[1, :]) / 2
        # middle left cheek
        new_points[4, :] = (points[54, :] + points[13, :]) / 2
        new_points[5, :] = (points[35, :] + points[15, :]) / 2
        # concatenate ALL 68+9+6=83
        boosted_landmarks = np.concatenate([points, aux_fh, new_points], axis=0)  # 按行拼起来83个点，二维
        # 根据灰度化的frame1和扩充后的boosted_landmarks得到人脸mask,各个region的value不同。regions_keys有各个region的key
        masks, region_values, regions_keys1 = get_mask_region_values(frame1, boosted_landmarks)
        show_frame1 = frame1.copy()  # 为什么要这样做复制一下？
        left, right, top, bottom = rect.left(), rect.right(), rect.top(), rect.bottom()
        # 得到扩大后的人脸rectangle，l-1 level
        left_e = int(left - (right - left) * 0.1)
        right_e = int(right + (right - left) * 0.1)
        top_e = int(top - (bottom - top) * 0.5)
        bottom_e = int(bottom + (bottom - top) * 0.5)
        cv2.rectangle(show_frame1, (int(left_e), int(top_e)), (int(right_e), int(bottom_e)), (255, 255, 0), 1)  # 叠加绘制扩大化后的整个人脸r
        for j, part in enumerate(face_points.parts()):  # numerate用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
            points[j] = (part.x, part.y)
            cv2.circle(show_frame1, (part.x, part.y), 1, (255, 0, 0), 2)
        return masks, region_values, show_frame1, regions_keys1


# 1.1 得到mask和region的值?
def get_mask_region_values(frame1, landmarks_t):
    """
    得到一张图上25个mask和25个region的key
    :param frame1: 一帧frame,二维
    :param landmarks_t: 扩充后的landmarks共83个，二维
    :return:返回二维mask，各个region的数值不同、人脸分区一维的key长度25，array里面放的是对应字符串
    """
    region = dict()  # 新建一个字典用来放25个region
    # Forehead
    region['LFH1'] = np.array([20, 21, 22, 78, 79, 72, 71, 70]) - 1
    region['RFH1'] = np.array([78, 23, 24, 25, 75, 74, 73, 79]) - 1
    region['LFH2'] = np.array([1, 18, 19, 20, 70, 69]) - 1
    region['RFH2'] = np.array([17, 77, 76, 75, 25, 26, 27]) - 1
    # eyes region
    region['LE1'] = np.array([1, 37, 38, 20, 19, 18]) - 1
    region['RE1'] = np.array([17, 27, 26, 25, 45, 46]) - 1
    region['LE2'] = np.array([38, 39, 40, 28, 78, 22, 21, 20]) - 1
    region['RE2'] = np.array([28, 43, 44, 45, 25, 24, 23, 78]) - 1
    # nose region
    region['LN1'] = np.array([32, 33, 34, 31, 30, 29, 28, 40]) - 1
    region['RN1'] = np.array([34, 35, 36, 43, 28, 29, 30, 31]) - 1
    # check region
    region['LCU1'] = np.array([81, 32, 40, 41, 42]) - 1
    region['LCU2'] = np.array([81, 42, 37, 1, 2, 3]) - 1
    region['RCU1'] = np.array([36, 83, 47, 48, 43]) - 1
    region['RCU2'] = np.array([83, 15, 16, 17, 46, 47]) - 1
    region['LCD1'] = np.array([81, 32, 49, 80]) - 1
    region['LCD2'] = np.array([80, 81, 3, 4]) - 1
    region['RCD1'] = np.array([36, 83, 82, 55]) - 1
    region['RCD2'] = np.array([83, 82, 14, 15]) - 1
    # mouth region
    region['LM'] = np.array([49, 50, 51, 52, 34, 33, 32]) - 1
    region['RM'] = np.array([52, 53, 54, 55, 36, 35, 34]) - 1
    # chin region
    region['LC1'] = np.array([4, 5, 6, 60, 49]) - 1
    region['RC1'] = np.array([55, 14, 13, 12, 56]) - 1
    region['LC2'] = np.array([6, 7, 8, 59, 60]) - 1
    region['RC2'] = np.array([12, 56, 57, 10, 11]) - 1
    region['CC'] = np.array([10, 57, 58, 59, 8, 9]) - 1

    # 一帧大小zero array用来放mask
    mask1 = np.zeros(frame1.shape)
    region_values = np.zeros([25, 1])
    # Build Region Masks 区域mask
    cont = 1
    for k in region.keys():
        mask_k, region_value = mask_generation_simple(frame1, landmarks_t, land_index=region[k])
        mask1 = np.maximum(cont * mask_k, mask1)  # 逐元素比较两个array的大小，在这里迭代后就成了每个区域的mask相加到一张图上，二维
        region_values[cont-1] = region_value
        cont += 1  # 每块的mask数值都不一样

    labels = np.array([key for key in region.keys()])  # keys作为字符串也可以array化
    mask1[mask1 > 0] = 1  # mask区域置1，本来应该是1到25的
    return mask1, region_values, labels
    # 返回mask二维、region_values是25个region的均值 region的key的array长度应该是25


# 1.1.1 得到mask和这个region的value of average
def mask_generation_simple(frame1, land_drift, land_index=None):  # 输入是一帧video，扩充的landmarks
    """
    得到一个region的1填充mask；还有一种用法，输入land_index=None直接人脸mask减去眼睛嘴巴等
    :param frame1:一帧灰度化后的图片
    :param land_drift:扩充后的landmarks，二维
    :param land_index:单个region的value，是个一维array
    :return:返回单个region的1填充的ROI.单个region的均值
    """
    if land_index is not None:  # 如果在计算某个区域就画出该区域的mask，返回为mask_l
        polygon = [(land_drift[a, 0], land_drift[a, 1]) for a in land_index]  # list化? 边界点坐标值
        imgmask = Image.new('L', (frame1.shape[1], frame1.shape[0]), 0)  # 参数0则代表填充黑色
        ImageDraw.Draw(imgmask).polygon(polygon, outline=1, fill=1)  # fill是用于填充的颜色为1，outline是边界颜色
        mask_l = np.array(imgmask)  # Image类型array化
    else:  # 这个else作用? region是没有None的value的吧
        ms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 25, 24, 19, 18, 17]
        e1 = [36, 37, 38, 39, 40, 41]
        e2 = [42, 43, 44, 45, 46, 47]
        bc = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
        polygon_ms = [(land_drift[a, 0], land_drift[a, 1]) for a in ms]
        polygon_e1 = [(land_drift[a, 0], land_drift[a, 1]) for a in e1]
        polygon_e2 = [(land_drift[a, 0], land_drift[a, 1]) for a in e2]
        polygon_bc = [(land_drift[a, 0], land_drift[a, 1]) for a in bc]
        # 功能？先画出整个的mask?
        imgmask = Image.new('L', (frame1.shape[1], frame1.shape[0]), 0)
        ImageDraw.Draw(imgmask).polygon(polygon_ms, outline=1, fill=1)
        mask_l = np.array(imgmask)
        # remove eye 1
        imgmask = Image.new('L', (frame1.shape[1], frame1.shape[0]), 0)
        ImageDraw.Draw(imgmask).polygon(polygon_e1, outline=1, fill=1)
        mask_l -= np.array(imgmask)
        # remove eye 2
        imgmask = Image.new('L', (frame1.shape[1], frame1.shape[0]), 0)
        ImageDraw.Draw(imgmask).polygon(polygon_e2, outline=1, fill=1)
        mask_l -= np.array(imgmask)
        # remove mouth
        imgmask = Image.new('L', (frame1.shape[1], frame1.shape[0]), 0)
        ImageDraw.Draw(imgmask).polygon(polygon_bc, outline=1, fill=1)
        mask_l -= np.array(imgmask)

    region_value = np.mean(frame1[mask_l > 0])  # 待验证
    return mask_l, region_value  # 返回mask的array，是一帧图像shape，单个region的ROI填充1.和单个region的均值


# 返回人脸mask和ROI均值，太慢了,待修改
def get_mask(frame1, detector, predictor):
    """
    得到整张人脸mask、average_value
    :param frame1: 输入一张人脸图片
    :param detector:
    :param predictor:
    :return:
    """
    rects = detector(frame1, 0)
    if len(rects) == 1:
        rect = rects[0]
        face_points = predictor(frame1, rect)
        points = np.zeros((len(face_points.parts()), 2))  # 新建81行2列array，0填充
        for m, part in enumerate(face_points.parts()):
            points[m] = (part.x, part.y)

        # add forehead添加上额头部分
        face_length = np.max(points[:, 1]) - np.min(points[:, 1])  # 计算最高点与最低点的距离
        idx_forehead = np.arange(18, 27)  # 返回一维array：18 19 ...26
        aux_fh = np.array(points[idx_forehead, :])  # 18到26行array化，赋予aux_fh，9个
        aux_fh[:, 1] = aux_fh[:, 1] - face_length * 0.25  # aux_fh全部减去四分之一张脸高度？用作参考
        # 新增六个
        new_points = np.zeros([6, 2])  # 六行两列0array
        # between eyebrows
        new_points[0, :] = (points[21, :] + points[22, :]) / 2
        new_points[1, :] = (aux_fh[3, :] + aux_fh[4, :]) / 2
        # middle left cheek
        new_points[2, :] = (points[48, :] + points[3, :]) / 2
        new_points[3, :] = (points[31, :] + points[1, :]) / 2
        # middle left cheek
        new_points[4, :] = (points[54, :] + points[13, :]) / 2
        new_points[5, :] = (points[35, :] + points[15, :]) / 2
        # concatenate ALL 68+9+6=83
        land_drift = np.concatenate([points, aux_fh, new_points], axis=0)  # 按行拼起来83个点，二维

        ms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 25, 24, 19, 18, 17]
        e1 = [36, 37, 38, 39, 40, 41]
        e2 = [42, 43, 44, 45, 46, 47]
        bc = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
        polygon_ms = [(land_drift[a, 0], land_drift[a, 1]) for a in ms]
        polygon_e1 = [(land_drift[a, 0], land_drift[a, 1]) for a in e1]
        polygon_e2 = [(land_drift[a, 0], land_drift[a, 1]) for a in e2]
        polygon_bc = [(land_drift[a, 0], land_drift[a, 1]) for a in bc]
        # 功能？先画出整个的mask?
        imgmask = Image.new('L', (frame1.shape[1], frame1.shape[0]), 0)
        ImageDraw.Draw(imgmask).polygon(polygon_ms, outline=1, fill=1)
        mask_l = np.array(imgmask)
        # remove eye 1
        imgmask = Image.new('L', (frame1.shape[1], frame1.shape[0]), 0)
        ImageDraw.Draw(imgmask).polygon(polygon_e1, outline=1, fill=1)
        mask_l -= np.array(imgmask)
        # remove eye 2
        imgmask = Image.new('L', (frame1.shape[1], frame1.shape[0]), 0)
        ImageDraw.Draw(imgmask).polygon(polygon_e2, outline=1, fill=1)
        mask_l -= np.array(imgmask)
        # remove mouth
        imgmask = Image.new('L', (frame1.shape[1], frame1.shape[0]), 0)
        ImageDraw.Draw(imgmask).polygon(polygon_bc, outline=1, fill=1)
        mask_l -= np.array(imgmask)

    region_value = np.mean(frame1[mask_l])  # 待验证
    frame1[mask_l == 0] = 0
    return mask_l, region_value, frame1  # 返回mask的array,是一帧图像shape,单个region的ROI填充1.和单个region的均值,show frame


# 2得到多尺度ROI和背景区域信号各自的平均值
def get_face_value(frame1, detector):
    """

    :param frame1:   输入的一帧灰度化后的图片，单通道
    :param detector:
    :return:
    show_frame1:     plot图
    level_mean:      各个尺度的平均值,长度为5的list
    background_mean: 背景平均值
    """
    rects = detector(frame1, 0)  # rectangles[[(323, 54) (410, 141)]]，是一个set还是dictionary？
    if len(rects) == 1:  # 确保只有一个人脸被检测到
        rect = rects[0]  # rects[0]应该是 [(323, 54) (410, 141)] class 'dlib.dlib.rectangle'

        # 得到检测到的人脸，l0 level 的rectangle
        left, right, top, bottom = rect.left(), rect.right(), rect.top(), rect.bottom()

        # 得到扩大后的人脸rectangle，l-1 level
        left_e = int(left - (right - left)*0.1)
        right_e = int(right + (right - left)*0.1)
        top_e = int(top - (bottom - top)*0.5)
        bottom_e = int(bottom + (bottom - top)*0.5)

        # l1 level的rectangle
        left1 = int(left + (right - left) * 0.2)
        right1 = int(right - (right - left) * 0.2)
        top1 = int(top + (bottom - top) * 0.2)
        bottom1 = int(bottom - (bottom - top) * 0.2)

        # l2 level的rectangle
        left2 = int(left + (right - left) * 0.3)
        right2 = int(right - (right - left) * 0.3)
        top2 = int(top + (bottom - top) * 0.3)
        bottom2 = int(bottom - (bottom - top) * 0.3)

        # l3 level的rectangle
        left3 = int(left + (right - left) * 0.4)
        right3 = int(right - (right - left) * 0.4)
        top3 = int(top + (bottom - top) * 0.4)
        bottom3 = int(bottom - (bottom - top) * 0.4)

        show_frame1 = frame1.copy()  # 为什么要这样做复制一下？

        cv2.rectangle(show_frame1, (int(left_e), int(top_e)), (int(right_e), int(bottom_e)), (255, 255, 0), 1)  # 叠加绘制扩大化后的整个人脸rectangle
        cv2.rectangle(show_frame1, (int(left), int(top)), (int(right), int(bottom)), (255, 255, 0), 1)  # 叠加绘制额头roi矩形
        cv2.rectangle(show_frame1, (int(left1), int(top1)), (int(right1), int(bottom1)), (255, 255, 0), 1)  # 叠加绘制 l1 level 的rectangle
        cv2.rectangle(show_frame1, (int(left2), int(top2)), (int(right2), int(bottom2)), (255, 255, 0), 1)  # 叠加绘制 l2
        cv2.rectangle(show_frame1, (int(left3), int(top3)), (int(right3), int(bottom3)), (255, 255, 0), 1)  # 叠加绘制 l3

        level_e = frame1[int(top_e):int(bottom_e), int(left_e):int(right_e)]  # 得到额头ROI 是一个2维array
        level_0 = frame1[int(top):int(bottom), int(left):int(right)]  # 得到额头ROI 是一个2维array
        level_1 = frame1[int(top1):int(bottom1), int(left1):int(right1)]  # 得到额头ROI 是一个2维array
        level_2 = frame1[int(top2):int(bottom2), int(left2):int(right2)]  # 得到额头ROI 是一个2维array
        level_3 = frame1[int(top3):int(bottom3), int(left3):int(right3)]  # 得到额头ROI 是一个2维array
        level_e_mean = np.mean(level_e)  #
        level_0_mean = np.mean(level_0)  #
        level_1_mean = np.mean(level_1)  #
        level_2_mean = np.mean(level_2)  #
        level_3_mean = np.mean(level_3)  #
        level_mean = [level_e_mean, level_0_mean, level_1_mean, level_2_mean, level_3_mean]  # list只能是一维的吧

        # 获得背景信号的平均值
        mask = np.ones_like(frame1, dtype=bool)  # 生成一个同shape的1填充的array
        mask[int(top_e):int(bottom_e), int(left_e):int(right_e)] = False  # mask部分填充0
        background = frame1[mask]  # 返回一维，从整张帧图中删去人脸rectangle的部分,长度上是对的
        background_mean = np.mean(background)  # 对背景区域求平均

        return show_frame1, level_mean, background_mean  # 返回show图、roi平均值、背景信号平均值
    else:
        level_zero = [0, 0, 0, 0, 0]
        background_zero = 0
        return frame1, level_zero, background_zero


# 3_1对一帧灰度frame处理，返回额头ROI数组(耗费时间)
def get_forehead_value_gray(frame, detector, predictor):
    """
    获取灰度帧额头矩形ROI均值
    :param frame:
    :param detector:
    :param predictor:
    :return:
    """
    rects = detector(frame, 0)  # 人脸数rects（耗时），必须是灰度化的图像呢
    if len(rects) == 1:
        face_points = predictor(frame, rects[0])  #
        points = np.zeros((len(face_points.parts()), 2))  # 将这些点存储在Numpy数组array中，这样我们就可以很容易地通过切片得到x和y的最小值和最大值68行2列
        for i, part in enumerate(face_points.parts()):  # numerate用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
            points[i] = (part.x, part.y)  # 把特征点的X坐标和Y坐标存进points数组
        # 得到额头矩形的左右上下边界值
        min_x = int(points[21, 0])
        min_y = int(min(points[21, 1], points[22, 1]))
        max_x = int(points[22, 0])
        max_y = int(max(points[21, 1], points[22, 1]))
        left = min_x
        right = max_x
        top = min_y - (max_x - min_x)
        bottom = max_y * 0.98
        # 为什么要这样做复制一下？
        show_frame = frame.copy()
        # 用于在任何图像上绘制矩形，返回带矩形的原图
        cv2.rectangle(show_frame, (int(left), int(top)), (int(right), int(bottom)), (255, 255, 0), 3)
        fh_roi = frame[int(top):int(bottom), int(left):int(right)]  # 得到额头ROI 是一个三维矩阵，厚度为3
        fh_mean = np.mean(fh_roi)
        return show_frame, fh_mean  # 返回值是额头ROI的平均值和画出矩形框的图


# 3对一帧frame处理，返回额头ROI三维数组(耗费时间)
def get_forehead_value(frame, detector, predictor):
    rects = detector(frame, 0)  # 人脸数rects（耗时），必须是灰度化的图像呢
    if len(rects) == 1:
        face_points = predictor(frame, rects[0])  #
        points = np.zeros((len(face_points.parts()), 2))  # 将这些点存储在Numpy数组array中，这样我们就可以很容易地通过切片得到x和y的最小值和最大值68行2列
        for i, part in enumerate(face_points.parts()):  # numerate用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
            points[i] = (part.x, part.y)  # 把特征点的X坐标和Y坐标存进points数组
        # 得到额头矩形的左右上下边界值
        min_x = int(points[21, 0])
        min_y = int(min(points[21, 1], points[22, 1]))
        max_x = int(points[22, 0])
        max_y = int(max(points[21, 1], points[22, 1]))
        left = min_x
        right = max_x
        top = min_y - (max_x - min_x)
        bottom = max_y * 0.98
        # 为什么要这样做复制一下？
        show_frame = frame.copy()
        # 用于在任何图像上绘制矩形，返回带矩形的原图
        cv2.rectangle(show_frame, (int(left), int(top)), (int(right), int(bottom)), (255, 255, 0), 3)
        fh_roi = frame[int(top):int(bottom), int(left):int(right)]  # 得到额头ROI 是一个二维矩阵，厚度为1
        roi_blue = fh_roi[:, :, 0]  #
        roi_green = fh_roi[:, :, 1]  # 取绿色通道,array数组形式
        roi_red = fh_roi[:, :, 2]
        bluemean = np.mean(roi_blue)
        greenmean = np.mean(roi_green)  # 对array数组取平均,结果是个数值
        redmean = np.mean(roi_red)
        return show_frame, bluemean, greenmean, redmean  # 返回值是额头ROI的平均值和画出矩形框的图


# 用哈尔级联面分类器，得到人脸的两个区域的边界值
def getProcessRegion(frame1):
    """
    来源于：2013年那篇用于BCG最初的特征点区域定位
    Gets the upper and lower facial regions of interest from the frame.从帧中获取感兴趣的上、下面部区域
    """
    face_cascade = cv2.CascadeClassifier('module/haarcascade_frontalface_default.xml')  #
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 这是一种什么方法？得到的是四个边界值,ndarray,1行4列，但是二维
    # for (x, y, w, h) in faces:
    face = faces.squeeze()  # 压缩掉长度只有1的维度
    x, y, w, h = face[0:4]
    # Calculate subrectangle based on dimensions given in the paper 根据文中给出的尺寸计算子矩形
    xx = int(x + w * 0.25)  # 写变量的时候，如果没有指明nonlocal/global, 就是在局部作用域定义一个新的变量
    yy = int(y + h * 0.05)
    ww = int(w / 2)
    hh = int(h * 0.9)

    # Calculate the middle rectangle (eye region)
    ex = xx
    ey = int(yy + hh * 0.20)
    ew = ww
    eh = int(hh * 0.35)

    # Calculate the upper rectangle (forehead region)
    ux = xx
    uy = ey + eh
    uw = ww
    uh = int(hh * 0.55)

    # Calculate the lower rectangle (mouth region)
    dx = xx
    dy = yy
    dw = ww
    dh = int(hh * 0.20)
    cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 255, 0), 1)  # 叠加绘制整个人脸roi矩形
    return ux, uy, uw, uh, ex, ey, ew, eh, dx, dy, dw, dh



# Homomorphic filter class 同态滤波
class HomomorphicFilter:
    """
    博客： https://www.pythonf.cn/read/142147
    Homomorphic filter implemented with diferents filters and an option to an external filter.
    High-frequency filters implemented:
        butterworth
        gaussian
    Attributes:
        a, b: Floats used on emphasis filter:
            H = a + b*H

        .
    """

    def __init__(self, a=0.5, b=1.5):
        self.a = float(a)
        self.b = float(b)

    # Filters
    def __butterworth_filter(self, I_shape, filter_params):
        P = I_shape[0] / 2
        Q = I_shape[1] / 2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U - P) ** 2 + (V - Q) ** 2)).astype(float)
        H = 1 / (1 + (Duv / filter_params[0] ** 2) ** filter_params[1])
        return (1 - H)

    def __gaussian_filter(self, I_shape, filter_params):
        P = I_shape[0] / 2
        Q = I_shape[1] / 2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U - P) ** 2 + (V - Q) ** 2)).astype(float)
        H = np.exp((-Duv / (2 * (filter_params[0]) ** 2)))
        return (1 - H)

    # Methods
    def __apply_filter(self, I, H):
        H = np.fft.fftshift(H)
        I_filtered = (self.a + self.b * H) * I
        return I_filtered

    def filter(self, I, filter_params, filter='butterworth', H=None):
        """
        Method to apply homormophic filter on an image
        Attributes:
            I: Single channel image
            filter_params: Parameters to be used on filters:
                butterworth:
                    filter_params[0]: Cutoff frequency
                    filter_params[1]: Order of filter
                gaussian:
                    filter_params[0]: Cutoff frequency
            filter: Choose of the filter, options:
                butterworth
                gaussian
                external
            H: Used to pass external filter
        """

        #  Validating image
        if len(I.shape) != 2:
            raise Exception('Improper image')

        # Take the image to log domain and then to frequency domain
        I_log = np.log1p(np.array(I, dtype="float"))
        I_fft = np.fft.fft2(I_log)

        # Filters
        if filter == 'butterworth':
            H = self.__butterworth_filter(I_shape=I_fft.shape, filter_params=filter_params)
        elif filter == 'gaussian':
            H = self.__gaussian_filter(I_shape=I_fft.shape, filter_params=filter_params)
        elif filter == 'external':
            print('external')
            if len(H.shape) != 2:
                raise Exception('Invalid external filter')
        else:
            raise Exception('Selected filter not implemented')

        # Apply filter on frequency domain then take the image back to spatial domain
        I_fft_filt = self.__apply_filter(I=I_fft, H=H)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt)) - 1
        return np.uint8(I)
# End of class HomomorphicFilter