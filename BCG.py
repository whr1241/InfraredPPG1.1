#!/usr/bin/env python
# -*- coding  : utf-8 -*-
# @Time       : 2022/3/11 21:37
# @Author     : wanghaoran
# @Site       : SCNU
# @File       : BCG.py
# @Description: 对图片格式存储的dataset+BCG
# @Software   : PyCharm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import video_tools as vtools
import signal_tools as stools
import decomposition as dtools


def opticalFlow(old_gray, p0, current_frame):  # 输入上一张frame的gray图，特征角点P0，还有当前的frame
    """
    根据旧的帧和特征点， 使用Lucas-Kanade算法去追踪当前帧中的特征点，并返回新的特征点
    # Calculate optical flow https://www.cnblogs.com/my-love-is-python/p/10447917.html
    """
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # pl表示光流检测后的角点位置，st的shape(56,1)表示是否是运动的角点，err表示是否出错
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, current_gray, p0, None, **lk_params)  # 根据给出的前一帧特征点坐标计算当前视频帧上的特征点坐标

    # 读取运动了的角点，st == 1表示检测到的运动物体
    good_new = p1[st == 1]  # shape(56,2)|返回运动的角点
    good_old = p0[st == 1]

    # 绘制轨迹
    for i, (new, old) in enumerate(zip(good_new, good_old)):  # zip将对象中对应的元素打包成一个个tuple，然后返回由这些元组组成的list
        a, b = new.ravel()  # ravel将多维数组降位一维？a、b是横纵坐标，只有在old和new中都出现的特征点才会取它的横纵坐标
        current_frame = cv2.circle(current_frame, (int(a), int(b)), 1, (255, 0, 0), -1)   # 给特征点画圈描述一下

    # 现在更新之前的帧和之前的点
    old_gray = current_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)  # 表示自适应该维度大小，reshape(-1, 1, 2) 方法将返回一个 shape = (n, 1, 2) 的 ndarray
    return old_gray, p0, good_old
    # P0是此帧的角点，old_gray是此帧画圈后的图片，good_old就是输入的上帧角点P0



if __name__ == "__main__":

    # 加载模型和全局参数
    face_cascade = cv2.CascadeClassifier('module/haarcascade_frontalface_default.xml')  #

    # Parameters for Shi-Tomasi corner detection Shi-Tomasi角点检测参数
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.05,
                          minDistance=5,
                          blockSize=3)
    # Parameters for Lucas-Kanade optical flow  Lucas-Kanade光流参数
    lk_params = dict(winSize=(15, 15),  # winSize表示选择多少个点进行u和v的求解
                     maxLevel=2,  # 金字塔层数
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    video_path = r'I:\DataBase\ir_heartrate_database\videos\04front'
    # video_path = r'I:\WHR\Dataset\1-Myself\2022.4.21\3heh\3heh_ppg\3.1'
    # video_path = r'I:\WHR\Dataset\1-Myself\2022.4.21\sit\sit_ppg'
    save_file_name = "04front"

    framerate = 30
    start_index = 31
    end_index = 5399
    i = start_index
    plot = True
    isFirstFrame = True
    # for i in range(1, 5400):
    while start_index <= i <= end_index:
        # 前1s曝光时间不算
        frame = cv2.imread(video_path+r'\{}.pgm'.format(i))  # 从31开始

        if isFirstFrame:
            uxx, uyy, uww, uhh, exx, eyy, eww, ehh, dxx, dyy, dww, dhh = vtools.getProcessRegion(frame)  # 在第一帧中，得到脸部的三个区域
            old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 彩色转灰度图  为感兴趣的区域(前额和嘴)创建一个mask
            face_mask = np.zeros_like(old_gray)  # 生成一个和你所给数组相同shape的全0数组
            face_mask[uyy:uyy + uhh, uxx:uxx + uww] = 1  # 生成mask
            face_mask[dyy:dyy + dhh, dxx:dxx + dww] = 1

            p0 = cv2.goodFeaturesToTrack(old_gray, mask=face_mask, **feature_params)  # 从灰度图的mask区域检测角点向量
            num_feature_points = len(p0)  # 特征角点的个数56，P0的shape(56,1,2)
            data = np.zeros((num_feature_points, 1))  # np.zeros返回来一个给定形状和类型的用0填充的array|n行1列
            isFirstFrame = False
            i = i + 1
        else:  # 在第一帧之后
            old_gray, p0, good_old = opticalFlow(old_gray, p0, frame)  # 在后续帧上，使用Lucas-Kanade算法跟踪特征点，并返回新的特征点

            new_column = np.zeros((num_feature_points, 1))
            for j in range(len(good_old)):
                new_column[j, 0] = good_old[j, 1]  # 将每个特征点的y值添加到相应的时间序列y(t)中
            data = np.append(data, new_column, axis=1)  # 依次添加上每一列数据，每一列长度是多少呢？
            cv2.imshow('Face', frame)  # 显示视频跟踪点，角点显示在哪里加上的？应该是currentframe
            # cv2.waitKey(0)
            i = i + 1
        if cv2.waitKey(25) & 0xFF == ord('q'):  # cv2.waitKey(25)：在25ms内根据键盘输入返回一个值
            break
    cv2.destroyAllWindows()

    bcg_data = data[:, 1:]  # 从数据中删除第一个虚列：原来是501列，第一列是0
    print('初始信号bcg_data的shape：', bcg_data.shape)
    # BCG_data = bcg_data.tolist()
    np.save('Output/video_signal/BCG_{}.npy'.format(save_file_name), bcg_data)

    r, c = bcg_data.shape
    t = np.linspace(0, c - 1, num=c)  # np.linspace主要用来创建等差数列，这里创建横轴坐标
    if plot:
        plt.figure(1)  # 只绘制第一行的数据，一条直线是因为没有归一化吧？
        plt.plot(t, bcg_data[0, :], 'b')  # 绘图,将original信号展示出来,g表示绿色显示
        plt.title("original bcg signal")

    # Signal processing 信号处理
    filtered_data = np.zeros((r, c))  # 设置一个同形状的array来装过滤后的信号array
    for time_series in range(r):  # 用巴特沃斯带通滤波器滤除低频运动
        filtered_data[time_series, :] = stools.bandPassFilter(bcg_data[time_series, :], framerate)  # 带通滤波
    if plot:
        plt.figure(2)
        plt.title('filtered bcg signal')  # 只绘制第一行的数据
        plt.plot(t, filtered_data[0, :], 'b')

    # PCA计算主成分分析的投影
    s = dtools.PCA_compute(filtered_data)  # s的shape:(500,5)，PCA计算
    print('PCA后的信号shape:', s)
    bpm, selected_bcg = stools.signal_selection(s)  # 以BPM为单位计算心率
    # selected_bcg是numpy的数据类型
    if plot:
        plt.figure(3)
        plt.title('selected bcg signal')  # 只绘制第一行的数据
        plt.plot(t, selected_bcg, 'b')
        plt.show()
    selected_bcg = selected_bcg.tolist()
    # np.save('Output/oldset/BCG_{}.npy'.format(save_file_name), selected_bcg)
    np.save('Output/video_signal/BCG_{}.npy'.format(save_file_name), selected_bcg)
    print('Heart rate = {:.2f} BPM'.format(bpm))

