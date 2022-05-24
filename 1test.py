#!/usr/bin/env python
# -*- coding  : utf-8 -*-
# @Time       : 2022/5/1 11:20
# @Author     : wanghaoran
# @Site       : SCNU
# @File       : 1test.py
# @description: 
# @Software   : PyCharm
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import signal_tools as stools
import decomposition as dc

# 对数据进行分析，生成插图
# data = np.load("output/video_signal/BVP_smooth_17front.npy")
data = np.load(r"output\video_signal\BVP_smooth_subject1.1.npy")
plt.figure("original regions_mean")
# x = np.arange(0, data.shape[1])  # 返回一个有终点和起点的固定步长的排列做x轴
for i in range(data.shape[0]):
    # plt.plot(x, data[i, :], color=color_name[i], label=level_name[i])  # 绘制第i行,并贴出标签
    # plt.plot(data[i, :])  # 绘制第i行,并贴出标签
    plt.plot(data[i, 3500:4500])  # 绘制第i行,并贴出标签
# plt.legend()
plt.title("original regions_mean")
plt.show()


# SPA趋势去除
data = stools.SPA(data, Plot=False)
for i in range(data.shape[0]):
    # plt.plot(x, data[i, :], color=color_name[i], label=level_name[i])  # 绘制第i行,并贴出标签
    # plt.plot(data[i, :])  # 绘制第i行,并贴出标签
    plt.plot(data[i, 3500:4500])  # 绘制第i行,并贴出标签
# plt.legend()
plt.title("SPA")
plt.show()


# Filter
data = stools.BandPassFilter(data, Plot=False)
for i in range(data.shape[0]):
    # plt.plot(x, data[i, :], color=color_name[i], label=level_name[i])  # 绘制第i行,并贴出标签
    # plt.plot(data[i, :])  # 绘制第i行,并贴出标签
    plt.plot(data[i, 3500:4500])  # 绘制第i行,并贴出标签
# plt.legend()
plt.title("bandpass filter")
plt.show()

# data = data[3]
# plt.plot(data[3500:4500])  # 绘制第i行,并贴出标签
# # plt.legend()
# plt.title("Filter0")
# plt.show()


# PCA计算
data = dc.PCA_compute(data, Plot=False).T  # PCA后shape是(5368, 5)
for i in range(data.shape[0]):
    # plt.plot(x, data[i, :], color=color_name[i], label=level_name[i])  # 绘制第i行,并贴出标签
    # plt.plot(data[i, :])  # 绘制第i行,并贴出标签
    plt.plot(data[i, 3500:4500])  # 绘制第i行,并贴出标签
# plt.legend()
plt.title("PCA")
plt.show()


data = data[0]
plt.plot(data[3500:4500])  # 绘制第i行,并贴出标签
# plt.legend()
plt.title("PCA0")
plt.show()

data = data.tolist()
data = stools.Wavelet(data, Plot=False)
plt.plot(data[3500:4500])  # 绘制第i行,并贴出标签
plt.title("DWT")
plt.show()
