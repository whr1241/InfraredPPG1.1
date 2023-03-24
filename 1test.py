#!/usr/bin/env python
# -*- coding  : utf-8 -*-
# @Time       : 2022/5/1 11:20
# @Author     : wanghaoran
# @Site       : SCNU
# @File       : 1test.py
# @description: 生成文章插图
# @Software   : PyCharm
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import signal_tools as stools
import decomposition as dc

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
SPAdata = stools.SPA(data, Plot=False)
for i in range(SPAdata.shape[0]):
    # plt.plot(x, data[i, :], color=color_name[i], label=level_name[i])  # 绘制第i行,并贴出标签
    # plt.plot(data[i, :])  # 绘制第i行,并贴出标签
    plt.plot(SPAdata[i, 3500:4500])  # 绘制第i行,并贴出标签
# plt.legend()
plt.title("SPA")
plt.show()


# Filter
Filterdata = stools.BandPassFilter(SPAdata, Plot=False)
for i in range(Filterdata.shape[0]):
    # plt.plot(x, data[i, :], color=color_name[i], label=level_name[i])  # 绘制第i行,并贴出标签
    # plt.plot(data[i, :])  # 绘制第i行,并贴出标签
    plt.plot(Filterdata[i, 3500:4500])  # 绘制第i行,并贴出标签
# plt.legend()
plt.title("bandpass filter")
plt.show()

# data = data[3]
# plt.plot(data[3500:4500])  # 绘制第i行,并贴出标签
# # plt.legend()
# plt.title("Filter0")
# plt.show()


# PCA计算
PCAdata = dc.PCA_compute(Filterdata, Plot=False).T  # PCA后shape是(5368, 5)
for i in range(PCAdata.shape[0]):
    # plt.plot(x, data[i, :], color=color_name[i], label=level_name[i])  # 绘制第i行,并贴出标签
    # plt.plot(data[i, :])  # 绘制第i行,并贴出标签
    plt.plot(PCAdata[i, 3500:4500])  # 绘制第i行,并贴出标签
# plt.legend()
plt.title("PCA")
plt.show()


PCAdata0 = PCAdata[0]
plt.plot(PCAdata0[3500:4500])  # 绘制第i行,并贴出标签
# plt.legend()
plt.title("PCA0")
plt.show()

PCAdata0 = PCAdata0.tolist()
Waveletdata = stools.Wavelet(PCAdata0, Plot=False)
plt.plot(Waveletdata[3500:4500])  # 绘制第i行,并贴出标签
plt.title("DWT")
plt.show()
