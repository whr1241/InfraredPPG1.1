#!/usr/bin/env python
# -*- coding  : utf-8 -*-
# @Time       : 2022/5/1 15:12
# @Author     : wanghaoran
# @Site       : SCNU
# @File       : evaluate_tools.py
# @description: 结果评估函数
# @Software   : PyCharm
import numpy as np
import matplotlib.pyplot as plt


def bland_altman_plot(data1, data2, *args, **kwargs):
    """
    Bland-Altman 图是反应数据一致性很简单直观的图示方法
    横轴表示两种方法测量每个样本的结果平均值，纵轴表示两种方法测量结果的差值。
    :param data1:
    :param data2:
    :param args:
    :param kwargs:
    :return:
    """
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.figure("bland", figsize=(7, 7))
    plt.title('Bland-altman plot')
    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')