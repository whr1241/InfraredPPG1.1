#!/usr/bin/env python
# -*- coding  : utf-8 -*-
# @Time       : 2022/4/25 15:11
# @Author     : wanghaoran
# @Site       : SCNU
# @File       : Fusion.py
# @description: PPG和BCG的融合
# @Software   : PyCharm
import numpy as np
from biosppy.signals import ecg
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.linalg as sl
import signal_tools as stools
import decomposition as dtools
from sklearn.metrics import max_error, mean_squared_error, mean_absolute_error

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


def fftTransfer(data, N=1024):
    """
    FFT变换
    :return:
    """
    if len(data) < N:
        for _ in range(N - len(data)):  # 补零补至N点
            data.append(0)
    else:
        data = data[0:N]
    df = [30 / N * i for i in range(N)]  # 频谱分辨率
    fft_data = np.abs(np.fft.fft(data))

    hr = df[fft_data.tolist()[0:120].index(max(fft_data.tolist()[0:120]))]*60
    return hr



if __name__ == '__main__':

    # 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）

    # bcg_data = np.load("output/video_signal/BCG_3heh3.1.npy")
    data = np.load("output/video_signal/BVP_smooth_08front.npy")
    # ppg_data = np.load("output/video_signal/BVP_smooth_3heh_ppg3.1.npy")
    ppg_data = np.array(data)

    plt.figure('raw ppg data')
    color_name = ['r', 'g', 'b', 'c', 'm']
    level_name = ['level_e_mean', 'level_0_mean', 'level_1_mean', 'level_2_mean', 'level_3_mean']
    x = np.arange(0, ppg_data.shape[1])  # 返回一个有终点和起点的固定步长的排列做x轴
    for i in range(ppg_data.shape[0]):
        plt.plot(x, ppg_data[i, :], color=color_name[i], label=level_name[i])  # 绘制第i行,并贴出标签
    plt.legend()
    plt.title("raw ppg data")

    # 滤波
    r, c = ppg_data.shape
    filtered_data = np.zeros((r, c))
    for time_series in range(r):  # 用巴特沃斯带通滤波器滤除低频运动
        filtered_data[time_series, :] = stools.SPA_detrending1(ppg_data[time_series, :])
        filtered_data[time_series, :] = stools.bandPassFilter(filtered_data[time_series, :])  # 带通滤波

    s = dtools.PCA_compute(filtered_data)  # s的shape:(?,5)，PCA计算
    pca_data = s.T
    plt.figure('PCA ppg data')
    color_name = ['r', 'g', 'b', 'c', 'm']
    level_name = ['level_e_mean', 'level_0_mean', 'level_1_mean', 'level_2_mean', 'level_3_mean']
    x = np.arange(0, pca_data.shape[1])  # 返回一个有终点和起点的固定步长的排列做x轴
    for i in range(pca_data.shape[0]):
        plt.plot(x, pca_data[i, :], color=color_name[i], label=level_name[i])  # 绘制第i行,并贴出标签
    plt.legend()
    plt.title("PCA ppg data")

    final_data = pca_data[0].tolist()

    # 真值
    ecgdata = np.loadtxt(r"I:\DataBase\ir_heartrate_database\ecg\08\front_ecg.txt")
    # ecgdata = np.loadtxt(r"I:\WHR\Dataset\1-Myself\2022.4.21\3heh\3heh_ecg\3.1.txt")
    ecg_signal = ecgdata[:, 0]  # type? 应该是list
    ecg_signal = ecg_signal[1000 * 1:]
    out = ecg.ecg(ecg_signal, sampling_rate=1000., show=False)  # biosppy库功能 Tuple,应该是默认采样率1000
    times = out['heart_rate_ts']  # times是时间，长176
    bpm = out['heart_rate']  # 实时心率，对应时间的心率，长176

    video_BPM = []
    real_BPM = []
    win_start = 0
    win_end = 30 * 10  # 5s时间窗口
    realtime_win_start = 0
    realtime_win_end = 10
    while win_end < 5369:
        averageHR = fftTransfer(final_data[win_start:win_end])
        video_BPM.append(averageHR)

        real_average_heartrate_win = []
        for idx, tm in enumerate(times):
            if realtime_win_start < tm < realtime_win_end:
                real_average_heartrate_win.append(bpm[idx])
        real_BPM.append(np.mean(real_average_heartrate_win))  # 对前5秒ECG心率取平均

        # 步进为1s
        win_start += 30 * 1
        win_end += 30 * 1
        realtime_win_start += 1
        realtime_win_end += 1

    # 计算MSE
    MSE = mean_squared_error(real_BPM, video_BPM)
    MAE = mean_absolute_error(real_BPM, video_BPM)
    # 根均方误差(RMSE)
    RMSE = np.sqrt(MSE)
    # 最大残差
    MRE = max_error(real_BPM, video_BPM)
    print("MSE", MSE)
    print("MAE", MAE)
    print("RMSE", RMSE)
    print("MRE", MRE)

    # 绘制Bland-Altman图
    bland_altman_plot(video_BPM, real_BPM)

    # 画散点图
    fig, ax = plt.subplots(1, 1)
    # plt.figure("scatter")
    plt.title("PPG ECG 方法之比")
    plt.xlabel("心率(bpm)")
    plt.ylabel("心率(bpm)")
    plt.scatter(video_BPM, real_BPM)  # 生成一个scatter散点图
    ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls='--', c='k', label="1:1 line")

    plt.figure("1")
    plt.title("PPG ECG 测出心率的曲线一致性")
    plt.xlabel("时间(sec)")
    plt.ylabel("心率(bpm)")
    plt.plot(video_BPM, label="PPG")
    plt.plot(real_BPM, label="ECG")
    plt.legend()  # 展示每个数据对应的图像名称
    plt.show()