# -*- encoding: utf-8 -*-
'''
@file        : BPM_calculate.py
@time        : 2022/03/07 20:49:50
@author      : Lin-sudo
@description : 庆麟写的程序，用于计算BPM
'''
import numpy as np
import matplotlib.pyplot as plt
from biosppy.signals import ecg  # https://github.com/PIA-Group/BioSPPy 
from scipy import signal
import scipy.linalg as sl
from sklearn.metrics import max_error, mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import biosppy.signals.tools as tools  # 提供多种信号时频域分析方法
import biosppy.signals.bvp as bvp  # Blood Volume Pulse (BVP) 血容量脉冲信号 处理
import biosppy.signals.resp	 as resp  # Respiration (Resp) 呼吸信号 处理
import biosppy.clustering as clustering  # 导入聚类方法
import biosppy.plotting as bplt
import biosppy.storage as storage  # 指定格式导入和存储数据
import biosppy.utils as utils  # 常用的辅助函数

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）


# 3主成分分析,得到一维主位置信号(降维)
def PCA_compute(data):
    """
    Perform PCA on the time series data and project the points onto the
    principal axes of variation (eigenvectors of covariance matrix) to get
    the principal 1-D position signals
    """
    # Object for principal component analysis主成分分析对象
    pca = PCA(n_components=5)
    temp = data.T  # 要每个信号都变成竖直排列
    l2_norms = np.linalg.norm(temp, ord=2, axis=1)  # Get L2 norms of each m_t
    # 抛弃在前25名中有L2标准的分数
    temp_with_abnormalities_removed = temp[l2_norms < np.percentile(l2_norms, 75)]
    # 拟合PCA模型
    pca.fit(temp_with_abnormalities_removed)
    # 将跟踪的点运动投影到主分量向量上
    projected = pca.transform(temp)
    return projected.T  # 返回的还是列信号,转换成行信号


def SPA_detrending(data, mu=1200):
    """
    平滑先验法去除趋势 (Smoothness Priors Approach, SPA)
    :param mu: 正则化系数
    :return:
    """
    N = len(data)

    D = np.zeros((N - 2, N))
    for n in range(N - 2):
        D[n, n], D[n, n + 1], D[n, n + 2] = 1.0, -2.0, 1.0
    D = mu * np.dot(D.T, D)
    for n in range(len(D)):
        D[n, n] += 1.0
    L = sl.cholesky(D, lower=True)
    Y = sl.solve_triangular(L, data, trans='N', lower=True)
    y = sl.solve_triangular(L, Y, trans='T', lower=True)
    data -= y

    return data.tolist()


def bandPassFilter(data):
    """
    带通滤波器 0.667--2.5Hz

    这里假设采样频率为1000hz,信号本身最大的频率为500hz，要滤除100hz以下，
    400hz以上频率成分，即截至频率为100，400hz,则wn1=2*100/1000=0.2，Wn1=0.2
    wn2=2*400/1000=0.8，Wn2=0.8。Wn=[0.2,0.8]
    :return:
    """
    wn1 = 2 * 0.8 / 30   # origin 0.667
    wn2 = 2 * 2.0 / 30
    b, a = signal.butter(N=8, Wn=[wn1, wn2], btype='bandpass')     # 8阶
    data = signal.filtfilt(b, a, data)                        # data为要过滤的信号
    data = data.tolist()                                 # ndarray --> list

    return data


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



if __name__ == "__main__":
    # 原始信号
    data = np.load("./output/video_signal/BVP_heh_ppg3.1.npy")  # BVP又称血容量脉搏，出现在数据集VIPL-HR的ground truth中
    # data1 = np.load(r"D:\3workspace\OneDrive - 华南师范大学\桌面\2020.3.28\30minute\30minutesPPG.npy", allow_pickle=True)

    data = np.array(data)
    plt.figure("raw signal")
    color_name = ['r', 'g', 'b', 'c', 'm']
    level_name = ['level_e_mean', 'level_0_mean', 'level_1_mean', 'level_2_mean', 'level_3_mean']
    x = np.arange(0, data.shape[1])  # 返回一个有终点和起点的固定步长的排列做x轴
    for i in range(data.shape[0]):
        plt.plot(x, data[i, :], color=color_name[i], label=level_name[i])  # 绘制第i行,并贴出标签
    plt.legend()
    plt.title("original regions_mean")

    data = data.tolist()
    for i in range(len(data)):  # 先进行趋势去除
        data[i] = SPA_detrending(data[i])
    data = np.array(data)
    plt.figure("SPA signal")
    color_name = ['r', 'g', 'b', 'c', 'm']
    level_name = ['level_e_mean', 'level_0_mean', 'level_1_mean', 'level_2_mean', 'level_3_mean']
    x = np.arange(0, data.shape[1])  # 返回一个有终点和起点的固定步长的排列做x轴
    for i in range(data.shape[0]):
        plt.plot(x, data[i, :], color=color_name[i], label=level_name[i])  # 绘制第i行,并贴出标签
    plt.legend()
    plt.title("original regions_mean")

    data = PCA_compute(data)
    plt.figure("PCA signal")
    color_name1 = ['r', 'g', 'b', 'c', 'm']
    level_name1 = ['P0', 'P1', 'P2', 'P3', 'P4']
    x = np.arange(0, data.shape[1])  # 返回一个有终点和起点的固定步长的排列做x轴
    for i in range(data.shape[0]):
        plt.plot(x, data[i, :], color=color_name1[i], label=level_name1[i])  # 绘制第i行,并贴出标签
    plt.legend()
    plt.title("original regions_mean")

    data = data[1, :].tolist()  # 只用第一个试试

    # data = SPA_detrending(data)
    data = bandPassFilter(data)

    # 真值，读取到的ecgdata是ndarray(180573, 3)
    ecgdata = np.loadtxt(r"I:\WHR\Dataset\1-Myself\2022.4.21\3heh\3heh_ecg\3.1.txt")
    ecg_signal = ecgdata[:, 0]  # type? 应该是list
    ecg_signal = ecg_signal[1000*1:]    # 舍去前1s数据
    out = ecg.ecg(ecg_signal, sampling_rate=1000., show=False)
    times = out['heart_rate_ts']   # times是时间，长176
    bpm = out['heart_rate']  # 每秒的心率
    # plt.plot(times, bpm)
    # plt.show()

    # Short-time Fourier Transfrom
    video_BPM = []
    real_BPM = []
    win_start = 0
    win_end = 30*10-1          # 5s时间窗口
    realtime_win_start = 0
    realtime_win_end = 10
    while win_end < 5369:

        averageHR = fftTransfer(data[win_start:win_end])
        video_BPM.append(averageHR)

        real_average_heartrate_win = []
        for idx, tm in enumerate(times):
            if realtime_win_start < tm < realtime_win_end:
                real_average_heartrate_win.append(bpm[idx])
        real_BPM.append(np.mean(real_average_heartrate_win))  # 对前5秒ECG心率取平均

        # 步进为1s
        win_start += 30*1
        win_end += 30*1
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
