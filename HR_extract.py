'''
Author: whr1241 2735535199@qq.com
Date: 2022-04-19 14:34:56
LastEditors: whr1241 2735535199@qq.com
LastEditTime: 2023-03-15 10:19:42
FilePath: \InfraredPPG1.1\HR_extract.py
Description: 最开始的实时心率
'''

# 对raw信号处理，得到估计心率
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import scipy.linalg as sl
from biosppy.signals import ecg
from sklearn.metrics import max_error, mean_squared_error, mean_absolute_error
import signal_tools as stools
# from filterpy.kalman import KalmanFilter  # 卡尔曼滤波
from signal_tools import KalmanFilter
import evaluate_tools as etools
import decomposition as dc
from mpl_toolkits.mplot3d import Axes3D

def find_nearest(data, target):
    """
    data:输入list
    target:输入目标值
    """
    array = np.asarray(data)
    idx = (np.abs(array - target)).argmin()  # 最小值的下表
    return array[idx]




if __name__ == "__main__":

    # 支持中文，但会影响全局的插图字体设置
    # plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
    # plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）

    # 载入ECG真值，并进行处理
    # ecgdata = np.loadtxt(r"I:\WHR\0-Dataset\DataBase\ir_heartrate_database\ecg\17\front_ecg.txt")
    ecgdata = np.loadtxt(r"I:\WHR\Dataset\1-Myself\5-haoran\ecg\subject3.1.txt")
    ecg_signal = ecgdata[:, 0]  # type? 应该是list
    ecg_signal = ecg_signal[1000*1:]
    out = ecg.ecg(ecg_signal, sampling_rate=1000., show=False)  # biosppy库功能 Tuple,应该是默认采样率1000
    times = out['heart_rate_ts']   # times是时间，长176
    bpm = out['heart_rate']  # 实时心率，对应时间的心率，长176
    # 载入原始PPG时间信号
    # data = np.load("output/video_signal/BVP_smooth_17front.npy")
    data = np.load(r"output\video_signal\BVP_smooth_subject3.1.npy")

    # 是否绘制出来图形
    Plot = True
    # show 原始时间数据
    stools.show_signal(data, Plot)
    #归一化
    # data = stools.Normalization(data, Plot)
    # SPA趋势去除
    data = stools.SPA(data, Plot)
    # Filter
    data = stools.BandPassFilter(data, Plot)
    # PCA计算，取第一个分量
    data = dc.PCA_compute(data, Plot).T  # PCA后shape是(5368, 5)
    data = data[0]
    # EMD计算,效果奇差，暂时不用
    # data = dc.EMD_compute(data, Plot)
    # data = data[0]
    # EEMD计算,效果奇差，暂时不用
    # data = dc.EEMD_compute(data, Plot)
    # data = data[3]
    data = data.tolist()  # numpy格式转化为list格式
    # 小波去噪，效果特别好
    data = stools.Wavelet(data, Plot)
    # 实时心率计算
    win_i = 0  # 第几个窗口
    video_BPM = []
    real_BPM = []
    averageHR = 0
    win_start = 0
    win_end = 30*10  # 10s时间窗口
    realtime_win_start = 0
    realtime_win_end = 10
    while win_end < 5369:
        # averageHR = stools.fftTransfer1(data[win_start:win_end])  
        averageHR, averageHRs = stools.fftTransfer(data[win_start:win_end], win_i) 
        # averageHR = stools.FindPeak_window(data[win_start:win_end], win_i)

        # print('最大五个：', averageHRs, '最大值：', averageHR)
        # 增加一个选择机制，看频域峰值哪个离上个最近
        if len(video_BPM) > 0:
            # 找到离上个BPM值最近的一个
            averageHR = find_nearest(averageHRs, video_BPM[-1])
            # 防止突变,要满足非主导频率再进行判断
            if len(averageHRs) > 1 and abs(averageHR - video_BPM[-1]) > 15:
                averageHR = video_BPM[-1]
            # print('第', win_i, '个时间窗口心率：', averageHR)
            # print('')

        win_i = win_i + 1
        video_BPM.append(averageHR)

        real_average_heartrate_win = []
        for idx, tm in enumerate(times):
            if realtime_win_start < tm < realtime_win_end:
                real_average_heartrate_win.append(bpm[idx])
        real_BPM.append(np.mean(real_average_heartrate_win))  # 对前5/10秒ECG心率取平均

        # 步进为1s
        win_start += 30*1
        win_end += 30*1
        realtime_win_start += 1
        realtime_win_end += 1
    print('real_BPM_len:', len(real_BPM))
    print('video_BPM_len:', len(video_BPM))

    # 四种结果评价
    MSE = mean_squared_error(real_BPM, video_BPM)
    MAE = mean_absolute_error(real_BPM, video_BPM)
    RMSE = np.sqrt(MSE)  # 根均方误差(RMSE)
    MRE = max_error(real_BPM, video_BPM)  # 最大残差
    PPMCC = etools.PearsonFirst(real_BPM, video_BPM)

    print("MSE", MSE)
    print("MAE", MAE)
    print("RMSE", RMSE)
    print("MRE", MRE)
    print("PPMCC", PPMCC)

    # Bland-Altman图
    etools.bland_altman_plot(video_BPM, real_BPM)

    # scatter散点图
    fig, ax = plt.subplots(1, 1)
    # plt.figure("scatter")
    plt.title("PPG ECG compare")
    plt.xlabel("HR(bpm)")
    plt.ylabel("HR(bpm)")
    plt.scatter(video_BPM, real_BPM)  # 生成一个scatter散点图
    ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls='--', c='k', label="1:1 line")

    # 心率曲线一致性
    plt.figure('TF', figsize=(7, 4), dpi=150)
    # plt.figure('1')
    # plt.title("Time-frequency domain comparison")
    plt.xlabel("time(sec)")
    plt.ylabel("heart rate(bpm)")
    plt.plot(video_BPM, label="iPPG")
    plt.plot(real_BPM, label="ECG")
    plt.legend(loc='lower left')  # 展示每个数据对应的图像名称
    # plt.savefig('output/TF_subject3.1.png')
    plt.show()