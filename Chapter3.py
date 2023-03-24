'''
Author: whr1241 2735535199@qq.com
Date: 2023-03-22 14:30:14
LastEditors: whr1241 2735535199@qq.com
LastEditTime: 2023-03-24 14:03:11
FilePath: \InfraredPPG1.1\Chapter3.py
Description: 第三章的实验，生成数据
'''
import cv2
import dlib
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import scipy.linalg as sl
from biosppy.signals import ecg
from sklearn.metrics import max_error, mean_squared_error, mean_absolute_error
import signal_tools as stools
from signal_tools import KalmanFilter
import evaluate_tools as etools
import decomposition as dc
from mpl_toolkits.mplot3d import Axes3D
import h5py




if __name__ == '__main__':
    # 加载并预处理ECG信号
    save_file_name = 'Face04front'
    ecgdata = np.loadtxt(r"I:\WHR\0-Dataset\DataBase\ir_heartrate_database\ecg\17\front_ecg.txt")
    ecg_signal = ecgdata[:, 0]  # type? 应该是list
    ecg_signal = ecg_signal[1000*1:]
    out = ecg.ecg(ecg_signal, sampling_rate=1000., show=False)  # biosppy库功能 Tuple,应该是默认采样率1000
    times = out['heart_rate_ts']   # times是时间，长176
    bpm = out['heart_rate']  # 实时心率，对应时间的心率，长176
    # 加载BVP信号
    data = np.load(r"output\video_signal3\17front.npy")
    
    # data = data[:, 1000:2500]
    # 信号处理
    # 去除0点，使用前后两帧的均值插入
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] == 0:
                data[i, j] = (data[i, j-1]+data[i, j+1])/2

    Plot = True
    stools.show_signal(data, Plot)  # 展示一下五个原始信号
    # SPA趋势去除
    data = stools.SPA(data, Plot)
    # Filter滤波
    data = stools.BandPassFilter(data, Plot)
    
    data = data[0]

    # # 试一下EMD
    # data = dc.EMD_compute(data, Plot)
    # data = data[2]
    data = dc.EEMD_compute(data, Plot)  # numpy.ndarray
    data = data[3]
    # data = data[2:4].sum(axis=0)
    print(data.shape)
    data = data.tolist()  # numpy格式转化为list格式

    # 实时心率计算
    win_i = 0  # 第几个窗口
    video_BPM = []
    real_BPM = []
    averageHR = 0
    win_start = 0
    win_end = 30*5  # 10s时间窗口
    realtime_win_start = 0
    realtime_win_end = 5
    while win_end < 5369:
        averageHR = stools.fftTransfer1(data[win_start:win_end])
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

    # 保存下滑动窗口得到的监测心率以及ECG真实心率
    # filename = 'output/FinalBPM3.h5'
    # h5f = h5py.File(filename, 'a')
    # h5f.create_dataset('real{}'.format(save_file_name), data=real_BPM)
    # h5f.create_dataset('video{}'.format(save_file_name), data=video_BPM)
    # h5f.close()


    # 五种结果评价
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
    