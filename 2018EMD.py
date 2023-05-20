'''
Author: whr1241 2735535199@qq.com
Date: 2022-06-25 21:13:19
LastEditors: whr1241 2735535199@qq.com
LastEditTime: 2023-03-25 06:04:42
FilePath: \InfraredPPG1.1\2018EMD.py
Description: 对2018年桂林电子、哈工大的的单通道EMD方法进行复现
好像很简单
'''
import numpy as np
from biosppy.signals import ecg
import signal_tools as stools
import decomposition as dc
import matplotlib.pyplot as plt
from sklearn.metrics import max_error, mean_squared_error, mean_absolute_error
import h5py
import evaluate_tools as etools

if __name__ == "__main__":

    save_file_name = 'subject10.2'
    # 真值
    # ecgdata = np.loadtxt(r"D:\1maydaystudy\0Github\ecg\17\front_ecg.txt")
    ecgdata = np.loadtxt(r"D:\1maydaystudy\0Github\ecg2\subject10.2.txt")
    ecg_signal = ecgdata[:, 0]  # type? 应该是list
    ecg_signal = ecg_signal[1000*1:]
    out = ecg.ecg(ecg_signal, sampling_rate=1000., show=False)  # biosppy库功能 Tuple,应该是默认采样率1000
    times = out['heart_rate_ts']   # times是时间，长176
    bpm = out['heart_rate']  # 实时心率，对应时间的心率，长176
    # 原始信号
    # data = np.load("output/video_signal/BVP_smooth_17front.npy")
    data = np.load(r"output\video_signal\BVP_smooth_subject10.2.npy")


    Plot = True

    # print(type(data))  # <class 'numpy.ndarray'>保存的时候不是list吗？
    data = data[2]
    # 归一化
    data = stools.Z_ScoreNormalization(data)
    # EMD
    data = dc.EMD_compute(data, Plot)
    data = np.sum(data[1:], axis=0)  # 信号重建，对第1行到第6行求和
    # # Filter
    data = data.tolist()  # 带通滤波需要输入list
    data = stools.Bandpass(data, w1=0.8, w2=3.2, N=8)
    
    # PPG与ECG结果对比
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

        print('最大十个：', averageHRs, '最大值：', averageHR)

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

    # 保存下滑动窗口得到的监测心率以及ECG真实心率
    filename = 'output/EMDFinalBPM.h5'
    h5f = h5py.File(filename, 'a')
    h5f.create_dataset('real{}'.format(save_file_name), data=real_BPM)
    h5f.create_dataset('video{}'.format(save_file_name), data=video_BPM)
    h5f.close()

    # 结果评价
    MSE = mean_squared_error(real_BPM, video_BPM)
    MAE = mean_absolute_error(real_BPM, video_BPM)
    RMSE = np.sqrt(MSE)  # 根均方误差(RMSE)
    MRE = max_error(real_BPM, video_BPM)  # 最大残差
    PPMCC = etools.PearsonFirst(real_BPM, video_BPM)
    print("MSE", MSE)
    print("RMSE", RMSE)
    print("MAE", MAE)
    print("MRE", MRE)
    print("PPMCC", PPMCC)

    # 直观时频结果对比
    # 心率曲线一致性
    plt.figure("1")
    plt.title("Time-frequency domain comparison")
    plt.xlabel("time(sec)")
    plt.ylabel("heart rate(bpm)")
    plt.plot(video_BPM, label="iPPG")
    plt.plot(real_BPM, label="ECG")
    plt.legend()  # 展示每个数据对应的图像名称
    plt.show()

