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

def find_nearest(data, target):
    """
    data:输入list
    target:输入目标值
    """
    array = np.asarray(data)
    idx = (np.abs(array - target)).argmin()  # 最小值的下表
    return array[idx]




if __name__ == "__main__":

    # 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）

    # 真值
    # ecgdata = np.loadtxt(r"I:\DataBase\ir_heartrate_database\ecg\16\front_ecg.txt")
    ecgdata = np.loadtxt(r"I:\WHR\Dataset\1-Myself\2022.4.21\3heh\3heh_ecg\3.0.txt")
    # ecgdata = np.loadtxt(r"I:\WHR\Dataset\1-Myself\5-haoran\ecg\subject1.1.txt")
    ecg_signal = ecgdata[:, 0]  # type? 应该是list
    ecg_signal = ecg_signal[1000*1:]
    out = ecg.ecg(ecg_signal, sampling_rate=1000., show=False)  # biosppy库功能 Tuple,应该是默认采样率1000
    times = out['heart_rate_ts']   # times是时间，长176
    bpm = out['heart_rate']  # 实时心率，对应时间的心率，长176

    # 原始信号
    # data = np.load("output/video_signal/BVP_02front.npy")
    # data = np.load("output/video_signal/BVP_smooth_16front.npy")
    # data = np.load("output/video_signal/BVP_3heh_ppg3.4.npy")
    data = np.load("output/video_signal/BVP_smooth_3heh_ppg3.0.npy")
    # data = np.load("output/video_signal/BVP_grid_heh3.0.npy")
    # data = np.vstack([np.array(data), np.array(data1)])
    # data = np.load(r"output\video_signal\BVP_smooth_subject1.1.npy")
    Plot = False

    # show 原始数据
    stools.show_signal(data, Plot)

    #归一化
    # data = stools.Normalization(data, Plot)

    # SPA趋势去除
    data = stools.SPA(data, Plot)

    # Filter
    data = stools.BandPassFilter(data, Plot)

    # PCA计算
    data = dc.PCA_compute(data, Plot).T  # PCA后shape是(5368, 5)
    data = data[0]

    # EMD计算,效果奇差，暂时不用
    # data = dc.EMD_compute(data, Plot)
    # data = data[0]

    # EEMD计算,效果奇差，暂时不用
    # data = dc.EEMD_compute(data, Plot)
    # data = data[3]

    data = data.tolist()

    # ICA计算
    # data = stools.ICA_compute(data)  # numpy.ndarray
    # print(data.shape)
    # data = data[3].tolist()
    # print(data)

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

        print('最大五个：', averageHRs, '最大值：', averageHR)
        # 增加一个选择机制，看频域峰值哪个离上个最近
        if len(video_BPM) > 0:
            # 找到离上个BPM值最近的一个
            averageHR = find_nearest(averageHRs, video_BPM[-1])
            # 防止突变,要满足非主导频率再进行判断
            if len(averageHRs) > 1 and abs(averageHR - video_BPM[-1]) > 15:
                averageHR = video_BPM[-1]
            print('第', win_i, '个时间窗口心率：', averageHR)
            print('')

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


    # print('video_BPM:', video_BPM)
    # 结果评价
    MSE = mean_squared_error(real_BPM, video_BPM)
    MAE = mean_absolute_error(real_BPM, video_BPM)
    RMSE = np.sqrt(MSE)  # 根均方误差(RMSE)
    MRE = max_error(real_BPM, video_BPM)  # 最大残差
    print("MSE", MSE)
    print("MAE", MAE)
    print("RMSE", RMSE)
    print("MRE", MRE)

    # Bland-Altman图
    etools.bland_altman_plot(video_BPM, real_BPM)

    # scatter散点图
    fig, ax = plt.subplots(1, 1)
    # plt.figure("scatter")
    plt.title("PPG ECG 方法之比")
    plt.xlabel("心率(bpm)")
    plt.ylabel("心率(bpm)")
    plt.scatter(video_BPM, real_BPM)  # 生成一个scatter散点图
    ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls='--', c='k', label="1:1 line")

    # 心率曲线一致性
    plt.figure("1")
    plt.title("PPG ECG 测出心率的曲线一致性")
    plt.xlabel("时间(sec)")
    plt.ylabel("心率(bpm)")
    plt.plot(video_BPM, label="PPG")
    plt.plot(real_BPM, label="ECG")
    plt.legend()  # 展示每个数据对应的图像名称
    plt.show()
