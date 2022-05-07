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
    ecgdata = np.loadtxt(r"I:\DataBase\ir_heartrate_database\ecg\03\front_ecg.txt")
    # ecgdata = np.loadtxt(r"I:\WHR\Dataset\1-Myself\2022.4.21\3heh\3heh_ecg\3.4.txt")
    ecg_signal = ecgdata[:, 0]  # type? 应该是list
    ecg_signal = ecg_signal[1000*1:]
    out = ecg.ecg(ecg_signal, sampling_rate=1000., show=False)  # biosppy库功能 Tuple,应该是默认采样率1000
    times = out['heart_rate_ts']   # times是时间，长176
    bpm = out['heart_rate']  # 实时心率，对应时间的心率，长176

    # 原始信号
    # data = np.load("output/video_signal/BVP_02front.npy")
    data = np.load("output/video_signal/BVP_smooth_03front.npy")
    # data = np.load("output/video_signal/BVP_3heh_ppg3.4.npy")
    # data = np.load("output/video_signal/BVP_smooth_3heh_ppg3.4.npy")
    # data = np.load("output/video_signal/BVP_grid_heh3.0.npy")
    # data = np.vstack([np.array(data), np.array(data1)])

    # color_name = ['r', 'g', 'b', 'c', 'm']
    # level_name = ['level_e_mean', 'level_0_mean', 'level_1_mean', 'level_2_mean', 'level_3_mean']
    # plt.figure("original regions_mean")
    # x = np.arange(0, data.shape[1])  # 返回一个有终点和起点的固定步长的排列做x轴
    # for i in range(data.shape[0]):
    #     # plt.plot(x, data[i, :], color=color_name[i], label=level_name[i])  # 绘制第i行,并贴出标签
    #     plt.plot(x, data[i, :])  # 绘制第i行,并贴出标签
    # # plt.legend()
    # plt.title("original regions_mean")


    # SPA趋势去除
    data = stools.SPA(data)


    # Filter
    data = stools.Filter(data)


    # PCA计算
    data = stools.PCA_compute(data).T  # PCA后shape是(5368, 5)
    data = data[0]
    

    # EMD计算
    data = dc.EMD_compute(data)
    data = data[0]

    data = data.tolist()
    # data = stools.arctan_Normalization(data)
    # plt.figure('arctan')
    # plt.plot(data)

    # ICA计算
    # data = stools.ICA_compute(data)  # numpy.ndarray
    # print(data.shape)
    # data = data[3].tolist()
    # print(data)

    # 小波去噪
    # data = stools.Wavelet(data)
    # plt.figure('Wavelet')
    # plt.plot(data)

    # 实时心率计算
    video_BPM = []
    real_BPM = []
    averageHR = 0
    win_start = 0
    win_end = 30*10  # 10s时间窗口
    realtime_win_start = 0
    realtime_win_end = 10
    while win_end < 5369:
        averageHR, averageHRs = stools.fftTransfer(data[win_start:win_end])  # 得到心率list,长度为5
        print('最大值：', averageHR)
        print('最大五个：', averageHRs)

        # 增加一个选择机制，看频域峰值哪个离上个最近
        if len(video_BPM) > 0:
            # 找到离上个BPM值最近的一个
            averageHR = find_nearest(averageHRs, video_BPM[-1])
            print('选择的心率值：', averageHR)

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


    print('video_BPM:', video_BPM)
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
