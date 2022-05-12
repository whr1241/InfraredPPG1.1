#!/usr/bin/env python
# -*- coding  : utf-8 -*-
# @Time       : 2022/3/9 22:19
# @Author     : wanghaoran
# @Site       : SCNU
# @File       : signal_tools.py
# @Description: 后续信号处理的各种函数
# @Software   : PyCharm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sl  # 应用于线性代数程序领域
from scipy import signal
# from filterpy.kalman import KalmanFilter
# from filterpy.common import Q_discrete_white_noise
import pywt

"""
1、巴特沃斯滤波
移动平均滤波
滑动窗口滤波算法
全局自相关滤波
SPA平滑先验趋势去除
幅值滤波
同态滤波
选择最具有周期性的信号
FFT变换
频域最大值对应的频率
均方根误差(RMSE)计算
"""


# 当前使用
def fftTransfer(data, win_i, N=1024):
    """
    FFT变换
    :return:
    TODO: 可能需要改进一下，从频谱图到功率谱图。好像又一样
    """
    if len(data) < N:
        for _ in range(N - len(data)):  # 补零补至N点
            data.append(0)
    else:
        data = data[0:N]
    df = [30 / N * i for i in range(N)]  # 频谱分辨率 N=1024,df也应该是1024长. 0到30
    fft_data = np.abs(np.fft.fft(data))  # type是array, shape是1024长

    fft_data = fft_data.tolist()  # 纵坐标

    # 前120个里面的极值点横坐标
    num_peak = signal.find_peaks(fft_data[0:120], distance=2) #distance表极大值点的距离至少大于等于2个水平单位
    num_peak_list_X = num_peak[0].tolist()

    # 得到极值点的纵坐标
    num_peak_list_Y = []
    for i in num_peak_list_X:
        num_peak_list_Y.append(fft_data[i])

    # 取最大的五个纵坐标
    num_peak_list_Y_final = sorted(num_peak_list_Y, reverse=True)[0:10]

    final_hr = []
    num_peak_list_X_final = []
    for i in num_peak_list_Y_final:    
        final_hr.append(df[fft_data[0:1024].index(i)]*60)
        num_peak_list_X_final.append(df[fft_data[0:1024].index(i)])  # 极值点横坐标

    hr = df[fft_data[0:1024].index(max(fft_data[0:120]))]*60
    x = fft_data[0:1024].index(max(fft_data[0:120]))  # 最高点对性的索引
    # 当信号很清晰时，主峰很明显，直接只取主峰好了
    # TODO：假峰有时候也会变得高出真峰很多倍
    # TODO：还是应该按照功率比值阈值来判断吧,这里应该用一个邻域窗口计算，不应只用一个频率点
    # TODO：这个判断条件还是不太行，有时候看着像主导频率但占比到不了20%，如何进行判断？
    # if num_peak_list_Y_final[0]/sum(fft_data[0:512]) > 0.07:
    if sum(fft_data[(x-5):(x+5)])/sum(fft_data[0:512]) > 0.20:
        final_hr = [hr]
    print('最大值窗口占总功率比值：', sum(fft_data[(x-5):(x+5)])/sum(fft_data[0:512]))
    # print('最大值所在窗口功率占总功率比值：', num_peak_list_Y_final[0]/sum(fft_data[0:512]))

    # show一下
    plt.figure('FFT')
    plt.plot(df, fft_data)
    plt.axis([0, 5, 0, np.max(fft_data)*2])
    plt.title('窗口{}: Heart Rate estimate: {:.2f}'.format(win_i, hr))
    for ii in range(len(num_peak_list_Y_final)):  # 画出5个极值点
        plt.plot(num_peak_list_X_final[ii], num_peak_list_Y_final[ii],'*',markersize=10)
    plt.pause(0.1)	# pause 1 second
    plt.clf()		# clear the current figure
    return hr, final_hr



# 对全局信号找峰值点
def FindPeak(data, Plot=False):
    num_peak = signal.find_peaks(data, distance=2)
    num_peak = num_peak[0].tolist()
    if Plot:
        plt.plot(data)

# 对窗口进行检测并plot,返回峰值数量
def FindPeak_window(data, win_i):
    num_peak = signal.find_peaks(data, distance=10)
    num_peak = num_peak[0].tolist()
    sum_peak = len(num_peak)
    hr = sum_peak/10*60  # 10s的窗口
    
    plt.figure('FindPeak')
    plt.plot(data)
    for i in num_peak:
        plt.plot(i, data[i], '*')
    plt.title('window{}: Heart Rate estimate: {:.2f}'.format(win_i, hr))
    plt.pause(0.5)	# pause 1 second
    plt.clf()		# clear the current figure
    
    return hr



# 归一化
def Normalization(data, Plot=False):
    """
    输入：array，行信号
    输出：
    """
    data_final = np.zeros_like(data)
    for i in range(data.shape[0]):
        # data_final[i] = np.array(Z_ScoreNormalization(data[i]))
        data_final[i] = np.array(Arctan_Normalization(data[i]))
    if Plot:
        plt.figure("Normalization")
        N = data.shape[0]
        for n, s in enumerate(data):
            plt.subplot(N,1,n+1)
            plt.plot(s, 'g')
            plt.title("Normalization "+str(n))
            plt.xlabel("frames")
    return data_final

# 归一化:Z-score标准化
def Z_ScoreNormalization(x):
    mu = np.mean(x)
    sigma = np.std(x)
    x = (x - mu) / sigma;
    return x

# Min-Max Normalization归一化
def Min_MaxNormalization(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x

# arctan归一化
def Arctan_Normalization(data):
    """
    输入：
    输出：
    """
    """反正切归一化，反正切函数的值域就是[-pi/2,pi/2]
    公式：反正切值 * (2 / pi)
    :return 值域[-1,1]，原始大于0的数被映射到[0,1]，小于0的数被映射到[-1,0]
    """
    new_value = np.arctan(data) * (2 / np.pi)
    return new_value
    # data = [x*4 for x in data]  # ？
    # data = np.arctan(data)
    # return data

# log函数转换归一化
def Log_Normalization(data):
    """
    负值怎么办？
    """
    data = np.log10(data)/np.log10(max(data))
    return data

def proportional_normalization(value):
    """比例归一
    公式：值/总和
    :return 值域[0,1]  又不是全是负数？
    """
    new_value = value / value.sum()
    return new_value



# 平滑先验趋势去除
def SPA(data, Plot=False):
    """
    输入是ndarry，行信号
    输出还要ndarry，行信号
    """
    data_final = np.zeros_like(data)
    for i in range(data.shape[0]):
        data_final[i] = np.array(SPA_detrending(data[i].tolist()))
    if Plot:
        plt.figure("SPA")
        N = data.shape[0]
        for n, spa in enumerate(data):
            plt.subplot(N,1,n+1)
            plt.plot(spa, 'g')
            plt.title("SPA "+str(n))
            plt.xlabel("frames")
    # plt.tight_layout()
    # x = np.arange(0, data.shape[1])
    # for i in range(data.shape[0]):
    #     plt.plot(x, data[i, :])  
    return data_final

def SPA_detrending(data, mu=1200):
    """
    平滑先验法去除趋势 (Smoothness Priors Approach, SPA) 
    基于正则化最小二乘法的平滑先验法 http://www.cqvip.com/qk/97964x/201810/7000867680.html
    # 芬兰库奥皮奥大学的Karjalainen博士提出的一种信号非线性去趋势方法：https://zhuanlan.zhihu.com/p/336228933
    :param mu: 正则化系数
    :return:
    输入是一维list
    """
    N = len(data)
    D = np.zeros((N - 2, N))
    for n in range(N - 2):
        D[n, n], D[n, n + 1], D[n, n + 2] = 1.0, -2.0, 1.0
    D = mu * np.dot(D.T, D)  # 内积
    for n in range(len(D)):
        D[n, n] += 1.0
    L = sl.cholesky(D, lower=True)
    Y = sl.solve_triangular(L, data, trans='N', lower=True)
    y = sl.solve_triangular(L, Y, trans='T', lower=True)
    data -= y
    return data.tolist()



# 带通滤波
def BandPassFilter(data, Plot=False):
    """
    输入是ndarry，行信号
    输出还要ndarry，行信号
    """
    data_final = np.zeros_like(data)
    for i in range(data.shape[0]):
        data_final[i] = np.array(Filter(data[i].tolist()))
    if Plot:
        plt.figure("Filter")
        N = data.shape[0]
        for n, fil in enumerate(data):
            plt.subplot(N,1,n+1)
            plt.plot(fil, 'g')
            plt.title("Filter "+str(n))
            plt.xlabel("frames")
    # plt.tight_layout()
    # x = np.arange(0, data.shape[1])  
    # for i in range(data.shape[0]):
    #     plt.plot(x, data[i, :])  
    return data_final

def Filter(data):
    """
    带通滤波器 0.667--2.5Hz
    这里假设采样频率为1000hz,信号本身最大的频率为500hz，要滤除100hz以下，
    400hz以上频率成分，即截至频率为100，400hz,则wn1=2*100/1000=0.2，Wn1=0.2
    wn2=2*400/1000=0.8，Wn2=0.8。Wn=[0.2,0.8]
    :return:
    """
    wn1 = 2 * 0.7 / 30   # origin 0.667
    wn2 = 2 * 4.0 / 30
    b, a = signal.butter(N=8, Wn=[wn1, wn2], btype='bandpass')     # 8阶
    data = signal.filtfilt(b, a, data)                        # data为要过滤的信号
    data = data.tolist()                                 # ndarray --> list
    return data



def Wavelet2(ppg):
    """
    小波去噪 https://www.jianshu.com/p/74f8e6d35ad5
    """
    y = ppg
    x= range(len(y))
    coeffs = pywt.wavedec(y, 'db4', level=4)  # 4阶小波分解
    ya4 = pywt.waverec(np.multiply(coeffs, [1, 0, 0, 0, 0]).tolist(), 'db4')
    yd4 = pywt.waverec(np.multiply(coeffs, [0, 1, 0, 0, 0]).tolist(), 'db4')
    yd3 = pywt.waverec(np.multiply(coeffs, [0, 0, 1, 0, 0]).tolist(), 'db4')
    yd2 = pywt.waverec(np.multiply(coeffs, [0, 0, 0, 1, 0]).tolist(), 'db4')
    yd1 = pywt.waverec(np.multiply(coeffs, [0, 0, 0, 0, 1]).tolist(), 'db4')



def Wavelet(ppg, Plot=False):
    """
    小波去噪 https://www.cnblogs.com/sggggr/p/12381164.html
    输入：list
    输出：list
    """
    index = []
    data = []
    for i in range(len(ppg) - 1):
        X = float(i)
        Y = float(ppg[i])
        index.append(X)
        data.append(Y)

    # Create wavelet object and define parameters
    w = pywt.Wavelet('db8')  # 选用Daubechies8小波
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    print("maximum level is " + str(maxlev))  # ?
    threshold = 0.01  # Threshold for filtering  去噪阈值

    # Decompose into wavelet components, to the level selected:
    coeffs = pywt.wavedec(data, 'db8', level=maxlev)  # 将信号进行小波分解

    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))  # 将噪声滤波

    mintime = 0
    maxtime = mintime + len(data)

    datarec = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构
    # ?为什么呢？ 为什么过滤掉的信号反而是好的呢？
    final_data = np.array(data[mintime:maxtime]) - np.array(datarec[mintime:maxtime])
    # plot一下
    if Plot:
        plt.figure('Wavelet')
        plt.plot(data)
    return final_data.tolist()



class KalmanFilter(object):
    """
    https://github.com/zziz/kalman-filter
    """
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P),
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)




# 移动平均滤波
def M_A_Filter(data):
    """
    输入是ndarry，行信号
    输出还要ndarry，行信号
    """
    data_final = np.zeros_like(data)
    for i in range(data.shape[0]):
        data_final[i] = np.array(moveAverageFilter(data[i].tolist()))
    return data_final

# 移动平均滤波, 滤波器长度=3，并去直流，相当于抑制高频？
def moveAverageFilter(data):
    """
    输出是?
    输入是list?
    """
    data1 = []
    j = 0
    while j < len(data):
        a = data[j] + data[j + 1] + data[j + 2] if j + 1 < (len(data) - 1) else data[j] * 3  # 语法?，最后两个元素不变了
        data1.append(a / 3)
        j += 1
    data = data1
    # 去直流
    direct_current = np.mean(data)  # 对移动平均滤波后的数据取平均
    data1 = [j - direct_current for j in data]  # ?
    data = data1
    return data



# 滑动窗口滤波算法？归一化，在0坐标轴附近震荡
def sliding_window_demean(signal_values, num_windows):
    """
    输入类型
    """
    window_size = int(round(len(signal_values) / num_windows))
    demeaned = np.zeros(signal_values.shape)
    for i in range(0, len(signal_values), window_size):
        if i + window_size > len(signal_values):
            window_size = len(signal_values) - i
        curr_slice = signal_values[i: i + window_size]
        demeaned[i:i + window_size] = curr_slice - np.mean(curr_slice)
    return demeaned



# 全局自相关滤波，用自相关处理包含随机噪声的信号。应该是出自：
# RealSense = Real Heart Rate: Illumination Invariant Heart Rate Estimation from Videos
def globalSelfsimilarityFilter(data):
    acf1 = np.correlate(data, data, mode='full')  # np.correlate用于计算两个一维序列的互相关
    acf1 = acf1[len(data) - 1:]
    acf1 = acf1 / np.arange(len(data), 0, -1)
    acf1 = acf1 / acf1[0]
    data = acf1
    return data.tolist()



def amplitudeSelectiveFiltering(C_rgb, amax=0.002, delta=0.0001):
    """
    幅值滤波
    https://github.com/PartyTrix/Amplitude_selective_filtering
    Input: Raw RGB signals with dimensions 3xL, where the R channel is column 0
    Output:
    C = Filtered RGB-signals with added global mean,
    raw = Filtered RGB signals
    """
    C_rgb  = np.array(C_rgb)
    C_rgb = np.expand_dims(C_rgb, axis=1)

    s = C_rgb.shape[1]  # 列数，即时间序列长度
    c = (1 / (np.mean(C_rgb, 1)))  # 对各行求均值，再求倒数， 是一维数组了，长度为3

    # line 1
    c = np.transpose(np.array([c, ] * s)) * C_rgb - 1  # 类似对每个时序信号进行归一化

    # line 2
    f = abs(np.fft.fft(c, n=s, axis=1) / s)  # L -> C_rgb.shape[0],对每一行分别进行FFT变换，再除以时序信号长度，最后取绝对值

    # line 3
    w = (delta / np.abs(f[0, :]))  # F[0,:]  is the R-channel ，红色通道取倒数再乘以🔺，shape和type是？

    # line 4
    w[np.abs(f[0, :] < amax)] = 1  # 小于最大值的取1
    w = w.reshape([1, s])

    # line 5
    # ff = np.multiply(f, (np.tile(w, [3, 1])))  # 复制三行，再与f逐元素相乘
    ff = np.multiply(f, (np.tile(w, [1, 1])))  # 复制三行，再与f逐元素相乘

    # line 6
    c = np.transpose(np.array([(np.mean(C_rgb, 1)), ] * s)) * np.abs(np.fft.ifft(ff) + 1)  # 均值array与 FFT逆变换加1的值 逐元素相乘
    raw = np.abs(np.fft.ifft(ff) + 1)

    return c.T, raw.T  # 转置后再输出



# 选择最具有周期性的信号，即功率谱密度峰值点横坐标对应的频率
def signal_selection(s):
    """
    心率是通过选择最具有周期信号的最大频率来计算的
    s：列信号
    """
    fs = 30
    percentages = np.zeros(5)  # 一维度，5个0
    # 本文只分析了前5个信号
    for i in range(5):
        s_i = s[:, i]  # 果然还是列
        # 计算源信号的功率谱，return的是两个array，不知道维度 flattop是窗口类型，spectrum是功率谱密度
        f, Pxx_spec = signal.periodogram(s_i, fs, 'flattop', scaling='spectrum')  # f的shape一维len：251,范围0到15. Pxx_spec也是，但好像竖着排列
        total_power = np.sum(Pxx_spec)  # 计算总功率
        # 得到最大功率的频率及其一次谐波，还有它们的坐标索引
        [minPower, maxPower, minLoc, maxLoc] = cv2.minMaxLoc(Pxx_spec)  # ？
        freqMaxPower = f[maxLoc[1]]  # maxLoc是tuple类型(0, 17)
        firstHarmonic = 2 * freqMaxPower
        firstHarmonicLoc = np.where(f == firstHarmonic)
        firstHarmonicPower = Pxx_spec[firstHarmonicLoc]
        #  计算最大频率占总功率的百分比 取前五个
        percentages[i] = (maxPower + firstHarmonicPower) / total_power
    # 在五个分量中，从最具周期的信号来计算BPM
    most_periodic_signal = np.argmax(percentages)  # 返回最大值索引
    selected_signal = s[:, most_periodic_signal]  # 挑出来的最具有周期性的信号
    f, Pxx_spec = signal.periodogram(selected_signal, fs, 'flattop', scaling='spectrum')
    [minVal, maxVal, minLoc, maxLoc] = cv2.minMaxLoc(Pxx_spec)  # 获得最大功率最大值的索引？
    f_pulse = f[maxLoc[1]]
    bpm = 60.0 * f_pulse
    return bpm, selected_signal





# 找到频域最大值对应的频率，找最大两种方法结果竟然不一样
# 这个方法较差,可能是f精度不够
def get_HR(f_signal, samplerate):  # f, pxx_spec都是251长度一维ndarray
    f, pxx_spec = signal.periodogram(f_signal, samplerate, 'flattop', scaling='spectrum')
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(pxx_spec)
    hr = f[max_loc[1]]
    bpm = 60.0 * hr
    return bpm


# 信噪比
def SNR(data):
    pass


# 12均方根误差（RMSE）计算
def compute_RMSE_every_n(signal1, signal2, sampling_time, mean_time):
    signal1_m, time_m = strided_mean(signal1, sampling_time, mean_time)
    signal2_m, _ = strided_mean(signal2, sampling_time, mean_time)

    RMSE = np.sqrt(np.mean((signal1_m - signal2_m) ** 2))
    rRMSE = np.mean(np.abs(signal1_m - signal2_m) / signal1_m)
    return RMSE, rRMSE


# 均方根误差（RMSE）计算调用
def strided_mean(signal,sampling_time, mean_time):
   block_len = np.ceil(mean_time/sampling_time)
   n_blocks = np.floor(signal.shape[0]/block_len)
   mean_sig = np.zeros([int(n_blocks),1])
   mean_t =np.arange(n_blocks)*mean_time
   for i in np.arange(n_blocks):
       mean_sig[int(i)]= signal[int(i*block_len):int((i+1)*block_len),...].mean()
   return mean_sig, mean_t




def show_signal(data, Plot=False):
    color_name = ['r', 'g', 'b', 'c', 'm']
    level_name = ['level_e_mean', 'level_0_mean', 'level_1_mean', 'level_2_mean', 'level_3_mean']
    if Plot:
        plt.figure("original regions_mean")
        x = np.arange(0, data.shape[1])  # 返回一个有终点和起点的固定步长的排列做x轴
        for i in range(data.shape[0]):
            # plt.plot(x, data[i, :], color=color_name[i], label=level_name[i])  # 绘制第i行,并贴出标签
            plt.plot(x, data[i, :])  # 绘制第i行,并贴出标签
        # plt.legend()
        plt.title("original regions_mean")
        # plt.show()
        # cv2.waitKey(10000)


# 2 平滑先验法 (Smoothness Priors Approach, SPA)|param mu: 正则化系数|去除趋势，相当于去除低频
# 芬兰库奥皮奥大学的Karjalainen博士提出的一种信号非线性去趋势方法：https://m.hanspub.org/journal/paper/28723
def SPA_detrending1(data, mu=1200):
    """
    平滑先验法去除趋势 (Smoothness Priors Approach, SPA),庆麟后来的
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
    return data



# FFT变换,计算脉搏率并展示结果
def fftTransfer1(filtered, framerate=30):  # 输入数据和帧率:信号抽样率
    n = 512
    if len(filtered) < n:
        for _ in range(n - len(filtered)):  # 补零补至N点
            filtered.append(0)
    else:
        filtered = filtered[0:n]
    df = [framerate / n * i for i in range(n)]  # 频谱分辨率？
    fft_data = np.abs(np.fft.fft(filtered))

    hr = df[fft_data.tolist()[0:100].index(max(fft_data.tolist()[0:100]))] * 60  # 峰值横坐标
    print('HR estimate:', hr)
    return hr