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
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import FastICA
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
def fftTransfer(data, N=1024):
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

    # 当信号很清晰时，主峰很明显，直接只取主峰好了
    # TODO：假峰有时候也会变得高出真峰很多倍
    # TODO：还是应该按照功率比值阈值来判断吧
    # if num_peak_list_Y_final[0] > num_peak_list_Y_final[1]*2:
    #     final_hr = [hr]

    # show一下
    plt.figure('FFT')
    plt.plot(df, fft_data)
    plt.axis([0, 5, 0, np.max(fft_data)*2])
    plt.title('Heart Rate estimate: {:.2f}'.format(hr))
    for ii in range(len(num_peak_list_Y_final)):  # 画出5个极值点
        plt.plot(num_peak_list_X_final[ii], num_peak_list_Y_final[ii],'*',markersize=10)
    plt.pause(0.1)	# pause 1 second
    plt.clf()		# clear the current figure

    return hr, final_hr



def SPA(data):
    """
    输入是ndarry，行信号
    输出还要ndarry，行信号
    """
    data_final = np.zeros_like(data)
    for i in range(data.shape[0]):
        data_final[i] = np.array(SPA_detrending(data[i].tolist()))
    return data_final

def SPA_detrending(data, mu=1200):
    """
    平滑先验法去除趋势 (Smoothness Priors Approach, SPA)
    # 芬兰库奥皮奥大学的Karjalainen博士提出的一种信号非线性去趋势方法：https://m.hanspub.org/journal/paper/28723
    :param mu: 正则化系数
    :return:
    输入是一维list
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



def Filter(data):
    """
    输入是ndarry，行信号
    输出还要ndarry，行信号
    """
    data_final = np.zeros_like(data)
    for i in range(data.shape[0]):
        data_final[i] = np.array(bandPassFilter(data[i].tolist()))
    return data_final

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



# IOU计算函数
def cal_iou(box1, box2):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return:
    """
    [[xmin1, ymin1], [xmax1, ymin1], [xmin1, ymax1], [xmax1, ymax1]] = box1[0, :, :]
    [[xmin2, ymin2], [xmax2, ymin2], [xmin2, ymax2], [xmax2, ymax2]] = box2[0, :, :]
    # xmin2, ymin2, xmax2, ymax2 = box2[0,:,:]
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # b1的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # b2的面积

    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    a1 = w * h  # C∩G的面积
    a2 = s1 + s2 - a1
    iou = a1 / a2  # iou = a1/ (s1 + s2 - a1)
    return iou



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




def Wavelet(ppg):
    """
    小波去噪 https://www.cnblogs.com/sggggr/p/12381164.html
    输入：
    输出：
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
    print("maximum level is " + str(maxlev))
    threshold = 0.01  # Threshold for filtering

    # Decompose into wavelet components, to the level selected:
    coeffs = pywt.wavedec(data, 'db8', level=maxlev)  # 将信号进行小波分解

    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))  # 将噪声滤波

    mintime = 0
    maxtime = mintime + len(data)

    datarec = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构
    final_data = np.array(data[mintime:maxtime]) - np.array(datarec[mintime:maxtime])

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



def ICA_compute(data):
    """
    输入：应该是array
    输出：
    """
    ica = FastICA(n_components=5)  # 得到几个输出?
    data = data.T  # 需要输入列信号
    u = ica.fit_transform(data)  # ica后也是列向量信号
    u = u.T
    return u  # 返回行信号array



# 移动平均滤波, 滤波器长度=3，并去直流，相当于抑制高频？
def moveAverageFilter(data):
    """
    输入是list?
    输出是
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


# arctan归一化
def arctan_Normalization(data):
    """
    输入：
    输出：
    """
    data = [x*4 for x in data]
    data = np.arctan(data)
    data = data.tolist()
    return data


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



# 5归一化
def arctanNormalization(self):
    self.data = [x*4 for x in self.data]
    self.data = np.arctan(self.data)
    self.data = self.data.tolist()

    plt.figure(2)  # 画图
    plt.plot(self.data)
    plt.title('arctan normalization')


# 3主成分分析,得到一维主位置信号(降维)
def PCA_compute(data):
    """
    Perform PCA on the time series data and project the points onto the
    principal axes of variation (eigenvectors of covariance matrix) to get
    the principal 1-D position signals
    输入是ndarray,行信号
    """
    # Object for principal component analysis主成分分析对象
    pca = PCA(n_components=5)
    temp = data.T  # Arrange the time series data in the format given in the paper
    l2_norms = np.linalg.norm(temp, ord=2, axis=1)  # Get L2 norms of each m_t
    # 抛弃在前25名中有L2标准的分数
    temp_with_abnormalities_removed = temp[l2_norms < np.percentile(l2_norms, 75)]
    # 拟合PCA模型
    pca.fit(temp_with_abnormalities_removed)
    # 将跟踪的点运动投影到主分量向量上
    projected = pca.transform(temp)
    return projected


# 10奇异值分解SVD
def optimal_svd(Y):
    ### OPTIMAL SHRINKAGE SVD
    # Implementation of algorithm proposed in:
    # based on the following paper: Gavish, Matan, and David L. Donoho. "Optimal shrinkage of singular values." IEEE Transactions on Information Theory 63.4 (2017): 2137-2152.
    #
    U, s, V = np.linalg.svd(Y, full_matrices=False)
    m, n = Y.shape
    beta = m / n

    y_med = np.median(s)

    beta_m = (1 - np.sqrt(beta)) ** 2
    beta_p = (1 + np.sqrt(beta)) ** 2

    t_array = np.linspace(beta_m, beta_p, 100000)
    dt = np.diff(t_array)[0]

    f = lambda t: np.sqrt((beta_p - t) * (t - beta_m)) / (2 * np.pi * t * beta)
    F = lambda t: np.cumsum(f(t) * dt)

    mu_beta = t_array[np.argmin((F(t_array) - 0.5) ** 2)]

    sigma_hat = y_med / np.sqrt(n * mu_beta)

    def eta(y, beta):
        mask = (y >= (1 + np.sqrt(beta)))
        aux_sqrt = np.sqrt((y[mask > 0] ** 2 - beta - 1) ** 2 - 4 * beta)
        aux = np.zeros(y.shape)
        aux[mask > 0] = aux_sqrt / y[mask > 0]
        return mask * aux

    def eta_sigma(y, beta, sigma):
        return sigma * eta(y / sigma, beta)

    s_eta = eta_sigma(s, beta, sigma_hat * np.sqrt(n))

    #     trim U,V
    aux = s_eta > 0

    return U[:, aux], s_eta[aux], V[aux, :]


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


# 3FFT变换,计算脉搏率并展示结果
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



# 找到频域最大值对应的频率，找最大两种方法结果竟然不一样
# 这个方法较差,可能是f精度不够
def get_HR(f_signal, samplerate):  # f, pxx_spec都是251长度一维ndarray
    f, pxx_spec = signal.periodogram(f_signal, samplerate, 'flattop', scaling='spectrum')
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(pxx_spec)
    hr = f[max_loc[1]]
    bpm = 60.0 * hr
    return bpm


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

