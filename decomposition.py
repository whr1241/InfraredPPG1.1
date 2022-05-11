#!/usr/bin/env python
# -*- coding  : utf-8 -*-
# @Time       : 2022/3/11 22:14
# @Author     : wanghaoran
# @Site       : SCNU
# @File       : decomposition.py
# @Description: 各种降维函数
# @Software   : PyCharm
import cv2
import scipy
import pywt
import numpy as np
# import Utils.sq_stft_utils as sq  # 导入同级自定义python模块
import matplotlib.pyplot as plt
from scipy.signal import butter  #
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from PyEMD import EEMD, EMD, Visualisation
"""
用来放各种降维函数
# todo:ICA
PCA
SVD
# todo:EEMD
等
"""
# 3主成分分析,得到一维主位置信号(降维)
def PCA_compute(data, Plot=False):
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
    # plot出来
    if Plot:
        plt.figure("PCA")
        N = data.shape[0]
        for n, p in enumerate(data):
            plt.subplot(N,1,n+1)
            plt.plot(p, 'g')
            plt.title("PCA "+str(n))
            plt.xlabel("frames")
    # x = np.arange(0, data.shape[1])  
    # for i in range(data.shape[0]):
    #     plt.plot(x, data[i, :])
    return projected



# 3主成分分析,得到一维主位置信号(降维)
def PCA_compute1(data):
    """
    Perform PCA on the time series data and project the points onto the
    principal axes of variation (eigenvectors of covariance matrix) to get
    the principal 1-D position signals
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



# SVD奇异值分解 输入是数组array，如shape是(420, 2000).使用方法参考杜克大学的工程
def optimal_svd(Y):
    # OPTIMAL SHRINKAGE SVD
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
    # trim U,V
    aux = s_eta > 0
    return U[:, aux], s_eta[aux], V[aux, :]
    # 输出是U,s_eta,V三个分解值，参考杜克大学近红外那篇

# 奇异值分解SVD
def optimal_svd1(Y):
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



# 经验模态分解
def EMD_compute(data, Plot=False):
    """
    EMD是输入一维信号array，分解为多个信号
    可以考虑一下输出裁剪一下
    """
    IMF = EMD().emd(data)  # 返回array(7, 5368)
    # 画出图
    if Plot:
        N = IMF.shape[0] + 1  
        plt.figure('EMD')
        plt.subplot(N,1,1)
        plt.plot(data, 'r')
        plt.title("Input signal")
        plt.xlabel("Time [s]")
        for n, imf in enumerate(IMF):
            plt.subplot(N,1,n+2)
            plt.plot(imf, 'g')
            plt.title("IMF "+str(n+1))
            plt.xlabel("Time [s]")
        plt.tight_layout()  # 自动调整子图间距
    return IMF  # n个行信号array



# 扩展的经验模态分解
def EEMD_compute(data, Plot=False):
    """
    输入：一维array
    输出：二维行信号array
    """
    eemd = EEMD() 
    E_IMFs = eemd.eemd(data)

    if Plot:
        N = E_IMFs.shape[0] + 1  
        plt.figure('EEMD')
        plt.subplot(N,1,1)
        plt.plot(data, 'r')
        plt.title("Input signal")
        plt.xlabel("Time [s]")
        for n, imf in enumerate(E_IMFs):
            plt.subplot(N,1,n+2)
            plt.plot(imf, 'g')
            plt.title("IMF "+str(n+1))
            plt.xlabel("Time [s]")
        plt.tight_layout()  # 自动调整子图间距

    return E_IMFs  # 行信号array

# 希尔伯特-黄变换获得时频谱图




# 短时傅里叶变换
def STFT(data):
    pass

def stftAnal(x, w, N, H):
    """
    从时域到频域
    x: 输入信号, w: 分析窗, N: FFT 的大小, H: hop 的大小
    返回 xmX, xpX: 振幅和相位，以 dB 为单位
    """

    M = w.size                                      # 分析窗的大小
    hM1 = (M+1)//2                                  
    hM2 = M//2                                      
    x = np.append(np.zeros(hM2),x)                  # 在信号 x 的最前面与最后面补零
    x = np.append(x,np.zeros(hM2))                  
    pin = hM1                                       # 初始化指针，用来指示现在指示现在正在处理哪一帧
    pend = x.size-hM1                               # 最后一帧的位置
    w = w / sum(w)                                  # 归一化分析窗
    xmX = []                                        
    xpX = []                                        
    while pin<=pend:                                    
        x1 = x[pin-hM1:pin+hM2]                     # 选择一帧输入的信号
        mX, pX = dftAnal(x1, w, N)              # 计算 DFT（这个函数不是库中的）
        xmX.append(np.array(mX))                    # 添加到 list 中
        xpX.append(np.array(pX))
        pin += H                                    # 更新指针指示的位置
    xmX = np.array(xmX)                             # 转换为 numpy 数组
    xpX = np.array(xpX)
    return xmX, xpX

def stftSynth(mY, pY, M, H) :
    """
    从频域到时域
    mY: 振幅谱以dB为单位, pY: 相位谱, M: 分析窗的大小, H: hop 的大小
    返回 y 还原后的信号
    """
    hM1 = (M+1)//2                                   
    hM2 = M//2                                       
    nFrames = mY[:,0].size                           # 计算帧的数量
    y = np.zeros(nFrames*H + hM1 + hM2)              # 初始化输出向量
    pin = hM1                  
    for i in range(nFrames):                         # 迭代所有帧     
        y1 = dftSynth(mY[i,:], pY[i,:], M)         # 计算IDFT（这个函数不是库中的）
        y[pin-hM1:pin+hM2] += H*y1                     # overlap-add
        pin += H                                       # pin是一个指针，用来指示现在指示现在正在处理哪一帧
    y = np.delete(y, range(hM2))                     # 删除头部在stftAnal中添加的部分
    y = np.delete(y, range(y.size-hM1, y.size))      # 删除尾部在stftAnal中添加的部分
    return y

def dftAnal(x, w, N):
	"""
	Analysis of a signal using the discrete Fourier transform
	x: input signal, w: analysis window, N: FFT size 
	returns mX, pX: magnitude and phase spectrum
	"""

	if not(UF.isPower2(N)):                                 # raise error if N not a power of two
		raise ValueError("FFT size (N) is not a power of 2")

	if (w.size > N):                                        # raise error if window size bigger than fft size
		raise ValueError("Window size (M) is bigger than FFT size")

	hN = (N//2)+1                                           # size of positive spectrum, it includes sample 0
	hM1 = (w.size+1)//2                                     # half analysis window size by rounding
	hM2 = w.size//2                                         # half analysis window size by floor
	fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
	w = w / sum(w)                                          # normalize analysis window
	xw = x*w                                                # window the input sound
	fftbuffer[:hM1] = xw[hM2:]                              # zero-phase window in fftbuffer
	fftbuffer[-hM2:] = xw[:hM2]        
	X = fft(fftbuffer)                                      # compute FFT
	absX = abs(X[:hN])                                      # compute ansolute value of positive side
	absX[absX<np.finfo(float).eps] = np.finfo(float).eps    # if zeros add epsilon to handle log
	mX = 20 * np.log10(absX)                                # magnitude spectrum of positive frequencies in dB
	X[:hN].real[np.abs(X[:hN].real) < tol] = 0.0            # for phase calculation set to 0 the small values
	X[:hN].imag[np.abs(X[:hN].imag) < tol] = 0.0            # for phase calculation set to 0 the small values         
	pX = np.unwrap(np.angle(X[:hN]))                        # unwrapped phase spectrum of positive frequencies
	return mX, pX

def dftSynth(mX, pX, M):
	"""
	Synthesis of a signal using the discrete Fourier transform
	mX: magnitude spectrum, pX: phase spectrum, M: window size
	returns y: output signal
	"""

	hN = mX.size                                            # size of positive spectrum, it includes sample 0
	N = (hN-1)*2                                            # FFT size
	if not(UF.isPower2(N)):                                 # raise error if N not a power of two, thus mX is wrong
		raise ValueError("size of mX is not (N/2)+1")

	hM1 = int(math.floor((M+1)/2))                          # half analysis window size by rounding
	hM2 = int(math.floor(M/2))                              # half analysis window size by floor
	fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
	y = np.zeros(M)                                         # initialize output array
	Y = np.zeros(N, dtype = complex)                        # clean output spectrum
	Y[:hN] = 10**(mX/20) * np.exp(1j*pX)                    # generate positive frequencies
	Y[hN:] = 10**(mX[-2:0:-1]/20) * np.exp(-1j*pX[-2:0:-1]) # generate negative frequencies
	fftbuffer = np.real(ifft(Y))                            # compute inverse FFT
	y[:hM2] = fftbuffer[-hM2:]                              # undo zero-phase window
	y[hM2:] = fftbuffer[:hM1]
	return y

