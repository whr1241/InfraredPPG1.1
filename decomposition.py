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
def PCA_compute(data):
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


def EMD_compute(data):
    """
    EMD是输入一维信号array，分解为多个信号
    可以考虑一下输出裁剪一下
    """
    IMF = EMD().emd(data)  # 返回array(7, 5368)
    
    # N = IMF.shape[0] + 1  
    # Plot results
    # plt.figure('EMD')
    # plt.subplot(N,1,1)
    # plt.plot(data, 'r')
    # plt.title("Input signal")
    # plt.xlabel("Time [s]")
    
    # for n, imf in enumerate(IMF):
    #     plt.subplot(N,1,n+2)
    #     plt.plot(imf, 'g')
    #     plt.title("IMF "+str(n+1))
    #     plt.xlabel("Time [s]")

    # plt.tight_layout()

    return IMF  # n个行信号array



def EEMD_compute(data):
    """
    输入：一维array
    输出：二维行信号array
    """
    eemd = EEMD() 
    E_IMFs = eemd.eemd(data)

    return E_IMFs  # 行信号array
