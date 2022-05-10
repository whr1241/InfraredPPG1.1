#!/usr/bin/env python
# -*- coding  : utf-8 -*-
# @Time       : 2022/5/1 11:20
# @Author     : wanghaoran
# @Site       : SCNU
# @File       : 1test.py
# @description: 
# @Software   : PyCharm
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

# stft
# 产生一个测试信号，振幅为2的正弦波，其频率在3kHZ缓慢调制，振幅以指数形式下降的白噪声
fs = 10e3
N = 1e5
amp = 2 * np.sqrt(2)
noise_power = 0.01 * fs / 2
time = np.arange(N)/float(fs)
mod = 500 * np.cos(2*np.pi*0.25*time)
carrier = amp * np.sin(2*np.pi*3e3*time+mod)
noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
noise *= np.exp(-time/5)
x = carrier + noise

# 计算并绘制STFT的大小
f, t, Zxx = signal.stft(x, fs, nperseg=1000)
plt.pcolormesh(t, f, np.abs(Zxx), vmin = 0, vmax = amp)
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
    