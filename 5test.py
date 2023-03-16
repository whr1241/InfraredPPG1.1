'''
Author: whr1241 2735535199@qq.com
Date: 2023-03-14 14:22:35
LastEditors: whr1241 2735535199@qq.com
LastEditTime: 2023-03-14 16:32:14
FilePath: \InfraredPPG1.1\5test.py
Description: 时频图绘制
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
# plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
plt.rc("font", family='Microsoft YaHei')# 增加

# 生成一个信号
t = np.linspace(0, 180, 180*30, endpoint=False)

x = np.load(r"output\video_signal\BVP_smooth_subject3.1.npy")[0]

# 使用滑动窗口分析信号
fs = 30  # 采样率
win = signal.windows.hann(50)  # 窗口函数
f, t, Sxx = signal.spectrogram(x, fs, window=win, nperseg=50, noverlap=25)

# 绘制3D时频图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(t, f)
ax.plot_surface(X, Y, 10*np.log10(Sxx), cmap='viridis')
ax.set_xlabel('时间 [sec]')
ax.set_ylabel('频率 [Hz]')
ax.set_zlabel('功率 [dB]')
plt.show()
