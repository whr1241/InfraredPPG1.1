'''
Author: whr1241 2735535199@qq.com
Date: 2023-03-13 10:53:52
LastEditors: whr1241 2735535199@qq.com
LastEditTime: 2023-03-14 14:22:05
FilePath: \InfraredPPG1.1\4test.py
Description: ECG图绘制
'''
import random
import pandas as pd
import matplotlib.pyplot as plt 
import math
import numpy as np
from biosppy.signals import ecg


ecgdata = np.loadtxt(r"I:\WHR\Dataset\1-Myself\5-haoran\ecg\subject3.1.txt")
ecg_signal = ecgdata[:, 0]  # type? 应该是list
ecg_signal = ecg_signal[1000*1:]
out = ecg.ecg(ecg_signal, sampling_rate=1000., show=True)  # biosppy库功能 Tuple,应该是默认采样率1000

