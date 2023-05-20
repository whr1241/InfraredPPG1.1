import numpy as np
import signal_tools as stools
import matplotlib.pyplot as plt
from biosppy.signals import ecg

# data = np.load(r"output\video_signal3\01front.npy")
# Plot = True
# # show 原始时间数据
# stools.show_signal(data, Plot)
# plt.show()
# print('hello')

ecgdata = np.loadtxt(r"D:\1maydaystudy\0Github\ecg\07\front_ecg.txt")
ecg_signal = ecgdata[:, 0]  # type? 应该是list
ecg_signal = ecg_signal[1000*1:]
out = ecg.ecg(ecg_signal, sampling_rate=1000., show=True)  # biosppy库功能 Tuple,应该是默认采样率1000
times = out['heart_rate_ts']   # times是时间，长176
bpm = out['heart_rate']  # 实时心率，对应时间的心率，长176

data = np.load(r"output\video_signal3\01front.npy")
