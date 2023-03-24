'''
Author: whr1241 2735535199@qq.com
Date: 2023-03-21 19:55:39
LastEditors: whr1241 2735535199@qq.com
LastEditTime: 2023-03-22 13:52:56
FilePath: \InfraredPPG1.1\6test.py
Description: 分别读取两个数据集，画出散点图等
'''

import numpy as np
import h5py
import matplotlib.pyplot as plt
import evaluate_tools as etools


# 读文件
# h5f = h5py.File('output/FinalBPM1.h5', 'r')
# print(h5f.keys())
# real_BPM = np.concatenate((h5f['real01front'][:], h5f['real04front'][:], h5f['real06front'][:], h5f['real07front'][:],
#                            h5f['real09front'][:], h5f['real10front'][:], h5f['real11front'][:], h5f['real12front'][:],
#                            h5f['real13front'][:], h5f['real15front'][:], h5f['real16front'][:], h5f['real17front'][:]), axis=0)


# video_BPM = np.concatenate((h5f['video01front'][:], h5f['video04front'][:], h5f['video06front'][:], h5f['video07front'][:],
#                            h5f['video09front'][:], h5f['video10front'][:], h5f['video11front'][:], h5f['video12front'][:],
#                            h5f['video13front'][:], h5f['video15front'][:], h5f['video16front'][:], h5f['video17front'][:]), axis=0)
# print(video_BPM)
# h5f.close()

h5f = h5py.File('output/FinalBPM2.h5', 'r')
print(h5f.keys())
real_BPM = np.concatenate((h5f['realsubject1.0'][:], h5f['realsubject1.1'][:], h5f['realsubject2.2'][:], h5f['realsubject3.1'][:],
                           h5f['realsubject4.2'][:], h5f['realsubject5.2'][:], h5f['realsubject6.2'][:], h5f['realsubject7.0'][:],
                           h5f['realsubject8.0'][:], h5f['realsubject9.0'][:], h5f['realsubject10.1'][:], h5f['realsubject10.2'][:]), axis=0)


video_BPM = np.concatenate((h5f['videosubject1.0'][:], h5f['videosubject1.1'][:], h5f['videosubject2.2'][:], h5f['videosubject3.1'][:],
                           h5f['videosubject4.2'][:], h5f['videosubject5.2'][:], h5f['videosubject6.2'][:], h5f['videosubject7.0'][:],
                           h5f['videosubject8.0'][:], h5f['videosubject9.0'][:], h5f['realsubject10.1'][:], h5f['realsubject10.2'][:]), axis=0)
print(video_BPM)
h5f.close()

etools.bland_altman_plot(video_BPM, real_BPM)


# scatter散点图
fig, ax = plt.subplots(1, 1)
# plt.figure("scatter")
plt.title("PPG ECG compare")
plt.xlabel("HR(bpm)")
plt.ylabel("HR(bpm)")
plt.scatter(video_BPM, real_BPM)  # 生成一个scatter散点图
ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls='--', c='k', label="1:1 line")
plt.show()