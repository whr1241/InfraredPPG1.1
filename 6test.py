'''
Author: whr1241 2735535199@qq.com
Date: 2023-03-21 19:55:39
LastEditors: whr1241 2735535199@qq.com
LastEditTime: 2023-03-25 05:04:53
FilePath: \InfraredPPG1.1\6test.py
Description: 分别读取两个数据集，画出散点图等
'''

import numpy as np
import h5py
import matplotlib.pyplot as plt
import evaluate_tools as etools


# 读取第四章数据静止数据集的数据
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

# 读取第四章数据体动数据集的数据
# h5f = h5py.File('output/FinalBPM2.h5', 'r')
# print(h5f.keys())
# real_BPM = np.concatenate((h5f['realsubject1.0'][:], h5f['realsubject1.1'][:], h5f['realsubject2.2'][:], h5f['realsubject3.1'][:],
#                            h5f['realsubject4.2'][:], h5f['realsubject5.2'][:], h5f['realsubject6.2'][:], h5f['realsubject7.0'][:],
#                            h5f['realsubject8.0'][:], h5f['realsubject9.0'][:], h5f['realsubject10.1'][:], h5f['realsubject10.2'][:]), axis=0)

# video_BPM = np.concatenate((h5f['videosubject1.0'][:], h5f['videosubject1.1'][:], h5f['videosubject2.2'][:], h5f['videosubject3.1'][:],
#                            h5f['videosubject4.2'][:], h5f['videosubject5.2'][:], h5f['videosubject6.2'][:], h5f['videosubject7.0'][:],
#                            h5f['videosubject8.0'][:], h5f['videosubject9.0'][:], h5f['realsubject10.1'][:], h5f['realsubject10.2'][:]), axis=0)
# print(video_BPM)
# h5f.close()

# 读取第三章没用EEMD的数据
# h5f = h5py.File('output/FinalBPM3.h5', 'r')
# print(h5f.keys())
# real_BPM = np.concatenate((h5f['realFace01front'][:], h5f['realFace04front'][:], h5f['realFace06front'][:], h5f['realFace07front'][:],
#                            h5f['realFace09front'][:], h5f['realFace10front'][:], h5f['realFace11front'][:], h5f['realFace12front'][:],
#                            h5f['realFace13front'][:], h5f['realFace15front'][:], h5f['realFace16front'][:], h5f['realFace17front'][:]), axis=0)

# video_BPM = np.concatenate((h5f['videoFace01front'][:], h5f['videoFace04front'][:], h5f['videoFace06front'][:], h5f['videoFace07front'][:],
#                            h5f['videoFace09front'][:], h5f['videoFace10front'][:], h5f['videoFace11front'][:], h5f['videoFace12front'][:],
#                            h5f['videoFace13front'][:], h5f['videoFace15front'][:], h5f['videoFace16front'][:], h5f['videoFace17front'][:]), axis=0)
# print(video_BPM)
# h5f.close()

# # 读取第三章使用了EEMD的数据
# h5f = h5py.File('output/FinalBPM3.h5', 'r')
# print(h5f.keys())
# real_BPM = np.concatenate((h5f['realEEMDFace01front'][:], h5f['realEEMDFace04front'][:], h5f['realEEMDFace06front'][:], h5f['realEEMDFace07front'][:],
#                            h5f['realEEMDFace09front'][:], h5f['realEEMDFace10front'][:], h5f['realEEMDFace11front'][:], h5f['realEEMDFace12front'][:],
#                            h5f['realEEMDFace13front'][:], h5f['realEEMDFace15front'][:], h5f['realEEMDFace16front'][:], h5f['realEEMDFace17front'][:]), axis=0)

# video_BPM = np.concatenate((h5f['videoEEMDFace01front'][:], h5f['videoEEMDFace04front'][:], h5f['videoEEMDFace06front'][:], h5f['videoEEMDFace07front'][:],
#                            h5f['videoEEMDFace09front'][:], h5f['videoEEMDFace10front'][:], h5f['videoEEMDFace11front'][:], h5f['videoEEMDFace12front'][:],
#                            h5f['videoEEMDFace13front'][:], h5f['videoEEMDFace15front'][:], h5f['videoEEMDFace16front'][:], h5f['videoEEMDFace17front'][:]), axis=0)
# print(video_BPM)
# h5f.close()


# 第三章EMD方法DATASET1
# h5f = h5py.File('output/EMDFinalBPM.h5', 'r')
# print(h5f.keys())
# real_BPM = np.concatenate((h5f['real01front'][:], h5f['real04front'][:], h5f['real06front'][:], h5f['real07front'][:],
#                            h5f['real09front'][:], h5f['real10front'][:], h5f['real11front'][:], h5f['real12front'][:],
#                            h5f['real13front'][:], h5f['real15front'][:], h5f['real16front'][:], h5f['real17front'][:]), axis=0)

# video_BPM = np.concatenate((h5f['video01front'][:], h5f['video04front'][:], h5f['video06front'][:], h5f['video07front'][:],
#                            h5f['video09front'][:], h5f['video10front'][:], h5f['video11front'][:], h5f['video12front'][:],
#                            h5f['video13front'][:], h5f['video15front'][:], h5f['video16front'][:], h5f['video17front'][:]), axis=0)
# print(video_BPM)
# h5f.close()


# 第三章EMD方法DATASET2
h5f = h5py.File('output/EMDFinalBPM.h5', 'r')
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