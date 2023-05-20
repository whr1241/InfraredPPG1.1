'''
Author: whr1241 2735535199@qq.com
Date: 2023-03-20 14:31:42
LastEditors: whr1241 2735535199@qq.com
LastEditTime: 2023-03-30 19:52:44
FilePath: \InfraredPPG1.1\7test.py
Description: 将各个评价指标画出箱线图
'''

import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#第三章方法的MAE
# GreenMAE1 = [0.98, 1.67, 1.20, 27.46, 1.18, 2.31, 1.07, 4.22, 30.73, 1.09, 5.21, 6.22]
# EEMDMAE1 = [0.99, 1.54, 1.20, 3.07, 0.82, 1.22, 0.66, 0.68, 2.23, 0.75, 1.42, 0.96]
# data = np.array([GreenMAE1, EEMDMAE1]).T
# 第四章方法的MAE
EMDMAE1 = [0.58, 0.68, 0.59, 0.56, 0.60, 0.62, 0.68, 0.54, 0.58, 0.68, 0.68, 1.09]
EMDMAE2 = [0.50, 1.60, 2.03, 5.29, 3.84, 2.57, 1.49, 2.77, 1.43, 1.24, 1.05, 0.89]
OurMAE1 = [0.63, 1.07, 0.81, 0.75, 0.92, 0.89, 1.04, 0.76, 0.74, 0.89, 0.97, 1.29]
OurMAE2 = [0.80, 0.79, 0.82, 1.10, 1.03, 0.90, 1.05, 0.90, 0.86, 0.76, 0.84, 0.77]
data = np.array([EMDMAE1+EMDMAE2, OurMAE1+OurMAE2]).T

# Create box plot
labels = ['EMD', 'Our']
fig, ax = plt.subplots()
bplot = ax.boxplot(data, notch=False, vert=True, patch_artist=True, labels=labels)  # 

# Set labels and title and grid
# ax.set_xlabel('Variable')
ax.set_ylabel('Value')
ax.set_title('Box plot')
ax.yaxis.grid(True)  

# fill with colors
colors = ['green', 'red']
markers = ['o', 'o']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
for flier, co, maker in zip(bplot['fliers'], colors, markers):
    flier.set(marker =maker, 
              color =co, 
              alpha = 0.8) 
# Show plot
plt.show()

