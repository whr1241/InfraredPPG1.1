'''
Author: whr1241 2735535199@qq.com
Date: 2023-03-13 10:53:52
LastEditors: whr1241 2735535199@qq.com
LastEditTime: 2023-03-25 05:42:12
FilePath: \InfraredPPG1.1\4test.py
Description: ECG图绘制
'''
import random
import pandas as pd
import matplotlib.pyplot as plt 
import math
import numpy as np
from biosppy.signals import ecg
import h5py

filename = 'output/EMDFinalBPM.h5'
h5f = h5py.File(filename, 'r+')
print(h5f.keys())
# print(h5f['video01front'][:])
# h5f.__delitem__('real01front')
# h5f.__delitem__('video01front')
# print(h5f.keys())
h5f.close()


