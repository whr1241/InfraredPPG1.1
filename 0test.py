import matplotlib.pyplot as plt
import pywt
import numpy as np
from PyEMD import EEMD, EMD, Visualisation


# 实验NeuroKit信号处理包 
# https://neurokit2.readthedocs.io/en/latest/introduction.html#photoplethysmography-ppg-bvp
# ippg38环境暂时存在问题
# Generate 15 seconds of PPG signal (recorded at 250 samples / second)
ppg = nk.ppg_simulate(duration=15, sampling_rate=250, heart_rate=70)

# Process it
signals, info = nk.ppg_process(ppg, sampling_rate=250)

# Visualize the processing
nk.ppg_plot(signals, sampling_rate=250)

